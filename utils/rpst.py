from copy import deepcopy

import torch
import torch.nn.functional as F

from utils.torch_utils import de_parallel

try:
    from torchvision.ops import roi_align as torchvision_roi_align
except Exception:
    torchvision_roi_align = None


class LocalEMATeacher:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.decay = float(decay)
        self.updates = 0
        self.ema.last_detect_features = None
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        student_state = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            source = student_state[k].detach().to(device=v.device)
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(source.to(dtype=v.dtype), alpha=1.0 - self.decay)
            else:
                v.copy_(source.to(dtype=v.dtype))


class RPSTLoss:
    def __init__(self, hyp, debug=False, rank=-1):
        self.hyp = hyp
        self.debug = debug
        self.rank = rank

    @staticmethod
    def _zero_stats(zero):
        return {
            'trajectory_loss': zero,
            'relation_loss': zero,
            'total_loss': zero,
            'weighted_loss': zero,
            'selected_objects': zero,
            'effective_weight': zero,
        }

    def _effective_weight(self, epoch):
        start = int(self.hyp.get('rpst_start_epoch', 10))
        warmup = max(float(self.hyp.get('rpst_warmup_epochs', 10)), 1e-6)
        progress = max(0.0, float(epoch - start + 1)) / warmup
        return float(self.hyp.get('rpst_loss_weight', 0.20)) * min(1.0, progress)

    def __call__(self, model, teacher, imgs, targets, epoch, ni):
        zero = imgs.new_zeros(())
        stats = self._zero_stats(zero)
        if teacher is None or not bool(self.hyp.get('rpst_enabled', False)):
            return zero, stats

        interval = max(int(self.hyp.get('rpst_interval', 4)), 1)
        if int(ni) % interval != 0:
            return zero, stats

        effective_weight = self._effective_weight(epoch)
        if effective_weight <= 0.0:
            return zero, stats

        student_model = de_parallel(model)
        student_features = getattr(student_model, 'last_detect_features', None)
        if student_features is None or len(student_features) == 0:
            return zero, stats
        student_features = self._sort_features(student_features)

        selected, selected_indices = self._select_tiny_objects(targets, imgs)
        if selected is None or selected.shape[0] == 0:
            return zero, stats
        stats['selected_objects'] = zero + float(selected.shape[0])

        crop_info = self._make_privileged_crops(imgs, selected, selected_indices)
        crop_images = crop_info['crop_images']
        if crop_images.shape[0] == 0:
            return zero, stats

        teacher_features = self._teacher_features(teacher, crop_images)
        if teacher_features is None or len(teacher_features) == 0:
            return zero, stats

        trajectory_loss, relation_loss = self._distill_losses(
            selected,
            crop_info,
            student_features,
            teacher_features,
            imgs.shape[-2:],
        )
        total_loss = (
            float(self.hyp.get('rpst_trajectory_weight', 1.0)) * trajectory_loss
            + float(self.hyp.get('rpst_relation_weight', 0.5)) * relation_loss
        )
        weighted_loss = total_loss * effective_weight
        stats.update({
            'trajectory_loss': trajectory_loss.detach(),
            'relation_loss': relation_loss.detach(),
            'total_loss': total_loss.detach(),
            'weighted_loss': weighted_loss.detach(),
            'effective_weight': zero + effective_weight,
        })
        return weighted_loss, stats

    def _select_tiny_objects(self, targets, imgs):
        if targets is None or targets.numel() == 0:
            return None, None

        bsz, _, img_h, img_w = imgs.shape
        device, dtype = imgs.device, imgs.dtype
        targets = targets.to(device=device, dtype=dtype)
        image_index = targets[:, 0].long()
        valid = (image_index >= 0) & (image_index < bsz)
        valid &= (targets[:, 4] > 0) & (targets[:, 5] > 0)
        if not valid.any():
            return None, None

        max_size = float(self.hyp.get('rpst_tiny_max_size', 32))
        max_per_image = int(self.hyp.get('rpst_max_objects_per_image', 2))
        max_per_batch = int(self.hyp.get('rpst_max_objects_per_batch', 16))
        if max_per_image <= 0 or max_per_batch <= 0:
            return None, None

        ids = torch.arange(targets.shape[0], device=device)
        width_pixels = targets[:, 4].clamp(min=0) * img_w
        height_pixels = targets[:, 5].clamp(min=0) * img_h
        area = width_pixels * height_pixels
        tiny = valid & (torch.maximum(width_pixels, height_pixels) <= max_size)

        selected_ids = []
        for b in range(bsz):
            image_mask = tiny & (image_index == b)
            image_ids = ids[image_mask]
            if image_ids.numel() == 0:
                continue
            order = torch.argsort(area[image_mask])
            selected_ids.append(image_ids[order[:max_per_image]])

        if not selected_ids:
            return None, None
        selected_ids = torch.cat(selected_ids, dim=0)
        if selected_ids.numel() > max_per_batch:
            order = torch.argsort(area[selected_ids])
            selected_ids = selected_ids[order[:max_per_batch]]
        return targets[selected_ids], selected_ids

    @staticmethod
    def _xywhn_to_xyxy(targets, image_hw):
        img_h, img_w = image_hw
        scale_xy = targets.new_tensor([float(img_w), float(img_h)])
        center = targets[:, 2:4] * scale_xy
        wh = targets[:, 4:6].clamp(min=0) * scale_xy
        xyxy = torch.cat((center - wh * 0.5, center + wh * 0.5), dim=1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, float(img_w))
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, float(img_h))
        return xyxy

    def _make_privileged_crops(self, imgs, selected, selected_indices):
        bsz, _, img_h, img_w = imgs.shape
        device, dtype = imgs.device, imgs.dtype
        crop_size = max(int(self.hyp.get('rpst_crop_size', 192)), 1)
        context = max(float(self.hyp.get('rpst_crop_context', 2.0)), 1e-6)

        source_image_indices = selected[:, 0].long().clamp(0, bsz - 1)
        boxes = self._xywhn_to_xyxy(selected, (img_h, img_w))
        bw = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
        side = (context * torch.maximum(bw, bh)).clamp_min(1.0)
        cx = selected[:, 2].clamp(0, 1) * img_w
        cy = selected[:, 3].clamp(0, 1) * img_h

        base = (torch.arange(crop_size, device=device, dtype=dtype) + 0.5) / crop_size - 0.5
        grid_x = cx.view(-1, 1, 1) + side.view(-1, 1, 1) * base.view(1, 1, -1)
        grid_y = cy.view(-1, 1, 1) + side.view(-1, 1, 1) * base.view(1, -1, 1)
        grid_x = grid_x.expand(-1, crop_size, crop_size)
        grid_y = grid_y.expand(-1, crop_size, crop_size)
        grid_x = (2.0 * grid_x + 1.0) / float(img_w) - 1.0
        grid_y = (2.0 * grid_y + 1.0) / float(img_h) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)

        crop_images = F.grid_sample(
            imgs[source_image_indices],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        crop_center = selected.new_full((selected.shape[0],), crop_size * 0.5)
        crop_w = (bw / side * crop_size).clamp(1.0, float(crop_size))
        crop_h = (bh / side * crop_size).clamp(1.0, float(crop_size))
        crop_boxes = torch.stack((
            crop_center - crop_w * 0.5,
            crop_center - crop_h * 0.5,
            crop_center + crop_w * 0.5,
            crop_center + crop_h * 0.5,
        ), dim=1)
        crop_boxes[:, 0::2] = crop_boxes[:, 0::2].clamp(0, float(crop_size))
        crop_boxes[:, 1::2] = crop_boxes[:, 1::2].clamp(0, float(crop_size))

        return {
            'crop_images': crop_images,
            'crop_boxes': crop_boxes,
            'source_image_indices': source_image_indices,
            'source_target_indices': selected_indices,
            'crop_scales': crop_size / side,
            'original_crop_sides': side,
        }

    def _teacher_features(self, teacher, crop_images):
        batch_size = max(int(self.hyp.get('rpst_teacher_batch_size', 16)), 1)
        features_by_level = None
        teacher.eval()
        with torch.no_grad():
            for start in range(0, crop_images.shape[0], batch_size):
                _ = teacher(crop_images[start:start + batch_size])
                features = getattr(teacher, 'last_detect_features', None)
                if features is None or len(features) == 0:
                    return None
                features = self._sort_features(features)
                if features_by_level is None:
                    features_by_level = [[] for _ in features]
                if len(features) != len(features_by_level):
                    return None
                for i, feature in enumerate(features):
                    features_by_level[i].append(feature.detach())
        return [torch.cat(chunks, dim=0) for chunks in features_by_level] if features_by_level else None

    @staticmethod
    def _sort_features(features):
        return tuple(sorted(list(features), key=lambda x: (x.shape[-2] * x.shape[-1], x.shape[-2]), reverse=True))

    def _roi_pool(self, feature, boxes, batch_indices, image_hw):
        roi_size = max(int(self.hyp.get('rpst_roi_size', 3)), 1)
        if boxes.numel() == 0:
            return feature.new_zeros((0, feature.shape[1], roi_size, roi_size))

        img_h, img_w = image_hw
        feat_h, feat_w = feature.shape[-2:]
        scale = boxes.new_tensor([
            feat_w / float(img_w),
            feat_h / float(img_h),
            feat_w / float(img_w),
            feat_h / float(img_h),
        ])
        feature_boxes = boxes * scale
        if torchvision_roi_align is not None:
            rois = torch.cat((batch_indices.to(dtype=boxes.dtype).view(-1, 1), feature_boxes), dim=1)
            try:
                return torchvision_roi_align(
                    feature,
                    rois,
                    output_size=(roi_size, roi_size),
                    spatial_scale=1.0,
                    sampling_ratio=-1,
                    aligned=True,
                )
            except Exception:
                pass
        return self._roi_pool_fallback(feature, feature_boxes, batch_indices, roi_size)

    @staticmethod
    def _roi_pool_fallback(feature, feature_boxes, batch_indices, roi_size):
        _, channels, feat_h, feat_w = feature.shape
        pooled = []
        for i in range(feature_boxes.shape[0]):
            b = int(batch_indices[i].detach().item())
            x1 = int(torch.floor(feature_boxes[i, 0]).clamp(0, feat_w - 1).detach().item())
            y1 = int(torch.floor(feature_boxes[i, 1]).clamp(0, feat_h - 1).detach().item())
            x2 = int(torch.ceil(feature_boxes[i, 2]).clamp(0, feat_w).detach().item())
            y2 = int(torch.ceil(feature_boxes[i, 3]).clamp(0, feat_h).detach().item())
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            region = feature[b:b + 1, :, y1:y2, x1:x2]
            if region.numel() == 0:
                pooled.append(feature.new_zeros((channels, roi_size, roi_size)))
            else:
                pooled.append(F.adaptive_avg_pool2d(region, (roi_size, roi_size)).squeeze(0))
        return torch.stack(pooled, dim=0) if pooled else feature.new_zeros((0, channels, roi_size, roi_size))

    @staticmethod
    def _expanded_boxes(boxes, image_hw, scale):
        img_h, img_w = image_hw
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        bw = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0) * float(scale)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0) * float(scale)
        expanded = torch.stack((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5), dim=1)
        expanded[:, 0::2] = expanded[:, 0::2].clamp(0, float(img_w))
        expanded[:, 1::2] = expanded[:, 1::2].clamp(0, float(img_h))
        return expanded

    @staticmethod
    def _project64(x):
        if x.shape[1] == 64:
            return x
        return F.adaptive_avg_pool1d(x.unsqueeze(1), 64).squeeze(1)

    def _relation_direction(self, foreground, expanded):
        context = expanded - foreground
        direction = foreground - context
        return F.normalize(self._project64(direction), p=2, dim=1, eps=1e-6)

    def _distill_losses(self, selected, crop_info, student_features, teacher_features, image_hw):
        zero = selected.new_zeros(())
        levels = min(len(student_features), len(teacher_features))
        if levels == 0:
            return zero, zero

        img_h, img_w = image_hw
        crop_size = crop_info['crop_images'].shape[-1]
        student_boxes = self._xywhn_to_xyxy(selected, image_hw)
        teacher_boxes = crop_info['crop_boxes']
        context_scale = float(self.hyp.get('rpst_context_scale', 1.5))
        student_context_boxes = self._expanded_boxes(student_boxes, image_hw, context_scale)
        teacher_context_boxes = self._expanded_boxes(teacher_boxes, (crop_size, crop_size), context_scale)
        student_batch = selected[:, 0].long()
        teacher_batch = torch.arange(selected.shape[0], device=selected.device)

        student_energy, teacher_energy = [], []
        student_foreground, teacher_foreground = [], []
        student_expanded, teacher_expanded = [], []
        for level in range(levels):
            s_fg = self._roi_pool(student_features[level], student_boxes, student_batch, (img_h, img_w))
            t_fg = self._roi_pool(teacher_features[level], teacher_boxes, teacher_batch, (crop_size, crop_size))
            s_exp = self._roi_pool(student_features[level], student_context_boxes, student_batch, (img_h, img_w))
            t_exp = self._roi_pool(teacher_features[level], teacher_context_boxes, teacher_batch, (crop_size, crop_size))

            student_energy.append(s_fg.abs().mean(dim=(1, 2, 3)))
            teacher_energy.append(t_fg.abs().mean(dim=(1, 2, 3)).detach())
            student_foreground.append(s_fg.mean(dim=(2, 3)))
            teacher_foreground.append(t_fg.mean(dim=(2, 3)).detach())
            student_expanded.append(s_exp.mean(dim=(2, 3)))
            teacher_expanded.append(t_exp.mean(dim=(2, 3)).detach())

        student_energy = torch.stack(student_energy, dim=1)
        teacher_energy = torch.stack(teacher_energy, dim=1)
        temperature = max(float(self.hyp.get('rpst_temperature', 1.0)), 1e-6)
        deltas = torch.round(torch.log2(crop_info['crop_scales'].clamp_min(1e-6))).long()

        trajectory_losses, relation_losses = [], []
        for obj_i in range(selected.shape[0]):
            delta = int(deltas[obj_i].detach().item())
            # Positive delta maps shallow student levels to deeper teacher levels after privileged enlargement.
            valid_student_levels = [s for s in range(levels) if 0 <= s + delta < levels]
            if not valid_student_levels:
                continue

            s_idx = torch.tensor(valid_student_levels, device=selected.device, dtype=torch.long)
            t_idx = s_idx + delta
            log_q_student = F.log_softmax(student_energy[obj_i].index_select(0, s_idx) / temperature, dim=0)
            q_teacher = F.softmax(teacher_energy[obj_i].index_select(0, t_idx) / temperature, dim=0).detach()
            trajectory_losses.append(F.kl_div(log_q_student, q_teacher, reduction='sum'))

            for s_level, t_level in zip(valid_student_levels, t_idx.tolist()):
                direction_student = self._relation_direction(
                    student_foreground[s_level][obj_i:obj_i + 1],
                    student_expanded[s_level][obj_i:obj_i + 1],
                )
                direction_teacher = self._relation_direction(
                    teacher_foreground[t_level][obj_i:obj_i + 1],
                    teacher_expanded[t_level][obj_i:obj_i + 1],
                ).detach()
                relation_losses.append(1.0 - F.cosine_similarity(direction_student, direction_teacher, dim=1).mean())

        trajectory_loss = torch.stack(trajectory_losses).mean() if trajectory_losses else zero
        relation_loss = torch.stack(relation_losses).mean() if relation_losses else zero
        return trajectory_loss, relation_loss
