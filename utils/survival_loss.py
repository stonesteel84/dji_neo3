import torch
import torch.nn.functional as F

from utils.torch_utils import de_parallel


def _zero_from(model, reference=None):
    if reference is not None:
        return reference.new_zeros(())
    return next(de_parallel(model).parameters()).new_zeros(())


def survival_modules(model):
    return [
        m for m in de_parallel(model).modules()
        if m.__class__.__name__ == 'TinyObjectSurvivalDownsample'
    ]


def has_survival_modules(model):
    return len(survival_modules(model)) > 0


def build_survival_target(risk_logits, targets, tau=4.0):
    bsz, _, h, w = risk_logits.shape
    device, dtype = risk_logits.device, risk_logits.dtype
    target = torch.zeros((bsz, 1, h, w), device=device, dtype=dtype)
    if targets is None or targets.numel() == 0 or bsz == 0 or h == 0 or w == 0:
        return target

    targets = targets.to(device=device, dtype=dtype)
    image_index = targets[:, 0].long()
    valid = (image_index >= 0) & (image_index < bsz)
    valid &= (targets[:, 4] > 0) & (targets[:, 5] > 0)
    if not valid.any():
        return target

    targets = targets[valid]
    image_index = image_index[valid]
    grid_y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    grid_x = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
    tau = max(float(tau), 1e-6)

    for b in range(bsz):
        image_targets = targets[image_index == b]
        if image_targets.numel() == 0:
            continue

        cx = (image_targets[:, 2].clamp(0, 1) * w - 0.5).clamp(0, w - 1).view(-1, 1, 1)
        cy = (image_targets[:, 3].clamp(0, 1) * h - 0.5).clamp(0, h - 1).view(-1, 1, 1)
        bw = image_targets[:, 4].clamp(min=0) * w
        bh = image_targets[:, 5].clamp(min=0) * h
        area = (bw * bh).clamp_min(1e-6)
        risk = torch.exp(-area / tau).view(-1, 1, 1)
        sigma = (0.25 * torch.sqrt(area)).clamp(1, 3).view(-1, 1, 1)

        dist2 = (grid_x - cx).pow(2) + (grid_y - cy).pow(2)
        gaussian = risk * torch.exp(-dist2 / (2.0 * sigma.pow(2)))
        target[b, 0] = torch.maximum(target[b, 0], gaussian.max(dim=0).values)

    return target


def compute_survival_loss(model, targets, hyp, enabled=True, reference=None):
    zero = _zero_from(model, reference)
    if not enabled:
        return zero, 0

    losses = []
    for module in survival_modules(model):
        risk_logits = getattr(module, 'last_risk_logits', None)
        if risk_logits is None:
            continue
        risk_target = build_survival_target(risk_logits, targets, tau=hyp.get('survival_tau', 4.0))
        bce = F.binary_cross_entropy_with_logits(risk_logits, risk_target, reduction='none')
        prob = torch.sigmoid(risk_logits)
        pt = risk_target * prob + (1.0 - risk_target) * (1.0 - prob)
        focal_weight = (1.0 - pt).pow(float(hyp.get('survival_focal_gamma', 2.0)))
        losses.append((focal_weight * bce).mean())

    if not losses:
        return zero, 0
    return torch.stack(losses).mean(), len(losses)
