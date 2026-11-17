"""TS-RPST building blocks ported to the Ultralytics (YOLO26) module API.

Two inference-time modules from the YOLOv5 implementation in ``models/common.py``:

* ``TinyObjectSurvivalDownsample`` (TOSD): stride-2 downsampling with a PixelUnshuffle detail path gated by a
  learned tiny-object extinction-risk map (paper Eq. 1-2). The risk logits are kept on the module after every forward
  so the survival loss can supervise them (Eq. 3-5).
* ``UAFeatureUpsample``: guided local feature reassembly used in the top-down neck in place of ``nn.Upsample``.

Both are drop-in ``nn.Module`` classes that use ``ultralytics.nn.modules.Conv`` so that fusing, AMP, and export keep
working. They are inserted into a YOLO26 model by :mod:`tsrpst_yolo26.model`, which swaps them in after Ultralytics'
own parser has built the surrounding layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, autopad

__all__ = ["TinyObjectSurvivalDownsample", "UAFeatureUpsample"]


class TinyObjectSurvivalDownsample(nn.Module):
    """Stride-2 downsampling with a detail path gated by tiny-object extinction risk (TOSD)."""

    def __init__(self, c1, c2, kernel_size=3, stride=2, risk_channels=16, gamma_init=0.0):
        super().__init__()
        if int(stride) != 2:
            raise ValueError(f"TinyObjectSurvivalDownsample only supports stride=2, got {stride}.")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"TinyObjectSurvivalDownsample requires a positive odd kernel_size, got {kernel_size}.")
        risk_channels = max(int(risk_channels), 1)

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.semantic = Conv(self.c1, self.c2, self.kernel_size, self.stride)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.detail = Conv(self.c1 * 4, self.c2, 1, 1)
        self.risk_dw = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, 3, 2, autopad(3), groups=self.c1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(),
        )
        self.risk_pw = Conv(self.c1, risk_channels, 1, 1)
        self.risk_logits = nn.Conv2d(risk_channels, 1, 1, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.last_risk_logits = None
        self.last_risk_map = None

    def __getstate__(self):
        # Never pickle / deepcopy transient activations (EMA copies, checkpoints).
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        state["last_risk_logits"] = None
        state["last_risk_map"] = None
        return state

    def clear_transient(self):
        self.last_risk_logits = None
        self.last_risk_map = None

    @staticmethod
    def _pad_to_even(x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        return F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x

    def forward(self, x):
        x = self._pad_to_even(x)
        x_sem = self.semantic(x)
        x_det = self.detail(self.pixel_unshuffle(x))
        risk_logits = self.risk_logits(self.risk_pw(self.risk_dw(x)))
        risk_map = torch.sigmoid(risk_logits)

        if self.training:
            self.last_risk_logits = risk_logits
            self.last_risk_map = risk_map
        else:
            self.last_risk_logits = risk_logits.detach()
            self.last_risk_map = risk_map.detach()

        gamma = self.gamma.to(dtype=x_sem.dtype).reshape(1, 1, 1, 1)
        return x_sem + gamma * risk_map.to(dtype=x_sem.dtype) * x_det


class UAFeatureUpsample(nn.Module):
    """Guided local feature reassembly for top-down upsampling (replaces ``nn.Upsample`` in the neck).

    Takes ``[x_low, x_guide]`` where ``x_guide`` is a backbone feature at the target resolution and returns ``x_low``
    upsampled by ``scale_factor`` with content-aware neighbour weights. ``residual_init=0`` makes it exactly bilinear at
    initialisation, so the module can be dropped into a pretrained neck without changing its outputs.
    """

    _LOGIT_CLAMP = 30.0
    _MIN_SCALE = 1e-3
    _MAX_SCALE = 32.0
    _GUIDANCE_SCORE_WEIGHT = 0.5
    _DEFAULT_SPATIAL_SCALE = 1.0
    _DEFAULT_RANGE_SCALE = 1.0

    def __init__(
        self,
        c_low: int,
        c_guide: int,
        guide_dim: int = 16,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        scale_factor: int = 2,
        residual_init: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"UAFeatureUpsample requires a positive odd kernel_size, got {kernel_size}.")
        if scale_factor <= 0:
            raise ValueError(f"UAFeatureUpsample requires a positive scale_factor, got {scale_factor}.")
        if c_low <= 0 or c_guide <= 0 or guide_dim <= 0 or hidden_dim <= 0:
            raise ValueError("UAFeatureUpsample channel and embedding dimensions must be positive.")

        self.c_low = int(c_low)
        self.c_guide = int(c_guide)
        self.guide_dim = int(guide_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.scale_factor = int(scale_factor)
        self.num_neighbors = self.kernel_size * self.kernel_size
        self.eps = float(eps)

        self.guide_proj = nn.Sequential(
            nn.Conv2d(self.c_guide, self.hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.guide_dim, 1, 1, bias=True),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(self.c_low, self.hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.guide_dim, 1, 1, bias=True),
        )
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(self.c_guide, self.hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.num_neighbors, 1, 1, bias=True),
        )
        nn.init.zeros_(self.kernel_predictor[-1].weight)
        nn.init.zeros_(self.kernel_predictor[-1].bias)

        self.spatial_scale_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(self._DEFAULT_SPATIAL_SCALE), dtype=torch.float32)
        )
        self.range_scale_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(self._DEFAULT_RANGE_SCALE), dtype=torch.float32)
        )
        self.gamma = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))

        radius = self.kernel_size // 2
        self._offset_pairs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.register_buffer("offsets", torch.tensor(self._offset_pairs, dtype=torch.float32), persistent=False)

    @staticmethod
    def _inverse_softplus(value):
        return math.log(math.expm1(float(value)))

    def _positive_scale(self, raw):
        return F.softplus(raw.float()).clamp(min=self._MIN_SCALE, max=self._MAX_SCALE)

    @staticmethod
    def _shift_low(x, dy, dx):
        h, w = x.shape[-2:]
        pad_y = abs(int(dy))
        pad_x = abs(int(dx))
        padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
        start_y = pad_y + int(dy)
        start_x = pad_x + int(dx)
        return padded[..., start_y : start_y + h, start_x : start_x + w]

    @staticmethod
    def _check_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"UAFeatureUpsample {name} contains NaN or Inf values.")

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("UAFeatureUpsample expects a list or tuple: [x_low, x_guide].")
        x_low, x_guide = inputs
        if x_low.ndim != 4 or x_guide.ndim != 4:
            raise ValueError("UAFeatureUpsample inputs must be 4D tensors shaped [B, C, H, W].")
        if x_low.shape[0] != x_guide.shape[0]:
            raise ValueError(f"UAFeatureUpsample batch size mismatch: low={x_low.shape[0]}, guide={x_guide.shape[0]}.")
        if x_low.shape[1] != self.c_low:
            raise ValueError(f"UAFeatureUpsample expected x_low with {self.c_low} channels, got {x_low.shape[1]}.")
        if x_guide.shape[1] != self.c_guide:
            raise ValueError(f"UAFeatureUpsample expected x_guide with {self.c_guide} channels, got {x_guide.shape[1]}.")
        h_low, w_low = x_low.shape[-2:]
        expected = (h_low * self.scale_factor, w_low * self.scale_factor)
        if tuple(x_guide.shape[-2:]) != expected:
            raise ValueError(
                "UAFeatureUpsample spatial ratio mismatch: "
                f"low={tuple(x_low.shape[-2:])}, guide={tuple(x_guide.shape[-2:])}, expected guide={expected}."
            )
        return x_low, x_guide

    def _spatial_logits(self, target_hw, device):
        h, w = target_hw
        sf = float(self.scale_factor)
        offsets = self.offsets.to(device=device, dtype=torch.float32)
        dy = offsets[:, 0].reshape(1, self.num_neighbors, 1, 1)
        dx = offsets[:, 1].reshape(1, self.num_neighbors, 1, 1)
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        sub_y = (torch.remainder(y, sf) + 0.5) / sf - 0.5
        sub_x = (torch.remainder(x, sf) + 0.5) / sf - 0.5
        dist2 = (sub_y.reshape(1, 1, h, 1) - dy).pow(2) + (sub_x.reshape(1, 1, 1, w) - dx).pow(2)
        spatial_scale = self._positive_scale(self.spatial_scale_logit).reshape(1, 1, 1, 1)
        return -dist2 / spatial_scale

    def _compute_weights(self, x_low, x_guide):
        target_hw = x_guide.shape[-2:]
        guide_embed = F.normalize(self.guide_proj(x_guide).float(), p=2, dim=1, eps=self.eps)
        low_embed = F.normalize(self.low_proj(x_low).float(), p=2, dim=1, eps=self.eps)
        range_scale = self._positive_scale(self.range_scale_logit).reshape(1, 1, 1)

        guidance_logits = []
        for dy, dx in self._offset_pairs:
            neighbor_embed = self._shift_low(low_embed, dy, dx)
            neighbor_embed = F.interpolate(neighbor_embed, size=target_hw, mode="nearest")
            feature_dist2 = (guide_embed - neighbor_embed).pow(2).sum(dim=1)
            guidance_logits.append(-self._GUIDANCE_SCORE_WEIGHT * feature_dist2 / range_scale)
        guidance_logits = torch.stack(guidance_logits, dim=1)

        correction_logits = self.kernel_predictor(x_guide).float()
        logits = self._spatial_logits(target_hw, x_guide.device) + guidance_logits + correction_logits
        self._check_finite("weight logits", logits)
        logits = logits.clamp(min=-self._LOGIT_CLAMP, max=self._LOGIT_CLAMP)
        weights = F.softmax(logits, dim=1)
        self._check_finite("normalized weights", weights)
        return weights

    def forward(self, inputs, return_weights=False):
        x_low, x_guide = self._validate_inputs(inputs)
        target_hw = x_guide.shape[-2:]

        weights = self._compute_weights(x_low, x_guide)
        base = F.interpolate(x_low, size=target_hw, mode="bilinear", align_corners=False)
        adaptive = torch.zeros_like(base)
        for neighbor_index, (dy, dx) in enumerate(self._offset_pairs):
            candidate = self._shift_low(x_low, dy, dx)
            candidate = F.interpolate(candidate, size=target_hw, mode="nearest")
            weight = weights[:, neighbor_index : neighbor_index + 1].to(dtype=candidate.dtype)
            adaptive = adaptive + candidate * weight

        gamma = self.gamma.to(dtype=base.dtype).reshape(1, 1, 1, 1)
        output = base + gamma * (adaptive - base)
        self._check_finite("output", output)
        return (output, weights) if return_weights else output
