import gc
import sys
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:
    class _MiniMark:
        @staticmethod
        def parametrize(_argnames, _argvalues):
            def decorator(fn):
                return fn
            return decorator

        @staticmethod
        def skipif(condition, reason=''):
            def decorator(fn):
                fn.__skipif__ = condition
                fn.__skip_reason__ = reason
                return fn
            return decorator

    class _MiniPytest:
        mark = _MiniMark()

        @staticmethod
        @contextmanager
        def raises(exc_type, match=None):
            try:
                yield
            except exc_type as exc:
                if match is not None:
                    assert match in str(exc)
                return
            raise AssertionError(f'{exc_type.__name__} was not raised')

    pytest = _MiniPytest()


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import UAFeatureUpsample  # noqa: E402
from models.yolo import Model  # noqa: E402


SITE_SHAPES = [
    (2, 192, 40, 40, 192, 80, 80),
    (2, 96, 80, 80, 96, 160, 160),
]
MODEL_CFGS = [
    'models/FFCA-YOLO-Bilinear.yaml',
    'models/FFCA-YOLO-UA-L11.yaml',
    'models/FFCA-YOLO-UA-L15.yaml',
    'models/FFCA-YOLO-UA-Dual.yaml',
]


def _assert_finite(tensor):
    assert torch.isfinite(tensor).all()


def _assert_param_grads_finite(module, require_all=True):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            assert not require_all, f'{name} has no gradient'
            continue
        _assert_finite(param.grad)


def _loss_from_output(output):
    if isinstance(output, torch.Tensor):
        return output.float().mean()
    if isinstance(output, (list, tuple)):
        loss = None
        for item in output:
            item_loss = _loss_from_output(item)
            loss = item_loss if loss is None else loss + item_loss
        return loss
    raise TypeError(type(output))


def _assert_detect_eval_output(output, img_size):
    assert isinstance(output, tuple)
    pred, features = output
    expected_cells = 3 * ((img_size // 4) ** 2 + (img_size // 8) ** 2 + (img_size // 16) ** 2)
    assert pred.shape == (1, expected_cells, 7)
    assert len(features) == 3
    assert [tuple(x.shape[2:4]) for x in features] == [
        (img_size // 4, img_size // 4),
        (img_size // 8, img_size // 8),
        (img_size // 16, img_size // 16),
    ]
    _assert_finite(pred)
    for feature in features:
        _assert_finite(feature)


@pytest.mark.parametrize('shape', SITE_SHAPES)
def test_module_cpu_forward_backward(shape):
    torch.manual_seed(0)
    b, c_low, h_low, w_low, c_guide, h_guide, w_guide = shape
    module = UAFeatureUpsample(c_low, c_guide).train()
    x_low = torch.randn(b, c_low, h_low, w_low, requires_grad=True)
    x_guide = torch.randn(b, c_guide, h_guide, w_guide)

    output = module([x_low, x_guide])
    assert output.shape == (b, c_low, h_guide, w_guide)
    _assert_finite(output)

    output.float().mean().backward()
    _assert_finite(x_low.grad)
    _assert_param_grads_finite(module, require_all=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('shape', SITE_SHAPES)
def test_module_cuda_forward_backward_amp(shape):
    torch.manual_seed(0)
    device = torch.device('cuda')
    b, c_low, h_low, w_low, c_guide, h_guide, w_guide = shape
    module = UAFeatureUpsample(c_low, c_guide).to(device).train()
    x_low = torch.randn(b, c_low, h_low, w_low, device=device, requires_grad=True)
    x_guide = torch.randn(b, c_guide, h_guide, w_guide, device=device)

    output = module((x_low, x_guide))
    assert output.shape == (b, c_low, h_guide, w_guide)
    _assert_finite(output)
    output.float().mean().backward()
    _assert_finite(x_low.grad)
    _assert_param_grads_finite(module, require_all=True)

    module.zero_grad(set_to_none=True)
    x_low.grad = None
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        amp_output = module((x_low, x_guide))
        amp_loss = amp_output.float().mean()
    scaler.scale(amp_loss).backward()
    scaler.unscale_(optimizer)
    _assert_finite(x_low.grad)
    _assert_param_grads_finite(module, require_all=True)
    scaler.step(optimizer)
    scaler.update()


def test_module_validation_bilinear_initialization_and_weight_sum():
    torch.manual_seed(1)
    module = UAFeatureUpsample(192, 192)
    x_low = torch.randn(2, 192, 40, 40)
    x_guide = torch.randn(2, 192, 80, 80)

    output, weights = module([x_low, x_guide], return_weights=True)
    base = F.interpolate(x_low, size=x_guide.shape[-2:], mode='bilinear', align_corners=False)
    assert torch.allclose(output, base, atol=1e-6, rtol=1e-6)
    assert weights.shape == (2, 9, 80, 80)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)
    _assert_finite(weights)

    with pytest.raises(ValueError, match='batch size mismatch'):
        module([torch.randn(1, 192, 40, 40), x_guide])
    with pytest.raises(ValueError, match='spatial ratio mismatch'):
        module([x_low, torch.randn(2, 192, 79, 80)])
    with pytest.raises(ValueError, match='exactly two'):
        module([x_low])


@pytest.mark.parametrize('cfg', MODEL_CFGS)
def test_full_model_cpu_forward_backward(cfg):
    torch.manual_seed(2)
    model = Model(str(ROOT / cfg))
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 640, 640))
    _assert_detect_eval_output(output, 640)

    model.train()
    small = torch.randn(1, 3, 256, 256)
    train_output = model(small)
    loss = _loss_from_output(train_output)
    _assert_finite(loss)
    loss.backward()
    _assert_param_grads_finite(model, require_all=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('cfg', MODEL_CFGS)
def test_full_model_cuda_forward_amp(cfg):
    torch.manual_seed(3)
    device = torch.device('cuda')
    model = Model(str(ROOT / cfg)).to(device).eval()
    input_tensor = torch.randn(1, 3, 640, 640, device=device)

    with torch.no_grad():
        output = model(input_tensor)
    _assert_detect_eval_output(output, 640)

    with torch.no_grad(), torch.cuda.amp.autocast():
        amp_output = model(input_tensor)
    _assert_detect_eval_output(amp_output, 640)

    del model, input_tensor, output, amp_output
    gc.collect()
    torch.cuda.empty_cache()


def _run_direct(name, fn, *args):
    if getattr(fn, '__skipif__', False):
        print(f'SKIPPED {name}: {getattr(fn, "__skip_reason__", "")}')
        return
    fn(*args)
    print(f'PASSED {name}')


if __name__ == '__main__':
    for shape in SITE_SHAPES:
        _run_direct(f'module_cpu_forward_backward_{shape[1]}', test_module_cpu_forward_backward, shape)
    for shape in SITE_SHAPES:
        _run_direct(f'module_cuda_forward_backward_amp_{shape[1]}', test_module_cuda_forward_backward_amp, shape)
    _run_direct('module_validation_bilinear_initialization_and_weight_sum',
                test_module_validation_bilinear_initialization_and_weight_sum)
    for cfg in MODEL_CFGS:
        _run_direct(f'full_model_cpu_forward_backward_{Path(cfg).stem}', test_full_model_cpu_forward_backward, cfg)
    for cfg in MODEL_CFGS:
        _run_direct(f'full_model_cuda_forward_amp_{Path(cfg).stem}', test_full_model_cuda_forward_amp, cfg)
