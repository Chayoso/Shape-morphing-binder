import torch
from physmorph.pipeline.image_metric import ScreenedImageResidual


def test_screened_image_metric_preserves_area_signal():
    metric = ScreenedImageResidual(32, 40)
    constant = torch.full((32, 40), .3)
    assert torch.allclose(metric(constant), constant, atol=1e-6)


def test_nonoverlapping_images_produce_a_physical_translation_direction():
    # The image itself is translated; the metric has no access to control/state.
    y, x = torch.meshgrid(torch.arange(32, dtype=torch.float64), torch.arange(32, dtype=torch.float64), indexing="ij")
    shift = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    current = torch.exp(-((x-8-shift)**2+(y-16)**2)/2)
    target = torch.exp(-((x-22)**2+(y-16)**2)/2)
    difference = current-target
    local = .5*difference.square().mean()
    metric = ScreenedImageResidual(32, 32, .15, dtype=torch.float64)
    nonlocal_loss = .5*metric(difference).square().mean()
    gl = torch.autograd.grad(local, shift, retain_graph=True)[0]
    gn = torch.autograd.grad(nonlocal_loss, shift)[0]
    assert abs(float(gl)) < 1e-10
    assert float(gn) < -1e-7  # descent moves right, toward the target
