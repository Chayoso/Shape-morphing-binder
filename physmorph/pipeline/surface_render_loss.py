"""P302 hybrid coverage/edge guidance on the actual shared surface primitives.

Targets, calibration, views and crop rectangles are frozen for one inner solve.
Detail images are rasterized at the declared full camera resolution BEFORE
cropping, retaining complete-scene occlusion/tails and the true pixel footprint.
No intermediate frame is compared with a fabricated final-target ground truth.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..render.studio import StudioRaster, raster_identity
from ..render.surface_gaussians import SurfaceGaussians


def image_edges(image):
    kernels = image.new_tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]) / 8.
    return F.conv2d(F.pad(image[None, None], (1, 1, 1, 1), mode='replicate'),
                    kernels[:, None])[0]


def coverage_error(image, target, deficit_weight=2., excess_weight=1.):
    delta = image-target
    return (deficit_weight*delta.clamp_max(0).square() +
            excess_weight*delta.clamp_min(0).square()).mean()


def edge_roi(target, size):
    """Deterministic target-only edge patch, fixed throughout candidate trials."""
    h, w = target.shape
    size = min(int(size), h, w)
    stride = max(1, size//2)
    energy = image_edges(target).square().sum(0)
    pooled = F.avg_pool2d(energy[None, None], size, stride=stride)[0, 0]
    index = int(pooled.argmax())
    y, x = (index//pooled.shape[1])*stride, (index % pooled.shape[1])*stride
    return int(y), int(x), size


@dataclass(frozen=True)
class ViewTarget:
    coarse: torch.Tensor
    detail: torch.Tensor
    edge: torch.Tensor
    roi: tuple


def select_surface_views(views, count):
    """Greedy angular separation on the camera sphere, starting at the export view."""
    selected = [(math.radians(35.), math.radians(18.))]
    candidates = list(dict.fromkeys(views))
    def vector(view):
        az, el = view
        return (math.cos(el)*math.sin(az), math.sin(el), math.cos(el)*math.cos(az))
    def separation(view):
        a = vector(view)
        return min(1.-sum(x*y for x, y in zip(a, vector(b))) for b in selected)
    while candidates and len(selected) < count:
        chosen = max(candidates, key=separation)
        candidates.remove(chosen)
        if separation(chosen) > 1e-10:
            selected.append(chosen)
    return selected


class SurfaceRenderViews:
    def __init__(self, reference, views, *, coarse_height=256, detail_height=2160,
                 patch_size=256, view_count=4, deficit_weight=2., excess_weight=1., raster_backend='legacy'):
        if not reference.is_cuda:
            raise ValueError('Surface Gaussian raster loss requires CUDA; no CPU fallback')
        self.geometry = SurfaceGaussians.from_reference(reference)
        self.raster_backend = raster_backend
        # Include the delivered studio camera, then distribute additional views.
        self.views = select_surface_views(views, view_count)
        self.coarse_height, self.detail_height = coarse_height, detail_height
        self.patch_size = patch_size
        self.deficit_weight, self.excess_weight = deficit_weight, excess_weight
        self.coarse, self.detail = [], []
        for az, el in self.views:
            for cameras, height in ((self.coarse, coarse_height), (self.detail, detail_height)):
                width = 2*round(height*16/9/2)
                cameras.append(StudioRaster(self.geometry.center, self.geometry.radius,
                    width, height, math.degrees(az), math.degrees(el), direct_covariance=True,
                    coverage_only=True, raster_backend=raster_backend))

    def prepare(self, reference, target_kind):
        targets = []
        with torch.no_grad():
            primitives = self.geometry(reference)
            for coarse, detail in zip(self.coarse, self.detail):
                low = coarse.coverage(reference, primitives).detach()
                high = detail.coverage(reference, primitives)
                roi = edge_roi(high, self.patch_size)
                y, x, size = roi
                patch = high[y:y+size, x:x+size].clone().detach()
                targets.append(ViewTarget(low, patch, image_edges(patch).detach(), roi))
        return SurfaceWindowLoss(self, targets, target_kind)

    def metadata(self):
        return dict(**self.geometry.metadata(), coarse_height=self.coarse_height,
                    detail_height=self.detail_height, aspect=16/9, patch_size=self.patch_size,
                    views_radians=self.views, raster_backend=self.raster_backend,
                    raster_files=raster_identity(self.raster_backend),
                    detail_raster='full camera then crop; not a resized crop')


class SurfaceWindowLoss:
    def __init__(self, shared, targets, target_kind):
        self.shared, self.targets, self.target_kind = shared, tuple(targets), target_kind

    def __call__(self, x):
        primitive = self.shared.geometry(x)
        values = {'coverage': x.new_zeros(()), 'detail_coverage': x.new_zeros(()),
                  'detail_edge': x.new_zeros(())}
        for low, high, target in zip(self.shared.coarse, self.shared.detail, self.targets):
            low_image = low.coverage(x, primitive)
            high_image = high.coverage(x, primitive)
            y, left, size = target.roi
            patch = high_image[y:y+size, left:left+size]
            values['coverage'] = values['coverage'] + coverage_error(low_image, target.coarse,
                self.shared.deficit_weight, self.shared.excess_weight)
            values['detail_coverage'] = values['detail_coverage'] + coverage_error(patch, target.detail,
                self.shared.deficit_weight, self.shared.excess_weight)
            values['detail_edge'] = values['detail_edge'] + (image_edges(patch)-target.edge).square().mean()
        values = {key: value/len(self.targets) for key, value in values.items()}
        return sum(values.values()), values

    def metadata(self):
        return dict(**self.shared.metadata(), target_kind=self.target_kind,
                    rois_y_x_size=[list(t.roi) for t in self.targets],
                    outer_acceptance='unchanged fixed-target CIC/physics gate; not a GS convergence certificate')
