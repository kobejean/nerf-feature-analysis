import random
from typing import Optional, Sequence

import shutil
import os
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.volrend import rendering

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def render_image_with_propnet(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
    img: Optional[torch.Tensor] = None,
    a_vec: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    # prepare rays
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        # Case when rendering full image
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs, img=img, a_vec=a_vec)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        extras,
    )


def setup_exp_dir(exp_dir):
    os.makedirs(exp_dir)

    # Save the Python scripts
    shutil.copy('train.py', exp_dir)
    shutil.copytree('radiance_fields', os.path.join(exp_dir, 'radiance_fields'))
    shutil.copytree('datasets', os.path.join(exp_dir, 'datasets'))