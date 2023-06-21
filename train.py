"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import argparse
import itertools
import pathlib
import time
from typing import Callable

import json
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import os
from lpips import LPIPS
from radiance_fields.ngp_appearance import NGPDensityField, NGPRadianceField

from utils import (
    render_image_with_propnet,
    set_random_seed,
    setup_exp_dir
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default="/home/ccl/Datasets/NeRF/aizu-student-hall/processed",
    help="the root dir of the dataset",
)
parser.add_argument(
    "--config",
    type=str,
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=str(pathlib.Path.cwd() / "output"),
    help="the output dir",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="experiment",
    help="experiment name",
)
args = parser.parse_args()


device = "cuda:0"
set_random_seed(42)

from datasets.nerf_colmap import SubjectLoader

with open(args.config, "r") as config_file:
    cfg = json.load(config_file)

# training parameters
max_steps = cfg['max_steps']
init_batch_size = cfg['init_batch_size']
weight_decay = cfg['weight_decay']
lr=cfg['lr']
# scene parameters
unbounded = cfg['unbounded']
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
near_plane = cfg['near_plane']
far_plane = cfg['far_plane']
# dataset parameters
train_dataset_kwargs = cfg["train_dataset_kwargs"]
test_dataset_kwargs = cfg["test_dataset_kwargs"]
# model parameters
proposal_networks = [ 
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            **params,
        ).to(device) for params in cfg["proposal_networks_params"]]

# render parameters
num_samples = cfg['num_samples']
num_samples_per_prop = cfg["num_samples_per_prop"]
sampling_type = cfg['sampling_type']
opaque_bkgd = cfg['opaque_bkgd']

exp_dir = os.path.join(args.out_dir, args.exp_name)
setup_exp_dir(exp_dir, args.config)

train_dataset = SubjectLoader(
    subject_id=0,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=0,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=lr,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(cfg=cfg['radiance_field'], aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=lr,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    img = data["img"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
        img=img,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 20000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        if step == 0:
            continue

    # if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        for p in proposal_networks:
            p.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                img = data["img"]

                # rendering
                rgb, acc, depth, _, = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays,
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=render_bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                    img=img,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                if torch.isnan(psnr):
                    break


                torch.save(radiance_field.state_dict(), os.path.join(exp_dir, 'radiance_field.pth'))
                # torch.save(estimator.state_dict(), os.path.join(exp_dir, 'estimator.pth'))
                for j, net in enumerate(proposal_networks):
                    torch.save(net.state_dict(), os.path.join(exp_dir, f'proposal_network_{j}.pth'))


                imageio.imwrite(
                    os.path.join(exp_dir, f"rgb_{i:08}_render.png"),
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(exp_dir, f"rgb_{i:08}_ground_truth.png"),
                    (pixels.cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(exp_dir, f"rgb_{i:08}_error.png"),
                    (
                        (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                    ).astype(np.uint8),
                )
                vis_depth = torch.log(depth)
                vis_depth -= torch.min(vis_depth)
                vis_depth /= torch.max(vis_depth)
                imageio.imwrite(
                    os.path.join(exp_dir, f"rgb_{i:08}_depth.png"),
                    (
                        vis_depth.cpu().numpy() * 255
                    ).astype(np.uint8),
                )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")