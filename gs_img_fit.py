import math
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor
from gsplat import rasterization, rasterization_2dgs


# ============================================================================
# SimpleTrainer — ONLY stores Gaussians + forward()
# ============================================================================
class SimpleTrainer:
    """Stores Gaussian parameters and performs forward rendering only."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.model_type = model_type

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians()

        # choose renderer
        if model_type == "3dgs":
            self.raster_fn = rasterization
        else:
            self.raster_fn = rasterization_2dgs

    # ----------------------------------------------------------------------
    def _init_gaussians(self):
        """Initialize Gaussian parameters."""
        bd = 2
        d = 3

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        # random quaternion initialization
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)
        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            dim=-1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        # camera pose
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # intrinsics
        self.K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        # mark trainable parameters
        for p in [self.means, self.scales, self.rgbs, self.quats, self.opacities]:
            p.requires_grad = True

    # ----------------------------------------------------------------------
    # FORWARD ONLY — no loss, no backward, no step
    # ----------------------------------------------------------------------
    def forward(self) -> Tensor:
        """Render Gaussians to an RGB image."""
        q_norm = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)

        renders = self.raster_fn(
            self.means,
            q_norm,
            self.scales,
            torch.sigmoid(self.opacities),
            torch.sigmoid(self.rgbs),
            self.viewmat[None],
            self.K[None],
            self.W,
            self.H,
            packed=False,
        )[0]

        return renders[0]


# ============================================================================
# Utility
# ============================================================================
def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    t = transforms.ToTensor()
    return t(img).permute(1, 2, 0)[..., :3]


# ============================================================================
# MAIN — training loop lives entirely here
# ============================================================================
def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 20000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
):
    # -----------------------------
    # Load GT image
    # -----------------------------
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3))
        gt_image[: height // 2, : width // 2] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :] = torch.tensor([0.0, 0.0, 1.0])

    # -----------------------------
    # Init trainer (forward only)
    # -----------------------------
    trainer = SimpleTrainer(
        gt_image=gt_image,
        num_points=num_points,
        model_type=model_type,
    )

    # -----------------------------
    # Optimizer and loss in main()
    # -----------------------------
    optimizer = torch.optim.Adam(
        [trainer.means, trainer.scales, trainer.rgbs, trainer.opacities, trainer.quats],
        lr=lr,
    )
    loss_fn = torch.nn.MSELoss()

    frames = []

    # -----------------------------
    # Training loop (entirely here)
    # -----------------------------
    for it in range(iterations):
        optimizer.zero_grad()

        out_img = trainer.forward()
        loss = loss_fn(out_img, trainer.gt_image)

        loss.backward()
        optimizer.step()

        print(f"Iter {it+1}/{iterations} | Loss: {loss.item():.6f}")

        # save frame
        if save_imgs and it % 5 == 0:
            with torch.no_grad():
                frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame))

    # save gif
    if save_imgs:
        os.makedirs("results", exist_ok=True)
        frames[0].save(
            "results/training.gif",
            save_all=True,
            append_images=frames[1:],
            duration=5,
            loop=0,
        )


if __name__ == "__main__":
    tyro.cli(main)
