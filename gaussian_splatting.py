import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat import rasterization, rasterization_2dgs


class GaussianSplatting2D(nn.Module):
    """
    2D Gaussian Splatting image network using gsplat library.
    """

    def __init__(
        self,
        width: int,
        height: int = None,
        num_gaussians: int = 1000,
        model_type: str = "2dgs",
        grayscale: bool = True,
    ):
        """
        Args:
            width: Image width
            height: Image height (defaults to width if None)
            num_gaussians: Number of Gaussians
            model_type: "3dgs" or "2dgs"
            grayscale: If True, output 1 channel; otherwise 3 channels (RGB)
        """
        super().__init__()
        if height is None:
            height = width

        self.width = width
        self.height = height
        self.num_gaussians = num_gaussians
        self.model_type = model_type
        self.grayscale = grayscale

        if model_type == "3dgs":
            self.raster_fn = rasterization
        else:
            self.raster_fn = rasterization_2dgs

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize Gaussian parameters as nn.Parameter."""
        num_channels = 1 if self.grayscale else 3
        bd = 2

        if self.model_type == "2dgs":
            # 2DGS: 2D Gaussians embedded in 3D space (z=0)
            # Initialize with grid + noise for uniform coverage (like aberration basis)
            grid_size = int(math.sqrt(self.num_gaussians))
            if grid_size * grid_size < self.num_gaussians:
                grid_size += 1

            # Create grid in [-bd, bd] to match camera view
            x = torch.linspace(-bd, bd, grid_size)
            y = torch.linspace(-bd, bd, grid_size)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[:self.num_gaussians]

            # Add small random noise for variation
            noise = (torch.rand_like(grid_xy) - 0.5) * (2 * bd / grid_size) * 0.5
            means_xy = grid_xy + noise
            means_z = torch.zeros(self.num_gaussians, 1)

            self.means = nn.Parameter(
                torch.cat([means_xy, means_z], dim=1),
                requires_grad=True
            )

            # 2D scales (width, height) + small z scale
            scales_xy = torch.rand(self.num_gaussians, 2) * 0.5 + 0.1
            scales_z = torch.ones(self.num_gaussians, 1) * 0.01
            self.scales = nn.Parameter(
                torch.cat([scales_xy, scales_z], dim=1),
                requires_grad=True
            )

            # Rotation angles (in radians) for 2DGS
            self.quats = nn.Parameter(
                torch.rand(self.num_gaussians) * 2 * math.pi,
                requires_grad=True
            )
        else:
            # 3DGS: full 3D representation
            self.means = nn.Parameter(
                bd * (torch.rand(self.num_gaussians, 3) - 0.5),
                requires_grad=True
            )

            self.scales = nn.Parameter(
                torch.rand(self.num_gaussians, 3),
                requires_grad=True
            )

            # Quaternions for 3D rotation
            u = torch.rand(self.num_gaussians, 1)
            v = torch.rand(self.num_gaussians, 1)
            w = torch.rand(self.num_gaussians, 1)
            quats_init = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
                ],
                dim=-1,
            )
            self.quats = nn.Parameter(quats_init, requires_grad=True)

        # Initialize RGBs in logit space to avoid white-washing
        initial_brightness = 0.1 if self.grayscale else 0.2
        self.rgbs = nn.Parameter(
            torch.logit(torch.rand(self.num_gaussians, num_channels) * 0.3 + initial_brightness),
            requires_grad=True
        )

        # Initialize opacities in logit space for better training
        self.opacities = nn.Parameter(
            torch.logit(torch.rand(self.num_gaussians) * 0.5 + 0.1),
            requires_grad=True
        )

        fov_x = math.pi / 2.0
        focal = 0.5 * float(self.width) / math.tan(0.5 * fov_x)

        viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        self.register_buffer("viewmat", viewmat)

        K = torch.tensor(
            [
                [focal, 0, self.width / 2],
                [0, focal, self.height / 2],
                [0, 0, 1],
            ],
        )
        self.register_buffer("K", K)

    def forward(self):
        """Render Gaussians to an image."""
        if self.model_type == "2dgs":
            # 2DGS rendering
            renders = self.raster_fn(
                self.means,
                self.quats,  # rotation angles, not quaternions
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                self.K[None],
                self.width,
                self.height,
                packed=False,
            )[0]
        else:
            # 3DGS rendering
            q_norm = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)
            renders = self.raster_fn(
                self.means,
                q_norm,
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                self.K[None],
                self.width,
                self.height,
                packed=False,
            )[0]

        # renders is [1, H, W, C], convert to [1, C, H, W]
        output = renders[0].permute(2, 0, 1).unsqueeze(0)
        return output


def image_path_to_tensor(image_path):
    """Load image from path and convert to tensor."""
    import torchvision.transforms as transforms
    from PIL import Image

    img = Image.open(image_path)
    t = transforms.ToTensor()
    return t(img).permute(1, 2, 0)[..., :3]


def main(
    height: int = 256,
    width: int = 256,
    num_gaussians: int = 2000,
    save_imgs: bool = True,
    img_path: str = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: str = "2dgs",
    grayscale: bool = False,
):
    """Test the GaussianSplatting2D module."""
    import os
    import numpy as np
    from PIL import Image
    from datetime import datetime

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3))
        gt_image[: height // 2, : width // 2] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :] = torch.tensor([0.0, 0.0, 1.0])

    gt_image = gt_image.to(device)

    model = GaussianSplatting2D(
        width=width,
        height=height,
        num_gaussians=num_gaussians,
        model_type=model_type,
        grayscale=grayscale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    frames = []

    for it in range(iterations):
        optimizer.zero_grad()

        out_img = model()
        out_img_hwc = out_img[0].permute(1, 2, 0)

        if grayscale:
            gt_gray = gt_image.mean(dim=-1, keepdim=True)
            loss = loss_fn(out_img_hwc, gt_gray)
        else:
            loss = loss_fn(out_img_hwc, gt_image)

        loss.backward()
        optimizer.step()

        print(f"Iter {it+1}/{iterations} | Loss: {loss.item():.6f}")

        if save_imgs and it % 5 == 0:
            with torch.no_grad():
                if grayscale:
                    frame = (out_img_hwc.repeat(1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
                else:
                    frame = (out_img_hwc.detach().cpu().numpy() * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame))

    if save_imgs:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_path = f"results/gs_training_{timestamp}.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=5,
            loop=0,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_gaussians", type=int, default=2000)
    parser.add_argument("--save_imgs", action="store_true", default=True)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model_type", type=str, choices=["2dgs", "3dgs"], default="2dgs")
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    main(
        height=args.height,
        width=args.width,
        num_gaussians=args.num_gaussians,
        save_imgs=args.save_imgs,
        img_path=args.img_path,
        iterations=args.iterations,
        lr=args.lr,
        model_type=args.model_type,
        grayscale=args.grayscale,
    )
