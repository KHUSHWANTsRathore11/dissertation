"""
Anomaly Synthesis Module for AAND Framework.

Generates synthetic anomalies by blending foreign textures into normal images
using Perlin noise masks. Supports two texture sources:
  - Procedural: Random noise, Perlin textures, color patterns (default)
  - DTD: Real-world textures from the Describable Textures Dataset

Reference: AAND paper (arxiv 2405.02068v2), Section III-B "Anomaly synthesis"
"""

import os
import math
import random
import glob
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms


# =============================================================================
# Perlin Noise Generation
# =============================================================================

def _fade(t):
    """Smoothstep fade function for Perlin noise."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a, b, t):
    """Linear interpolation."""
    return a + t * (b - a)


def generate_perlin_noise_2d(shape, res, seed=None):
    """
    Generate 2D Perlin noise.

    Args:
        shape: (H, W) tuple for the output noise map.
        res: (res_y, res_x) tuple — number of periods of noise along each axis.
        seed: Optional random seed for reproducibility.

    Returns:
        np.ndarray of shape (H, W) with values in roughly [-1, 1].
    """
    if seed is not None:
        np.random.seed(seed)

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Tile gradients for each cell
    g00 = gradients[:-1, :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g10 = gradients[1:, :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g01 = gradients[:-1, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g11 = gradients[1:, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)

    # Dot products
    t = grid[:shape[0], :shape[1]]
    n00 = np.sum(np.stack([t[:, :, 0], t[:, :, 1]], axis=-1) * g00[:shape[0], :shape[1]], axis=-1)
    n10 = np.sum(np.stack([t[:, :, 0] - 1, t[:, :, 1]], axis=-1) * g10[:shape[0], :shape[1]], axis=-1)
    n01 = np.sum(np.stack([t[:, :, 0], t[:, :, 1] - 1], axis=-1) * g01[:shape[0], :shape[1]], axis=-1)
    n11 = np.sum(np.stack([t[:, :, 0] - 1, t[:, :, 1] - 1], axis=-1) * g11[:shape[0], :shape[1]], axis=-1)

    # Interpolation
    u = _fade(t)
    return _lerp(
        _lerp(n00, n10, u[:, :, 0]),
        _lerp(n01, n11, u[:, :, 0]),
        u[:, :, 1]
    )


def generate_perlin_mask(height, width, threshold=0.0, min_res=2, max_res=8):
    """
    Generate a binary anomaly mask using multi-octave Perlin noise.

    Args:
        height, width: Dimensions of the output mask.
        threshold: Binarization threshold (higher = smaller anomaly regions).
        min_res, max_res: Range of noise resolutions to combine.

    Returns:
        np.ndarray of shape (H, W) with values in {0, 1}.
    """
    noise = np.zeros((height, width))
    res = random.randint(min_res, max_res)

    # Ensure dimensions are divisible by resolution
    h = (height // res) * res
    w = (width // res) * res

    perlin = generate_perlin_noise_2d((h, w), (res, res))

    # Pad to original size if needed
    noise[:h, :w] = perlin

    # Binarize
    mask = (noise > threshold).astype(np.float32)
    return mask


# =============================================================================
# Texture Sources
# =============================================================================

class ProceduralTextureSource:
    """
    Generates synthetic textures procedurally (no external dataset needed).
    Uses random color noise, Perlin noise patterns, and gradient textures.
    """

    def __init__(self, image_size=518):
        self.image_size = image_size

    def get_texture(self):
        """Return a random procedural texture as a PIL Image."""
        method = random.choice(['noise', 'perlin', 'gradient', 'checkerboard'])

        if method == 'noise':
            # Random color noise
            arr = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            # Optionally blur for smoother textures
            if random.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5)))
            return img

        elif method == 'perlin':
            # Colored Perlin noise
            channels = []
            for _ in range(3):
                res = random.randint(2, 8)
                h = (self.image_size // res) * res
                w = (self.image_size // res) * res
                noise = generate_perlin_noise_2d((h, w), (res, res))
                # Pad to target size
                full = np.zeros((self.image_size, self.image_size))
                full[:h, :w] = noise
                # Normalize to [0, 255]
                full = ((full - full.min()) / (full.max() - full.min() + 1e-8) * 255).astype(np.uint8)
                channels.append(full)
            arr = np.stack(channels, axis=-1)
            return Image.fromarray(arr)

        elif method == 'gradient':
            # Linear or radial color gradient
            arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            for i in range(self.image_size):
                t = i / self.image_size
                arr[i, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            return Image.fromarray(arr)

        else:  # checkerboard
            block_size = random.randint(4, 32)
            arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            for i in range(0, self.image_size, block_size):
                for j in range(0, self.image_size, block_size):
                    color = c1 if ((i // block_size) + (j // block_size)) % 2 == 0 else c2
                    arr[i:i+block_size, j:j+block_size] = color
            return Image.fromarray(arr)


class DTDTextureSource:
    """
    Loads textures from the Describable Textures Dataset (DTD).
    DTD must be downloaded and extracted to the specified path.

    Download: https://www.robots.ox.ac.uk/~vgg/data/dtd/
    """

    def __init__(self, dtd_path, image_size=518):
        self.image_size = image_size
        self.texture_paths = []

        if not os.path.exists(dtd_path):
            raise FileNotFoundError(
                f"DTD dataset not found at '{dtd_path}'. "
                f"Download it from https://www.robots.ox.ac.uk/~vgg/data/dtd/ "
                f"and extract to '{dtd_path}'."
            )

        # Collect all texture images
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.texture_paths.extend(glob.glob(os.path.join(dtd_path, '**', ext), recursive=True))

        if len(self.texture_paths) == 0:
            raise FileNotFoundError(f"No texture images found in '{dtd_path}'.")

        print(f"DTDTextureSource: loaded {len(self.texture_paths)} texture images from {dtd_path}")

    def get_texture(self):
        """Return a random DTD texture as a PIL Image."""
        path = random.choice(self.texture_paths)
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img


# =============================================================================
# Anomaly Synthesis Pipeline
# =============================================================================

class AnomalySynthesizer:
    """
    Generates synthetic anomaly images by blending foreign textures into normal images.

    Args:
        image_size: Target image size (square).
        texture_source: 'procedural' (default) or 'dtd'.
        dtd_path: Path to DTD dataset (required if texture_source='dtd').
        anomaly_opacity: Blending opacity for the synthetic anomaly (0-1).
    """

    def __init__(self, image_size=518, texture_source='procedural', dtd_path=None,
                 anomaly_opacity=0.7):
        self.image_size = image_size
        self.anomaly_opacity = anomaly_opacity

        if texture_source == 'dtd':
            if dtd_path is None:
                raise ValueError("dtd_path must be provided when texture_source='dtd'")
            self.texture_gen = DTDTextureSource(dtd_path, image_size)
        else:
            self.texture_gen = ProceduralTextureSource(image_size)

    def __call__(self, normal_image):
        """
        Generate a synthetic anomaly from a normal image.

        Args:
            normal_image: PIL Image (RGB) or torch.Tensor (C, H, W).

        Returns:
            corrupted_image: PIL Image with synthetic anomaly blended in.
            anomaly_mask: np.ndarray (H, W) binary mask, 1 = anomaly region.
        """
        if isinstance(normal_image, torch.Tensor):
            # Convert tensor to PIL
            if normal_image.dim() == 3:
                normal_image = transforms.ToPILImage()(normal_image)

        normal_image = normal_image.resize((self.image_size, self.image_size))
        normal_np = np.array(normal_image).astype(np.float32)

        # Generate anomaly mask using Perlin noise
        threshold = random.uniform(-0.2, 0.3)
        anomaly_mask = generate_perlin_mask(self.image_size, self.image_size, threshold=threshold)

        # Ensure the mask is not empty or too large
        mask_ratio = anomaly_mask.sum() / anomaly_mask.size
        max_retries = 5
        retries = 0
        while (mask_ratio < 0.01 or mask_ratio > 0.5) and retries < max_retries:
            threshold = random.uniform(-0.2, 0.3)
            anomaly_mask = generate_perlin_mask(self.image_size, self.image_size, threshold=threshold)
            mask_ratio = anomaly_mask.sum() / anomaly_mask.size
            retries += 1

        # Get texture
        texture = self.texture_gen.get_texture()
        texture_np = np.array(texture.resize((self.image_size, self.image_size))).astype(np.float32)

        # Blend: corrupted = normal * (1 - mask * opacity) + texture * (mask * opacity)
        mask_3d = anomaly_mask[:, :, np.newaxis]
        opacity = self.anomaly_opacity
        corrupted_np = normal_np * (1 - mask_3d * opacity) + texture_np * (mask_3d * opacity)
        corrupted_np = np.clip(corrupted_np, 0, 255).astype(np.uint8)

        corrupted_image = Image.fromarray(corrupted_np)

        return corrupted_image, anomaly_mask
