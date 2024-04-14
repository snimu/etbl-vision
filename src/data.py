
from typing import Literal

import torch
import torchvision.transforms as transforms
import einops


def make_noise_mask(batch_size: int, h: int, w: int, c: int, num_pixels: int = 1):
    """Turn the outermost pixels of an image into Gaussian noise."""
    if c:
        mask = torch.zeros(batch_size, h, w, c)
        mask[:, :num_pixels, :, :] = torch.randn(batch_size, num_pixels, w, c)
        mask[:, :, :num_pixels, :] = torch.randn(batch_size, h, num_pixels, c)
        mask[:, -num_pixels:, :, :] = torch.randn(batch_size, num_pixels, w, c)
        mask[:, :, -num_pixels:, :] = torch.randn(batch_size, h, num_pixels, c)

    else:
        mask = torch.zeros(batch_size, h, w)
        mask[:, :num_pixels, :] = torch.randn(batch_size, num_pixels, w)
        mask[:, -num_pixels:, :] = torch.randn(batch_size, num_pixels, w)
        mask[:, :, :num_pixels] = torch.randn(batch_size, h, num_pixels)
        mask[:, :, -num_pixels:] = torch.randn(batch_size, h, num_pixels)
    return mask


def add_noise_border(image: torch.Tensor, random_seed: int | None = None, shuffle_batch: bool = False):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: The image tensor to add noise to.
        random_seed: The random seed to use for adding noise.
                        It is not used for shuffling the batch.
                        The reason is that I want to be able to add the same noise but to different images.
        shuffle_batch: Whether to shuffle the batch of noise_masks.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    if len(image.shape) == 3:
        batch_size, h, w = image.shape
        c = 0
    else:
        batch_size, h, w, c = image.shape
    noise_mask = make_noise_mask(batch_size, h, w, c)
    if shuffle_batch:
        torch.seed()  # The seed for batch-shuffling should be random
        noise_mask = noise_mask[torch.randperm(batch_size)]
    noisy_image = torch.where(noise_mask != 0, noise_mask, image)

    return noisy_image


class AddNoiseBorder:
    def __init__(self, random_seed: int | None = None, shuffle_batch: bool = False):
        self.random_seed = random_seed
        self.shuffle_batch = shuffle_batch

    def __call__(self, image: torch.Tensor):
        return add_noise_border(image, self.random_seed, self.shuffle_batch)
    

def get_transforms(how: Literal["border", "additivey"] = "border", noise_level: float = 0.05):
    return transforms.Compose([
        transforms.ToTensor(),
        AddNoiseBorder() if how == "border" else AddNoiseAdditivey(noise_level),
    ])


def add_noise_additivey(image: torch.Tensor, noise_level: float):
    """
    Add Gaussian noise to an image. Quadratically decay the noise towards the center of the image,
    so that the noise mostly impacts the borders.
    
    Args:
        image: The image tensor to add noise to.
        noise_level: The standard deviation of the Gaussian noise.
    """
    noise = torch.randn_like(image) * noise_level

    # Quadratically decay the noise towards the center of the image
    h, w = image.shape[-2:]
    x = torch.arange(w).float()
    y = torch.arange(h).float()
    X, Y = torch.meshgrid(x, y)
    center_x = w / 2
    center_y = h / 2
    
    distance_x = (X - center_x) / center_x
    distance_y = (Y - center_y) / center_y
    distance_x = einops.repeat(distance_x, "h w -> 1 1 h w")
    distance_y = einops.repeat(distance_y, "h w -> 1 1 h w")
    distance = distance_x * distance_y
    if len(image.shape) == 3:
        distance = einops.rearrange(distance, "1 1 h w -> 1 h w")

    try:
        noise *= distance
    except RuntimeError as e:
        print(f"{image.shape=}, {noise.shape=}, {distance.shape=}")
        raise e

    noisy_image = image + noise
    if len(image.shape) == 4:
        noisy_image = einops.rearrange(noisy_image, "b c h w -> b h w c")
    return noisy_image


class AddNoiseAdditivey:
    def __init__(self, noise_level: float):
        self.noise_level = noise_level

    def __call__(self, image: torch.Tensor):
        return add_noise_additivey(image, self.noise_level)
    

def generate_transforms(
        same_seed_for_n_batches: int = 1,
        shuffle_batch: bool = False,
):
    random_seed = None
    idx = 0
    while True:
        if (same_seed_for_n_batches > 1) and (idx % same_seed_for_n_batches == 0):
            torch.seed()
            random_seed = torch.randint(0, 2**32, (1,)).item()
        else:
            random_seed = None

        yield transforms.Compose([
            transforms.ToTensor(),
            AddNoiseBorder(random_seed=random_seed, shuffle_batch=shuffle_batch),
        ])
