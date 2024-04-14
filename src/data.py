import torch
import torchvision.transforms as transforms


def make_noise_mask(batch_size: int, h: int, w: int, c: int, num_pixels: int = 1):
    """Turn the outermost pixels of an image into Gaussian noise."""
    mask = torch.zeros(batch_size, h, w, c)
    mask[:, :num_pixels, :, :] = torch.randn(batch_size, num_pixels, w, c)
    mask[:, :, :num_pixels, :] = torch.randn(batch_size, h, num_pixels, c)
    mask[:, -num_pixels:, :, :] = torch.randn(batch_size, num_pixels, w, c)
    mask[:, :, -num_pixels:, :] = torch.randn(batch_size, h, num_pixels, c)
    return mask


def add_noise(image: torch.Tensor, random_seed: int | None = None, shuffle_batch: bool = False):
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
    batch_size, h, w, c = image.shape
    noise_mask = make_noise_mask(batch_size, h, w, c)
    if shuffle_batch:
        torch.seed()  # The seed for batch-shuffling should be random
        noise_mask = noise_mask[torch.randperm(batch_size)]
    noisy_image = torch.where(noise_mask != 0, noise_mask, image)

    return noisy_image


class AddNoise:
    def __init__(self, random_seed: int | None = None, shuffle_batch: bool = False):
        self.random_seed = random_seed
        self.shuffle_batch = shuffle_batch

    def __call__(self, image: torch.Tensor):
        return add_noise(image, self.random_seed, self.shuffle_batch)
    

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
            AddNoise(random_seed=random_seed, shuffle_batch=shuffle_batch),
        ])
