"""
Vision datasets for OrScale experiments.

Provides builders for CIFAR-10 (via torchvision) and ImageNet-1K (via
``ImageFolder`` over a standard ILSVRC directory). Both return
``torch.utils.data.DataLoader`` instances with DDP-compatible samplers when
``world_size > 1``.

CIFAR-10 augmentation follows the standard recipe (RandomCrop(32, pad=4) +
RandomHorizontalFlip + Normalize), matching LAMB's Table 6 setup.

ImageNet augmentation follows Goyal et al. (2017) / LAMB Table 5:
    - train: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize
    - val:   Resize(256) + CenterCrop(224) + Normalize
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def build_cifar10_loaders(
    root: str = "data/cifar10",
    batch_size: int = 512,
    eval_batch_size: int | None = None,
    augment: bool = True,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    download: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train/val loaders.

    Args:
        root: Data directory (created if missing).
        batch_size: Training batch size *per rank*.
        eval_batch_size: Eval batch size per rank (defaults to ``batch_size``).
        augment: Apply RandomCrop+HFlip to the training split.
        num_workers: Worker processes per loader.
        rank, world_size: Set for DDP; uses ``DistributedSampler``.
        download: Download the dataset if it is not present.
        seed: Seed for the DistributedSampler.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as err:
        raise ImportError(
            "torchvision is required for CIFAR-10 loaders. pip install torchvision"
        ) from err

    eval_batch_size = eval_batch_size or batch_size
    Path(root).mkdir(parents=True, exist_ok=True)

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if augment:
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tfm = transforms.Compose([transforms.ToTensor(), normalize])

    val_tfm = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(root=root, train=True, transform=train_tfm, download=download)
    val_set = datasets.CIFAR10(root=root, train=False, transform=val_tfm, download=download)

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=seed,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# ImageNet-1K
# ---------------------------------------------------------------------------

def build_imagenet_loaders(
    root: str,
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    image_size: int = 224,
    num_workers: int = 8,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build ImageNet train/val loaders using torchvision ``ImageFolder``.

    The ``root`` directory must contain ``train/`` and ``val/`` subdirectories
    in the standard ILSVRC layout (one folder per class).

    Args:
        root: Path to ImageNet root (containing ``train`` and ``val``).
        batch_size: Training batch size *per rank* (total = batch_size * world_size).
        eval_batch_size: Eval batch size per rank (defaults to ``batch_size``).
        image_size: Crop size for training and eval (default 224).
        num_workers: Worker processes per loader.
        rank, world_size: DDP setup; uses ``DistributedSampler`` when >1.
        seed: Seed for the DistributedSampler.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as err:
        raise ImportError(
            "torchvision is required for ImageNet loaders. pip install torchvision"
        ) from err

    eval_batch_size = eval_batch_size or batch_size

    train_dir = Path(root) / "train"
    val_dir = Path(root) / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet root {root!r} must contain 'train/' and 'val/' subdirectories."
        )

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.ImageFolder(str(train_dir), transform=train_tfm)
    val_set = datasets.ImageFolder(str(val_dir), transform=val_tfm)

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=seed,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vision_dataloaders(
    data_config: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Dispatch to CIFAR-10 / ImageNet builders based on ``data_config.name``."""
    name = data_config.get("name", "cifar10").lower()
    num_workers = int(data_config.get("num_workers", 4))
    eval_batch_size = data_config.get("eval_batch_size")

    if name == "cifar10":
        return build_cifar10_loaders(
            root=data_config.get("root", "data/cifar10"),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            augment=data_config.get("augment", True),
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            download=data_config.get("download", True),
            seed=seed,
        )
    if name == "imagenet":
        return build_imagenet_loaders(
            root=data_config["root"],
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            image_size=int(data_config.get("image_size", 224)),
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
    raise ValueError(f"Unknown vision dataset: {name!r} (choose from cifar10, imagenet)")
