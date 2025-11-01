"""
DataModule abstraction for flexible dataset loading.
Supports built-in torchvision datasets and ImageFolder custom datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    name: str = "mnist"  # mnist | fashion_mnist | cifar10 | cifar100 | image_folder
    root: str = "./data"
    batch_size: int = 64
    val_split: float = 0.2
    num_workers: int = 0
    download: bool = True
    image_folder_root: Optional[str] = None  # for name == image_folder


class DataModule:
    def __init__(self, config: DatasetConfig | Dict[str, Any]):
        if isinstance(config, dict):
            self.config = DatasetConfig(**config)
        else:
            self.config = config

    def build_loaders(self) -> Tuple[DataLoader, DataLoader]:
        name = self.config.name.lower()

        if name in ["mnist", "fashion_mnist"]:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        if name == "mnist":
            full = datasets.MNIST(self.config.root, train=True, download=self.config.download, transform=transform)
        elif name == "fashion_mnist":
            full = datasets.FashionMNIST(self.config.root, train=True, download=self.config.download, transform=transform)
        elif name == "cifar10":
            full = datasets.CIFAR10(self.config.root, train=True, download=self.config.download, transform=transform)
        elif name == "cifar100":
            full = datasets.CIFAR100(self.config.root, train=True, download=self.config.download, transform=transform)
        elif name == "image_folder":
            if not self.config.image_folder_root:
                raise ValueError("image_folder_root must be set for name='image_folder'")
            full = datasets.ImageFolder(self.config.image_folder_root, transform=transform)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        val_size = int(self.config.val_split * len(full))
        train_size = len(full) - val_size
        train_ds, val_ds = random_split(full, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        return train_loader, val_loader