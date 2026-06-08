from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def get_dataloaders(
    dataset_path,
    image_size=224,
    batch_size=16,
    random_state=42
):
    dataset_path = Path(dataset_path)

    base_dataset = datasets.ImageFolder(root=dataset_path)

    class_names = base_dataset.classes
    targets = base_dataset.targets
    indices = np.arange(len(base_dataset))

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        targets,
        test_size=0.30,
        random_state=random_state,
        stratify=targets
    )

    val_idx, test_idx, val_y, test_y = train_test_split(
        temp_idx,
        temp_y,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_y
    )

    train_transform, test_transform = get_transforms(image_size)

    train_full = datasets.ImageFolder(
        root=dataset_path,
        transform=train_transform
    )

    val_full = datasets.ImageFolder(
        root=dataset_path,
        transform=test_transform
    )

    test_full = datasets.ImageFolder(
        root=dataset_path,
        transform=test_transform
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)
    test_dataset = Subset(test_full, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, class_names