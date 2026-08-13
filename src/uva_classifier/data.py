from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


CLASS_NAMES = ["background", "UAV"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class BinaryImageDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples = []

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        for class_name in CLASS_NAMES:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            image_paths = sorted(
                path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            label = CLASS_TO_INDEX[class_name]
            self.samples.extend((image_path, label) for image_path in image_paths)

        self.class_counts = self._compute_class_counts()

    def _compute_class_counts(self):
        counts = {class_name: 0 for class_name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path)


def build_train_transform(image_size, model_name):
    if model_name == "simple_cnn":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size, model_name):
    if model_name == "simple_cnn":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_inference_transform(image_size, model_name):
    return build_eval_transform(image_size=image_size, model_name=model_name)


def _build_weighted_sampler(dataset):
    counts = dataset.class_counts
    sample_weights = []
    for _, label in dataset.samples:
        class_name = CLASS_NAMES[label]
        sample_weights.append(1.0 / counts[class_name])
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _compute_class_weights(dataset):
    counts = dataset.class_counts
    total = sum(counts.values())
    weights = [total / (len(CLASS_NAMES) * counts[class_name]) for class_name in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir,
    image_size,
    batch_size,
    num_workers=0,
    imbalance_strategy="none",
    model_name="simple_cnn",
    eval_batch_size=None,
):
    data_dir = Path(data_dir)
    eval_batch_size = eval_batch_size or batch_size
    train_dataset = BinaryImageDataset(data_dir / "train", transform=build_train_transform(image_size, model_name))
    val_dataset = BinaryImageDataset(data_dir / "val", transform=build_eval_transform(image_size, model_name))
    test_dataset = BinaryImageDataset(data_dir / "test", transform=build_eval_transform(image_size, model_name))

    sampler = _build_weighted_sampler(train_dataset) if imbalance_strategy == "weighted_sampler" else None
    shuffle = sampler is None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": CLASS_NAMES,
        "class_counts": train_dataset.class_counts,
        "class_weights": _compute_class_weights(train_dataset),
    }


def build_eval_dataloader(data_dir, split, image_size, batch_size, num_workers=0, model_name="simple_cnn"):
    dataset = BinaryImageDataset(Path(data_dir) / split, transform=build_eval_transform(image_size, model_name))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
