import os

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


TRAIN_DIR = "./dataset1/train"
VAL_DIR = "./dataset1/val"
TEST_DIR = "./dataset1/test"


class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None, return_paths=False):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return image, label, img_path
        return image, label


def build_cnn_dataloaders(train_batch_size=128, eval_batch_size=100):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageDataset(
        os.path.join(TRAIN_DIR, "UAV"),
        os.path.join(TRAIN_DIR, "background"),
        transform=transform,
        return_paths=True,
    )
    val_dataset = ImageDataset(
        os.path.join(VAL_DIR, "UAV"),
        os.path.join(VAL_DIR, "background"),
        transform=transform,
        return_paths=True,
    )
    test_dataset = ImageDataset(
        os.path.join(TEST_DIR, "UAV"),
        os.path.join(TEST_DIR, "background"),
        transform=transform,
        return_paths=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def build_resnet_dataloaders(train_batch_size=128, eval_batch_size=100):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageDataset(
        os.path.join(TRAIN_DIR, "UAV"),
        os.path.join(TRAIN_DIR, "background"),
        transform=transform,
        return_paths=False,
    )
    val_dataset = ImageDataset(
        os.path.join(VAL_DIR, "UAV"),
        os.path.join(VAL_DIR, "background"),
        transform=transform,
        return_paths=False,
    )
    test_dataset = ImageDataset(
        os.path.join(TEST_DIR, "UAV"),
        os.path.join(TEST_DIR, "background"),
        transform=transform,
        return_paths=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
