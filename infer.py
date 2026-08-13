import argparse
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
from src.uav_classifier.data import build_inference_transform
from src.uav_classifier.models import build_model

INFER_CONFIG = {
    "checkpoint": r"checkpoints\resnet-18\epoch_011.pt",
    "image": r"dataset1\test\UAV\phantom05_0018.jpg",
    "device": "cuda",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override. Default comes from INFER_CONFIG in infer.py.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image override. Default comes from INFER_CONFIG in infer.py.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override: cpu or cuda.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_value = args.checkpoint or INFER_CONFIG["checkpoint"]
    image_value = args.image or INFER_CONFIG["image"]
    device_override = args.device or INFER_CONFIG["device"]

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_value)
    image_path = Path(image_value)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    model_name = metadata.get("model_name", "resnet18")
    image_size = metadata.get("image_size", 224)
    dropout = metadata.get("dropout", 0.3)
    class_names = metadata.get("class_names", ["background", "UAV"])

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        dropout=dropout,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = build_inference_transform(image_size=image_size, model_name=model_name)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        predicted_index = int(torch.argmax(logits, dim=1).item())

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[predicted_index]}")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
