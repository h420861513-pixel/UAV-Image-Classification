import argparse
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
from src.uav_classifier.data import build_eval_dataloader
from src.uav_classifier.engine import evaluate
from src.uav_classifier.models import build_model
from src.uav_classifier.utils import save_json

EVAL_CONFIG = {
    "checkpoint": r"checkpoints\resnet-18\epoch_011.pt",
    "data_dir": "dataset1",
    "split": "test",
    "batch_size": 100,
    "num_workers": 4,
    "device": "cuda",
    "save_path": r"outputs\resnet-18\test_metrics.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved UAV classifier checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override. Default comes from EVAL_CONFIG in evaluate.py.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional split override.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader workers override.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override: cpu or cuda.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional metrics JSON output path override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_value = args.checkpoint or EVAL_CONFIG["checkpoint"]
    data_dir = args.data_dir or EVAL_CONFIG["data_dir"]
    split = args.split or EVAL_CONFIG["split"]
    batch_size = args.batch_size or EVAL_CONFIG["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else EVAL_CONFIG["num_workers"]
    device_override = args.device or EVAL_CONFIG["device"]
    save_path_override = args.save_path if args.save_path is not None else EVAL_CONFIG["save_path"]

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

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

    dataloader = build_eval_dataloader(
        data_dir=PROJECT_ROOT / data_dir,
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
    )
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, dataloader, criterion, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(
        f"Loss={metrics['loss']:.4f} Acc={metrics['accuracy']:.4f} "
        f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    if save_path_override:
        save_path = Path(save_path_override)
        if not save_path.is_absolute():
            save_path = PROJECT_ROOT / save_path
    else:
        save_path = PROJECT_ROOT / "outputs" / model_name / f"{split}_metrics.json"
    save_json(save_path, metrics)
    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
