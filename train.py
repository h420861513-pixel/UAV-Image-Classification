import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_SCRIPT_MAP = {
    "simple_cnn": PROJECT_ROOT / "CNN.py",
    "resnet18": PROJECT_ROOT / "resnet-18.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dispatch training to the original UAV classification scripts."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=sorted(MODEL_SCRIPT_MAP.keys()),
        help="Which original training script to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = MODEL_SCRIPT_MAP[args.model]

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    print(f"Selected model: {args.model}")
    print(f"Running script: {script_path.name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
