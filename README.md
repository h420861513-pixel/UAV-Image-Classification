# UAV Image Classification

A PyTorch-based binary image classification project for detecting whether a UAV is present in an image.
The repository keeps the original CNN and ResNet18 experiment scripts while adding cleaner evaluation and inference entry points for reproducibility and GitHub presentation.

## Features

- Two training options: `simple_cnn` and `resnet18`
- Original experiment scripts preserved for reproducibility
- Unified evaluation script: `evaluate.py`
- Unified single-image inference script: `infer.py`
- Ready-to-use default checkpoint configuration for testing and demo
- Checkpoints and evaluation outputs are stored separately

## Dataset Structure

The project uses the following directory layout:

```text
dataset1/
  train/
    UAV/
    background/
  val/
    UAV/
    background/
  test/
    UAV/
    background/
```

Verified dataset statistics in the current workspace:

- `train`
  - `UAV`: 1000
  - `background`: 10000
- `val`
  - `UAV`: 400
  - `background`: 396
- `test`
  - `UAV`: 600
  - `background`: 594

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU acceleration, make sure your environment already has a CUDA-enabled PyTorch installation.

## Training

Train the simple CNN model:

```bash
python train.py --model simple_cnn
```

Train the ResNet18 model:

```bash
python train.py --model resnet18
```

Notes:

- `train.py` is a unified launcher
- `simple_cnn` maps to [CNN.py](/D:/UAV-dataset/CNN.py)
- `resnet18` maps to [resnet-18.py](/D:/UAV-dataset/resnet-18.py)
- Model checkpoints are saved under `checkpoints/simple_cnn/` and `checkpoints/resnet-18/`

## Evaluation

Run evaluation with the default checkpoint configuration:

```bash
python evaluate.py
```

Current default evaluation config in [evaluate.py](/D:/UAV-dataset/evaluate.py):

- Checkpoint: `checkpoints/resnet-18/epoch_011.pt`
- Dataset: `dataset1`
- Split: `test`
- Device: `cuda`
- Metrics output: `outputs/resnet-18/test_metrics.json`

If you want to test a different checkpoint or split, edit `EVAL_CONFIG` at the top of `evaluate.py`.

## Inference

Run single-image inference with the default configuration:

```bash
python infer.py
```

Current default inference config in [infer.py](/D:/UAV-dataset/infer.py):

- Checkpoint: `checkpoints/resnet-18/epoch_011.pt`
- Image: `dataset1/test/UAV/phantom05_0018.jpg`
- Device: `cuda`

If you want to test another image or another checkpoint, edit `INFER_CONFIG` at the top of `infer.py`.

## Current Result

Default evaluation target:

- Checkpoint: `checkpoints/resnet-18/epoch_011.pt`

Test set performance:

- `Loss = 0.0036`
- `Accuracy = 0.9983`
- `Precision = 1.0000`
- `Recall = 0.9967`
- `F1 = 0.9983`

Confusion matrix:

```python
{'tn': 594, 'fp': 0, 'fn': 2, 'tp': 598}
```

## Metric Explanation

- `Accuracy`: overall percentage of correct predictions. `0.9983` means `99.83%` of all test images were classified correctly.
- `Precision`: among all images predicted as `UAV`, how many were actually `UAV`. `1.0000` means there were no false positives.
- `Recall`: among all real `UAV` images, how many were successfully detected. `0.9967` means only 2 UAV images were missed.
- `F1`: the harmonic mean of precision and recall, useful for balancing false positives and false negatives.
- `Loss`: optimization objective value. Lower is generally better, although the classification metrics above are more intuitive for presentation.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- train.py
|-- evaluate.py
|-- infer.py
|-- CNN.py
|-- resnet-18.py
|-- checkpoints/
|-- outputs/
`-- src/
    `-- uav_classifier/
        |-- data.py
        |-- engine.py
        |-- metrics.py
        |-- models.py
        `-- utils.py
```
