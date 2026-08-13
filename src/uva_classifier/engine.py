import torch

from src.uav_classifier.metrics import compute_binary_metrics


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")
        self.should_stop = False

    def step(self, current_value):
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    metrics = compute_binary_metrics(predictions, targets)
    metrics["loss"] = running_loss / max(len(dataloader), 1)
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

    metrics = compute_binary_metrics(predictions, targets)
    metrics["loss"] = running_loss / max(len(dataloader), 1)
    return metrics
