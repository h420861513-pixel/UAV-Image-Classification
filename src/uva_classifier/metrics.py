def compute_binary_metrics(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    tp = tn = fp = fn = 0
    for prediction, target in zip(predictions, targets):
        if prediction == 1 and target == 1:
            tp += 1
        elif prediction == 0 and target == 0:
            tn += 1
        elif prediction == 1 and target == 0:
            fp += 1
        elif prediction == 0 and target == 1:
            fn += 1

    total = max(len(targets), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

