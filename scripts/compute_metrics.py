#!/usr/bin/env python3
"""Compute precision/recall/F1 comparing LogAI predictions to ground truth.

Reads:
  - data/runs.csv (ground truth labels)
  - results/logai_scores.csv (LogAI predictions)

Outputs:
  - results/metrics.csv (summary metrics)
  - Prints a results table to stdout
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def load_ground_truth(path: Path) -> dict[str, dict]:
    """Load ground truth labels from runs.csv."""
    labels = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row['run_id']
            labels[run_id] = {
                'outcome': row['outcome'],
                'failure_class': row['failure_class'],
                'is_failure': row['outcome'] == 'failure'
            }
    return labels


def load_predictions(path: Path) -> dict[str, dict]:
    """Load LogAI predictions from logai_scores.csv."""
    predictions = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row['run_id']
            predictions[run_id] = {
                'anomaly_score': float(row['anomaly_score']),
                'predicted_anomaly': row['predicted_anomaly'].lower() == 'true'
            }
    return predictions


def compute_binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Compute precision, recall, F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4)
    }


def compute_per_class_metrics(labels: dict, predictions: dict) -> dict:
    """Compute metrics per failure class."""
    class_metrics = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    for run_id, truth in labels.items():
        if run_id in predictions:
            pred = predictions[run_id]['predicted_anomaly']
            failure_class = truth['failure_class']
            is_failure = truth['is_failure']

            class_metrics[failure_class]['y_true'].append(is_failure)
            class_metrics[failure_class]['y_pred'].append(pred)

    results = {}
    for cls, data in class_metrics.items():
        if data['y_true']:
            results[cls] = compute_binary_metrics(data['y_true'], data['y_pred'])

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--ground-truth", default="data/runs.csv", help="Ground truth CSV")
    parser.add_argument("--predictions", default="results/logai_scores.csv", help="Predictions CSV")
    parser.add_argument("--output", default="results/metrics.csv", help="Output metrics CSV")
    args = parser.parse_args()

    labels = load_ground_truth(Path(args.ground_truth))
    predictions = load_predictions(Path(args.predictions))

    # Match runs present in both
    common_runs = set(labels.keys()) & set(predictions.keys())
    print(f"Evaluating {len(common_runs)} runs (of {len(labels)} labeled, {len(predictions)} predicted)")

    if not common_runs:
        print("ERROR: No common runs between ground truth and predictions")
        return

    # Overall metrics
    y_true = [labels[r]['is_failure'] for r in common_runs]
    y_pred = [predictions[r]['predicted_anomaly'] for r in common_runs]
    overall = compute_binary_metrics(y_true, y_pred)

    print("\n=== Overall Metrics ===")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1_score']:.4f}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={overall['true_positives']} FP={overall['false_positives']}")
    print(f"  FN={overall['false_negatives']} TN={overall['true_negatives']}")

    # Per-class metrics
    per_class = compute_per_class_metrics(labels, predictions)
    print("\n=== Per-Class Metrics ===")
    for cls, metrics in sorted(per_class.items()):
        print(f"\n{cls}:")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['precision', overall['precision']])
        writer.writerow(['recall', overall['recall']])
        writer.writerow(['f1_score', overall['f1_score']])
        writer.writerow(['accuracy', overall['accuracy']])
        writer.writerow(['true_positives', overall['true_positives']])
        writer.writerow(['false_positives', overall['false_positives']])
        writer.writerow(['false_negatives', overall['false_negatives']])
        writer.writerow(['true_negatives', overall['true_negatives']])

    print(f"\nMetrics written to {output_path}")


if __name__ == "__main__":
    main()
