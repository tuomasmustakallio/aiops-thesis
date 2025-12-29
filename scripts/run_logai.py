#!/usr/bin/env python3
"""Run LogAI anomaly detection on normalized CI logs.

Outputs anomaly scores per run to results/logai_scores.csv.

Requires: logai package (pip install logai)
"""
import argparse
import csv
from pathlib import Path

# TODO: Import LogAI components once installed
# from logai.algorithms.anomaly_detection_algo import AnomalyDetectionConfig
# from logai.analysis.anomaly_detector import AnomalyDetector


def load_logs_for_run(run_dir: Path) -> list[str]:
    """Load all normalized log lines for a run."""
    lines = []
    for log_file in run_dir.rglob('*.log'):
        with open(log_file, 'r') as f:
            lines.extend(f.readlines())
    return lines


def compute_anomaly_score(logs: list[str]) -> float:
    """Compute anomaly score using LogAI.

    TODO: Implement actual LogAI integration.
    Current placeholder returns a dummy score based on error count.
    """
    # Placeholder implementation
    error_count = sum(1 for line in logs if line.startswith('ERROR'))
    total = len(logs) if logs else 1

    # Dummy score: higher error ratio = higher anomaly score
    score = min(1.0, error_count / total * 10)

    return score


def main():
    parser = argparse.ArgumentParser(description="Run LogAI anomaly detection")
    parser.add_argument("--input", default="artifacts/normalized", help="Normalized logs directory")
    parser.add_argument("--output", default="results/logai_scores.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for run_dir in sorted(input_dir.iterdir()):
        if run_dir.is_dir():
            run_id = run_dir.name
            print(f"Processing run {run_id}...")

            logs = load_logs_for_run(run_dir)
            score = compute_anomaly_score(logs)
            predicted_anomaly = score >= args.threshold

            results.append({
                'run_id': run_id,
                'anomaly_score': round(score, 4),
                'predicted_anomaly': predicted_anomaly,
                'log_lines': len(logs)
            })
            print(f"  Score: {score:.4f}, Anomaly: {predicted_anomaly}")

    # Write results
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'anomaly_score', 'predicted_anomaly', 'log_lines'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
