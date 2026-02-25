#!/usr/bin/env python3
"""Evaluate Dynatrace anomaly detection on CI/CD pipeline logs.

Uses simple static threshold: runs with error_count > 0 are flagged as anomalous.
This evaluates Dynatrace's log aggregation and querying capabilities rather than
ML-based anomaly detection (which would require longer baselines).

The evaluation uses DQL queries to retrieve error counts per run_id, then compares
against ground truth labels.

Usage:
    python scripts/evaluate_dynatrace.py --threshold 0

Requirements:
    pip install requests
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def load_ground_truth(path: Path) -> dict[str, dict]:
    """Load ground truth metadata for each run from runs.csv."""
    metadata = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["run_id"]] = {
                "outcome": row["outcome"],
                "failure_class": row["failure_class"],
                "timestamp": row["timestamp"],
            }
    return metadata


def query_dynatrace_logs(url: str, token: str, query: str) -> list[dict]:
    """Execute a DQL query against Dynatrace logs."""
    endpoint = f"{url.rstrip('/')}/api/v1/queryExecutionEngine/query/execute"
    headers = {
        "Authorization": f"Api-Token {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "fetchTimeoutSeconds": 60,
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            # DQL returns results in data["result"]["records"]
            if "result" in data and "records" in data["result"]:
                return data["result"]["records"]
            else:
                print(f"WARNING: Unexpected response structure: {data}")
                return []
        else:
            print(f"ERROR {resp.status_code}: {resp.text[:500]}")
            return []

    except requests.exceptions.RequestException as exc:
        print(f"ERROR: Request failed: {exc}")
        return []


def _load_dotenv():
    """Load .env file into os.environ."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate Dynatrace log analysis on CI/CD pipeline failures",
    )
    parser.add_argument(
        "--url", default=os.environ.get("DYNATRACE_URL"),
        help="Dynatrace environment URL (or set DYNATRACE_URL env var)",
    )
    parser.add_argument(
        "--token", default=os.environ.get("DYNATRACE_API_TOKEN"),
        help="Dynatrace API token (or set DYNATRACE_API_TOKEN env var)",
    )
    parser.add_argument(
        "--ground-truth", default="data/runs.csv",
        help="Ground truth CSV (default: data/runs.csv)",
    )
    parser.add_argument(
        "--threshold", type=int, default=0,
        help="ERROR count threshold to flag run as anomalous (default: 0 = any error)",
    )
    parser.add_argument(
        "--output", default="results/dynatrace_scores.csv",
        help="Output CSV path (default: results/dynatrace_scores.csv)",
    )
    args = parser.parse_args()

    if not REQUESTS_AVAILABLE:
        print("ERROR: requests package is not installed")
        print("Install with: pip install requests")
        sys.exit(1)

    if not args.url:
        print("ERROR: No Dynatrace URL provided.")
        print("Set DYNATRACE_URL in .env or pass --url")
        sys.exit(1)

    if not args.token:
        print("ERROR: No Dynatrace API token provided.")
        print("Set DYNATRACE_API_TOKEN in .env or pass --token")
        sys.exit(1)

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        metadata = load_ground_truth(gt_path)
        print(f"Loaded ground truth for {len(metadata)} runs")
    else:
        print(f"ERROR: Ground truth not found at {gt_path}")
        sys.exit(1)

    # Query Dynatrace for error counts
    print("\nQuerying Dynatrace for ERROR line counts per run_id…")
    query = """fetch logs
| filter loglevel == "ERROR"
| summarize error_count = count(), by: {run_id}
| sort error_count desc"""

    records = query_dynatrace_logs(args.url, args.token, query)

    if not records:
        print("ERROR: No records returned from Dynatrace query")
        sys.exit(1)

    print(f"Retrieved {len(records)} runs with data from Dynatrace")

    # Build results with ground truth labels
    results = []
    for record in records:
        run_id = record.get("run_id")
        error_count = record.get("error_count", 0)

        # Get ground truth
        gt = metadata.get(run_id, {})
        outcome = gt.get("outcome", "unknown")
        failure_class = gt.get("failure_class", "unknown")

        # Classify: error_count > threshold means anomalous
        predicted_anomaly = error_count > args.threshold

        results.append({
            "run_id": run_id,
            "error_count": error_count,
            "predicted_anomaly": predicted_anomaly,
            "outcome": outcome,
            "failure_class": failure_class,
        })

    # Include runs with 0 errors (success runs)
    runs_in_results = {r["run_id"] for r in results}
    for run_id, gt in metadata.items():
        if run_id not in runs_in_results:
            results.append({
                "run_id": run_id,
                "error_count": 0,
                "predicted_anomaly": False,
                "outcome": gt.get("outcome", "unknown"),
                "failure_class": gt.get("failure_class", "unknown"),
            })

    # Sort by run_id
    results.sort(key=lambda r: r["run_id"])

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "error_count", "predicted_anomaly", "outcome", "failure_class"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}")

    # Compute metrics (using only labeled runs)
    labeled_results = [r for r in results if r["outcome"] != "unknown"]

    tp = sum(1 for r in labeled_results
             if r["predicted_anomaly"] and r["outcome"] == "failure")
    fp = sum(1 for r in labeled_results
             if r["predicted_anomaly"] and r["outcome"] == "success")
    fn = sum(1 for r in labeled_results
             if not r["predicted_anomaly"] and r["outcome"] == "failure")
    tn = sum(1 for r in labeled_results
             if not r["predicted_anomaly"] and r["outcome"] == "success")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labeled_results) if len(labeled_results) > 0 else 0.0

    print(f"\n--- Metrics (threshold={args.threshold}) ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Detection by failure class
    print(f"\n--- Detection by Failure Class ---")
    failure_classes = set(r["failure_class"] for r in labeled_results if r["failure_class"] != "unknown")
    for fc in sorted(failure_classes):
        runs = [r for r in labeled_results if r["failure_class"] == fc]
        detected = sum(1 for r in runs if r["predicted_anomaly"])
        total = len(runs)
        error_counts = [r["error_count"] for r in runs]
        print(f"{fc:18s}: {detected}/{total} detected, error_count range {min(error_counts)}-{max(error_counts)}")

    # Write metrics
    metrics_path = Path(str(args.output).replace("_scores.csv", "_metrics.csv"))
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "value"],
        )
        writer.writeheader()
        writer.writerows([
            {"metric": "precision", "value": precision},
            {"metric": "recall", "value": recall},
            {"metric": "f1_score", "value": f1},
            {"metric": "accuracy", "value": accuracy},
            {"metric": "true_positives", "value": tp},
            {"metric": "false_positives", "value": fp},
            {"metric": "false_negatives", "value": fn},
            {"metric": "true_negatives", "value": tn},
        ])
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
