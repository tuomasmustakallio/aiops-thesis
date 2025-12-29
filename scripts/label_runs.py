#!/usr/bin/env python3
"""Helper script to label pipeline runs interactively.

Fetches recent runs from GitHub and prompts for labels.
Appends to data/runs.csv.
"""
import argparse
import csv
import subprocess
import json
from pathlib import Path
from datetime import datetime


FAILURE_CLASSES = ['none', 'backend_test', 'frontend_build', 'dependency', 'deploy', 'infra']


def run_gh(args: list[str]) -> str:
    """Run gh CLI command."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True, check=True)
    return result.stdout


def get_existing_run_ids(csv_path: Path) -> set[str]:
    """Get set of already-labeled run IDs."""
    if not csv_path.exists():
        return set()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return {row['run_id'] for row in reader}


def fetch_runs(repo: str, limit: int) -> list[dict]:
    """Fetch recent workflow runs from GitHub."""
    output = run_gh([
        "run", "list",
        "--repo", repo,
        "--limit", str(limit),
        "--json", "databaseId,headSha,createdAt,status,conclusion,name"
    ])
    return json.loads(output)


def prompt_label(run: dict) -> dict | None:
    """Interactively prompt for run label."""
    print(f"\n{'='*60}")
    print(f"Run ID:     {run['databaseId']}")
    print(f"Commit:     {run['headSha'][:8]}")
    print(f"Workflow:   {run['name']}")
    print(f"Created:    {run['createdAt']}")
    print(f"Status:     {run['status']}")
    print(f"Conclusion: {run['conclusion']}")

    # Determine outcome
    if run['conclusion'] == 'success':
        outcome = 'success'
        default_class = 'none'
    else:
        outcome = 'failure'
        default_class = None

    print(f"\nOutcome: {outcome}")

    if outcome == 'failure':
        print(f"\nFailure classes: {', '.join(FAILURE_CLASSES)}")
        while True:
            failure_class = input(f"Enter failure class: ").strip().lower()
            if failure_class in FAILURE_CLASSES:
                break
            print(f"Invalid class. Choose from: {FAILURE_CLASSES}")
    else:
        failure_class = 'none'

    note = input("Note (optional): ").strip()

    confirm = input("\nSave this label? [Y/n]: ").strip().lower()
    if confirm == 'n':
        return None

    return {
        'run_id': str(run['databaseId']),
        'commit_sha': run['headSha'],
        'timestamp': run['createdAt'],
        'outcome': outcome,
        'failure_class': failure_class,
        'note': note
    }


def append_to_csv(csv_path: Path, label: dict):
    """Append a label to the CSV file."""
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'commit_sha', 'timestamp', 'outcome', 'failure_class', 'note'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(label)


def main():
    parser = argparse.ArgumentParser(description="Label pipeline runs interactively")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument("--output", default="data/runs.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=20, help="Max runs to fetch")
    args = parser.parse_args()

    csv_path = Path(args.output)
    existing = get_existing_run_ids(csv_path)
    print(f"Already labeled: {len(existing)} runs")

    runs = fetch_runs(args.repo, args.limit)
    print(f"Fetched {len(runs)} recent runs")

    new_runs = [r for r in runs if str(r['databaseId']) not in existing]
    print(f"New runs to label: {len(new_runs)}")

    if not new_runs:
        print("No new runs to label.")
        return

    labeled = 0
    for run in new_runs:
        label = prompt_label(run)
        if label:
            append_to_csv(csv_path, label)
            labeled += 1
            print(f"Labeled run {label['run_id']} as {label['outcome']}/{label['failure_class']}")

        cont = input("\nContinue to next run? [Y/n]: ").strip().lower()
        if cont == 'n':
            break

    print(f"\nLabeled {labeled} runs. Total in CSV: {len(existing) + labeled}")


if __name__ == "__main__":
    main()
