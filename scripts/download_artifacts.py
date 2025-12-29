#!/usr/bin/env python3
"""Download GitHub Actions artifacts for analysis.

Requires: gh CLI authenticated with repo access.

Usage:
    python scripts/download_artifacts.py --repo owner/repo --output artifacts/
    python scripts/download_artifacts.py --repo owner/repo --run-id 12345 --output artifacts/
"""
import argparse
import subprocess
import json
import os
from pathlib import Path


def run_gh(args: list[str]) -> str:
    """Run gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def list_runs(repo: str, limit: int = 100) -> list[dict]:
    """List recent workflow runs."""
    output = run_gh([
        "run", "list",
        "--repo", repo,
        "--limit", str(limit),
        "--json", "databaseId,headSha,createdAt,status,conclusion"
    ])
    return json.loads(output)


def download_artifacts(repo: str, run_id: int, output_dir: Path) -> list[str]:
    """Download all artifacts for a run."""
    run_dir = output_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_gh([
        "run", "download", str(run_id),
        "--repo", repo,
        "--dir", str(run_dir)
    ])

    return list(run_dir.iterdir())


def main():
    parser = argparse.ArgumentParser(description="Download GitHub Actions artifacts")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument("--run-id", type=int, help="Specific run ID (optional)")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    parser.add_argument("--limit", type=int, default=100, help="Max runs to fetch")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        runs = [{"databaseId": args.run_id}]
    else:
        runs = list_runs(args.repo, args.limit)
        print(f"Found {len(runs)} runs")

    for run in runs:
        run_id = run["databaseId"]
        print(f"Downloading artifacts for run {run_id}...")
        try:
            downloaded = download_artifacts(args.repo, run_id, output_dir)
            print(f"  Downloaded: {[p.name for p in downloaded]}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e.stderr}")


if __name__ == "__main__":
    main()
