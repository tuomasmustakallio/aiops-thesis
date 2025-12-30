#!/usr/bin/env python3
"""Generate pipeline runs for experiment dataset collection.

This script:
1. Checks out each experiment branch
2. Creates N no-op commits by bumping RUN_MARKER
3. Pushes to trigger CI runs
4. Records placeholder entries in data/runs.csv

Usage:
    python scripts/generate_runs.py                    # Generate all runs (default counts)
    python scripts/generate_runs.py --dry-run          # Preview without making changes
    python scripts/generate_runs.py --branch success --count 5  # Generate 5 success runs only
    python scripts/generate_runs.py --trigger-deploy   # Trigger deploy job for deploy-fail runs

Target counts (default):
    - success: 25
    - backend-fail: 6
    - frontend-fail: 6
    - dep-fail: 6
    - deploy-fail: 2 (requires --trigger-deploy for workflow_dispatch)
"""
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MARKER_FILE = REPO_ROOT / "experiments" / "run_marker.txt"
RUNS_CSV = REPO_ROOT / "data" / "runs.csv"

# Default target counts per branch
DEFAULT_COUNTS = {
    "experiment/success": 25,
    "experiment/backend-fail": 6,
    "experiment/frontend-fail": 6,
    "experiment/dep-fail": 6,
    "experiment/deploy-fail": 2,
}

FAILURE_CLASS_MAP = {
    "experiment/success": "none",
    "experiment/backend-fail": "backend_test",
    "experiment/frontend-fail": "frontend_build",
    "experiment/dep-fail": "dependency",
    "experiment/deploy-fail": "deploy",
}


def run_cmd(args: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    return subprocess.run(args, check=check, capture_output=capture, text=True)


def get_current_branch() -> str:
    """Get current git branch name."""
    result = run_cmd(["git", "branch", "--show-current"])
    return result.stdout.strip()


def read_marker() -> int:
    """Read current RUN_MARKER value."""
    if not MARKER_FILE.exists():
        return 0
    content = MARKER_FILE.read_text().strip()
    for line in content.split('\n'):
        if line.startswith('RUN_MARKER='):
            return int(line.split('=')[1])
    return 0


def write_marker(value: int) -> None:
    """Write new RUN_MARKER value."""
    MARKER_FILE.write_text(f"RUN_MARKER={value}\n")


def bump_and_commit() -> int:
    """Bump marker and commit. Returns new marker value."""
    current = read_marker()
    new_value = current + 1
    write_marker(new_value)

    run_cmd(["git", "add", str(MARKER_FILE)])
    run_cmd(["git", "commit", "-m", f"chore: bump run marker to {new_value}"])

    return new_value


def append_run_placeholder(branch: str, marker: int, commit_sha: str) -> None:
    """Append a placeholder row to runs.csv."""
    failure_class = FAILURE_CLASS_MAP.get(branch, "unknown")
    outcome = "success" if failure_class == "none" else "failure"

    row = {
        "run_id": "",  # To be filled after run completes
        "commit_sha": commit_sha,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": outcome,
        "failure_class": failure_class,
        "note": f"RUN_MARKER={marker}",
    }

    file_exists = RUNS_CSV.exists() and RUNS_CSV.stat().st_size > 0

    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "commit_sha", "timestamp", "outcome", "failure_class", "note"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_commit_sha() -> str:
    """Get current commit SHA."""
    result = run_cmd(["git", "rev-parse", "HEAD"])
    return result.stdout.strip()


def branch_exists_remote(branch: str) -> bool:
    """Check if branch exists on remote."""
    result = run_cmd(["git", "ls-remote", "--heads", "origin", branch], check=False)
    return bool(result.stdout.strip())


def generate_runs_for_branch(branch: str, count: int, dry_run: bool = False, trigger_deploy: bool = False) -> int:
    """Generate N runs for a specific branch. Returns number of runs created."""
    print(f"\n{'='*60}")
    print(f"Branch: {branch}")
    print(f"Target runs: {count}")
    print(f"{'='*60}")

    if not branch_exists_remote(branch):
        print(f"  WARNING: Branch {branch} does not exist on remote.")
        print(f"  Run ./scripts/setup_experiment_branches.sh first")
        return 0

    if dry_run:
        print(f"  [DRY RUN] Would generate {count} runs")
        return count

    # Checkout branch
    run_cmd(["git", "checkout", branch])
    run_cmd(["git", "pull", "origin", branch], check=False)  # May fail if no upstream

    runs_created = 0
    for i in range(count):
        print(f"  Run {i+1}/{count}...", end=" ", flush=True)

        # Bump marker and commit
        marker = bump_and_commit()
        sha = get_commit_sha()

        # Push
        run_cmd(["git", "push", "-u", "origin", branch])

        # Record placeholder
        append_run_placeholder(branch, marker, sha)

        print(f"pushed (marker={marker})")
        runs_created += 1

    # For deploy-fail, trigger via workflow_dispatch if requested
    if branch == "experiment/deploy-fail" and trigger_deploy:
        print(f"\n  Triggering deploy job via workflow_dispatch...")
        print(f"  NOTE: Use GitHub UI or gh CLI to trigger with run_deploy=true")
        print(f"  Command: gh workflow run ci.yml --ref {branch} -f run_deploy=true")

    return runs_created


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline runs for experiment")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--branch", help="Generate runs for specific branch only")
    parser.add_argument("--count", type=int, help="Override run count for --branch")
    parser.add_argument("--trigger-deploy", action="store_true",
                        help="Print instructions for triggering deploy job")
    args = parser.parse_args()

    # Save current branch
    original_branch = get_current_branch()

    # Check for clean working directory
    status = run_cmd(["git", "status", "--porcelain"])
    if status.stdout.strip():
        print("ERROR: Working directory not clean. Commit or stash changes first.")
        sys.exit(1)

    try:
        if args.branch:
            # Single branch mode
            branch = args.branch if args.branch.startswith("experiment/") else f"experiment/{args.branch}"
            count = args.count or DEFAULT_COUNTS.get(branch, 5)
            generate_runs_for_branch(branch, count, args.dry_run, args.trigger_deploy)
        else:
            # All branches mode
            total = 0
            for branch, count in DEFAULT_COUNTS.items():
                if args.count:
                    count = args.count
                created = generate_runs_for_branch(branch, count, args.dry_run, args.trigger_deploy)
                total += created

            print(f"\n{'='*60}")
            print(f"Total runs generated: {total}")
            print(f"{'='*60}")

    finally:
        # Return to original branch
        run_cmd(["git", "checkout", original_branch])
        print(f"\nReturned to branch: {original_branch}")


if __name__ == "__main__":
    main()
