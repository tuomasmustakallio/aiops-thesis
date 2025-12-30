#!/usr/bin/env python3
"""Bump the run marker and commit the change.

Usage:
    python scripts/bump_marker.py           # Bump and commit
    python scripts/bump_marker.py --no-commit  # Bump only, don't commit
    python scripts/bump_marker.py --value      # Print current value
"""
import argparse
import subprocess
import sys
from pathlib import Path

MARKER_FILE = Path(__file__).parent.parent / "experiments" / "run_marker.txt"


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


def git_commit(marker_value: int) -> bool:
    """Stage and commit the marker file."""
    try:
        subprocess.run(
            ["git", "add", str(MARKER_FILE)],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"chore: bump run marker to {marker_value}"],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e.stderr.decode()}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Bump the run marker")
    parser.add_argument("--no-commit", action="store_true", help="Don't commit after bumping")
    parser.add_argument("--value", action="store_true", help="Print current value and exit")
    args = parser.parse_args()

    current = read_marker()

    if args.value:
        print(current)
        return

    new_value = current + 1
    write_marker(new_value)
    print(f"Bumped RUN_MARKER: {current} -> {new_value}")

    if not args.no_commit:
        if git_commit(new_value):
            print(f"Committed: chore: bump run marker to {new_value}")
        else:
            print("Commit failed", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
