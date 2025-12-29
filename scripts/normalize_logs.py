#!/usr/bin/env python3
"""Normalize downloaded CI logs into a consistent format for LogAI.

Reads artifacts from artifacts/ directory and outputs normalized logs
to artifacts/normalized/.
"""
import argparse
import re
from pathlib import Path
from datetime import datetime


def normalize_line(line: str) -> str:
    """Remove ANSI codes, timestamps variations, and normalize whitespace."""
    # Remove ANSI escape codes
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    line = ansi_pattern.sub('', line)

    # Normalize multiple spaces
    line = re.sub(r' +', ' ', line)

    return line.strip()


def extract_log_level(line: str) -> str:
    """Attempt to extract log level from line."""
    line_upper = line.upper()
    if 'ERROR' in line_upper or 'FAILED' in line_upper:
        return 'ERROR'
    elif 'WARN' in line_upper:
        return 'WARN'
    elif 'DEBUG' in line_upper:
        return 'DEBUG'
    return 'INFO'


def normalize_log_file(input_path: Path, output_path: Path) -> int:
    """Normalize a single log file. Returns line count."""
    lines = []
    with open(input_path, 'r', errors='replace') as f:
        for line in f:
            normalized = normalize_line(line)
            if normalized:  # Skip empty lines
                level = extract_log_level(normalized)
                lines.append(f"{level}\t{normalized}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(lines)


def process_run_directory(run_dir: Path, output_base: Path) -> dict:
    """Process all log files in a run directory."""
    stats = {'files': 0, 'lines': 0}

    for log_file in run_dir.rglob('*.log'):
        relative = log_file.relative_to(run_dir)
        output_path = output_base / run_dir.name / relative

        line_count = normalize_log_file(log_file, output_path)
        stats['files'] += 1
        stats['lines'] += line_count
        print(f"  {relative}: {line_count} lines")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize CI logs for LogAI")
    parser.add_argument("--input", default="artifacts", help="Input artifacts directory")
    parser.add_argument("--output", default="artifacts/normalized", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    total_stats = {'runs': 0, 'files': 0, 'lines': 0}

    for run_dir in sorted(input_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name != 'normalized':
            print(f"Processing run {run_dir.name}...")
            stats = process_run_directory(run_dir, output_dir)
            total_stats['runs'] += 1
            total_stats['files'] += stats['files']
            total_stats['lines'] += stats['lines']

    print(f"\nTotal: {total_stats['runs']} runs, {total_stats['files']} files, {total_stats['lines']} lines")


if __name__ == "__main__":
    main()
