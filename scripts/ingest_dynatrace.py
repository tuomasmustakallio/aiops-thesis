#!/usr/bin/env python3
"""Ingest normalized CI/CD logs into Dynatrace for evaluation.

Uploads all normalized log lines via the Dynatrace Log Ingestion API v2.
Each log line is sent as a JSON object with run metadata.

IMPORTANT: Dynatrace rejects log records older than 24 hours from
ingestion time. Logs must be from recent pipeline runs.

Usage:
    # Set credentials in .env (recommended):
    echo 'DYNATRACE_URL=https://{env-id}.live.dynatrace.com' >> .env
    echo 'DYNATRACE_API_TOKEN=dt0c01.xxx...' >> .env
    python scripts/ingest_dynatrace.py

    # Or pass on command line:
    python scripts/ingest_dynatrace.py --url URL --token TOKEN

    # Ingest only specific runs (by run_id):
    python scripts/ingest_dynatrace.py --run-ids 123456 789012

Requirements:
    pip install requests
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Dynatrace Log Ingestion API limits
MAX_PAYLOAD_BYTES = 1_048_576  # 1 MB per request
MAX_ENTRIES_PER_REQUEST = 50_000
LOG_AGE_LIMIT_HOURS = 24

# Map our log levels to Dynatrace severity
SEVERITY_MAP = {
    "ERROR": "ERROR",
    "WARN": "WARN",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
}


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


def build_log_entries(input_dir: Path, metadata: dict,
                      run_id_filter: list[str] | None = None,
                      use_current_time: bool = False) -> list[dict]:
    """Build Dynatrace log entry objects from normalized logs.

    Each log line becomes one entry with:
      - content: the log message
      - timestamp: ISO 8601 from run metadata (with per-line offset)
      - severity: mapped from log level
      - custom attributes: run_id, job_name, log_file, line_number

    NOTE: outcome and failure_class are excluded from ingestion to avoid
    leaking ground truth to the tool.

    If use_current_time=True, all timestamps are set to now (useful for
    testing with old logs that would be rejected by the 24h age limit).
    """
    entries = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=LOG_AGE_LIMIT_HOURS)
    skipped_old = 0
    run_index = 0

    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name

        if run_id_filter and run_id not in run_id_filter:
            continue

        run_meta = metadata.get(run_id, {})

        # Parse base timestamp or use current time
        if use_current_time:
            base_ts = now
            run_index += 1
        else:
            base_ts_str = run_meta.get("timestamp", "")
            try:
                base_ts = datetime.fromisoformat(base_ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                base_ts = now - timedelta(minutes=10)  # Recent default

        # Check age limit (skip if not using current_time and too old)
        if not use_current_time and base_ts < cutoff:
            print(f"  WARNING: Run {run_id} timestamp {base_ts.isoformat()} "
                  f"is older than 24h, skipping")
            skipped_old += 1
            continue

        line_num = 0
        for log_file in sorted(run_dir.rglob("*.log")):
            job_name = log_file.parent.name
            with open(log_file, "r") as f:
                for raw_line in f:
                    raw_line = raw_line.rstrip("\n")
                    if not raw_line:
                        continue

                    if "\t" in raw_line:
                        level, content = raw_line.split("\t", 1)
                    else:
                        level, content = "INFO", raw_line

                    level = level.strip()
                    content = content.strip()

                    # Sub-second offset for ordering within a run
                    ts = base_ts + timedelta(milliseconds=line_num * 10)

                    entry = {
                        "content": content,
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                        "severity": SEVERITY_MAP.get(level, "INFO"),
                        "log.source": f"/cicd/{run_id}/{log_file.name}",
                        "run_id": run_id,
                        "job_name": job_name,
                        "log_file": log_file.name,
                        "line_number": str(line_num),
                    }
                    entries.append(entry)
                    line_num += 1

        if line_num > 0:
            print(f"  {run_id}: {line_num} lines (ts: {base_ts.isoformat()})")

    if skipped_old:
        print(f"\n  WARNING: Skipped {skipped_old} runs older than 24 hours")

    return entries


def send_batch(url: str, token: str, entries: list[dict],
               verify_ssl: bool = True) -> tuple[int, int]:
    """Send a batch of log entries to Dynatrace.

    Returns (success_count, error_count).
    """
    endpoint = f"{url.rstrip('/')}/api/v2/logs/ingest"
    headers = {
        "Authorization": f"Api-Token {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    # Split into chunks that fit under the payload limit
    success_total = 0
    error_total = 0

    chunk = []
    chunk_size = 0

    for entry in entries:
        entry_json = json.dumps(entry)
        entry_size = len(entry_json.encode("utf-8"))

        # If adding this entry would exceed limit, send current chunk
        if chunk and (chunk_size + entry_size > MAX_PAYLOAD_BYTES - 1024
                      or len(chunk) >= MAX_ENTRIES_PER_REQUEST):
            ok, err = _send_chunk(endpoint, headers, chunk, verify_ssl)
            success_total += ok
            error_total += err
            chunk = []
            chunk_size = 0
            time.sleep(0.1)  # Rate limit courtesy

        chunk.append(entry)
        chunk_size += entry_size

    # Send remaining
    if chunk:
        ok, err = _send_chunk(endpoint, headers, chunk, verify_ssl)
        success_total += ok
        error_total += err

    return success_total, error_total


def _send_chunk(endpoint: str, headers: dict, entries: list[dict],
                verify_ssl: bool) -> tuple[int, int]:
    """Send a single chunk of entries. Returns (success, error) counts."""
    payload = json.dumps(entries)

    try:
        resp = requests.post(
            endpoint,
            data=payload,
            headers=headers,
            verify=verify_ssl,
            timeout=30,
        )

        if resp.status_code in (200, 204):
            # Success — all entries accepted
            return len(entries), 0
        else:
            error_msg = resp.text[:500] if resp.text else f"HTTP {resp.status_code}"
            print(f"    WARNING: {resp.status_code}: {error_msg}")
            return 0, len(entries)

    except requests.exceptions.RequestException as exc:
        print(f"    ERROR: Request failed: {exc}")
        return 0, len(entries)


def _load_dotenv():
    """Load .env file into os.environ (no dependency needed)."""
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
        description="Ingest CI/CD logs into Dynatrace",
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
        "--input", default="artifacts/normalized",
        help="Normalized logs directory (default: artifacts/normalized)",
    )
    parser.add_argument(
        "--ground-truth", default="data/runs.csv",
        help="Ground truth CSV (default: data/runs.csv)",
    )
    parser.add_argument(
        "--run-ids", nargs="+",
        help="Only ingest specific run IDs (space-separated)",
    )
    parser.add_argument(
        "--use-current-time", action="store_true",
        help="Use current timestamp for all logs (useful for testing old logs that would be rejected by 24h age limit)",
    )
    parser.add_argument(
        "--no-verify-ssl", action="store_true",
        help="Disable SSL certificate verification",
    )
    args = parser.parse_args()

    if not REQUESTS_AVAILABLE:
        print("ERROR: requests package is not installed")
        print("Install with: pip install requests")
        sys.exit(1)

    if not args.url:
        print("ERROR: No Dynatrace URL provided.")
        print("Set DYNATRACE_URL in .env or pass --url")
        print("Format: https://{environment-id}.live.dynatrace.com")
        sys.exit(1)

    if not args.token:
        print("ERROR: No Dynatrace API token provided.")
        print("Set DYNATRACE_API_TOKEN in .env or pass --token")
        print("Token needs 'logs.ingest' scope")
        sys.exit(1)

    # Load run metadata
    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        metadata = load_ground_truth(gt_path)
        print(f"Loaded metadata for {len(metadata)} runs from {gt_path}")
    else:
        print(f"WARNING: Ground truth not found at {gt_path}")
        metadata = {}

    # Build log entries
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"\nBuilding log entries from {input_dir} …")
    if args.use_current_time:
        print("  (using current timestamp for all logs)")
    entries = build_log_entries(input_dir, metadata, args.run_ids, args.use_current_time)

    if not entries:
        print("ERROR: No log entries to ingest.")
        print("Check that runs are within the 24-hour age limit.")
        sys.exit(1)

    print(f"\nTotal entries to ingest: {len(entries)}")
    payload_size = sum(len(json.dumps(e).encode("utf-8")) for e in entries)
    print(f"Estimated payload size: {payload_size / 1024:.1f} KB")

    # Send to Dynatrace
    print(f"\nIngesting into {args.url} …")
    verify_ssl = not args.no_verify_ssl
    success, errors = send_batch(args.url, args.token, entries, verify_ssl)

    print(f"\nIngestion complete:")
    print(f"  Accepted: {success}")
    print(f"  Rejected: {errors}")

    if errors:
        print("\nSome entries were rejected. Common causes:")
        print("  - Log timestamp older than 24 hours")
        print("  - Invalid API token or missing 'logs.ingest' scope")
        print("  - Payload too large")

    # Print next steps
    print("\n--- Next steps in Dynatrace ---")
    print("1. Go to Logs & Events > Logs")
    print("   Verify ingested logs appear (filter by run_id)")
    print("2. Create a log-based metric:")
    print("   Settings > Log Monitoring > Create metric")
    print("   e.g., count of ERROR severity lines")
    print("3. Create a metric event rule:")
    print("   Settings > Event Processing > Custom metric events")
    print("   e.g., alert when error_count > 0")
    print("4. Use DQL to query:")
    print('   fetch logs | filter run_id == "YOUR_RUN_ID"')
    print("   fetch logs | summarize count(), by:{run_id, severity}")


if __name__ == "__main__":
    main()
