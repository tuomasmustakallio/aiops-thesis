#!/usr/bin/env python3
"""Ingest normalized CI/CD logs into Elasticsearch for ML anomaly detection.

Uploads all normalized log lines as individual documents, tagged with
run metadata (run_id, outcome, failure_class) from ground truth.

After ingestion, use Kibana ML to create anomaly detection jobs:
  - Log rate anomaly detection
  - Log categorization

Usage:
    # Set credentials in .env (recommended):
    echo 'ELASTIC_URL=https://your-deployment.es.cloud.es.io:443' >> .env
    echo 'ELASTIC_API_KEY=your-api-key' >> .env
    python scripts/ingest_elastic.py

    # Or pass on command line:
    python scripts/ingest_elastic.py --url URL --api-key KEY

Requirements:
    pip install elasticsearch>=8.0.0
"""
import argparse
import os
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

DEFAULT_INDEX = "cicd-pipeline-logs"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "run_id":        {"type": "keyword"},
            "job_name":      {"type": "keyword"},
            "log_file":      {"type": "keyword"},
            "log_level":     {"type": "keyword"},
            "message":       {"type": "text"},
            "timestamp":     {"type": "date"},
            "line_number":   {"type": "integer"},
            "outcome":       {"type": "keyword"},
            "failure_class": {"type": "keyword"},
        }
    }
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


def generate_documents(input_dir: Path, metadata: dict, index_name: str):
    """Yield Elasticsearch bulk-upload actions from normalized logs.

    Each log line becomes one document. Within a run, lines get
    incrementing sub-second timestamps so Kibana ML can see ordering.
    """
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        run_meta = metadata.get(run_id, {})

        # Parse base timestamp or use a default
        base_ts_str = run_meta.get("timestamp", "2025-01-01T00:00:00Z")
        try:
            base_ts = datetime.fromisoformat(base_ts_str.replace("Z", "+00:00"))
        except ValueError:
            base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        line_num = 0
        for log_file in sorted(run_dir.rglob("*.log")):
            job_name = log_file.parent.name  # e.g. backend-logs-20596283825
            with open(log_file, "r") as f:
                for raw_line in f:
                    raw_line = raw_line.rstrip("\n")
                    if not raw_line:
                        continue

                    if "\t" in raw_line:
                        level, content = raw_line.split("\t", 1)
                    else:
                        level, content = "INFO", raw_line

                    # Sub-second offset so lines within a run have unique timestamps
                    ts = base_ts + timedelta(milliseconds=line_num * 10)

                    yield {
                        "_index": index_name,
                        "_source": {
                            "run_id": run_id,
                            "job_name": job_name,
                            "log_file": log_file.name,
                            "log_level": level.strip(),
                            "message": content.strip(),
                            "timestamp": ts.isoformat(),
                            "line_number": line_num,
                            "outcome": run_meta.get("outcome", "unknown"),
                            "failure_class": run_meta.get("failure_class", "unknown"),
                        },
                    }
                    line_num += 1

        if line_num > 0:
            print(f"  {run_id}: {line_num} lines")


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
        description="Ingest CI/CD logs into Elasticsearch",
    )
    parser.add_argument(
        "--url", default=os.environ.get("ELASTIC_URL"),
        help="Elasticsearch URL (or set ELASTIC_URL env var)",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ELASTIC_API_KEY"),
        help="Elasticsearch API key (or set ELASTIC_API_KEY env var)",
    )
    parser.add_argument("--username", default="elastic", help="Username")
    parser.add_argument("--password", help="Password")
    parser.add_argument(
        "--input", default="artifacts/normalized",
        help="Normalized logs directory (default: artifacts/normalized)",
    )
    parser.add_argument(
        "--ground-truth", default="data/runs.csv",
        help="Ground truth CSV (default: data/runs.csv)",
    )
    parser.add_argument(
        "--index", default=DEFAULT_INDEX,
        help=f"Elasticsearch index name (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--delete-existing", action="store_true",
        help="Delete and recreate the index before ingestion",
    )
    args = parser.parse_args()

    if not ES_AVAILABLE:
        print("ERROR: elasticsearch package is not installed")
        print("Install with: pip install 'elasticsearch>=8.0.0'")
        sys.exit(1)

    if not args.url:
        print("ERROR: No Elasticsearch URL provided.")
        print("Set ELASTIC_URL in .env or pass --url")
        sys.exit(1)

    # Connect
    if args.api_key:
        es = Elasticsearch(args.url, api_key=args.api_key, verify_certs=True)
    elif args.password:
        es = Elasticsearch(
            args.url, basic_auth=(args.username, args.password), verify_certs=True,
        )
    else:
        print("ERROR: Provide --api-key or --password for authentication")
        sys.exit(1)

    # Verify connection
    try:
        info = es.info()
        print(f"Connected to Elasticsearch {info['version']['number']}")
    except Exception as exc:
        print(f"ERROR: Cannot connect to Elasticsearch: {exc}")
        sys.exit(1)

    index_name = args.index

    # Index management
    if args.delete_existing and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Deleted existing index '{index_name}'")

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=INDEX_MAPPING)
        print(f"Created index '{index_name}' with mapping")

    # Load run metadata
    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        metadata = load_ground_truth(gt_path)
        print(f"Loaded metadata for {len(metadata)} runs from {gt_path}")
    else:
        print(f"WARNING: Ground truth not found at {gt_path}, proceeding without metadata")
        metadata = {}

    # Bulk upload
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"\nIngesting logs from {input_dir} …")
    success, errors = bulk(
        es,
        generate_documents(input_dir, metadata, index_name),
        chunk_size=500,
        raise_on_error=False,
    )

    print(f"\nIndexed {success} documents")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  {err}")

    # Verify
    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    print(f"Total documents in '{index_name}': {count}")

    # Print next steps
    print("\n--- Next steps in Kibana ---")
    print("1. Go to Stack Management > Index Patterns")
    print(f"   Create pattern: {index_name}")
    print("2. Go to Machine Learning > Anomaly Detection > Create job")
    print("   - Single metric job: high_count on log_level=ERROR")
    print("   - Or: Advanced job with log categorization")
    print("3. Let the job run, then check Anomaly Explorer")


if __name__ == "__main__":
    main()
