#!/usr/bin/env python3
"""Evaluate LogAI anomaly detection on CI/CD pipeline logs.

Feeds normalized CI/CD logs into LogAI's provided anomaly detection
components and exports per-run anomaly scores.

LogAI is used as-is -- no custom ML algorithms are implemented.
This is glue code that connects our log dataset to LogAI's anomaly
detection functionality.

Two approaches are available (--mode flag):

  app     - High-level LogAnomalyDetection API (simplest, may not
            support custom log formats)
  blocks  - Step-by-step building blocks (Drain parser, Word2Vec,
            IsolationForest etc.) -- more reliable for custom logs

Requirements:
    Python 3.10 (LogAI has known issues with 3.11+)
    pip install logai
    python -m nltk.downloader punkt

Usage:
    python scripts/run_logai_eval.py
    python scripts/run_logai_eval.py --mode app
    python scripts/run_logai_eval.py --algorithm one_class_svm
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# ---------------------------------------------------------------------------
# LogAI imports -- these are attempted lazily so the script can still
# provide useful error messages if LogAI is not installed.
# ---------------------------------------------------------------------------

def _check_logai():
    """Return True if LogAI can be imported, print help otherwise."""
    try:
        import logai  # noqa: F401
        return True
    except ImportError as exc:
        print(f"ERROR: Cannot import LogAI: {exc}")
        print()
        print("Install LogAI (requires Python 3.10):")
        print("  python3.10 -m venv .venv-logai")
        print("  source .venv-logai/bin/activate")
        print("  pip install logai")
        print("  python -m nltk.downloader punkt")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_logs(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load all normalized log lines from all runs.

    Returns (DataFrame with columns [run_id, level, content], sorted_run_ids).
    """
    rows = []
    run_ids = []

    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        run_ids.append(run_id)

        for log_file in sorted(run_dir.rglob("*.log")):
            with open(log_file, "r") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if "\t" in line:
                        level, content = line.split("\t", 1)
                    else:
                        level, content = "INFO", line
                    rows.append({
                        "run_id": run_id,
                        "level": level.strip(),
                        "content": content.strip(),
                    })

    df = pd.DataFrame(rows)
    return df, run_ids


def aggregate_scores(df: pd.DataFrame, run_ids: list[str],
                     threshold: float) -> list[dict]:
    """Aggregate per-line anomaly scores to per-run scores.

    Uses the continuous 'anomaly_score' column (from sklearn's
    decision_function, inverted so higher = more anomalous).
    Falls back to binary 'is_anomalous' fraction if scores are missing.
    """
    use_continuous = "anomaly_score_line" in df.columns
    results = []
    for run_id in run_ids:
        mask = df["run_id"] == run_id
        run_lines = df.loc[mask]
        n_total = len(run_lines)

        if n_total == 0:
            results.append({
                "run_id": run_id,
                "anomaly_score": 0.0,
                "predicted_anomaly": False,
                "log_lines": 0,
            })
            continue

        if use_continuous:
            # Mean of per-line scores (already inverted: higher = more anomalous)
            score = round(float(run_lines["anomaly_score_line"].mean()), 4)
        else:
            n_anomalous = run_lines["is_anomalous"].sum()
            score = round(float(n_anomalous) / n_total, 4)

        results.append({
            "run_id": run_id,
            "anomaly_score": score,
            "predicted_anomaly": score >= threshold,
            "log_lines": n_total,
        })
    return results


# ---------------------------------------------------------------------------
# Approach 1: LogAI Application-level API
# ---------------------------------------------------------------------------

def run_app_api(input_dir: Path, algorithm: str) -> pd.DataFrame | None:
    """Try the high-level LogAnomalyDetection API.

    Concatenates all logs into a temp file and runs the built-in workflow.
    Returns anomaly_results or None if it fails.
    """
    from logai.applications.application_interfaces import WorkFlowConfig
    from logai.applications.log_anomaly_detection import LogAnomalyDetection

    # Concatenate all log content into a single temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, prefix="logai_"
    )
    line_map = []  # (line_index, run_id)
    idx = 0
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for log_file in sorted(run_dir.rglob("*.log")):
            with open(log_file, "r") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # Write only the content (strip level prefix)
                    if "\t" in line:
                        _, content = line.split("\t", 1)
                    else:
                        content = line
                    tmp.write(content.strip() + "\n")
                    line_map.append((idx, run_id))
                    idx += 1
    tmp.close()
    tmp_path = tmp.name
    print(f"  Wrote {idx} lines to {tmp_path}")

    config = {
        "open_set_data_loader_config": {
            "dataset_name": "HealthApp",
            "filepath": tmp_path,
        },
        "preprocessor_config": {
            "custom_delimiters_regex": [],
        },
        "log_parser_config": {
            "parsing_algorithm": "drain",
            "parsing_algo_params": {
                "sim_th": 0.5,
                "depth": 5,
            },
        },
        "feature_extractor_config": {
            "group_by_category": ["Level"],
            "group_by_time": "1s",
        },
        "log_vectorizer_config": {
            "algo_name": "word2vec",
        },
        "categorical_encoder_config": {
            "name": "label_encoder",
        },
        "anomaly_detection_config": {
            "algo_name": algorithm,
        },
    }

    workflow_config = WorkFlowConfig.from_dict(config)
    app = LogAnomalyDetection(workflow_config)
    app.execute()

    res = app.anomaly_results
    print(f"  type(anomaly_results) = {type(res)}")
    if isinstance(res, pd.DataFrame):
        print(f"  columns = {list(res.columns)}")
        print(f"  shape   = {res.shape}")
        print(res.head(10))
    elif isinstance(res, pd.Series):
        print(f"  length  = {len(res)}")
        print(res.head(10))
    else:
        print(f"  value   = {res}")

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)
    return res


# ---------------------------------------------------------------------------
# Approach 2: Building-blocks pipeline
# ---------------------------------------------------------------------------

def run_building_blocks(df: pd.DataFrame, algorithm: str) -> pd.Series:
    """Run LogAI anomaly detection using individual components.

    Pipeline:
      Preprocessor -> Drain parser -> Word2Vec vectorizer
      -> CategoricalEncoder (level) -> FeatureExtractor
      -> AnomalyDetector (IsolationForest / OC-SVM)

    All components are from the LogAI library.
    """
    from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
    from logai.information_extraction.log_parser import LogParser, LogParserConfig
    from logai.algorithms.parsing_algo.drain import DrainParams
    from logai.information_extraction.log_vectorizer import (
        LogVectorizer, VectorizerConfig,
    )
    from logai.information_extraction.categorical_encoder import (
        CategoricalEncoder, CategoricalEncoderConfig,
    )
    from logai.information_extraction.feature_extractor import (
        FeatureExtractor, FeatureExtractorConfig,
    )
    from logai.analysis.anomaly_detector import (
        AnomalyDetector, AnomalyDetectionConfig,
    )

    content = df["content"]

    # Synthetic timestamps (1 s apart) -- required by feature extractor
    t0 = datetime(2025, 1, 1)
    timestamps = pd.Series(
        [t0 + timedelta(seconds=i) for i in range(len(df))]
    )

    # 1. Preprocess
    print("  [1/6] Preprocessing …")
    pre_cfg = PreprocessorConfig(
        custom_replace_list=[
            [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
            [r"0x[0-9a-fA-F]+", "<HEX>"],
        ],
    )
    preprocessor = Preprocessor(pre_cfg)
    clean_content, _ = preprocessor.clean_log(content)

    # 2. Parse (Drain)
    print("  [2/6] Parsing templates (Drain) …")
    drain_params = DrainParams(sim_th=0.5, depth=5)
    parser_cfg = LogParserConfig(
        parsing_algorithm="drain", parsing_algo_params=drain_params,
    )
    log_parser = LogParser(parser_cfg)
    parsed = log_parser.parse(clean_content)
    parsed_loglines = parsed["parsed_logline"]

    # 3. Vectorize (Word2Vec)
    print("  [3/6] Vectorizing (Word2Vec) …")
    vec_cfg = VectorizerConfig(algo_name="word2vec")
    vectorizer = LogVectorizer(vec_cfg)
    vectorizer.fit(parsed_loglines)
    log_vectors = vectorizer.transform(parsed_loglines)

    # 4. Encode categorical attributes (Level)
    print("  [4/6] Encoding log levels …")
    enc_cfg = CategoricalEncoderConfig(name="label_encoder")
    encoder = CategoricalEncoder(enc_cfg)
    level_encoded = encoder.fit_transform(df[["level"]])

    # 5. Feature extraction (combine vector + encoded attributes)
    print("  [5/6] Extracting features …")
    fe_cfg = FeatureExtractorConfig(max_feature_len=200)
    fe = FeatureExtractor(fe_cfg)
    _, feature_vector = fe.convert_to_feature_vector(
        log_vectors, level_encoded, timestamps,
    )

    # Workaround: LogAI's FeatureExtractor includes the timestamp column
    # as datetime64, which sklearn cannot handle.  Convert any datetime
    # columns to numeric (epoch seconds).
    import numpy as np
    for col in feature_vector.columns:
        if pd.api.types.is_datetime64_any_dtype(feature_vector[col]):
            feature_vector[col] = (
                feature_vector[col].astype(np.int64) // 10**9
            )

    # 6. Anomaly detection
    print(f"  [6/6] Running {algorithm} …")
    ad_cfg = AnomalyDetectionConfig(algo_name=algorithm)

    # Try to set algorithm-specific params if available
    if algorithm == "isolation_forest":
        try:
            from logai.algorithms.anomaly_detection_algo.isolation_forest import (
                IsolationForestParams,
            )
            ad_cfg = AnomalyDetectionConfig(
                algo_name=algorithm,
                algo_params=IsolationForestParams(
                    n_estimators=200, max_features=1.0,
                ),
            )
        except ImportError:
            pass

    detector = AnomalyDetector(ad_cfg)

    # Workaround: LogAI passes warm_start=0 (int) but scikit-learn >= 1.4
    # requires a proper bool.  Fix it on the underlying sklearn model.
    sklearn_model = getattr(detector, "anomaly_detector", None)
    if sklearn_model is not None:
        inner = getattr(sklearn_model, "model", None)
        if inner is not None and hasattr(inner, "warm_start"):
            inner.warm_start = bool(inner.warm_start)

    detector.fit(feature_vector)
    predictions = detector.predict(feature_vector)

    # Also get continuous anomaly scores from the underlying sklearn model
    # (decision_function: lower = more anomalous)
    raw_scores = None
    if sklearn_model is not None:
        inner = getattr(sklearn_model, "model", None)
        if inner is not None and hasattr(inner, "decision_function"):
            raw_scores = inner.decision_function(feature_vector)

    return predictions, raw_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LogAI anomaly detection on CI/CD logs",
    )
    parser.add_argument(
        "--input", default="artifacts/normalized",
        help="Directory containing normalized logs (default: artifacts/normalized)",
    )
    parser.add_argument(
        "--output", default="results/logai_scores.csv",
        help="Output CSV path (default: results/logai_scores.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.095,
        help="Anomaly score threshold to flag a run (default: 0.095)",
    )
    parser.add_argument(
        "--algorithm", default="isolation_forest",
        choices=["isolation_forest", "one_class_svm"],
        help="Anomaly detection algorithm (default: isolation_forest)",
    )
    parser.add_argument(
        "--mode", default="blocks", choices=["blocks", "app"],
        help=(
            "blocks = step-by-step building blocks (default, recommended); "
            "app = high-level LogAnomalyDetection API"
        ),
    )
    args = parser.parse_args()

    _check_logai()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Load logs
    print(f"Loading logs from {input_dir} …")
    df, run_ids = load_all_logs(input_dir)
    print(f"Loaded {len(df)} lines from {len(run_ids)} runs")

    if df.empty:
        print("ERROR: No log data found")
        sys.exit(1)

    # ---- Run anomaly detection ----

    if args.mode == "app":
        print("\n--- Application-level API (LogAnomalyDetection) ---")
        try:
            res = run_app_api(input_dir, args.algorithm)
            if res is None:
                print("Application API returned None. No results to export.")
                sys.exit(1)
            # Attempt to map results back to runs
            # (This is best-effort; the structure depends on LogAI version)
            print("\nApplication API finished. Inspect the output above.")
            print("If the results are usable, adapt the aggregation logic.")
            print("Falling through to export stage …")
            # We cannot reliably aggregate without knowing the result schema,
            # so we print diagnostics and exit.
            sys.exit(0)
        except Exception as exc:
            print(f"\nApplication API failed: {exc}")
            print("This is expected for custom log formats.")
            print("Use --mode blocks instead (default).")
            sys.exit(1)

    # ---- Building blocks approach ----
    print(f"\n--- Building blocks: {args.algorithm} ---")
    try:
        predictions, raw_scores = run_building_blocks(df, args.algorithm)
    except Exception as exc:
        print(f"\nERROR during LogAI pipeline: {exc}")
        print()
        print("Common causes:")
        print("  - Python version: LogAI requires 3.10 (3.11+ has known issues)")
        print("  - Missing NLTK data: python -m nltk.downloader punkt")
        print("  - Incompatible numpy/scipy: try pinning numpy<2")
        raise

    # Map predictions to anomaly flags.
    # LogAI may return a 2D result (labels + scores) or 1D (labels only).
    import numpy as np
    pred_values = predictions.values if hasattr(predictions, "values") else predictions
    pred_values = np.asarray(pred_values)

    print(f"  Prediction shape: {pred_values.shape}")
    if pred_values.ndim == 2:
        # First column is typically the label (-1 / 1)
        labels = pred_values[:, 0]
    else:
        labels = pred_values

    # IsolationForest / OC-SVM: -1 = anomaly, 1 = normal
    df["is_anomalous"] = labels < 0

    # Use continuous scores from decision_function if available.
    # sklearn convention: lower score = more anomalous.
    # Invert so higher = more anomalous, then normalize to [0, 1].
    if raw_scores is not None:
        inverted = -np.asarray(raw_scores)
        lo, hi = inverted.min(), inverted.max()
        if hi > lo:
            normalized = (inverted - lo) / (hi - lo)
        else:
            normalized = np.zeros_like(inverted)
        df["anomaly_score_line"] = normalized
        print(f"  Continuous scores: min={inverted.min():.4f} max={inverted.max():.4f}")
        print(f"  Lines flagged anomalous (binary): {int(df['is_anomalous'].sum())}/{len(df)}")

    # Aggregate to per-run scores
    results = aggregate_scores(df, run_ids, args.threshold)

    # Write output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "anomaly_score", "predicted_anomaly", "log_lines"],
        )
        writer.writeheader()
        writer.writerows(results)

    n_anomalous = sum(1 for r in results if r["predicted_anomaly"])
    print(f"\nResults written to {output_path}")
    print(f"Total runs:          {len(results)}")
    print(f"Predicted anomalies: {n_anomalous}/{len(results)}")

    # Quick summary per score bucket
    print("\nScore distribution:")
    for lo, hi, label in [(0, 0.1, "0.0-0.1"), (0.1, 0.3, "0.1-0.3"),
                          (0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"),
                          (0.7, 1.01, "0.7-1.0")]:
        n = sum(1 for r in results if lo <= r["anomaly_score"] < hi)
        if n:
            print(f"  {label}: {n} runs")


if __name__ == "__main__":
    main()
