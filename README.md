# AI for Failure Prediction in CI/CD Pipelines

Master's thesis evaluating AI-enabled observability tools for detecting CI/CD pipeline failures in a realistic DevOps environment.

## Quick Results

Three AI tools evaluated on 47 labeled pipeline runs (27 success, 20 failures):

| Tool | Precision | Recall | F1 Score | Integration Effort |
|------|-----------|--------|----------|-------------------|
| **Dynatrace Davis AI** | 1.000 | 1.000 | 1.000 | Medium |
| **LogAI** | 0.778 | 0.778 | 0.778 | Low |
| **Elastic Observability** | 0.182 | 1.0 | 0.182 | High |

## Project Overview

This repository contains:

1. **Reproducible CI/CD Experiment Environment** - GitHub Actions workflow with controlled failure injection
2. **Web Application** - Backend API (Python/Flask) + Frontend (TypeScript/React) deployed to Azure
3. **Ground Truth Dataset** - 47 manually labeled pipeline runs across 5 failure branches
4. **Log Collection Pipeline** - Download, normalize, and prepare logs from GitHub Actions artifacts
5. **Tool Integration** - Scripts to ingest the same dataset into LogAI, Elastic, and Dynatrace
6. **Evaluation Results** - Performance metrics, detailed analysis per tool

## Project Structure

```
├── README.md                          # This file (knowledge bank)
│
├── docs/
│   ├── DEPLOYMENT.md                  # Azure infrastructure setup
│   ├── logai_evaluation.md            # LogAI full evaluation (F1: 0.778)
│   ├── elastic_evaluation.md          # Elastic full evaluation (F1: 0.182)
│   └── dynatrace_evaluation.md        # Dynatrace full evaluation (F1: 1.0)
│
├── data/
│   ├── README.md                      # Dataset schema documentation
│   └── runs.csv                       # Ground truth labels (47 runs)
│
├── backend/                           # Python Flask API
│   ├── main.py                        # API server
│   └── tests/
│       ├── test_api.py                # API endpoint tests
│       └── test_experiment_fail.py    # Tests for failure injection
│
├── frontend/                          # TypeScript React application
│   ├── package.json
│   ├── tsconfig.json
│   └── src/                           # (React components)
│
├── experiments/
│   └── failure-toggles/               # Failure injection mechanisms
│       ├── README.md
│       └── backend-fail.py            # Pytest failure injection
│
└── scripts/
    ├── generate_runs.py               # Trigger CI runs on experiment branches
    ├── download_artifacts.py          # Download logs from GitHub Actions
    ├── normalize_logs.py              # Normalize logs to standard format
    ├── label_runs.py                  # CLI helper for ground truth labeling
    ├── compute_metrics.py             # Calculate precision/recall/F1
    │
    ├── ingest_logai.py                # LogAI ingestion pipeline
    ├── run_logai_eval.py              # LogAI evaluation
    │
    ├── ingest_elastic.py              # Elastic ingestion and evaluation
    │
    ├── ingest_dynatrace.py            # Dynatrace log ingestion
    │
    ├── run_baseline.py                # Baseline: naive ERROR keyword heuristic
    ├── bump_marker.py                 # CI: inject RUN_MARKER for log attribution
    └── azure_setup.sh                 # Azure infrastructure provisioning

[Generated/regenerable: artifacts/, results/*.csv]
```

## Evaluation Dataset

**47 Pipeline Runs** across 5 experiment branches:

| Branch | Failure Class | Count | Outcome |
|--------|---------------|-------|---------|
| `experiment/success` | none | 25 | ✓ Success |
| `experiment/backend-fail` | backend_test | 6 | ✗ Failure |
| `experiment/frontend-fail` | frontend_build | 6 | ✗ Failure |
| `experiment/dep-fail` | dependency | 6 | ✗ Failure |
| `experiment/deploy-fail` | deploy | 2 | ✗ Failure |

**Totals**: 27 success, 20 failures | **13,065 normalized log lines** ingested into each tool

**Ground truth** in `data/runs.csv` with schema: `run_id, commit_sha, timestamp, outcome, failure_class, note`

## Key Findings

### Dynatrace Davis AI - Best Performance ⭐

**Precision: 1.0 | Recall: 1.0 | F1: 1.0**

- Detected all 20 failures with zero false positives
- Per-run anomaly detection via DQL time-series queries
- Automated event creation and lifecycle management
- **Tradeoff**: Requires time-series aggregation; not direct log analysis

### LogAI - Strong, Practical, Low-Overhead

**Precision: 0.778 | Recall: 0.778 | F1: 0.778**

- Unsupervised clustering on raw logs (no training required)
- Lowest integration effort; works out-of-the-box
- Best for quick analysis with minimal configuration
- **Tradeoff**: Lower precision than Dynatrace; threshold tuning needed

### Elastic Observability - High Recall, Many False Positives

**Precision: 0.182 | Recall: 1.0 | F1: 0.182**

- Detects all failures but generates excessive false alerts
- High integration complexity (ML feature setup)
- **Best for**: Scenarios where strict recall is mandatory and FPs are acceptable cost
- **Tradeoff**: Impractical false positive rate for small deployments; needs production-scale data

## How It Works

### 1. CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/...`) with 3 jobs:
- **backend**: Python pytest tests
- **frontend**: TypeScript build (npm)
- **deploy**: Docker build and Azure deployment

Each job uploads logs as artifacts with run metadata.

### 2. Failure Injection

5 experiment branches with **deterministic, controlled failures**:
- `experiment/backend-fail` - pytest fails
- `experiment/frontend-fail` - TypeScript compilation error
- `experiment/dep-fail` - Invalid pip package
- `experiment/deploy-fail` - Dockerfile error
- `experiment/success` - Clean run (baseline)

### 3. Log Collection & Normalization

```bash
# Download artifacts from GitHub Actions
python3 scripts/download_artifacts.py --repo owner/repo

# Normalize to standard format
python3 scripts/normalize_logs.py

# Result: normalized logs in artifacts/normalized/<run_id>/
```

### 4. Ground Truth Labeling

```bash
# Label runs manually after each batch
python3 scripts/label_runs.py --run-id 20596283825 --outcome failure --class backend_test

# Runs stored in data/runs.csv (binary success/failure + failure class)
```

### 5. Tool Evaluation

Each tool receives the same normalized dataset:

```bash
# LogAI
python3 scripts/run_logai_eval.py

# Elastic
python3 scripts/ingest_elastic.py

# Dynatrace
python3 scripts/ingest_dynatrace.py --use-current-time
```

### 6. Metrics Computation

```bash
# Calculate precision/recall/F1 per tool
python3 scripts/compute_metrics.py --tool logai --ground-truth data/runs.csv

# Results in results/metrics_*.csv
```

## Tool Architecture Comparison

| Aspect | LogAI | Elastic | Dynatrace |
|--------|-------|---------|-----------|
| **Input Type** | Raw logs | Raw logs | Time-series (logs aggregated) |
| **Detection Model** | Clustering | ML classification | Statistical anomaly detection |
| **Training Required** | No | Yes (on-platform) | No (static threshold) |
| **Baseline Data Needed** | None | 7-14 days | Varies by analyzer |
| **Failure Granularity** | Global clustering | Per-pattern | Per entity (e.g., run_id) |
| **Configuration Complexity** | Low | High | Medium |
| **Time to Insight** | Minutes | Hours (model training) | Minutes |

## Detailed Evaluation Docs

See the per-tool evaluation documents for complete methodology, results, and analysis:

- **`docs/logai_evaluation.md`**
  - Environment setup, pipeline architecture, integration notes
  - Per-failure-class breakdown, comparison table
  - Limitations: lower precision, threshold sensitivity

- **`docs/elastic_evaluation.md`**
  - ML feature configuration, model tuning, integration complexity
  - High false positive analysis
  - Why precision is low despite perfect recall

- **`docs/dynatrace_evaluation.md`**
  - Davis AI setup via Notebooks and Anomaly Detection app
  - Per-run alert configuration with DQL makeTimeseries
  - Perfect detection results analysis

## Evaluation Constraints & Methodology

### Constraints (from CLAUDE.md)

- **No custom ML implementations** - Tools evaluated as-provided by vendors
- **No AI algorithm reimplementation** - Focus on integration and practical use
- **Single evaluation window** - All data ~2 minutes (limits multi-day baseline models)
- **No ground truth leakage** - Outcome fields excluded from tool inputs
- **Uniform dataset** - Same normalized logs used across all tools

### Metrics Computed

For each tool:
1. **Precision**: Of predicted failures, how many were actually failures?
2. **Recall**: Of actual failures, how many were detected?
3. **F1 Score**: Harmonic mean balancing precision and recall
4. **Per-failure-class breakdown**: Detection rate by failure type

## Reproducibility Guide

To reproduce the full evaluation:

### Step 1: Set up Azure Infrastructure

```bash
./scripts/azure_setup.sh
# or follow docs/DEPLOYMENT.md for manual setup
```

### Step 2: Generate Fresh Runs

```bash
# Trigger CI runs across all experiment branches
python3 scripts/generate_runs.py
```

### Step 3: Download & Normalize Logs

```bash
python3 scripts/download_artifacts.py --repo owner/repo
python3 scripts/normalize_logs.py
```

### Step 4: Label Ground Truth

```bash
# After reviewing each run
python3 scripts/label_runs.py --run-id <id> --outcome success|failure --class <class>
```

### Step 5: Evaluate Each Tool

```bash
python3 scripts/run_logai_eval.py
python3 scripts/ingest_elastic.py
python3 scripts/ingest_dynatrace.py --use-current-time
```

### Step 6: Compute Metrics

```bash
python3 scripts/compute_metrics.py --tool all --ground-truth data/runs.csv
```

## Data Files Reference

- **`data/runs.csv`** - Ground truth: 47 runs with labels, outcomes, failure classes
- **`data/README.md`** - Schema documentation and failure class definitions

(Generated files: `results/metrics_*.csv` and `results/*_scores.csv` are regenerated per tool evaluation)

## References

- **`docs/DEPLOYMENT.md`** - Azure infrastructure setup and troubleshooting
- **`docs/logai_evaluation.md`, `elastic_evaluation.md`, `dynatrace_evaluation.md`** - Tool-specific evaluation documentation
- **`experiments/failure-toggles/README.md`** - Failure injection mechanisms
- **`data/README.md`** - Dataset schema and failure classification

## Key Insights for Thesis

1. **Dynatrace excels** at per-entity anomaly detection when configured properly
2. **LogAI provides practical value** with minimal configuration overhead
3. **Elastic requires significant tuning** but guarantees failure detection
4. **No single tool is universally best** - tradeoffs depend on operational priorities
5. **Tool architecture matters** - time-series vs. raw logs, ML vs. heuristics

## Next Steps

1. **For thesis writing**: Reference the per-tool docs for detailed technical sections
2. **For future extensions**: Generate more runs, evaluate additional tools, extend baseline heuristics
3. **For reproducibility**: Follow the Reproducibility Guide above
4. **For deep dives**: See individual tool evaluation docs and inline code comments

---

*This repository serves as a complete audit trail of the evaluation process: ground truth tracking, reproducible scripts, detailed documentation, and transparent results.*
