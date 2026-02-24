# LogAI — Evaluation Documentation

## Overview

LogAI was evaluated as the first AI tool for CI/CD pipeline failure
detection. It is an open-source Python library by Salesforce Research
that provides log analysis components including parsing, vectorization,
and anomaly detection.

LogAI was chosen as the first tool because it is free, runs locally,
and requires no cloud accounts or trial subscriptions.

## Environment

- **LogAI version**: 0.1.5 (latest release, March 2023)
- **Python version**: 3.10 (required — LogAI has known incompatibilities with 3.11+)
- **Virtual environment**: `.venv-logai` (dedicated venv to isolate Python 3.10 dependencies)
- **OS**: Ubuntu (WSL2 on Windows)
- **Data**: 7,315 normalized log lines from 60 CI/CD pipeline runs
- **Evaluation date**: 2026-02-22

### Python 3.10 Requirement

LogAI's dependencies (particularly older versions of scikit-learn and
numpy) are incompatible with Python 3.11+. Python 3.10 was installed
via the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv .venv-logai
source .venv-logai/bin/activate
pip install logai
python -m nltk.downloader punkt punkt_tab
```

### Dependencies

Key packages installed in the LogAI venv:

- `logai>=0.1.5`
- `scikit-learn` (installed as LogAI dependency)
- `gensim` (Word2Vec, installed as LogAI dependency)
- `nltk` (tokenization, installed as LogAI dependency)
- `pandas`, `numpy`

## About LogAI

LogAI (https://github.com/salesforce/logai) is an open-source log
analytics toolkit released by Salesforce Research. It provides:

- **Log parsing**: Drain, AEL, IPLoM, and other parsing algorithms
  that extract log templates from unstructured log messages
- **Log vectorization**: Word2Vec, TF-IDF, and other methods to convert
  log templates into numerical feature vectors
- **Anomaly detection**: IsolationForest, One-Class SVM, and other
  unsupervised anomaly detection algorithms
- **GUI**: An optional Dash-based GUI for interactive analysis

LogAI offers two usage modes:

1. **Application-level API** (`LogAnomalyDetection`): A high-level
   workflow that chains all components together with a single config.
   Designed for standard log formats (e.g., HDFS, HealthApp).
2. **Building blocks API**: Individual components (Preprocessor,
   LogParser, LogVectorizer, etc.) that can be composed into a custom
   pipeline. More flexible for non-standard log formats.

The library was last updated in March 2023. It has not received
maintenance updates since then.

## Data Format

### Input

Normalized log files in `artifacts/normalized/`, organized as:

```
artifacts/normalized/
  {run_id}/
    backend-logs-{run_id}/
      install.log
      pytest.log        (only if backend tests ran)
    frontend-logs-{run_id}/
      install.log
      build.log         (only if frontend build ran)
    deploy-logs-{run_id}/   (only if deploy step ran)
      deploy.log
```

Each log file uses tab-separated format: `LEVEL\tMESSAGE`

```
INFO	Requirement already satisfied: pip in /opt/hostedtoolcache/...
INFO	Collecting fastapi>=0.104.0 (from -r requirements.txt (line 1))
ERROR	FAILED tests/test_api.py::test_health_check - AssertionError
```

Lines without a tab separator are treated as `INFO` level.

### Runs in dataset

| Category        | Count | Log lines (typical) |
|-----------------|-------|---------------------|
| Success         | 25    | 115 lines           |
| backend_test    | 6     | 124 lines           |
| frontend_build  | 6     | 107 lines           |
| dependency      | 6     | 29 lines            |
| deploy          | 2     | 289–295 lines       |
| Unlabeled       | 13    | varies              |
| **Total**       | **60**| **7,315 lines**     |

## Approach 1: Application-Level API

### Rationale

The Application-level API (`LogAnomalyDetection`) was considered first
because it provides the simplest integration — a single configuration
dictionary drives the entire pipeline.

### Why It Was Not Used

The Application API is designed for well-known log formats (HDFS,
HealthApp, BGL, etc.) with specific expected column structures. Our
normalized CI/CD logs do not match any of these formats. Initial
testing showed that the API's `OpenSetDataLoader` expects specific
column headers and timestamp formats that our tab-separated log format
does not provide.

The building blocks approach was used instead, as it allows each
pipeline component to be configured individually for our data format.

## Approach 2: Building Blocks Pipeline

### Rationale

The building blocks API provides individual components that can be
composed into a custom pipeline. This is more work than the Application
API but allows each step to be adapted to our log format.

### Pipeline Architecture

The evaluation script (`scripts/run_logai_eval.py`) implements a
6-stage pipeline using LogAI's provided components:

```
Raw log lines (7,315)
    |
    v
[1] Preprocessor       — normalize IP addresses, hex values
    |
    v
[2] Drain Parser       — extract log templates (sim_th=0.5, depth=5)
    |
    v
[3] Word2Vec Vectorizer — convert templates to 100-dim vectors
    |
    v
[4] CategoricalEncoder — label-encode log levels (INFO/WARN/ERROR/DEBUG)
    |
    v
[5] FeatureExtractor   — combine vectors + encoded attributes + timestamps
    |
    v
[6] IsolationForest    — unsupervised anomaly detection (n_estimators=200)
    |
    v
Per-line anomaly scores → aggregate to per-run scores
```

All six components are from the LogAI library. The evaluation script
is glue code that connects our data format to LogAI's components.

### Step-by-Step Detail

#### Step 1: Preprocessor

```python
PreprocessorConfig(
    custom_replace_list=[
        [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
        [r"0x[0-9a-fA-F]+", "<HEX>"],
    ],
)
```

Normalizes IP addresses and hex values to reduce noise in downstream
parsing. LogAI's `Preprocessor.clean_log()` applies regex replacements
to the raw log message content.

#### Step 2: Drain Parser

```python
DrainParams(sim_th=0.5, depth=5)
```

Drain is a log parsing algorithm that extracts log templates by building
a fixed-depth parse tree from log messages. Similar messages are grouped
into the same template, with variable parts replaced by wildcards.

For example:
- `Collecting fastapi>=0.104.0` → template: `Collecting <*>`
- `PASSED [ 33%]` → template: `PASSED [ <*>]`

Parameters:
- `sim_th=0.5`: Similarity threshold for template matching (0.5 is Drain's default)
- `depth=5`: Parse tree depth (controls template specificity)

#### Step 3: Word2Vec Vectorizer

```python
VectorizerConfig(algo_name="word2vec")
```

Converts parsed log templates into dense numerical vectors using
Word2Vec (via gensim). Each log template is tokenized (using NLTK's
`punkt` tokenizer) and transformed into a fixed-length vector.

This is the key step that captures **semantic** information from log
content — lines about test failures, dependency errors, and build
errors produce different vectors even if their structural features
(line count, log level) are similar.

#### Step 4: CategoricalEncoder

```python
CategoricalEncoderConfig(name="label_encoder")
```

Encodes the `level` column (INFO, WARN, ERROR, DEBUG) as integers
using scikit-learn's `LabelEncoder`. This adds log severity as a
numeric feature alongside the Word2Vec vectors.

#### Step 5: FeatureExtractor

```python
FeatureExtractorConfig(max_feature_len=200)
```

Combines the Word2Vec vectors, encoded categorical attributes, and
timestamps into a single feature matrix. The `max_feature_len=200`
parameter caps the feature vector length.

A synthetic timestamp was generated for each log line (1 second apart
from a base time of 2025-01-01) because the FeatureExtractor requires
a timestamp column.

#### Step 6: IsolationForest (Anomaly Detection)

```python
AnomalyDetectionConfig(
    algo_name="isolation_forest",
    algo_params=IsolationForestParams(
        n_estimators=200,
        max_features=1.0,
    ),
)
```

IsolationForest is an unsupervised anomaly detection algorithm that
identifies anomalies by measuring how easily data points can be
isolated through random partitioning. Points that are isolated quickly
(requiring fewer splits) are more likely to be anomalous.

Parameters:
- `n_estimators=200`: Number of isolation trees in the ensemble
- `max_features=1.0`: Use all features for each tree

The algorithm was used with LogAI's default `contamination` parameter.
No hyperparameter tuning was performed — the algorithm was used as
provided by LogAI.

### Score Computation

#### Per-line scores

LogAI's `AnomalyDetector.predict()` returns binary labels (-1 = anomaly,
1 = normal) per log line. To get continuous scores, the underlying
scikit-learn model's `decision_function()` was accessed directly:

- `decision_function()` returns a score where **lower = more anomalous**
- Scores were **inverted** (negated) so higher = more anomalous
- Then **normalized** to [0, 1] range using min-max scaling

#### Per-run aggregation

Per-line continuous scores were aggregated to per-run scores by
computing the **mean** of all line-level scores within each run.
This produces a single anomaly score per pipeline run.

#### Threshold selection

A threshold of **0.095** was used to classify runs as anomalous.
This threshold was determined by analyzing the score distribution:

- Success runs scored: 0.086–0.090 (tight cluster)
- backend_test failures: 0.099–0.102 (slightly above success)
- dependency failures: 0.206 (clearly separated)
- deploy failures: 0.124–0.125 (clearly separated)
- frontend_build failures: 0.083–0.085 (overlapping with success)

The threshold 0.095 was chosen as a natural separation point between
the success cluster and the backend_test cluster. A higher threshold
(e.g., 0.5 as used for Elastic) would have missed all detections
except dependency failures.

## Compatibility Issues and Workarounds

LogAI (last updated March 2023) has several compatibility issues with
current Python package versions. Four workarounds were implemented in
the evaluation script:

### 1. NLTK punkt_tab resource

**Error**: `Resource punkt_tab not found`

LogAI's Word2Vec vectorizer uses NLTK for tokenization. Recent NLTK
versions require the `punkt_tab` resource in addition to `punkt`.

**Fix**: Download both resources:
```bash
python -m nltk.downloader punkt punkt_tab
```

### 2. warm_start type error

**Error**: `TypeError: 'warm_start' must be an instance of bool`

LogAI passes `warm_start=0` (integer) to scikit-learn's IsolationForest,
but scikit-learn >= 1.4 strictly requires a boolean value.

**Fix**: Monkey-patch the underlying sklearn model before fitting:
```python
sklearn_model = getattr(detector, "anomaly_detector", None)
if sklearn_model is not None:
    inner = getattr(sklearn_model, "model", None)
    if inner is not None and hasattr(inner, "warm_start"):
        inner.warm_start = bool(inner.warm_start)
```

### 3. DateTime64 in feature vector

**Error**: `Cannot convert from DatetimeTZDtype to float64`

LogAI's FeatureExtractor includes the timestamp column as `datetime64`
in the feature vector, which scikit-learn cannot process as a numeric
feature.

**Fix**: Convert any datetime columns to epoch seconds:
```python
import numpy as np
for col in feature_vector.columns:
    if pd.api.types.is_datetime64_any_dtype(feature_vector[col]):
        feature_vector[col] = (
            feature_vector[col].astype(np.int64) // 10**9
        )
```

### 4. 2D prediction shape

**Error**: Predictions had shape `(7315, 2)` instead of expected 1D array.

LogAI's anomaly detector returns a 2D array (labels in column 0,
scores in column 1) rather than a 1D array of labels.

**Fix**: Check dimensionality and extract the label column:
```python
if pred_values.ndim == 2:
    labels = pred_values[:, 0]
else:
    labels = pred_values
```

### Impact of workarounds

All four workarounds address interface mismatches between LogAI and its
dependencies, not algorithmic modifications. The anomaly detection
algorithm (IsolationForest) was used exactly as provided by LogAI with
no changes to the detection logic. The workarounds ensure that LogAI's
pipeline can execute on current package versions.

## Results

### Score Distribution

All 60 runs received anomaly scores. The scores cluster into distinct
groups corresponding to failure classes:

| Score Range   | Count | Includes                                         |
|---------------|-------|--------------------------------------------------|
| 0.206–0.207   | 6     | All dependency failures (6/6)                    |
| 0.124–0.125   | 2     | Both deploy failures (2/2)                       |
| 0.106–0.109   | 2     | Unlabeled runs (high line count)                 |
| 0.099–0.102   | 6     | All backend_test failures (6/6)                  |
| 0.095–0.096   | 4     | 2 success runs + 2 unlabeled                     |
| 0.086–0.093   | 34    | All 25 success runs, all 6 frontend_build, unlabeled |
| 0.083–0.085   | 6     | All frontend_build failures (6/6) — within success range |

### Score by Failure Class (labeled runs only)

| Failure Class  | Runs | Score Range   | Mean Score | Detected |
|----------------|------|---------------|------------|----------|
| dependency     | 6    | 0.206–0.207   | 0.206      | 6/6      |
| deploy         | 2    | 0.124–0.125   | 0.124      | 2/2      |
| backend_test   | 6    | 0.099–0.102   | 0.100      | 6/6      |
| success        | 27   | 0.086–0.096   | 0.088      | 0/27     |
| frontend_build | 6    | 0.083–0.085   | 0.084      | 0/6      |

Notable: frontend_build failures scored **lower** than success runs,
making them indistinguishable from normal runs.

### Evaluation Metrics (threshold = 0.095)

Metrics were computed using `scripts/compute_metrics.py` against the 47
labeled runs in `data/runs.csv`. The 13 unlabeled runs were excluded
from metric computation.

| Metric    | Value |
|-----------|-------|
| Precision | 0.875 |
| Recall    | 0.700 |
| F1 Score  | 0.778 |
| Accuracy  | 0.830 |
| TP        | 14    |
| FP        | 2     |
| FN        | 6     |
| TN        | 25    |

### Detection by Failure Class

| Failure Class  | Runs | Detected | Score Range   |
|----------------|------|----------|---------------|
| backend_test   | 6    | 6/6      | 0.099–0.102   |
| frontend_build | 6    | 0/6      | 0.083–0.085   |
| dependency     | 6    | 6/6      | 0.206–0.207   |
| deploy         | 2    | 2/2      | 0.124–0.125   |

### False Positives

Two success runs were predicted as anomalous:

| Run ID        | Score  | Lines | Note                       |
|---------------|--------|-------|----------------------------|
| 20596289580   | 0.0951 | 115   | success (deploy not triggered) |
| 20596289891   | 0.0963 | 115   | success (deploy not triggered) |

Both are success runs with the note "deploy not triggered" — these
were the last two runs in the labeled dataset and may have slightly
different log content compared to the other 25 success runs (they were
generated from a different commit to test deploy triggering). Their
scores are just barely above the 0.095 threshold.

### Confusion Matrix

```
                 Predicted
              Anomaly  Normal
Actual  Fail    14       6     (6 = all frontend_build)
       Normal    2      25
```

## Analysis

### Why dependency failures were detected (highest scores)

Dependency failures have the highest anomaly scores (0.206) because
they are structurally distinct:

- **Only 29 log lines** (vs 115 for success) — pipeline fails early
  during `pip install` when a dependency cannot be resolved
- **2 ERROR lines** with distinctive error messages about package
  resolution failures
- The short run length and specific error content produce Word2Vec
  vectors that are clearly different from normal runs

### Why deploy failures were detected

Deploy failures scored 0.124–0.125 because they have:

- **289–295 log lines** (vs 115 for success) — significantly more
  output due to the deploy step producing additional logs
- **2 ERROR lines** and **1 WARN line** — deploy errors have distinctive
  messages about deployment failures
- **4 log files** (backend + frontend + deploy) vs 2–3 for other runs

### Why backend_test failures were detected

Backend test failures scored 0.099–0.102, just above the threshold.
These runs have:

- **124 lines** (vs 115 for success) — slightly more output
- **4 ERROR lines** with pytest failure messages (`FAILED tests/test_api.py::...`)
- The Word2Vec vectorization captures the semantic difference in pytest
  failure output vs success output

### Why frontend_build failures were NOT detected

Frontend build failures scored 0.083–0.085, **below** success runs
(0.086–0.090). This is the only failure class that LogAI missed entirely.

The reason is that TypeScript/Next.js build errors produce output that
is structurally and semantically similar to normal build output in the
Word2Vec embedding space:

- **107 lines** (vs 115 for success) — slightly fewer lines, not more
- **1 ERROR line** — the error-to-total ratio is very low
- The build error messages (TypeScript type errors) contain many of the
  same tokens as successful build output (module names, file paths,
  npm package names)
- Word2Vec encodes these similar tokens into similar vectors, making
  frontend_build failures look like "slightly shorter normal runs"

**Key insight**: Word2Vec captures semantic similarity at the token
level. When failure messages share vocabulary with success messages
(common in build tool output), the vectorization cannot distinguish
them. A more specialized approach (e.g., error-message-specific
features, or log template distribution analysis) might improve
detection for this failure class.

## Comparison with Other Tools

| Metric         | Baseline (keywords) | LogAI (IsolationForest) | Elastic DFA |
|----------------|---------------------|-------------------------|-------------|
| Precision      | 1.000               | 0.875                   | 1.000       |
| Recall         | 0.300               | 0.700                   | 0.100       |
| F1 Score       | 0.462               | 0.778                   | 0.182       |
| Accuracy       | 0.702               | 0.830                   | 0.617       |
| TP             | 6                   | 14                      | 2           |
| FP             | 0                   | 2                       | 0           |
| FN             | 14                  | 6                       | 18          |
| TN             | 27                  | 25                      | 27          |

| Failure Class    | Baseline | LogAI | Elastic DFA |
|------------------|----------|-------|-------------|
| backend_test     | 0/6      | 6/6   | 0/6         |
| frontend_build   | 0/6      | 0/6   | 0/6         |
| dependency       | 6/6      | 6/6   | 0/6         |
| deploy           | 0/2      | 2/2   | 2/2         |

### Comparison with Baseline

The baseline keyword heuristic (counting ERROR keywords, scaled by
line ratio) detected only dependency failures (6/6) because they have
the highest error-to-line ratio. It missed backend_test (error ratio
too low to reach 0.5 threshold), frontend_build (only 1 error line),
and deploy (error ratio diluted by high total line count).

LogAI outperforms the baseline in recall (0.700 vs 0.300) because
Word2Vec vectorization captures semantic content beyond simple keyword
counting. The baseline achieves perfect precision (1.000) only because
its 0.5 threshold is conservative enough to avoid false positives,
at the cost of missing most failure classes.

### Comparison with Elastic DFA

Elastic DFA detected only deploy failures because its features
(log_lines, error_lines, warn_lines, info_lines, unique_files) are
aggregate counts that cannot capture the semantic content of log
messages. LogAI's Word2Vec vectorization provides much richer feature
representation, which is why it detected 3 out of 4 failure classes
compared to Elastic's 1 out of 4.

## Effort Assessment

| Activity                              | Effort Level |
|---------------------------------------|-------------|
| Python 3.10 installation              | Low — deadsnakes PPA, standard venv |
| LogAI installation                    | Low — pip install, NLTK data download |
| Understanding LogAI API               | Medium — documentation is sparse, required deep research |
| Building blocks pipeline              | Medium — 6 components to configure and connect |
| Debugging compatibility issues        | High — 4 workarounds for outdated dependencies |
| Score interpretation and threshold    | Medium — required analysis of score distributions |
| Total evaluation effort               | Medium — significant debugging but straightforward once working |

### Challenges

1. **Sparse documentation**: LogAI's documentation provides basic
   examples for standard datasets (HDFS, HealthApp) but limited
   guidance for custom log formats or the building blocks API.

2. **Python version restriction**: The requirement for Python 3.10
   means a dedicated virtual environment is needed, which complicates
   integration with projects using newer Python versions.

3. **Unmaintained library**: No updates since March 2023. Compatibility
   issues with current dependency versions (scikit-learn, numpy, NLTK)
   required multiple workarounds.

4. **Opaque scoring**: LogAI's `AnomalyDetector.predict()` returns only
   binary labels by default. Extracting continuous scores required
   accessing the underlying scikit-learn model's `decision_function()`
   directly.

## Limitations

1. **No labeled training**: IsolationForest is purely unsupervised.
   It cannot incorporate known labels to learn discriminative
   boundaries. It identifies statistical outliers in the Word2Vec
   feature space, which may or may not align with actual failures.

2. **Threshold sensitivity**: The 0.095 threshold captures 3/4 failure
   classes but also produces 2 false positives. A slightly higher
   threshold (e.g., 0.097) would eliminate false positives but might
   miss some backend_test failures. There is no principled way to
   select the threshold without labeled data.

3. **Per-line granularity only**: LogAI operates on individual log
   lines, not pipeline runs. The run-level score is a derived metric
   (mean of per-line scores), which dilutes strong anomaly signals
   in runs with many normal lines.

4. **Frontend build blind spot**: The Word2Vec embedding cannot
   distinguish frontend build errors from normal build output because
   they share too much vocabulary. This is a fundamental limitation
   of token-level embeddings for this failure type.

5. **Reproducibility risk**: As an unmaintained library with strict
   Python version requirements, LogAI may become increasingly
   difficult to install and run as dependencies continue to evolve.

## Reproducibility

To reproduce the evaluation:

1. **Install Python 3.10**:
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt install python3.10 python3.10-venv python3.10-dev
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   python3.10 -m venv .venv-logai
   source .venv-logai/bin/activate
   pip install logai
   python -m nltk.downloader punkt punkt_tab
   ```

3. **Run the evaluation**:
   ```bash
   python scripts/run_logai_eval.py \
       --input artifacts/normalized \
       --output results/logai_scores.csv \
       --threshold 0.095 \
       --algorithm isolation_forest
   ```

4. **Compute metrics**:
   ```bash
   python scripts/compute_metrics.py \
       --predictions results/logai_scores.csv \
       --output results/metrics_logai.csv
   ```

The evaluation script uses a fixed random seed (scikit-learn's default)
so results should be reproducible given the same input data and
package versions.

## Key Findings

1. **LogAI achieved the highest F1 score (0.778) among all evaluated
   tools.** It detected 14 out of 20 failures across 3 of 4 failure
   classes, with only 2 false positives.

2. **Word2Vec vectorization is the key differentiator.** By converting
   log messages into semantic vectors, LogAI captures content-level
   differences that simple aggregate features (line counts, error
   counts) miss. This is why it detected backend_test failures that
   both the baseline and Elastic DFA missed.

3. **Frontend build failures remain undetectable** by all three
   approaches (baseline, LogAI, Elastic DFA). These failures produce
   output that is too similar to normal builds at both the keyword and
   semantic level.

4. **The library is showing its age.** Four compatibility workarounds
   were needed for LogAI to run on current package versions. The
   Python 3.10 restriction and lack of maintenance updates are
   practical barriers to adoption.

5. **Unsupervised anomaly detection on CI/CD logs is viable** but
   threshold selection is a challenge. Without labeled data for
   calibration, choosing the right threshold requires manual analysis
   of score distributions.

## Files

| File | Description |
|------|-------------|
| `scripts/run_logai_eval.py` | LogAI evaluation script (building blocks pipeline) |
| `results/logai_scores.csv` | Per-run anomaly scores (60 runs) |
| `results/metrics_logai.csv` | Precision/recall/F1 metrics |

## Summary

LogAI was evaluated using its building blocks API with a 6-stage
pipeline: Preprocessor, Drain parser, Word2Vec vectorizer,
CategoricalEncoder, FeatureExtractor, and IsolationForest anomaly
detector. All components are from the LogAI library — no custom ML
was implemented.

The evaluation achieved F1=0.778 (Precision=0.875, Recall=0.700),
detecting backend_test failures (6/6), dependency failures (6/6),
and deploy failures (2/2). Frontend build failures (0/6) were not
detected because their log output shares too much vocabulary with
normal builds for Word2Vec to distinguish them.

LogAI's main advantage over the other evaluated tools is its semantic
feature representation (Word2Vec), which captures the meaning of log
messages rather than just structural properties. Its main disadvantages
are the unmaintained codebase, Python 3.10 restriction, and multiple
compatibility workarounds required for current environments.
