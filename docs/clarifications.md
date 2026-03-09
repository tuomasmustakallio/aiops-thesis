# Thesis Clarifications

Answers to specific questions about the evaluation methodology, definitions, and decisions.

---

## 1. Evaluation Practicalities

### a. Dynatrace — Log ingestion only, or also time-series metrics?

**Log ingestion only.** No infrastructure metrics (CPU, memory, latency) were used. The pipeline was:

1. Normalized CI/CD log lines ingested via Dynatrace Log Ingestion API v2
2. Logs converted to a time series using DQL: `fetch logs | filter matchesValue(loglevel, "ERROR") | makeTimeseries error_count = count(), interval: 1m, by:{run_id}`
3. Davis Anomaly Detection evaluated the resulting time series with a static threshold

The "time-series aggregation" happened on the log data itself (counting ERROR-level lines per minute per run), not on external metrics. Davis AI cannot analyze raw logs directly — it requires a time-series input, so the DQL `makeTimeseries` step is a mandatory part of its architecture.

### b. Elastic — ML anomaly detection or simple threshold?

**Both were attempted. ML anomaly detection produced no results; DFA outlier detection was used for final metrics.**

Two Elastic ML approaches were tried:

1. **ML Anomaly Detection (time-series)**: Two jobs were created — a `high_count` detector partitioned by `run_id` and a log categorization job with `count by mlcategory`. Both produced zero anomalies because the dataset (47 runs in a ~17-minute window) did not provide enough temporal variation for Elastic to learn a baseline. Elastic ML needs hours-to-weeks of continuous data. This is an architectural limitation, not a configuration error.

2. **Data Frame Analytics (DFA) outlier detection**: A batch-mode ML approach. Logs were aggregated into per-run feature vectors (log_lines, error_lines, warn_lines, info_lines, unique_files) via an Elasticsearch Transform, then DFA's ensemble outlier detection algorithm scored each run. This is Elastic's built-in unsupervised ML — not a manual threshold. The F1=0.182 result comes from this approach.

No simple rule/threshold heuristic was used for the Elastic evaluation. The baseline keyword heuristic (`run_baseline.py`) is a separate reference point, not part of the Elastic evaluation.

### c. LogAI — Existing model or trained/tuned?

**No pre-trained model was reused. No tuning was performed.** LogAI's IsolationForest is an unsupervised algorithm — it does not use labeled training data. The model was fit on the full dataset (7,315 log lines) in a single pass and produced anomaly scores without any labeled supervision.

The pipeline used LogAI's provided components with default or near-default parameters:
- Drain parser: `sim_th=0.5, depth=5` (Drain defaults)
- Word2Vec vectorizer: default settings
- IsolationForest: `n_estimators=200, max_features=1.0`, default contamination

The only manually selected parameter was the **classification threshold** (0.095), chosen by analyzing the score distribution after the model ran. This is a post-hoc decision point, not model tuning — the model itself was not modified.

---

## 2. Definition of a "Detection"

### a. Dynatrace

A run is counted as **detected** if Davis AI creates a `CUSTOM_ALERT` event whose name contains that run's `run_id`.

Specifically: the DQL query `makeTimeseries error_count = count(), interval: 1m, by:{run_id}` creates a separate time series per run. The static threshold alert fires when `error_count > 0` for any run_id dimension. Each fired alert creates a Davis event named `CI/CD Error Per Run <run_id>`. The event's presence in `dt.davis.events` constitutes detection.

**Operational definition**: Event tied to run_id exists in `dt.davis.events` → run is detected as anomalous.

### b. Elastic

A run is counted as **detected** if the DFA outlier detection assigns an `outlier_score >= 0.5` to that run's summary document in the `cicd-run-outliers` index.

The 0.5 threshold is generous — the actual score distribution has a clear gap between detected (0.994–0.995) and undetected (0.038) runs, so any threshold between 0.04 and 0.94 would produce the same binary result.

**Operational definition**: `outlier_score >= 0.5` in DFA results → run is detected as anomalous.

### c. LogAI

A run is counted as **detected** if its aggregated anomaly score exceeds the threshold of 0.095.

The per-line IsolationForest scores (inverted `decision_function()` values, normalized to [0,1]) are averaged across all log lines in a run to produce a per-run score. This per-run score is compared against the threshold.

**Operational definition**: Mean per-line anomaly score > 0.095 → run is detected as anomalous.

---

## 3. Temporal Detection Window

### When were logs sampled?

**Entire run logs after completion.** For all three tools, the complete set of log lines from each pipeline run was collected after the run finished. No partial or streaming ingestion was performed during the run.

The log collection pipeline:
1. GitHub Actions produces log artifacts upon workflow completion
2. `scripts/download_artifacts.py` downloads all artifacts
3. `scripts/normalize_logs.py` converts them to a standard format
4. The normalized logs (entire runs) are then ingested into each tool

### Was early detection attempted?

**No.** This evaluation is **reactive** (post-run analysis), not **proactive** (mid-run detection). The question "can this tool detect a failing run?" was evaluated on complete run logs, not on partial logs mid-execution.

This is an important framing distinction for the thesis:
- The evaluation measures whether the tool can **correctly classify** a completed run as failed or succeeded
- It does **not** measure whether the tool could detect the failure **before** the pipeline finishes
- Early/proactive detection would require streaming log ingestion during pipeline execution, which was outside the scope

### Why perfect recall for some tools?

Dynatrace achieved perfect recall because its detection mechanism (ERROR count > 0 per run) is fundamentally binary: failure runs produce ERROR-level log lines, success runs do not. Given the controlled experiment design (deterministic failure injection), this alignment between detection signal and ground truth is expected.

Elastic's perfect recall (1.0) in the DFA approach applies only to deploy failures — it detected 2/2 deploy failures but missed all other 18 failures, giving an overall recall of 0.100.

---

## 4. Ground Truth Labelling

### Who labeled the runs?

**The thesis author (single labeler).** Ground truth labels were assigned by the person who designed and executed the experiment.

### How were labels determined?

Labels are **deterministic from the experiment design**, not subjective human judgment:

- Each experiment branch (`experiment/success`, `experiment/backend-fail`, etc.) produces a **predetermined outcome**. The failure injection is controlled and deterministic.
- `experiment/success` → always succeeds → labeled `success`
- `experiment/backend-fail` → always fails at pytest → labeled `failure, backend_test`
- `experiment/frontend-fail` → always fails at TypeScript build → labeled `failure, frontend_build`
- `experiment/dep-fail` → always fails at pip install → labeled `failure, dependency`
- `experiment/deploy-fail` → always fails at Docker build → labeled `failure, deploy`

The labeling was cross-checked against the actual GitHub Actions run outcomes (pass/fail status) using `scripts/label_runs.py`.

### Were there disagreements?

**No disagreements possible.** Since labels are determined by the experiment branch (not by subjective log interpretation), there is no ambiguity. A run on `experiment/backend-fail` is always labeled as `failure, backend_test` because the branch deterministically injects a failing test.

### Validity

The ground truth is valid because:
1. Failure injection is deterministic and controlled
2. Labels match the branch design by construction
3. Labels were verified against actual GitHub Actions outcomes
4. No borderline or ambiguous cases exist in this controlled setup

---

## 5. Failure Class Usage

### Are per-class results available?

**Yes**, for all three tools. Per-class breakdowns are documented in each tool's evaluation doc.

Summary:

| Failure Class    | LogAI   | Elastic DFA | Dynatrace | Baseline |
|------------------|---------|-------------|-----------|----------|
| backend_test     | 6/6     | 0/6         | 6/6       | 0/6      |
| frontend_build   | 0/6     | 0/6         | 6/6       | 0/6      |
| dependency       | 6/6     | 0/6         | 6/6       | 6/6      |
| deploy           | 2/2     | 2/2         | 2/2       | 0/2      |
| **Total**        | **14/20** | **2/20**  | **20/20** | **6/20** |

### Where are per-class results stored?

- **LogAI**: `docs/logai_evaluation.md` (Detection by Failure Class section) and `results/logai_scores.csv` (per-run scores joinable with `data/runs.csv`)
- **Elastic**: `docs/elastic_evaluation.md` (Detection by Failure Class section) and `results/elastic_scores.csv`
- **Dynatrace**: `docs/dynatrace_evaluation.md` (Per-Failure-Class Breakdown section) and `results/table-data.csv`

### Should per-class results be in the thesis?

**Yes, strongly recommended.** The per-class breakdown reveals the most important insight: different tools detect different failure types. The aggregate F1 score alone misses this nuance. For example:
- Only Dynatrace detected frontend_build failures
- Only LogAI and Dynatrace detected backend_test failures
- The baseline only detected dependency failures (highest error ratio)

---

## 6. Baseline Comparison

### Was a baseline computed?

**Yes.** `scripts/run_baseline.py` implements a naive ERROR keyword heuristic: it counts ERROR occurrences per run, computes an error-to-line ratio (scaled by 10, capped at 1.0), and classifies runs with score >= 0.5 as anomalous.

### Baseline results

| Metric    | Value |
|-----------|-------|
| Precision | 1.000 |
| Recall    | 0.300 |
| F1 Score  | 0.462 |
| TP        | 6     |
| FP        | 0     |
| FN        | 14    |
| TN        | 27    |

The baseline detected only **dependency failures** (6/6) because they have the highest error-to-line ratio (2 errors in 29 lines). It missed backend_test (4 errors in 124 lines — ratio too low), frontend_build (1 error in 107 lines), and deploy (2 errors in 289+ lines — ratio diluted).

### Where are baseline results stored?

- `results/baseline_scores.csv` — per-run scores
- `results/metrics_baseline.csv` — aggregate metrics

### How to contextualize

The baseline establishes a floor: any AI tool that performs worse than keyword counting (F1=0.462) provides negative value. All three AI tools should be compared against this baseline:
- **LogAI (F1=0.778)**: Significantly above baseline — Word2Vec captures semantics beyond keywords
- **Elastic DFA (F1=0.182)**: Below baseline — aggregate feature counts are less discriminative than keyword ratios
- **Dynatrace (F1=1.000)**: Far above baseline — per-run error presence is a strong signal when configured correctly

---

## 7. Screenshots

### What screenshots exist?

All screenshots are in `pics/`.

**Dynatrace** (5 screenshots):
1. `01_dynatrace_notebook_analyze_and_alert.png` — Notebooks: makeTimeseries query with "Analyze and alert" enabled
2. `02_dynatrace_notebook_errors.png` — Notebooks: ERROR log time series visualization
3. `03_dynatrace_notebook_alerts.png` — Notebooks: `fetch dt.davis.events` query showing all Davis events
4. `04_dynatrace_notebook_anomalies.png` — Notebooks: per-run event list with parsed run_ids
5. `05_anomaly_alerts.png` — Anomaly Detection app: custom alert configurations

**Elastic** (6 screenshots):
1. `01_elastic_dfa_results_explorer.png` — DFA Results Explorer showing outlier scores table
2. `02__elastic_dfa_scatterplot.png` — DFA scatterplot matrix
3. `03__elastic_dfa_job_details.png` — DFA job configuration and stats
4. `04__elastic_anomaly_detection_job.png` — ML Anomaly Detection job showing "no anomalies"
5. `05__elastic_discover_logs.png` — Discover view of ingested log documents
6. `06__elastic_transform_details.png` — Transform stats (7,315 → 60 documents)

**LogAI** — No GUI screenshots. LogAI was run entirely via Python scripts (CLI-based). The optional Dash GUI was not used.

### For thesis figures

Yes, screenshots should be included as figures. Suggested labeling:
- Dynatrace: makeTimeseries analysis (`pics/01_dynatrace`), Davis events (`pics/03_dynatrace` or `04_dynatrace`), alert config (`pics/05_anomaly`)
- Elastic: DFA results (`pics/01_elastic`), anomaly detection "no anomalies" (`pics/04_elastic`), transform (`pics/06_elastic`)

---

## 8. Script Execution Environment

### Where were evaluations run?

**Local machine running WSL2 (Windows Subsystem for Linux).**

| Property | Value |
|----------|-------|
| **OS** | Ubuntu on WSL2 |
| **Kernel** | Linux 5.15.133.1-microsoft-standard-WSL2 |
| **Host OS** | Windows |
| **Python** | 3.10 (LogAI), 3.12 (other scripts) |
| **Shell** | zsh |

### Tool-specific environments

- **LogAI**: Ran locally in a dedicated Python 3.10 venv (`.venv-logai`). All processing was local — no cloud resources.
- **Elastic**: Data ingested from local machine to Elastic Cloud (serverless deployment). ML jobs ran on Elastic Cloud infrastructure. Results viewed in Kibana web UI.
- **Dynatrace**: Data ingested from local machine to Dynatrace SaaS trial (urk56544.live.dynatrace.com). Anomaly detection ran on Dynatrace cloud infrastructure. Results queried via DQL in Dynatrace web UI.

### CI/CD pipeline environment

The CI/CD pipeline itself ran on **GitHub Actions** (GitHub-hosted runners). The web application was deployed to **Azure App Service** (Azure Container Registry + Web App for Containers). These are the systems being monitored — the evaluation tools analyzed logs produced by these environments.

### For the thesis methodology section

State: "Evaluations were executed on a local development machine running Ubuntu under WSL2. LogAI ran entirely locally. Elastic and Dynatrace evaluations used cloud-hosted trial instances with data ingested from the local machine via REST APIs. The CI/CD pipeline under observation ran on GitHub Actions with deployment to Azure App Service."
