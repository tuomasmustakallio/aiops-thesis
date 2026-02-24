# Elastic Observability — Evaluation Documentation

## Overview

Elastic Observability was evaluated as the second AI tool for CI/CD pipeline
failure detection. Two approaches were tested:

1. **ML Anomaly Detection** (time-series) — log categorization job
2. **Data Frame Analytics (DFA)** — batch outlier detection on run-level features

Only the DFA approach produced results. The time-series anomaly detection
found no anomalies due to insufficient temporal data.

## Environment

- **Elastic Cloud**: Serverless deployment (free trial, 14 days)
- **Elasticsearch version**: 8.11.0
- **Subscription tier**: Trial (includes ML features)
- **Data**: 7,315 normalized log lines from 60 CI/CD pipeline runs
- **Evaluation date**: 2026-02-22

## Data Ingestion

### Method

A Python script (`scripts/ingest_elastic.py`) bulk-uploaded all normalized
log lines to Elasticsearch via the `elasticsearch-py` client library.

Each log line became one document with the following schema:

| Field          | Type    | Description                        |
|----------------|---------|------------------------------------|
| run_id         | keyword | Pipeline run identifier            |
| job_name       | keyword | CI job name (e.g. backend-logs-*)  |
| log_file       | keyword | Source file (install.log, etc.)    |
| log_level      | keyword | ERROR, WARN, INFO, DEBUG           |
| message        | text    | Log line content                   |
| timestamp      | date    | Run timestamp + per-line offset    |
| line_number    | integer | Line position within run           |
| outcome        | keyword | success or failure (from ground truth) |
| failure_class  | keyword | none, backend_test, frontend_build, dependency, deploy |

### Index

- **Index name**: `cicd-pipeline-logs`
- **Documents indexed**: 7,315
- **Runs covered**: 60 (47 labeled + 13 from earlier test runs)

The 13 unlabeled runs originated from experiment infrastructure development
and testing before the final labeled dataset was generated. They were present
in the `artifacts/normalized/` directory and were ingested alongside the
labeled runs. These runs received `outcome: unknown` and
`failure_class: unknown` tags. They were excluded from metric computation
but were included in the DFA analysis (the algorithm sees all 60 documents).

### Command

```bash
python scripts/ingest_elastic.py \
    --url "$ELASTIC_URL" \
    --api-key "$ELASTIC_API_KEY"
```

Credentials stored in `.env` (not committed to repository).

## Approach 1: ML Anomaly Detection (Time-Series)

### Rationale

Elastic ML Anomaly Detection monitors data over time, learns a baseline of
"normal" behavior, and flags deviations. This is Elastic's primary ML
capability for log analysis. It was tested first because it is the most
commonly referenced Elastic ML feature for log analysis.

### Iterative Process

The anomaly detection evaluation went through several iterations due to
configuration issues and UI limitations in Elastic Cloud serverless mode:

1. **First attempt — Single-metric wizard**: Opened Kibana ML > Anomaly
   Detection > Create Job. The single-metric wizard only offered "Distinct
   count" for keyword fields like `log_level`. The basic "Count" (event rate)
   function was not listed. Abandoned this wizard.

2. **Second attempt — Multi-metric wizard**: Same limitation as single-metric.
   The multi-metric wizard did not expose the `high_count` detector function
   for the ingested data. Abandoned this wizard.

3. **Third attempt — Advanced Job wizard (Job 1)**: Used the Advanced Job
   wizard to manually configure a `high_count` detector partitioned by
   `run_id`. Encountered a datafeed query parse error on the first try
   (full job JSON was accidentally placed in the datafeed query field).
   Successfully created on the second try, but influencers were not saved.
   Job ran to completion but found no anomalies.

4. **Fourth attempt — Advanced Job wizard (Job 2)**: Configured a log
   categorization job with `count by mlcategory` detector and a 15-minute
   bucket span. This job ran successfully and categorized all 7,315 log
   lines into 98 templates. However, it still found no anomalies due to
   the same empty-bucket problem.

5. **Research phase**: Conducted deep research into Elastic ML data
   requirements and alternative approaches. This led to the discovery of
   DFA outlier detection as a batch-compatible alternative (see Approach 2).

### Configuration

Two jobs were created through the Kibana Advanced Job wizard:

#### Job 1: `cicd-log-anomaly-test` (high_count)

```json
{
  "analysis_config": {
    "bucket_span": "1m",
    "detectors": [{
      "function": "high_count",
      "partition_field_name": "run_id"
    }],
    "influencers": []
  }
}
```

- **Records processed**: 7,315
- **Bucket count**: 523,470
- **Empty buckets**: 523,467 (99.999%)
- **Model memory**: 28 MB (12 MB limit, but status "ok")
- **Result**: No anomalies detected

**Issues encountered during Job 1 creation:**

1. **Single/Multi-metric wizard limitations**: The Kibana wizard for
   single-metric and multi-metric jobs only showed "Distinct count" options
   for keyword fields. The plain "Count" (event rate) detector was not
   accessible through these wizards, requiring the Advanced job wizard.

2. **Validation warning**: `"run_id" is not an aggregatable field` — the
   `run_id` field was treated as non-aggregatable in the serverless
   environment. The job was created despite this warning.

3. **Datafeed query parse error**: The first creation attempt failed with
   `datafeed_config failed to parse field [query]` because the full job
   JSON was accidentally placed in the datafeed query field (which expects
   only an Elasticsearch query like `{"match_all": {}}`).

4. **Influencers not saved**: Despite configuring `run_id`, `log_level`,
   and `failure_class` as influencers in the UI, the resulting job had
   `"influencers": []` (empty array), likely due to the validation error
   during creation.

#### Job 2: `cicd-log-categorization-advanced-2` (log categorization)

```json
{
  "analysis_config": {
    "bucket_span": "15m",
    "categorization_field_name": "message",
    "detectors": [{
      "function": "count",
      "by_field_name": "mlcategory"
    }],
    "influencers": ["log_level", "failure_class", "run_id"]
  }
}
```

- **Records processed**: 7,315
- **Categorized documents**: 7,315 (all documents categorized successfully)
- **Categories found**: 98 total
  - 7 frequent categories
  - 0 rare categories
  - 2 dead categories
  - 0 failed categories
- **Bucket count**: 34,898
- **Empty buckets**: 34,896 (99.99%)
- **Model memory**: 34.8 MB (64 MB limit)
- **Result**: No anomalies detected

The categorization itself worked — Elastic successfully parsed all 7,315
log lines into 98 distinct log templates. However, the `count by mlcategory`
detector found no temporal anomalies in category frequencies because the
same empty-bucket problem persisted.

### Timestamp Distribution Issue

A critical factor in both jobs: the timestamp distribution was artificially
spread across nearly a full year. The 47 labeled runs had timestamps around
2025-12-30T12:13-12:30 UTC (~17 minute window), but 13 additional runs from
earlier testing had no ground truth entry and received a default timestamp of
2025-01-01T00:00:00 UTC in the ingestion script. This created:

- **Earliest record**: 2025-01-01 02:00:00
- **Latest record**: 2025-12-30 14:30:41
- **Apparent time span**: ~365 days
- **Actual data**: concentrated in two clusters (Jan 1 and Dec 30)
- **Empty time between**: 364 days of nothing

Even correcting this to the true 17-minute window would not resolve the
fundamental issue — Elastic ML needs much longer baselines.

### Why No Anomalies Were Found

The fundamental issue is data distribution, not tool configuration:

1. **Insufficient temporal spread**: All 47 labeled runs occurred within a
   ~17-minute window (2025-12-30 12:13–12:30 UTC). Elastic ML needs
   hours to weeks of data to establish a baseline of "normal."

2. **No streaming context**: Elastic ML is designed for continuous log
   streams where it can learn patterns over time. Our batch-uploaded
   historical data provides no temporal variation to model.

3. **Almost all buckets empty**: With 1-minute buckets spanning the full
   timestamp range (Jan 2025 – Dec 2025, due to 13 runs with default
   timestamps), 99.99% of buckets contained zero events.

4. **Minimum data requirements**: Elastic documentation states that
   count-based detectors need "4 non-empty bucket spans or 2 hours
   (whichever is greater)" and rare detectors need "~20 bucket spans."
   Our data does not meet these thresholds.

### Minimum Data Requirements (from Elastic documentation)

Research into Elastic's documentation and community forums confirmed
the following minimum data requirements for anomaly detection:

| Detector type | Minimum requirement |
|---------------|---------------------|
| Sampled metrics (mean/min/max/median) | 8 non-empty bucket spans or 2 hours, whichever is greater |
| Count-based (count, sum) | 4 non-empty bucket spans or 2 hours, whichever is greater |
| Rare/freq_rare | ~20 bucket spans |
| General recommendation | 1–3 weeks for reliable baselines |

Source: Elastic community discussions and official documentation.
Our ~17 minute dataset does not meet any of these thresholds.

### Conclusion (Anomaly Detection)

Elastic ML Anomaly Detection is architecturally unsuitable for offline
batch analysis of CI/CD logs from a narrow time window. The tool is
designed for operational monitoring of continuous data streams. This is
a design-level mismatch, not a configuration error. The categorization
engine itself worked (98 templates identified), but the time-series
anomaly scoring layer could not function without temporal variation.

## Approach 2: Data Frame Analytics — Outlier Detection

### Rationale

After the anomaly detection results, targeted research was conducted to
determine whether Elastic offers any batch-compatible ML approach. The
research (documented in `deep-research-report.md`) identified **Data Frame
Analytics (DFA) outlier detection** as Elastic's batch analysis capability:

- DFA outlier detection is explicitly designed as a **one-time batch
  analysis on a static snapshot** of data
- It does **not require time-series data** or temporal baselines
- It produces per-document `outlier_score` values and `feature_influence`
  explanations
- It works best on **entity-centric data** (one document per entity)

The research recommended creating a Transform to pivot log-level documents
into run-level summaries, then running DFA outlier detection on the
summaries. This is the approach we implemented.

### Method

#### Step 1: Transform — Aggregate logs to run-level features

Created an Elasticsearch Transform to pivot 7,315 log-level documents
into 60 run-level summary documents (one per pipeline run):

```json
PUT _transform/cicd-run-summary
{
  "source": { "index": "cicd-pipeline-logs" },
  "dest": { "index": "cicd-run-summary" },
  "pivot": {
    "group_by": {
      "run_id": { "terms": { "field": "run_id" } }
    },
    "aggregations": {
      "log_lines":    { "value_count": { "field": "run_id" } },
      "error_lines":  { "filter": { "term": { "log_level": "ERROR" } } },
      "warn_lines":   { "filter": { "term": { "log_level": "WARN" } } },
      "info_lines":   { "filter": { "term": { "log_level": "INFO" } } },
      "unique_files": { "cardinality": { "field": "log_file" } }
    }
  }
}
```

**Result**: 7,315 documents → 60 run-summary documents.

Features per run:

| Feature       | Description                        |
|---------------|------------------------------------|
| log_lines     | Total number of log lines          |
| error_lines   | Count of ERROR-level lines         |
| warn_lines    | Count of WARN-level lines          |
| info_lines    | Count of INFO-level lines          |
| unique_files  | Number of distinct log files       |

#### Step 2: DFA Outlier Detection

```json
PUT _ml/data_frame/analytics/cicd-run-outliers
{
  "source": { "index": "cicd-run-summary" },
  "dest": { "index": "cicd-run-outliers", "results_field": "ml" },
  "analysis": {
    "outlier_detection": {
      "compute_feature_influence": true
    }
  },
  "analyzed_fields": {
    "includes": ["log_lines", "error_lines", "warn_lines", "info_lines", "unique_files"]
  }
}
```

**Algorithm**: Ensemble method (automatic)
- `outlier_fraction`: 0.05
- `standardization_enabled`: true
- `compute_feature_influence`: true

**Result**: 60 documents analyzed, 0 skipped.

### Results

#### Outlier Scores

The DFA algorithm (ensemble method) assigned outlier scores to all 60 runs.
The scores show a clear separation into distinct tiers:

**Tier 1 — Strong outliers (score > 0.9):**

| Outlier Score | Run ID        | Lines | Errors | Warns | Files | Ground Truth     |
|---------------|---------------|-------|--------|-------|-------|------------------|
| 0.995         | 20596576671   | 289   | 2      | 1     | 4     | deploy failure   |
| 0.994         | 20596575942   | 295   | 2      | 1     | 4     | deploy failure   |
| 0.945         | 20596161099   | 361   | 0      | 4     | 4     | (not in GT)      |
| 0.943         | 20596223801   | 357   | 0      | 4     | 4     | (not in GT)      |
| 0.943         | 20596685733   | 359   | 0      | 4     | 4     | (not in GT)      |

These runs stand out due to high log_lines (289–361) and/or unique
combination of errors + warnings + 4 log files. The deploy failures are
the only labeled failures in this tier.

**Tier 2 — Moderate outliers (score 0.4–0.5):**

| Outlier Score | Run ID        | Lines | Errors | Warns | Files | Ground Truth     |
|---------------|---------------|-------|--------|-------|-------|------------------|
| 0.421         | 20596174271   | 117   | 0      | 0     | 3     | (not in GT)      |
| 0.421         | 20596174287   | 117   | 0      | 0     | 3     | (not in GT)      |

These are from earlier test runs not in ground truth.

**Tier 3 — Low scores (score < 0.06):**

| Outlier Score | Count | Includes                                          |
|---------------|-------|---------------------------------------------------|
| 0.052         | 2     | Earlier test runs                                 |
| 0.045         | 1     | Earlier test run                                  |
| 0.038         | 51    | All 25 success runs, all 6 backend_test, all 6 frontend_build, all 6 dependency, plus remaining unlabeled |

All four non-deploy failure classes (backend_test, frontend_build,
dependency) received the same score (0.038) as success runs — the DFA
could not distinguish them.

The top-scoring runs were dominated by `unique_files` and `log_lines`
feature influence — deploy failures had 4 log files (backend + frontend +
deploy) while most runs had only 2-3.

#### Feature Influence (Top Outlier: 20596576671)

| Feature      | Influence |
|--------------|-----------|
| unique_files | 0.264     |
| error_lines  | 0.211     |
| info_lines   | 0.180     |
| log_lines    | 0.173     |
| warn_lines   | 0.171     |

#### Evaluation Metrics (threshold = 0.5)

Metrics were computed using `scripts/compute_metrics.py` against the 47
labeled runs in `data/runs.csv`. The 13 unlabeled runs (from earlier
experiment infrastructure testing) were excluded from metric computation.
A threshold of 0.5 was used: any run with `outlier_score >= 0.5` was
predicted as anomalous.

| Metric    | Value |
|-----------|-------|
| Precision | 1.000 |
| Recall    | 0.100 |
| F1 Score  | 0.182 |
| Accuracy  | 0.617 |
| TP        | 2     |
| FP        | 0     |
| FN        | 18    |
| TN        | 27    |

Note: The 0.5 threshold is generous to the tool — the actual score gap
between detected failures (0.994–0.995) and undetected ones (0.038) is
so wide that any threshold between 0.04 and 0.94 would produce the same
binary classification result.

#### Detection by Failure Class

| Failure Class  | Runs | Detected | Score Range |
|----------------|------|----------|-------------|
| backend_test   | 6    | 0/6      | 0.038       |
| frontend_build | 6    | 0/6      | 0.038       |
| dependency     | 6    | 0/6      | 0.038       |
| deploy         | 2    | 2/2      | 0.994–0.995 |

### Analysis

DFA outlier detection successfully identified deploy failures because they
have a distinctive feature profile: more log files (4 vs 2-3), the
combination of errors AND warnings, and higher total line count.

The other three failure classes were not detected because their aggregate
features (log_lines, error_lines, warn_lines) are too similar to success
runs at the run level:

- **backend_test failures**: 124 lines, 4 errors — similar to success runs
  (115 lines, 0 errors). The error count difference is not extreme enough
  for the outlier ensemble to flag.
- **frontend_build failures**: 107 lines, 1 error — virtually identical to
  success runs in aggregate features.
- **dependency failures**: 29 lines, 2 errors — short runs (pipeline fails
  early), but several success-adjacent runs also have ~29 lines.

**Key insight**: DFA outlier detection is only as good as the features it
receives. Simple log-level counts lose the semantic information that LogAI's
Word2Vec vectorization captures. More sophisticated features (e.g., TF-IDF
on error messages, log template distributions) could improve results, but
would require custom feature engineering outside of Elastic's built-in
Transform capabilities.

### Limitations of the DFA Approach

1. **Feature expressiveness**: Elasticsearch Transforms only support
   standard aggregations (counts, sums, cardinality, min/max, etc.).
   There is no built-in way to compute text embeddings, TF-IDF vectors,
   or log template distributions within a Transform. This means the
   outlier detection can only operate on structural features (how many
   lines, how many errors) rather than semantic features (what the error
   messages say).

2. **No labeled training**: DFA outlier detection is purely unsupervised.
   It cannot incorporate the known labels (success/failure) to learn
   discriminative boundaries. It finds statistical outliers in feature
   space, which may or may not correspond to actual failures.

3. **Outlier fraction assumption**: The default `outlier_fraction` of 0.05
   assumes 5% of documents are outliers. In our dataset, 20 out of 47
   labeled runs (42.5%) are failures. This mismatch means the algorithm
   expects far fewer anomalies than actually exist.

4. **Single-index limitation**: DFA operates on a single destination index.
   The Transform must be pre-configured with the right aggregations before
   running DFA. There is no iterative feature selection — the features
   are fixed at Transform creation time.

5. **No real-time scoring**: DFA is a one-time batch job. New pipeline runs
   cannot be scored incrementally without re-running the entire analysis.
   For continuous CI/CD monitoring, this would need to be re-triggered
   after each new run is ingested.

## Comparison with Other Tools

| Metric         | Baseline (keywords) | LogAI (IsolationForest) | Elastic DFA |
|----------------|--------------------|-----------------------|-------------|
| Precision      | 1.000              | 0.875                 | 1.000       |
| Recall         | 0.300              | 0.700                 | 0.100       |
| F1 Score       | 0.462              | 0.778                 | 0.182       |
| TP             | 6                  | 14                    | 2           |
| FP             | 0                  | 2                     | 0           |

| Failure Class    | Baseline | LogAI | Elastic DFA |
|------------------|----------|-------|-------------|
| backend_test     | partial  | 6/6   | 0/6         |
| frontend_build   | 0/6      | 0/6   | 0/6         |
| dependency       | 6/6      | 6/6   | 0/6         |
| deploy           | 0/2      | 2/2   | 2/2         |

## Effort Assessment

| Activity                          | Effort Level |
|-----------------------------------|-------------|
| Account setup (Elastic Cloud)     | Low — free trial, ~5 min |
| Data ingestion (script)           | Low — standard REST API, bulk upload |
| Anomaly Detection job setup       | Medium — required multiple iterations |
| Understanding why AD failed       | High — required deep research into tool limitations |
| DFA setup (Transform + outlier)   | Medium — API commands in Dev Tools |
| Total evaluation effort           | High — significant troubleshooting |

## Reproducibility

To reproduce the evaluation from scratch:

1. **Create Elastic Cloud account**: Sign up at cloud.elastic.co for a free
   trial. A serverless deployment is sufficient (ML features are included
   in the trial tier).

2. **Configure credentials**: Create an API key in Kibana (Stack Management >
   API Keys) and store in `.env`:
   ```
   ELASTIC_URL=https://your-deployment.es.cloud.es.io:443
   ELASTIC_API_KEY=your-api-key
   ```

3. **Ingest data**:
   ```bash
   pip install 'elasticsearch>=8.0.0'
   python scripts/ingest_elastic.py
   ```

4. **Create Transform** (in Kibana Dev Tools console): Run the Transform
   PUT command from Step 1 above, then start it:
   ```
   POST _transform/cicd-run-summary/_start
   ```

5. **Run DFA outlier detection** (in Kibana Dev Tools console): Run the
   DFA PUT command from Step 2 above, then start it:
   ```
   POST _ml/data_frame/analytics/cicd-run-outliers/_start
   ```

6. **View results**: Navigate to Machine Learning > Data Frame Analytics >
   `cicd-run-outliers` > Results Explorer in Kibana.

7. **Export scores**: Query the results index and save to CSV:
   ```
   GET cicd-run-outliers/_search?size=100
   ```

All API commands were executed through Kibana Dev Tools (Management >
Dev Tools) rather than external REST calls.

## Serverless Limitations Encountered

The Elastic Cloud serverless deployment had several limitations compared
to a self-managed or standard cloud deployment:

- `number_of_shards` / `number_of_replicas` index settings not available
  (ingestion script had to be modified to remove these from the mapping)
- `/_ml/anomaly_detectors/{id}/results/buckets` API not available
  (returned HTTP 410 Gone — could not programmatically extract bucket results)
- Some ML result APIs blocked (status 410) — results had to be viewed
  through the Kibana UI rather than exported via API
- Single-metric and multi-metric ML job wizards had limited detector
  options compared to documentation examples

## Key Findings

1. **Elastic ML Anomaly Detection is unsuitable for offline batch log
   analysis.** It requires continuous, time-distributed data streams to
   establish behavioral baselines. A 17-minute batch of historical logs
   does not provide sufficient temporal context.

2. **Elastic DFA outlier detection works for batch data** but is limited
   by the features available through Elasticsearch aggregations. It
   detected deploy failures (distinctive feature profile) but missed
   failure classes with subtle differences from normal runs.

3. **Feature engineering is the bottleneck.** Elastic's Transform provides
   basic aggregations (counts, cardinality) but cannot extract semantic
   features from log text. This limits outlier detection to structural
   anomalies (unusual line counts, file counts) rather than content-based
   anomalies.

4. **Integration effort is moderate to high.** While Elastic Cloud setup
   is straightforward, understanding which ML approach to use and why
   certain approaches fail requires significant domain knowledge.

## Files

| File | Description |
|------|-------------|
| `scripts/ingest_elastic.py` | Log ingestion script |
| `results/elastic_scores.csv` | DFA outlier scores per run |
| `results/metrics_elastic.csv` | Precision/recall/F1 metrics |
| `docs/screenshots/elastic/` | Kibana screenshots (see below) |

## Summary

Elastic Observability was evaluated using two distinct ML approaches:

**Anomaly Detection (time-series)** was tested first as Elastic's primary
ML capability. It failed to produce results because it requires continuous,
temporally distributed data to learn behavioral baselines. Our batch-uploaded
dataset from a narrow 17-minute window provided no temporal variation. This
is an architectural mismatch — the tool is designed for operational monitoring,
not offline log analysis. The categorization engine worked correctly (98
templates from 7,315 lines), confirming that the issue was in the anomaly
scoring layer, not in log parsing.

**Data Frame Analytics (batch outlier detection)** was identified through
targeted research as Elastic's batch-compatible alternative. It successfully
detected the 2 deploy failures (outlier scores 0.994–0.995) but could not
distinguish the remaining 18 failures from success runs (all scored 0.038).
The limitation is in feature expressiveness: Elasticsearch Transforms provide
only basic aggregations (counts, cardinality), and these aggregate features
do not capture the semantic differences between failure types and normal runs.

Compared to LogAI (F1=0.778), Elastic DFA (F1=0.182) performed significantly
worse on this dataset. The key differentiator is feature representation:
LogAI's Word2Vec vectorization captures semantic content of log messages,
while Elastic's Transform-based features are limited to structural counts.

## Screenshots

The following screenshots were captured from Kibana:

1. `01_dfa_results_explorer.png` — DFA Results Explorer showing outlier scores table
2. `02_dfa_scatterplot.png` — Scatterplot matrix (if visible)
3. `03_dfa_job_details.png` — DFA job configuration and stats
4. `04_anomaly_detection_job.png` — ML Anomaly Detection job showing "no anomalies"
5. `05_discover_logs.png` — Discover view of ingested log documents
6. `06_transform_details.png` — Transform stats (7,315 → 60 documents)
