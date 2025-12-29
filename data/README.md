# Data Directory

This directory contains ground truth labels for pipeline runs.

## Schema: runs.csv

| Column | Type | Description |
|--------|------|-------------|
| run_id | int | GitHub Actions workflow run ID |
| commit_sha | string | Full commit SHA that triggered the run |
| timestamp | datetime | ISO 8601 timestamp (UTC) of run start |
| outcome | string | `success` or `failure` |
| failure_class | string | One of: `none`, `backend_test`, `frontend_build`, `dependency`, `deploy`, `infra` |
| note | string | Optional free-text note about the failure or run |

## Failure Classes

- `none`: Run succeeded
- `backend_test`: pytest failure (injected or real)
- `frontend_build`: TypeScript/build error (injected or real)
- `dependency`: npm/pip install failure
- `deploy`: Azure deployment failure
- `infra`: GitHub Actions infrastructure issue

## Population

Runs are labeled manually after each experiment batch. Use the `scripts/label_runs.py` helper for CLI-based labeling.
