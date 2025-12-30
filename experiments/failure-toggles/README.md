# Failure Toggle Files

These files contain the deterministic failures used in experiment branches.

## Usage

Each experiment branch applies ONE failure by copying/appending the relevant file:

| Branch | Failure File | Target | Effect |
|--------|--------------|--------|--------|
| `experiment/backend-fail` | `backend-fail.py` | `backend/tests/test_experiment_fail.py` | pytest fails |
| `experiment/frontend-fail` | `frontend-fail.ts` | `frontend/src/experiment-fail.ts` | TypeScript error |
| `experiment/dep-fail` | `dep-fail.txt` | Append to `backend/requirements.txt` | pip install fails |
| `experiment/deploy-fail` | `deploy-fail.dockerfile` | Append to `Dockerfile` | Docker build fails |

## Single-Cause Principle

Each branch should have exactly ONE failure cause. This ensures clean, classifiable failures for the experiment dataset.

## Branch Generation

Use `scripts/setup_experiment_branches.sh` to create all branches from main with their respective failures applied.
