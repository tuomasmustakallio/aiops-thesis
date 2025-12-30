#!/usr/bin/env bash
#
# Setup experiment branches with deterministic failures.
# Each branch has exactly ONE failure cause (single-cause principle).
#
# Usage:
#   ./scripts/setup_experiment_branches.sh          # Create all branches
#   ./scripts/setup_experiment_branches.sh success  # Create only success branch
#   ./scripts/setup_experiment_branches.sh backend-fail  # Create only backend-fail branch
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

ensure_clean_workdir() {
    if [[ -n $(git status --porcelain) ]]; then
        echo "ERROR: Working directory not clean. Commit or stash changes first."
        exit 1
    fi
}

create_branch() {
    local branch=$1
    local description=$2

    log "Creating branch: $branch"

    # Delete local branch if exists
    git branch -D "$branch" 2>/dev/null || true

    # Create from main
    git checkout -b "$branch" main

    log "  $description"
}

setup_success() {
    create_branch "experiment/success" "No failure - baseline success runs"
    git commit --allow-empty -m "experiment: baseline success branch"
}

setup_backend_fail() {
    create_branch "experiment/backend-fail" "Backend test failure"
    cp experiments/failure-toggles/backend-fail.py backend/tests/test_experiment_fail.py
    git add backend/tests/test_experiment_fail.py
    git commit -m "experiment: add failing backend test"
}

setup_frontend_fail() {
    create_branch "experiment/frontend-fail" "Frontend TypeScript error"
    cp experiments/failure-toggles/frontend-fail.ts frontend/src/experiment-fail.ts
    git add frontend/src/experiment-fail.ts
    git commit -m "experiment: add TypeScript error"
}

setup_dep_fail() {
    create_branch "experiment/dep-fail" "Dependency install failure"
    cat experiments/failure-toggles/dep-fail.txt >> backend/requirements.txt
    git add backend/requirements.txt
    git commit -m "experiment: add nonexistent dependency"
}

setup_deploy_fail() {
    create_branch "experiment/deploy-fail" "Docker build failure"
    cat experiments/failure-toggles/deploy-fail.dockerfile >> Dockerfile
    git add Dockerfile
    git commit -m "experiment: add Docker build failure"
}

# Main
ensure_clean_workdir

ORIGINAL_BRANCH=$(git branch --show-current)

case "${1:-all}" in
    success)
        setup_success
        ;;
    backend-fail)
        setup_backend_fail
        ;;
    frontend-fail)
        setup_frontend_fail
        ;;
    dep-fail)
        setup_dep_fail
        ;;
    deploy-fail)
        setup_deploy_fail
        ;;
    all)
        setup_success
        setup_backend_fail
        setup_frontend_fail
        setup_dep_fail
        setup_deploy_fail
        ;;
    *)
        echo "Usage: $0 {success|backend-fail|frontend-fail|dep-fail|deploy-fail|all}"
        exit 1
        ;;
esac

# Return to original branch
git checkout "$ORIGINAL_BRANCH"

log "Done. Branches created locally."
log ""
log "To push all experiment branches:"
log "  git push -u origin experiment/success experiment/backend-fail experiment/frontend-fail experiment/dep-fail experiment/deploy-fail"
