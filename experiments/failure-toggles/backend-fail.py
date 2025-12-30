"""
EXPERIMENT FAILURE FILE

Copy this file to backend/tests/test_experiment_fail.py to cause a test failure.
This is used on the experiment/backend-fail branch.
"""
import pytest


def test_experiment_backend_failure():
    """Intentional test failure for experiment dataset collection."""
    assert False, "EXPERIMENT: Intentional backend test failure for dataset collection"
