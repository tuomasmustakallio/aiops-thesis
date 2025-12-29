"""Tests for the FastAPI backend."""
import os

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    """Test that health endpoint returns healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "thesis-backend"


def test_health_check_response_time():
    """Test that health endpoint responds quickly."""
    import time
    start = time.time()
    response = client.get("/api/health")
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 1.0  # Should respond in under 1 second
