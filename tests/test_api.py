"""Basic API smoke test.

This checks that the health route responds. It does not test predictions or prove
that a real PostgreSQL server can be reached.
"""

import os

# Set this before importing src.api: src.db creates its engine during import.
# SQLite keeps this small test independent of a local PostgreSQL installation.
os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402


def test_health():
    """The health endpoint should return HTTP 200 and a status value."""
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
