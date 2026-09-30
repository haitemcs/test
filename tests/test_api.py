import os
os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient
from src.api import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
