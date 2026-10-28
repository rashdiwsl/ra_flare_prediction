
import importlib.util
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient


PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "models" / "flaredown_candidate_model.pkl"
DATA_PATH = PROJECT / "data" / "ra_processed_v3_candidate.csv"

FEATURES = [
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
]


@pytest.fixture()
def research_api(monkeypatch, tmp_path):
    """Load the API using an isolated temporary database."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "research_test.db"))

    import api

    return api


@pytest.fixture()
def client(research_api):
    with TestClient(research_api.app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload():
    df = pd.read_csv(DATA_PATH)
    row = df.iloc[0]

    return {
        "past_mean_severity": float(row["past_mean_severity"]),
        "past_std_severity": float(row["past_std_severity"]),
        "past_trend": float(row["past_trend"]),
        "past_last_severity": float(row["past_last_severity"]),
        "past_engagement": float(row["past_engagement"]),
        "personal_baseline": float(row["personal_baseline"]),
        "age": float(row["age"]),
        "sex": str(row["sex"]),
    }


def test_research_endpoint_returns_labeled_prediction(
    client, valid_payload
):
    response = client.post(
        "/api/research/symptom-worsening-prediction",
        json=valid_payload,
    )

    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert result["research_only"] is True
    assert result["target"] == "dataset_defined_symptom_worsening"
    assert result["prediction"]["class"] in (0, 1)
    assert 0.0 <= result["prediction"]["positive_class_model_score"] <= 1.0
    assert any("not a confirmed clinical RA flare" in item
               for item in result["limitations"])


def test_research_endpoint_rejects_missing_feature(
    client, valid_payload
):
    valid_payload.pop("past_mean_severity")

    response = client.post(
        "/api/research/symptom-worsening-prediction",
        json=valid_payload,
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("past_std_severity", -1),
        ("past_engagement", -1),
        ("age", 0),
        ("age", 121),
        ("sex", ""),
    ],
)
def test_research_endpoint_rejects_invalid_values(
    client, valid_payload, field, value
):
    valid_payload[field] = value

    response = client.post(
        "/api/research/symptom-worsening-prediction",
        json=valid_payload,
    )

    assert response.status_code == 422


def test_research_endpoint_reports_model_unavailable(
    client, research_api, valid_payload, monkeypatch
):
    monkeypatch.setattr(
        research_api,
        "RESEARCH_MODEL_PATH",
        research_api.BASE_DIR / "models" / "missing_test_model.pkl",
    )

    response = client.post(
        "/api/research/symptom-worsening-prediction",
        json=valid_payload,
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
