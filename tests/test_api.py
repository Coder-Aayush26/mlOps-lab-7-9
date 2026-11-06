import pytest
from fastapi.testclient import TestClient

from src import api as api_module
from src.predict import load_model, predict as local_predict


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", artifacts["cfg"]["paths"]["model_path"])
    with TestClient(api_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_input(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0.5 <= body["probability"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_api_matches_local_prediction(client, valid_payload, artifacts):
    """Same input -> same output through the API and through predict() directly."""
    api_body = client.post("/predict", json=valid_payload).json()
    model = load_model(artifacts["cfg"]["paths"]["model_path"])
    local = local_predict([valid_payload], model=model, class_names=artifacts["cfg"]["data"]["class_names"])[0]
    assert api_body["prediction"] == local["prediction"]
    assert api_body["probability"] == pytest.approx(local["probability"], abs=1e-9)


def test_missing_field_returns_422(client, valid_payload):
    del valid_payload["mean_area"]
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422
    assert any("mean_area" in str(e["loc"]) for e in r.json()["detail"])


def test_wrong_type_returns_422(client, valid_payload):
    valid_payload["mean_radius"] = "large"
    assert client.post("/predict", json=valid_payload).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("mean_radius", -5.0), ("mean_area", 0), ("mean_smoothness", 1.5), ("mean_concavity", -0.1),
])
def test_invalid_business_values_return_422(client, valid_payload, field, value):
    valid_payload[field] = value
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_unknown_field_rejected(client, valid_payload):
    valid_payload["surprise"] = 1
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_batch_predict(client, valid_payload):
    r = client.post("/predict/batch", json={"records": [valid_payload] * 3})
    assert r.status_code == 200 and r.json()["count"] == 3
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_info_endpoint(client):
    body = client.get("/info").json()
    assert body["model_loaded"] is True and body["model_type"] == "random_forest"


def test_model_missing_gives_503_not_crash(monkeypatch, tmp_path, valid_payload):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(api_module.app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert c.post("/predict", json=valid_payload).status_code == 503


def test_internal_errors_do_not_leak_details(artifacts, monkeypatch, valid_payload):
    monkeypatch.setenv("MODEL_PATH", artifacts["cfg"]["paths"]["model_path"])

    def boom(*a, **k):
        raise RuntimeError("secret internal path /etc/passwd")

    monkeypatch.setattr(api_module, "score", boom)
    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        r = c.post("/predict", json=valid_payload)
        assert r.status_code == 500
        assert r.json() == {"detail": "Internal server error."}
        assert "secret" not in r.text
        assert c.get("/health").status_code == 200  # service still alive
