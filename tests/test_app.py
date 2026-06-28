# tests/test_app.py
import app as app_module


def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_ok():
    resp = client().get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
    assert resp.get_json()["model_loaded"] is True


def test_predict_valid_returns_price():
    resp = client().post("/predict", json={"OverallQual": 8, "GrLivArea": 2000})
    assert resp.status_code == 200
    price = resp.get_json()["predicted_price"]
    assert 50_000 < price < 1_000_000


def test_predict_unknown_key_is_400():
    resp = client().post("/predict", json={"NotARealColumn": 5})
    assert resp.status_code == 400


def test_predict_non_object_body_is_400():
    resp = client().post("/predict", json=[1, 2, 3])
    assert resp.status_code == 400


def test_metadata_returns_expected_keys():
    resp = client().get("/metadata")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) >= {"neighborhoods", "price_range", "field_ranges"}
    assert len(data["neighborhoods"]) > 0
    assert data["price_range"]["min"] < data["price_range"]["max"]
