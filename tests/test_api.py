def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_requires_file(client):
    r = client.post("/predict")
    assert r.status_code == 422
