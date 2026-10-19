"""End to end: a fast training run, then the inference API on its artifacts."""

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from inbox.classify import export, predictor
from inbox.classify import model as M
from inbox.data import clinc

MSGS = [
    {"id": "1", "contact": "Anna", "text": "booked a table at Café Luna for Friday, 20:00", "sent_at": "2026-10-10T10:00:00"},
    {"id": "2", "contact": "Ben", "text": "dinner at the ramen bar on Saturday 19:30?", "sent_at": "2026-10-11T10:00:00"},
    {"id": "3", "contact": "Anna", "text": "they moved our booking to 21:00", "sent_at": "2026-10-12T10:00:00"},
    {"id": "4", "contact": "Anna", "text": "haha yes", "sent_at": "2026-10-12T11:00:00"},
]


@pytest.fixture(scope="module")
def client(trained):
    from inbox.serving.app import create_app

    return TestClient(create_app(trained / "models"))


def test_pipeline_report_and_registry(trained):
    report = json.loads((trained / "report.json").read_text())
    names = list(report["classifier"]["candidates"])
    assert names[:3] == ["tfidf_logreg", "ngram_fp32", "ngram_int8"]
    fp32, int8 = (report["classifier"]["candidates"][n]["metrics"] for n in ("ngram_fp32", "ngram_int8"))
    assert int8["size_mb"] < fp32["size_mb"] / 3  # int8 quantisation
    assert abs(int8["inscope_accuracy"] - fp32["inscope_accuracy"]) < 0.01  # without losing accuracy
    assert any("size" in reason for reason in report["classifier"]["candidates"]["ngram_fp32"]["reasons"])  # 67 MB fp32 is over budget
    r = report["retrieval"]
    assert r["candidates_plus_reranker"]["ndcg@10"] > r["hybrid"]["ndcg@10"]
    p = report["personalisation"]["models"]
    assert p["global_contextual"]["acceptance"] > p["popularity"]["acceptance"]
    assert p["federated_dp"]["raw_events_leave_device"] is False and p["federated_dp"]["epsilon"] < 20
    reg = json.loads((trained / "models" / "registry.json").read_text())
    assert all({"data_sha256", "gate_reasons", "status"} <= set(e) for e in reg)


def test_onnx_matches_pytorch(tmp_path):
    d = clinc.load()
    r = M.train(d["train"].texts[:1500], d["train"].labels[:1500], M.TrainConfig(epochs=1))
    export.to_onnx(r.model, tmp_path / "m.onnx")
    (tmp_path / "m.json").write_text(json.dumps({"labels": r.labels, "temperature": 1.0, "threshold": 0.0, "version": "t"}))
    texts = d["test"].texts[:50]
    onnx_logits = predictor.ActionClassifier(tmp_path / "m.onnx", tmp_path / "m.json").logits(texts)
    with torch.no_grad():
        torch_logits = M.logits(r.model, texts)
    assert np.allclose(onnx_logits, torch_logits, atol=1e-4)


def test_classify_and_out_of_scope(client):
    r = client.post("/v1/classify", json={"messages": ["remind me to buy milk tomorrow", "set an alarm for 6 am", "asdf qwerty zxcv"]})
    preds = r.json()["predictions"]
    assert preds[0]["action"] == "set_reminder" and preds[1]["action"] == "set_alarm"
    assert preds[2]["out_of_scope"] and preds[2]["action"] == "none"
    assert r.json()["model"]


def test_search_is_per_user_and_erasable(client):
    assert client.put("/v1/users/u1/messages", json={"messages": MSGS}).json()["indexed"] == 4
    res = client.post("/v1/users/u1/search", json={"query": "what time is dinner with Anna?", "k": 3}).json()
    assert {x["id"] for x in res["results"][:2]} == {"1", "3"}
    assert client.post("/v1/users/u2/search", json={"query": "dinner"}).status_code == 404
    assert client.delete("/v1/users/u1").json()["erased"] is True
    assert client.post("/v1/users/u1/search", json={"query": "dinner"}).status_code == 404


def test_suggestions_use_the_on_device_taste_without_storing_it(client):
    base = client.post("/v1/suggest", json={"message_type": "plan", "sender": "friend", "time_of_day": "evening"}).json()
    tasted = client.post(
        "/v1/suggest", json={"message_type": "plan", "sender": "friend", "time_of_day": "evening", "taste": [2.0, 0.5, -1.0, 0.0]}
    ).json()
    assert base["personalised"] is False and tasted["personalised"] is True
    assert base["suggestions"] != tasted["suggestions"]
    oos = client.post("/v1/suggest", json={"text": "asdf qwerty zxcv"}).json()
    assert oos["suggestions"] == []  # never act on a message it does not understand
    assert client.post("/v1/suggest", json={"message_type": "nonsense"}).status_code == 422


def test_metrics_and_drift_signals(client):
    client.post("/v1/classify", json={"messages": ["what's my balance"] * 256})
    text = client.get("/metrics").text
    assert "inbox_request_seconds_bucket" in text and "inbox_oos_rate_window" in text and "inbox_confidence_psi" in text
    assert client.post("/v1/feedback", json={"action": "add_event", "accepted": True}).json()["recorded"]
    assert 'inbox_feedback_total{accepted="true",action="add_event"}' in client.get("/metrics").text


def test_bearer_token_and_shadow_model(trained, monkeypatch):
    from inbox.serving.app import create_app

    monkeypatch.setenv("API_TOKEN", "s3cret")
    monkeypatch.setenv("SHADOW_MODEL", str(trained / "models" / "classifier" / "ngram_fp32.onnx"))
    c = TestClient(create_app(trained / "models"))
    assert c.post("/v1/classify", json={"messages": ["hi"]}).status_code == 401
    ok = c.post("/v1/classify", json={"messages": ["set a timer for 5 minutes"]}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and c.get("/healthz").json()["shadow"] == "ngram_fp32"
    assert "inbox_shadow_total" in c.get("/metrics").text
