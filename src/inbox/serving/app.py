"""Inference API. ONNX Runtime, LightGBM and numpy only — no PyTorch in the serving image.

Privacy by design:
* each user's messages live in their own in-memory index; a search can never see another user's
  data, and DELETE /v1/users/{id} erases it (GDPR right to erasure);
* the personal taste vector stays on the device and is sent per request; the server does not
  store it;
* feedback is aggregated per action without user identifiers.

Operations: Prometheus metrics with per-endpoint latency, model version on every response, an
optional shadow model scored on live traffic, and drift signals (out-of-scope rate and the
population stability index of classifier confidence) against the validation reference.
"""

# No `from __future__ import annotations`: FastAPI resolves dependency annotations at runtime.

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from .. import features
from ..classify.predictor import ActionClassifier
from ..config import models_dir
from ..retrieval import rerank
from ..retrieval.corpus import Message
from ..retrieval.index import UserIndex

# classifier action -> message type used by the suggestion model
ACTION_TO_MESSAGE_TYPE = {
    "add_event": "plan",
    "check_calendar": "plan",
    "set_reminder": "request",
    "add_task": "request",
    "check_tasks": "request",
    "set_alarm": "logistics",
    "share_location": "logistics",
    "start_call": "logistics",
    "send_message": "question",
    "pay_bill": "payment",
    "translate": "foreign_language",
    "answer": "question",
    "none": "chatter",
}


class ClassifyBody(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=256)


class MessageIn(BaseModel):
    id: str = Field(max_length=128)
    contact: str = Field(max_length=128)
    text: str = Field(max_length=4000)
    sent_at: datetime


class IndexBody(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=20_000)


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)


class SuggestBody(BaseModel):
    text: str | None = Field(default=None, max_length=4000)
    message_type: str | None = None
    sender: str = "friend"
    time_of_day: str = "daytime"
    weekend: bool = False
    taste: list[float] | None = Field(default=None, min_length=4, max_length=4, description="on-device personal vector; never stored")
    k: int = Field(default=3, ge=1, le=10)


class FeedbackBody(BaseModel):
    action: str
    accepted: bool


class Onnx:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            out.append(self.s.run(["embedding"], {"ids": features.padded(texts[i : i + 256])})[0])
        return np.concatenate(out) if out else np.zeros((0, 128), dtype=np.float32)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = (expected + 1) / (expected.sum() + len(expected))
    a = (actual + 1) / (actual.sum() + len(actual))
    return float(np.sum((a - e) * np.log(a / e)))


def create_app(models: Path | None = None) -> FastAPI:
    m = models or Path(os.environ.get("INBOX_MODELS", models_dir()))
    clf = ActionClassifier(m / "classifier" / "serving.onnx", m / "classifier" / "serving.json")
    meta = json.loads((m / "classifier" / "serving.json").read_text())
    shadow = None
    if os.environ.get("SHADOW_MODEL"):
        sp = Path(os.environ["SHADOW_MODEL"])
        shadow = ActionClassifier(sp, sp.with_suffix(".json"))
    encoder = Onnx(m / "retrieval" / "encoder.onnx")
    ranker = lgb.Booster(model_file=str(m / "retrieval" / "reranker.txt"))
    pers = json.loads((m / "personalisation" / "global.json").read_text())
    pb, pw, pv = np.array(pers["b"]), np.array(pers["w"]), np.array(pers["v"])
    token = os.environ.get("API_TOKEN")
    indexes: dict[str, UserIndex] = {}
    window: deque = deque(maxlen=2000)  # recent (out_of_scope, confidence) for drift signals

    reg = CollectorRegistry()
    latency = Histogram(
        "inbox_request_seconds", "Request latency", ["endpoint"], buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5), registry=reg
    )
    predictions = Counter("inbox_predictions_total", "Classifier predictions", ["action"], registry=reg)
    oos_rate = Gauge("inbox_oos_rate_window", "Out-of-scope share in the last 2000 messages", registry=reg)
    conf_psi = Gauge("inbox_confidence_psi", "PSI of classifier confidence vs validation reference", registry=reg)
    shadow_agree = Counter("inbox_shadow_total", "Shadow model comparisons", ["agree"], registry=reg)
    feedback = Counter("inbox_feedback_total", "Suggestion feedback", ["action", "accepted"], registry=reg)
    users = Gauge("inbox_indexed_users", "Users with an index in memory", registry=reg)

    app = FastAPI(title="Proactive Inbox ML", version="1.0.0")

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    def timed(name: str):
        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                latency.labels(name).observe(time.perf_counter() - self.t)

        return _T()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "classifier": clf.version, "gate_status": meta.get("gate_status"), "shadow": shadow.version if shadow else None}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.post("/v1/classify", dependencies=[Depends(auth)])
    def classify(body: ClassifyBody):
        with timed("classify"):
            preds = clf.predict(body.messages)
            for p in preds:
                predictions.labels(p["action"]).inc()
                window.append((p["out_of_scope"], p["confidence"]))
            if shadow:
                for a, b in zip(preds, shadow.predict(body.messages), strict=True):
                    shadow_agree.labels(str(a["intent"] == b["intent"]).lower()).inc()
            if len(window) >= 200:
                oos_rate.set(float(np.mean([w[0] for w in window])))
                conf = np.array([w[1] for w in window if not w[0]])
                conf_psi.set(psi(np.array(meta["reference"]["confidence_hist"], float), np.histogram(conf, bins=10, range=(0, 1))[0].astype(float)))
        return {"model": clf.version, "predictions": preds}

    @app.put("/v1/users/{user_id}/messages", dependencies=[Depends(auth)])
    def index_messages(user_id: str, body: IndexBody):
        with timed("index"):
            msgs = [Message(x.id, user_id, x.contact, x.text, x.sent_at) for x in body.messages]
            indexes[user_id] = UserIndex(msgs, encoder.embed)
            users.set(len(indexes))
        return {"user": user_id, "indexed": len(msgs)}

    @app.delete("/v1/users/{user_id}", dependencies=[Depends(auth)])
    def erase(user_id: str):
        existed = indexes.pop(user_id, None) is not None
        users.set(len(indexes))
        return {"user": user_id, "erased": existed}

    @app.post("/v1/users/{user_id}/search", dependencies=[Depends(auth)])
    def search(user_id: str, body: SearchBody):
        idx = indexes.get(user_id)
        if idx is None:
            raise HTTPException(404, "no index for this user; PUT /v1/users/{id}/messages first")
        with timed("search"):
            t0 = time.perf_counter()
            cands = idx.search(body.query, 50, mode="candidates")
            t1 = time.perf_counter()
            ranked = rerank.rerank(ranker, body.query, cands)[: body.k]
            t2 = time.perf_counter()
        return {
            "query": body.query,
            "results": [
                {"id": c.message.id, "contact": c.message.contact, "text": c.message.text, "sent_at": c.message.sent_at.isoformat()} for c in ranked
            ],
            "timing_ms": {"candidates": round((t1 - t0) * 1000, 2), "rerank": round((t2 - t1) * 1000, 2)},
        }

    @app.post("/v1/suggest", dependencies=[Depends(auth)])
    def suggest(body: SuggestBody):
        with timed("suggest"):
            intent = None
            mtype = body.message_type
            if body.text:
                intent = clf.predict([body.text])[0]
                if intent["out_of_scope"] and not mtype:
                    return {"suggestions": [], "reason": "message not understood with enough confidence; no proactive action", "intent": intent}
                mtype = mtype or ACTION_TO_MESSAGE_TYPE.get(intent["action"], "question")
            for value, allowed, field in [
                (mtype, pers["message_types"], "message_type"),
                (body.sender, pers["senders"], "sender"),
                (body.time_of_day, pers["times"], "time_of_day"),
            ]:
                if value not in allowed:
                    raise HTTPException(422, f"{field} must be one of {allowed}")
            x = np.zeros(pw.shape[1])
            x[pers["message_types"].index(mtype)] = 1
            o = len(pers["message_types"])
            x[o + pers["senders"].index(body.sender)] = 1
            x[o + len(pers["senders"]) + pers["times"].index(body.time_of_day)] = 1
            x[-1] = float(body.weekend)
            theta = np.array(body.taste) if body.taste else np.zeros(pv.shape[1])
            z = pb + pw @ x + pv @ theta
            p = 1 / (1 + np.exp(-z))
            order = np.argsort(-p)[: body.k]
        return {
            "message_type": mtype,
            "intent": intent,
            "personalised": body.taste is not None,
            "model": pers["model"],
            "epsilon": pers["epsilon"],
            "suggestions": [{"action": pers["actions"][i], "p_accept": round(float(p[i]), 4)} for i in order],
        }

    @app.post("/v1/feedback", dependencies=[Depends(auth)])
    def give_feedback(body: FeedbackBody):
        if body.action not in pers["actions"]:
            raise HTTPException(422, "unknown action")
        feedback.labels(body.action, str(body.accepted).lower()).inc()
        return {"recorded": True}

    return app
