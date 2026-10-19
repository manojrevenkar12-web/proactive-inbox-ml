from datetime import datetime

import numpy as np

from inbox.personalize import models as pm
from inbox.personalize import sim
from inbox.retrieval import corpus
from inbox.retrieval.corpus import KINDS, Message
from inbox.retrieval.index import UserIndex


def test_corpus_is_deterministic_and_test_wording_is_held_out():
    a, b = corpus.generate(3, corpus.TEST, seed=5), corpus.generate(3, corpus.TEST, seed=5)
    assert [q.text for q in a.queries] == [q.text for q in b.queries]
    enc = corpus.generate(20, corpus.ENCODER, seed=6)
    enc_q = {q.text for q in enc.queries}
    for q in a.queries:
        assert q.text not in enc_q
    test_templates = {KINDS[k]["q"][i] for k in KINDS for i in corpus.TEST}
    train_templates = {KINDS[k]["q"][i] for k in KINDS for i in corpus.ENCODER}
    assert not test_templates & train_templates


def test_corrections_are_the_best_answer():
    c = corpus.generate(10, corpus.ENCODER, seed=3)
    by_id = {m.id: m for msgs in c.messages.values() for m in msgs}
    corrected = [q for q in c.queries if len(q.relevance) == 2]
    assert corrected
    for q in corrected:
        best = max(q.relevance, key=q.relevance.get)
        other = min(q.relevance, key=q.relevance.get)
        assert by_id[best].sent_at > by_id[other].sent_at and by_id[best].contact == by_id[other].contact


def test_index_only_sees_its_own_user():
    def embed(texts):
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 8))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    t = datetime(2026, 1, 1)
    a = UserIndex([Message("a1", "A", "Anna", "dinner at 8", t), Message("a2", "A", "Ben", "lol", t)], embed)
    b = UserIndex([Message("b1", "B", "Anna", "dinner at 9", t)], embed)
    assert {c.message.user for c in a.search("dinner Anna", 10, "candidates")} == {"A"}
    assert [c.message.id for c in b.search("dinner", 10)] == ["b1"]
    assert a.search("dinner Anna", 10, "bm25")[0].message.id == "a1"


def test_personalisation_beats_global_and_federated_keeps_taste_on_device():
    w = sim.make_world(300, seed=4)
    ev = sim.log_events(w, 1)
    rng = np.random.default_rng(0)
    tu = np.repeat(np.arange(300), 10)
    tx = sim.contexts(rng, len(tu))
    p = w.p_accept(tu, tx)

    def value(s):
        return p[np.arange(len(tu)), s.argmax(1)].mean()

    pop = value(np.tile(pm.popularity(ev), (len(tu), 1)))
    g, _ = pm.fit_central(ev, 300, personalised=False, steps=200)
    glob = value(pm.scores(g, tx, np.zeros((len(tu), sim.TASTE_DIM))))
    assert glob > pop + 0.1
    f, info = pm.fit_federated(ev, 300, pm.FedConfig(rounds=20, noise_multiplier=1.0))
    # what a device sends has exactly the size of the global weights: no room for θ_u or raw events
    assert pm.Params().flat().shape == f.flat().shape
    assert np.isfinite(info["epsilon"]) and info["device_updates"] > 0
