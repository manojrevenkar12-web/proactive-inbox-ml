# proactive-inbox-ml

The machine-learning layer of an agent-first messaging app, built end to end and measured. When a message arrives, it:

1. **decides whether the message calls for an action** (set a reminder, add an event, add a task, pay a bill…) across 150 intents, and **declines to act** when it does not understand;
2. **finds the earlier messages the assistant needs as context** ("what time is dinner with Anna?" → the plan, and the later message that moved it);
3. **ranks which proactive suggestion this particular person will accept**, without their messages or taste profile leaving the phone;
4. and only **ships a model that passes a release gate** on quality, latency and size.

Everything trains on a CPU in about ten minutes (`inbox train`), and the inference service runs from an image with ONNX Runtime and no PyTorch.

```
 message ──▶ action classifier ──── out of scope? ──▶ do nothing
             hashed n-gram network       │ no
             ONNX int8, 0.1 ms           ▼
                              suggestion ranker ◀── on-device taste vector θ_u (never stored)
                              global model trained with federated averaging + differential privacy
                                         │
 question ─▶ per-user index: BM25 + dense encoder ─▶ top-50 ─▶ LightGBM LambdaRank ─▶ context for the agent

 training: data (SHA-256 pinned) → models → bootstrap metrics + slices → release gate → registry → serving bundle
 serving:  FastAPI · Prometheus latency · drift (out-of-scope rate, confidence PSI) · shadow model · per-user erasure
```

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel is enough
pip install -e ".[train,dev]"
inbox data                      # CLINC150 from a pinned commit, SHA-256 verified
inbox train                     # all models, gate, registry, reports/report.json (~10 min on 2 vCPU)
inbox train --with-transformer  # also the transformer challenger
inbox dp-scale                  # privacy cost vs number of users
inbox serve                     # API on :8000
pytest                          # 19 tests, including a fast end-to-end training run
docker build -t inbox-ml .      # inference image (after inbox train)
```

```bash
curl -X POST localhost:8000/v1/classify -H 'content-type: application/json' \
  -d '{"messages": ["can you remind me to pay rent on friday", "I like turtles"]}'
curl -X POST localhost:8000/v1/suggest -H 'content-type: application/json' \
  -d '{"text": "dinner at 8 on friday?", "sender": "friend", "time_of_day": "evening", "taste": [1.2, 0.3, -0.5, 0.0]}'
```

## Results

All numbers below are from one `inbox train --with-transformer` run on a 2-vCPU machine (fixed seeds; `artifacts/reports/report.json`).

### 1. Action classifier — CLINC150, real crowd-sourced queries

[CLINC150](https://github.com/clinc/oos-eval) (Larson et al., EMNLP 2019, CC BY 3.0): 150 intents across 10 domains plus out-of-scope queries. The test set has 4,500 in-scope and 1,000 out-of-scope queries. Out-of-scope matters for a proactive assistant: acting on a message it does not understand is worse than doing nothing. Following the paper's protocol, predicting "out of scope" for an in-scope query counts as an error.

| Model | In-scope accuracy (95% CI) | Out-of-scope recall (95% CI) | OOS precision | Macro-F1 | Size | Latency, 1 message, 1 core (p50 / p95) | Release gate |
|---|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 90.6% (89.8–91.5) | 73.7% (70.9–76.5) | 78.5% | 0.899 | 64 MB | 1.64 / 2.40 ms | baseline champion |
| Hashed n-gram network, fp32 | 90.8% (90.0–91.6) | 79.2% (76.6–81.5) | 79.3% | 0.906 | 67 MB | 0.10 / 0.18 ms | **rejected**: over the 50 MB size budget |
| **Hashed n-gram network, int8** | **90.8%** (89.9–91.6) | **79.2%** (76.5–81.6) | 79.0% | 0.906 | **17 MB** | **0.10 / 0.17 ms** | **champion**: out-of-scope recall +5.5 points (paired 95% CI +2.9…+8.1), no regression |
| 2-layer transformer from scratch | 86.4% (85.4–87.4) | 66.6% (63.6–69.3) | 69.4% | 0.859 | 35 MB | 0.27 / 0.39 ms | **rejected**: worse on both metrics and on 7 of 10 domains |

- **Abstention:** temperature scaling on the validation split, then the confidence threshold that maximises out-of-scope recall while giving up at most 2 points of in-scope accuracy. Test data is never used for tuning.
- **Quantisation:** int8 shrinks the model 4× with accuracy unchanged (−0.04 points). It does not make it faster here: per-message latency is dominated by featurisation, and dynamic quantisation is slightly slower at batch 64 (5.6 ms vs 4.2 ms per batch). The gain is memory and download size, which is what matters on a phone.
- **Why not the transformer:** with no pretrained weights available offline, a transformer trained from scratch on 15,000 sentences loses to n-grams. The gate rejected it with nine itemised reasons. A pretrained encoder fine-tuned on this data is the obvious next challenger, and the gate needs no changes for it.

**How much labelled data is enough?** Same model, same test set, with fewer training examples per intent:

| Examples per intent | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| In-scope accuracy | 77.5% | 85.1% | 88.3% | 90.8% |
| Out-of-scope recall | 67.0% | 73.2% | 79.7% | 79.2% |

The curve is still rising at 100 examples per intent, so more labels for the weakest domains (home 82.9%, kitchen and dining 86.7%) are worth more than a bigger model.

### 2. Context retrieval — hybrid search and a learned reranker

Real chat histories are private, and no public dataset pairs personal messages with the questions people later ask about them. So [`retrieval/corpus.py`](src/inbox/retrieval/corpus.py) generates seeded message histories that are hard in the ways real ones are:
- **lexical gap:** the question rarely uses the message's words;
- **look-alikes:** the same kind of fact from a different contact;
- **corrections:** "they moved our booking to 21:00" is the best answer, and the original plan only partly relevant;
- **noise:** about 85% small talk.

Message and question phrasings are split three ways: encoder training, reranker training, and test only. Users are disjoint across splits, so the test measures new people **and** new wording.

| First stage → reranking (840 test questions, 60 unseen users, unseen phrasing) | nDCG@10 | MRR@10 | Recall@1 | Recall@10 |
|---|---|---|---|---|
| BM25 | 0.203 | 0.191 | 0.155 | 0.267 |
| Dense encoder (trained here, contrastive, hard negatives) | 0.299 | 0.277 | 0.242 | 0.377 |
| Hybrid, reciprocal rank fusion | 0.258 | 0.245 | 0.223 | 0.308 |
| **Candidates (dense top-50 ∪ BM25 top-25) + LightGBM LambdaRank** | **0.720** | **0.618** | **0.432** | **0.982** |

- **Latency:** ~0.9 ms for the first stage and ~1.2 ms to rerank (p50) for a user with ~250 messages.
- **Isolation:** each user has their own index, so a query can never reach another user's messages.
- **Reranker signals:** BM25 and dense scores and ranks, whether the question names the sender, recency among candidates, and whether the message follows a strong candidate from the same contact (corrections).

**The honest part.** On new users written with the phrasing it was trained on, the same dense encoder reaches nDCG@10 0.936; on unseen phrasing it drops to 0.299. Trained from scratch, it learned the wording, not the meaning. Most of the test performance comes from the reranker combining weak signals, and the most important reranker feature is message length, which in this corpus separates facts from small talk. Both point the same way: swap in a pretrained multilingual sentence encoder (the index takes any ONNX encoder; Hugging Face downloads were blocked in this build environment) and re-check the features on real data.

### 3. Personalised suggestions — with the data left on the phone

When a message arrives the app offers one action. Whether a person accepts depends on the message, the context (sender, time of day) and the person: some live in their calendar, some never want a reminder. Real acceptance logs are private, so [`personalize/sim.py`](src/inbox/personalize/sim.py) simulates users with known acceptance probabilities. That way every policy is scored by the acceptance rate it would actually get, not only by offline AUC. The simulation has 2,000 users and 93,789 logged interactions under random exploration, median 28 per user (many cold-start users). The best possible policy (the oracle) gets an acceptance rate of 56.3%.

| Model | Acceptance of the top suggestion | % of oracle | Cold users (<15 events) | Heavy users (60+) | Offline AUC | Raw events leave the phone? |
|---|---|---|---|---|---|---|
| Most popular action | 26.7% | 47.5% | 27.2% | 26.3% | — | yes |
| Global contextual model | 51.1% | 90.8% | 50.6% | 51.4% | 0.693 | yes |
| Central personalised (per-user taste vectors on the server) | **52.8%** | **93.8%** | 51.4% | 53.7% | 0.736 | **yes** |
| Federated (taste vector on device, only clipped global updates sent) | 51.5% | 91.5% | 50.3% | 53.0% | 0.701 | no |
| Federated + differential privacy (ε = 6.2, δ = 2.3·10⁻⁴) | 50.7% | 90.0% | 49.3% | 52.1% | 0.703 | no |

How it works:
- **Federated training:** each round, about 5% of devices train on their own events and send a clipped update of the global weights (DP-FedAvg). The server adds Gaussian noise to the sum.
- **Privacy guarantee:** user-level (ε, δ), computed with a Rényi-DP accountant for the sampled Gaussian mechanism (`epsilon()`, unit-tested against the closed form).
- **The taste vector θ_u** is fitted and kept on the device. The API takes it per request and never stores it.

**The honest part, and what scale changes** (`inbox dp-scale`, same simulator):

| Users | Global model | Federated | Federated + DP | ε |
|---|---|---|---|---|
| 2,000 | 90.7% | 91.5% | 90.0% | 6.2 |
| 10,000 | 90.8% | 92.0% | **92.1%** | 6.8 |

With 2,000 users the privacy noise cancels the personalisation gain. With 10,000 users the same noise is spread over five times as many updates per round, and private federated personalisation matches the non-private federated model and beats the global one. Central training on uploaded logs (93.8% at 2,000 users) remains the upper bound. At app scale (millions of users), the privacy cost of this design should be small.

### 4. Release gate and registry

[`evaluation/gate.py`](src/inbox/evaluation/gate.py) compares a challenger with the current champion on the same test items:
- **Paired bootstrap per key metric:** reject if a metric is significantly worse beyond tolerance.
- **Slices:** reject if any domain is significantly worse by more than 3 points.
- **Budgets:** hard limits on p95 latency and model size.
- **Promotion:** a challenger is promoted only if it is significantly better somewhere, or equal and meaningfully cheaper.

Every decision is stored with its reasons in `artifacts/models/registry.json`, with the data SHA-256 and git commit. The records map one-to-one onto an MLflow registry. In the run above the gate:
- kept the TF-IDF baseline as the first champion;
- **rejected** the fp32 network for size;
- **promoted** the int8 network on out-of-scope recall;
- **rejected** the transformer.

### 5. Serving

`inbox serve` (FastAPI) loads only ONNX models, the LightGBM reranker and the federated model's global weights:

| Endpoint | What it does |
|---|---|
| `POST /v1/classify` | intents, actions, calibrated confidence, out-of-scope flag, model version |
| `PUT /v1/users/{id}/messages`, `POST /v1/users/{id}/search`, `DELETE /v1/users/{id}` | per-user index, reranked search, erasure (GDPR) |
| `POST /v1/suggest` | ranked proactive actions. Uses the classifier if text is given; returns nothing for out-of-scope messages; personalised if the device sends its taste vector |
| `POST /v1/feedback` | accept/dismiss counts per action, no user identifier |
| `GET /metrics` | latency histograms per endpoint, predictions per action, out-of-scope rate and confidence PSI against the validation reference (drift), shadow-model agreement (`SHADOW_MODEL=…`), feedback |

Load test (`inbox loadtest`, 16 concurrent clients on the same 2-vCPU machine as a single uvicorn worker): **479 requests/s, p50 17 ms, p95 99 ms**. The model is 0.1 ms of that; the rest is HTTP, JSON and the load generator competing for the same two cores. Scaling is horizontal: the image is stateless apart from the opt-in per-user indexes.

## Design decisions

- **Small models first, measured against a baseline.** A 17 MB classifier that answers in 0.1 ms and abstains when unsure beats a larger model that is not better, and the gate makes that trade-off explicit.
- **Thresholds and temperature are tuned on validation only, with an explicit budget** ("give up at most 2 points of accuracy to catch out-of-scope messages"), so the operating point is a product decision written in code.
- **Same features in training and serving.** Hashing needs no vocabulary file, and the identical numpy featuriser feeds PyTorch in training and ONNX Runtime in serving (a test checks the ONNX and PyTorch logits match).
- **Privacy by architecture, not policy:** per-user indexes, taste vectors that stay on the device, aggregate-only feedback, and a formal DP guarantee for the shared model.

## Limits

- Retrieval and personalisation are evaluated on simulated data with known ground truth (the reason is above). The classifier uses real data. None of it has served real users.
- No pretrained encoders were available in the build environment, and both the transformer and the dense encoder show it. That is the first upgrade.
- The federated training is simulated in one process. The algorithm, clipping, noise and accounting are real, but secure aggregation and device scheduling are not implemented.
- JAX is not used; everything is PyTorch, numpy and LightGBM.

## Layout

```
src/inbox/
  data/clinc.py          pinned download, SHA-256 check, intent → app action map
  features.py            hashed n-gram and word-sequence featurisers (shared by training and serving)
  classify/              PyTorch models, calibration and abstention, ONNX export + int8, ONNX predictor, TF-IDF baseline
  retrieval/             corpus generator, dual encoder, per-user hybrid index, LambdaRank reranker
  personalize/           user simulator, global / central / federated + DP models, RDP accountant
  evaluation/            classification and ranking metrics, bootstrap CIs, release gate, registry
  serving/               FastAPI app, load test
  pipeline.py            train → evaluate → gate → register → serving bundle
```

MIT licence. Built by Manoj Kumar Prakash.
