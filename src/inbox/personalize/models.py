"""Suggestion-ranking models, from a popularity baseline to federated learning with differential privacy.

All models share one form:

    score(u, x, a) = b_a + w_a · x + θ_u · v_a

b, w, v are global (learned from everyone); θ_u is a small per-user taste vector. The privacy
question is where θ_u and the raw events live:

* central personalised — all raw interaction logs are uploaded and θ_u is learned on the server;
* federated personalised — raw events never leave the phone. Each round, sampled devices train
  on their own events and send only a clipped update of the global weights; θ_u is trained and
  kept on the device. With Gaussian noise on the aggregate (DP-FedAvg) the global model has a
  user-level (ε, δ) differential-privacy guarantee, computed below with an RDP accountant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .sim import ACTIONS, CONTEXT_DIM, TASTE_DIM, Events

A = len(ACTIONS)


@dataclass
class Params:
    b: np.ndarray = field(default_factory=lambda: np.zeros(A))
    w: np.ndarray = field(default_factory=lambda: np.zeros((A, CONTEXT_DIM)))
    v: np.ndarray = field(default_factory=lambda: np.random.default_rng(0).normal(0, 0.1, (A, TASTE_DIM)))

    def flat(self) -> np.ndarray:
        return np.concatenate([self.b, self.w.ravel(), self.v.ravel()])

    @classmethod
    def from_flat(cls, f: np.ndarray) -> Params:
        b = f[:A]
        w = f[A : A + A * CONTEXT_DIM].reshape(A, CONTEXT_DIM)
        v = f[A + A * CONTEXT_DIM :].reshape(A, TASTE_DIM)
        return cls(b.copy(), w.copy(), v.copy())


def scores(p: Params, ctx: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """[N, A] logits for every action. theta: [N, TASTE_DIM] (zeros = no personalisation)."""
    return p.b[None, :] + ctx @ p.w.T + theta @ p.v.T


def _grads(p: Params, theta: np.ndarray, ev: Events, users_local: np.ndarray, l2: float, l2_theta: float):
    """Gradients of mean BCE on the shown actions. users_local indexes rows of theta."""
    a = ev.action
    z = p.b[a] + np.einsum("nd,nd->n", ev.ctx, p.w[a]) + np.einsum("nk,nk->n", theta[users_local], p.v[a])
    err = (1 / (1 + np.exp(-np.clip(z, -30, 30))) - ev.accepted) / len(a)
    gb = np.bincount(a, weights=err, minlength=A)
    gw = np.zeros_like(p.w)
    np.add.at(gw, a, err[:, None] * ev.ctx)
    gv = np.zeros_like(p.v)
    np.add.at(gv, a, err[:, None] * theta[users_local])
    gt = np.zeros_like(theta)
    np.add.at(gt, users_local, err[:, None] * p.v[a])
    # θ prior: λ_θ pseudo-observations' worth of shrinkage per user (per-event mean loss scale)
    return gb, gw + l2 * p.w, gv + l2 * p.v, gt + l2_theta * theta / len(a)


class _Adam:
    def __init__(self, lr: float):
        self.lr, self.m, self.v, self.t = lr, {}, {}, 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            m = self.m.setdefault(k, np.zeros_like(g))
            v = self.v.setdefault(k, np.zeros_like(g))
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            params[k] -= self.lr * (m / (1 - 0.9**self.t)) / (np.sqrt(v / (1 - 0.999**self.t)) + 1e-8)


def popularity(ev: Events) -> np.ndarray:
    """Acceptance rate per action (with a prior), ignoring context and user."""
    acc = np.bincount(ev.action, weights=ev.accepted, minlength=A)
    shown = np.bincount(ev.action, minlength=A)
    return (acc + 1) / (shown + 10)


def fit_central(
    ev: Events, n_users: int, personalised: bool, steps: int = 600, lr: float = 0.05, l2: float = 3e-3, l2_theta: float = 6.0
) -> tuple[Params, np.ndarray]:
    """Server-side training on all raw logs. With personalised=True, alternates an Adam step on the
    global weights with a per-user gradient step on θ_u scaled by that user's event count, so a
    user with three events cannot overfit their taste vector."""
    p = Params()
    l2 = l2 if personalised else 1e-4  # without θ there is nothing for the extra shrinkage on v to protect
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 0.05, (n_users, TASTE_DIM)) if personalised else np.zeros((n_users, TASTE_DIM))
    counts = np.maximum(np.bincount(ev.user, minlength=n_users), 1)[:, None]
    opt = _Adam(lr)
    for _ in range(steps):
        gb, gw, gv, gt = _grads(p, theta, ev, ev.user, l2, 0.0)
        opt.step({"b": p.b, "w": p.w, "v": p.v}, {"b": gb, "w": gw, "v": gv})
        if personalised:
            theta -= (gt * len(ev.action) + l2_theta * theta) / _curvature(p, counts, l2_theta)
    return p, theta


def personalise_on_device(p: Params, ev: Events, n_users: int, steps: int = 150, lr: float = 1.0, l2_theta: float = 6.0) -> np.ndarray:
    """Each device fits its own θ_u against the frozen global model, using only its own events.

    Computed here for all simulated users at once, but each user's update only ever touches that
    user's events — it is exactly what every phone would compute independently."""
    theta = np.zeros((n_users, TASTE_DIM))
    counts = np.maximum(np.bincount(ev.user, minlength=n_users), 1)[:, None]
    a = ev.action
    base = p.b[a] + np.einsum("nd,nd->n", ev.ctx, p.w[a])
    for _ in range(steps):
        z = base + np.einsum("nk,nk->n", theta[ev.user], p.v[a])
        err = 1 / (1 + np.exp(-np.clip(z, -30, 30))) - ev.accepted
        g = np.zeros_like(theta)
        np.add.at(g, ev.user, err[:, None] * p.v[a])
        theta -= lr * (g + l2_theta * theta) / _curvature(p, counts, l2_theta)
    return theta


def _curvature(p: Params, counts: np.ndarray, l2_theta: float) -> np.ndarray:
    """Upper bound on the per-user Hessian (logistic loss curvature ≤ 1/4): a safe Newton-like step size."""
    return 0.25 * counts * float((p.v**2).sum(1).max()) + l2_theta


@dataclass
class FedConfig:
    rounds: int = 300
    sample_rate: float = 0.05  # Poisson sampling of devices per round
    local_steps: int = 10
    local_lr: float = 0.5
    server_lr: float = 1.0
    clip: float = 1.0  # L2 norm bound of each device's update (user-level sensitivity)
    noise_multiplier: float = 0.0  # σ; 0 = federated without DP
    seed: int = 21


def fit_federated(ev: Events, n_users: int, cfg: FedConfig) -> tuple[Params, dict]:
    rng = np.random.default_rng(cfg.seed)
    g = Params()
    order = np.argsort(ev.user, kind="stable")
    bounds = np.searchsorted(ev.user[order], np.arange(n_users + 1))
    expected = cfg.sample_rate * n_users
    clipped = 0
    sent = 0
    device_theta = np.zeros((n_users, TASTE_DIM))  # lives on each phone; never sent
    for _ in range(cfg.rounds):
        chosen = np.flatnonzero(rng.random(n_users) < cfg.sample_rate)
        total = np.zeros_like(g.flat())
        for u in chosen:
            idx = order[bounds[u] : bounds[u + 1]]
            if len(idx) == 0:
                continue
            sub = Events(np.zeros(len(idx), dtype=int), ev.ctx[idx], ev.action[idx], ev.accepted[idx])
            local = Params.from_flat(g.flat())
            theta = device_theta[u : u + 1].copy()
            if not theta.any():
                theta = rng.normal(0, 0.05, theta.shape)
            for _ in range(cfg.local_steps):
                gb, gw, gv, gt = _grads(local, theta, sub, sub.user, 3e-3, 6.0)
                local.b -= cfg.local_lr * gb
                local.w -= cfg.local_lr * gw
                local.v -= cfg.local_lr * gv
                theta -= (gt * len(idx) + 6.0 * theta) / _curvature(local, np.array([[len(idx)]]), 6.0)
            device_theta[u] = theta[0]
            delta = local.flat() - g.flat()
            norm = np.linalg.norm(delta)
            if norm > cfg.clip:
                delta *= cfg.clip / norm
                clipped += 1
            sent += 1
            total += delta
        noise = rng.normal(0, cfg.noise_multiplier * cfg.clip, total.shape) if cfg.noise_multiplier > 0 else 0.0
        g = Params.from_flat(g.flat() + cfg.server_lr * (total + noise) / expected)
    eps = epsilon(cfg.noise_multiplier, cfg.sample_rate, cfg.rounds, delta=1 / n_users**1.1) if cfg.noise_multiplier > 0 else math.inf
    return g, {"device_updates": sent, "clipped_share": round(clipped / max(1, sent), 3), "epsilon": eps, "delta": 1 / n_users**1.1}


# --------------------------------------------------------------------------- privacy accounting


def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log A_α for the sampled Gaussian mechanism, integer α (Mironov, Talwar & Zhang 2019)."""
    if q >= 1.0:  # no sampling: plain Gaussian mechanism
        return (alpha * alpha - alpha) / (2 * sigma**2)
    terms = [
        math.lgamma(alpha + 1)
        - math.lgamma(k + 1)
        - math.lgamma(alpha - k + 1)
        + k * math.log(q)
        + (alpha - k) * math.log1p(-q)
        + (k * k - k) / (2 * sigma**2)
        for k in range(alpha + 1)
    ]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def epsilon(sigma: float, q: float, steps: int, delta: float) -> float:
    """User-level ε after `steps` rounds of Poisson-sampled Gaussian aggregation (RDP → (ε, δ))."""
    best = math.inf
    for alpha in range(2, 257):
        rdp = steps * _log_a_int(q, sigma, alpha) / (alpha - 1)
        best = min(best, rdp + math.log(1 / delta) / (alpha - 1))
    return round(best, 2)


# --------------------------------------------------------------------------- evaluation


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
