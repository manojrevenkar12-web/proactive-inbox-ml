"""Simulated users reacting to proactive suggestions.

When a message arrives, the app can offer one action (add it to the calendar, set a reminder,
reply with a quick answer, …). Whether a person accepts depends on the message, the context and
on the person: some people live in their calendar, some never want a reminder. The simulator
encodes this as

    P(accept | user u, context x, action a) = sigmoid(b_a + w_a · x + θ_u · v_a)

with θ_u drawn from a few personas plus individual noise, and a heavy-tailed number of events
per user (many people have only a handful of interactions — the cold-start problem). Because
the true acceptance probabilities are known, every policy can be scored by the acceptance rate
it would really achieve, not only by offline AUC on logged clicks.

The logged training data comes from a uniform-random exploration policy, so it is unbiased for
every action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = [
    "add_event",
    "set_reminder",
    "add_task",
    "quick_reply",
    "share_location",
    "pay_bill",
    "summarise_thread",
    "snooze",
    "translate",
    "set_alarm",
]
MESSAGE_TYPES = ["plan", "request", "question", "logistics", "payment", "chatter", "long_thread", "foreign_language"]
SENDERS = ["family", "friend", "work", "service"]
TIMES = ["morning", "daytime", "evening", "night"]
CONTEXT_DIM = len(MESSAGE_TYPES) + len(SENDERS) + len(TIMES) + 1  # + weekend flag
TASTE_DIM = 4

# Which actions make sense for which message type (others are possible but rarely accepted).
_AFFINITY = {
    "plan": {"add_event": 2.2, "set_reminder": 1.0, "quick_reply": 0.8},
    "request": {"add_task": 2.0, "set_reminder": 1.4, "quick_reply": 0.6},
    "question": {"quick_reply": 2.0, "snooze": 0.6},
    "logistics": {"share_location": 1.8, "quick_reply": 1.0, "set_alarm": 0.8},
    "payment": {"pay_bill": 2.4, "set_reminder": 0.8},
    "chatter": {"quick_reply": 0.9, "snooze": 0.4},
    "long_thread": {"summarise_thread": 2.2, "snooze": 0.8},
    "foreign_language": {"translate": 2.6},
}
PERSONAS = {  # taste vectors; v_a below maps actions into the same space
    "planner": [1.6, 0.2, -0.3, 0.0],
    "minimalist": [-1.4, -1.0, 0.0, 0.3],
    "work_focused": [0.4, 1.4, -0.6, 0.0],
    "social": [-0.2, -0.4, 1.5, 0.4],
}


@dataclass
class World:
    b: np.ndarray  # [A]
    w: np.ndarray  # [A, CONTEXT_DIM]
    v: np.ndarray  # [A, TASTE_DIM]
    theta: np.ndarray  # [U, TASTE_DIM]
    persona: list[str]
    n_events: np.ndarray  # [U]

    def p_accept(self, users: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """[N, A] true acceptance probabilities."""
        z = self.b[None, :] + ctx @ self.w.T + self.theta[users] @ self.v.T
        return 1 / (1 + np.exp(-z))


@dataclass
class Events:
    user: np.ndarray  # [N]
    ctx: np.ndarray  # [N, CONTEXT_DIM]
    action: np.ndarray  # [N] shown action
    accepted: np.ndarray  # [N] 0/1


def make_world(n_users: int = 2000, seed: int = 11) -> World:
    rng = np.random.default_rng(seed)
    a_n = len(ACTIONS)
    b = np.full(a_n, -2.2)
    w = rng.normal(0, 0.15, (a_n, CONTEXT_DIM))
    for mt, aff in _AFFINITY.items():
        for a, s in aff.items():
            w[ACTIONS.index(a), MESSAGE_TYPES.index(mt)] += s
    # sender and time effects: reminders/alarms at night, work messages -> tasks, family -> quick replies
    o = len(MESSAGE_TYPES)
    w[ACTIONS.index("add_task"), o + SENDERS.index("work")] += 0.9
    w[ACTIONS.index("quick_reply"), o + SENDERS.index("family")] += 0.7
    w[ACTIONS.index("pay_bill"), o + SENDERS.index("service")] += 0.8
    w[ACTIONS.index("set_alarm"), o + len(SENDERS) + TIMES.index("night")] += 1.0
    w[ACTIONS.index("snooze"), o + len(SENDERS) + TIMES.index("night")] += 0.7
    v = rng.normal(0, 0.25, (a_n, TASTE_DIM))
    v[ACTIONS.index("add_event")] += [1.0, 0.3, 0, 0]
    v[ACTIONS.index("set_reminder")] += [0.9, 0.6, 0, 0]
    v[ACTIONS.index("add_task")] += [0.4, 1.0, 0, 0]
    v[ACTIONS.index("quick_reply")] += [0, 0, 1.0, 0]
    v[ACTIONS.index("share_location")] += [0, 0, 0.8, 0.2]
    v[ACTIONS.index("summarise_thread")] += [0, 0.8, -0.3, 0]
    names = list(PERSONAS)
    persona = [names[i] for i in rng.integers(len(names), size=n_users)]
    theta = np.array([PERSONAS[p] for p in persona]) + rng.normal(0, 0.35, (n_users, TASTE_DIM))
    n_events = np.clip(rng.lognormal(3.4, 1.0, n_users).astype(int), 3, 600)
    return World(b, w, v, theta, persona, n_events)


def contexts(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.zeros((n, CONTEXT_DIM))
    x[np.arange(n), rng.integers(len(MESSAGE_TYPES), size=n)] = 1
    o = len(MESSAGE_TYPES)
    x[np.arange(n), o + rng.choice(len(SENDERS), size=n, p=[0.3, 0.35, 0.25, 0.1])] = 1
    x[np.arange(n), o + len(SENDERS) + rng.choice(len(TIMES), size=n, p=[0.25, 0.4, 0.28, 0.07])] = 1
    x[:, -1] = rng.random(n) < 2 / 7
    return x


def log_events(world: World, seed: int, per_user: np.ndarray | None = None) -> Events:
    """Interactions logged under uniform-random exploration."""
    rng = np.random.default_rng(seed)
    counts = world.n_events if per_user is None else per_user
    users = np.repeat(np.arange(len(counts)), counts)
    ctx = contexts(rng, len(users))
    actions = rng.integers(len(ACTIONS), size=len(users))
    p = world.p_accept(users, ctx)[np.arange(len(users)), actions]
    return Events(users, ctx, actions, (rng.random(len(users)) < p).astype(np.float64))
