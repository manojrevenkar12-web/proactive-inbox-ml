"""A seeded generator of personal message histories with graded relevance judgements.

Why synthetic: real chat histories are private, and no public dataset pairs personal messages
with the questions people later ask an assistant about them. The generator is built to make
retrieval hard in the ways real inboxes are:

* lexical gap — the question rarely uses the message's words ("when does my plane leave?" vs
  "LH 2045 boards 7:40 at B12");
* look-alike distractors — the same kind of fact with a different person or day;
* corrections — a later "actually, make it 8 instead" is the most relevant message, the original
  plan is only partly relevant;
* noise — most messages are small talk.

Message and question phrasings are split three ways — encoder training, reranker training and
test — and users are disjoint across splits, so the test measures generalisation to unseen
wording and unseen people, and the reranker learns how to combine signals on wording the encoder
has never seen (which is the situation it faces in production).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

NAMES = [
    "Anna",
    "Ben",
    "Clara",
    "David",
    "Elif",
    "Felix",
    "Greta",
    "Hamid",
    "Ines",
    "Jonas",
    "Kira",
    "Lukas",
    "Mia",
    "Noah",
    "Olga",
    "Paul",
    "Rosa",
    "Sami",
    "Tara",
    "Umut",
    "Vera",
    "Wen",
    "Yara",
    "Zoe",
    "Arjun",
    "Priya",
    "Lena",
    "Marco",
    "Nadia",
    "Omar",
]
PLACES = [
    "Café Luna",
    "the Thai place on Oranienstraße",
    "Mauerpark",
    "Markthalle Neun",
    "the pizzeria near the station",
    "Tempelhofer Feld",
    "Zur Letzten Instanz",
    "the ramen bar in Mitte",
    "Klunkerkranich",
    "Brlo Brwhouse",
    "the climbing gym",
    "Café Einstein",
]
STREETS = [
    "Kastanienallee",
    "Sonnenallee",
    "Bergmannstraße",
    "Torstraße",
    "Schönhauser Allee",
    "Karl-Marx-Allee",
    "Warschauer Straße",
    "Boxhagener Straße",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AIRLINES = ["LH", "EW", "FR", "U2", "KL", "OS"]
DOCTORS = ["dentist", "GP", "dermatologist", "eye doctor", "physio"]
ITEMS = ["oat milk", "eggs", "basil", "coffee beans", "dish soap", "bread", "tomatoes", "batteries", "rice", "lemons"]
EVENTS = ["birthday", "housewarming", "farewell drinks", "graduation party"]
SMALL_TALK = [
    "haha yes",
    "on my way",
    "see you soon",
    "thanks!",
    "ok 👍",
    "sounds good",
    "lol",
    "how was your weekend?",
    "miss you",
    "can't wait",
    "running 5 min late sorry",
    "did you watch the match?",
    "good night",
    "sure thing",
    "that's hilarious",
    "I'll check and get back to you",
    "happy friday!",
    "the weather is crazy today",
    "coffee later?",
    "ugh mondays",
    "nice photo!",
    "call you later",
    "congrats!!",
    "no worries",
]


def _time(rng: random.Random) -> str:
    return f"{rng.randint(7, 21)}:{rng.choice(['00', '15', '30', '45'])}"


# Each fact kind: slot sampler, message templates, correction templates, question templates.
# Templates are split by index: 0 and 2 train the encoder, 1 trains the reranker, 3 is test only.
KINDS: dict[str, dict] = {
    "dinner": {
        "slots": lambda r, c: {"place": r.choice(PLACES), "time": _time(r), "day": r.choice(DAYS)},
        "msg": [
            "dinner at {place} on {day} at {time}?",
            "let's eat at {place}, {day} {time} works for me",
            "booked a table at {place} for {day}, {time}",
            "{day} evening at {place}? say {time}",
        ],
        "fix": ["actually can we push it to {time2}?", "small change: {time2} instead, the table was gone", "they moved our booking to {time2}"],
        "q": [
            "what time am I having dinner with {contact}?",
            "where are {contact} and I eating?",
            "when is my dinner with {contact}",
            "remind me of the restaurant plan with {contact}",
        ],
    },
    "flight": {
        "slots": lambda r, c: {
            "flight": f"{r.choice(AIRLINES)} {r.randint(100, 9999)}",
            "gate": f"{r.choice('ABCD')}{r.randint(1, 40)}",
            "time": _time(r),
        },
        "msg": [
            "my flight {flight} departs {time} from gate {gate}",
            "boarding {flight} at {time}, gate {gate}",
            "{flight} is on time, leaves {time} gate {gate}",
            "I'm on {flight}, take-off {time}, gate {gate}",
        ],
        "fix": ["update: {flight} delayed to {time2}", "gate changed, now leaving {time2}", "they rescheduled {flight} to {time2}"],
        "q": [
            "when does {contact}'s plane leave?",
            "which gate is {contact} flying from",
            "what is {contact}'s flight number?",
            "when should I pick up {contact} from the airport",
        ],
    },
    "address": {
        "slots": lambda r, c: {"street": r.choice(STREETS), "no": r.randint(1, 180), "floor": r.randint(1, 6)},
        "msg": [
            "my new address is {street} {no}, {floor}th floor",
            "we moved! {street} {no}, ring the bell on floor {floor}",
            "come to {street} {no}, {floor}. OG",
            "the flat is at {street} {no}, floor {floor}",
        ],
        "fix": ["sorry wrong number, it's {street} {no2}", "correction: house number {no2}", "typo, it's {no2} not {no}"],
        "q": [
            "where does {contact} live now?",
            "what's {contact}'s address",
            "which street is {contact}'s flat on",
            "how do I get to {contact}'s place",
        ],
    },
    "wifi": {
        "slots": lambda r, c: {"code": "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))},
        "msg": ["wifi password is {code}", "the network key: {code}", "pw for the internet {code}", "to get online use {code}"],
        "fix": ["changed the router, new password {code2}", "new wifi key is {code2}", "reset it, now {code2}"],
        "q": [
            "what's the wifi password at {contact}'s?",
            "how do I get online at {contact}'s place",
            "network key {contact} sent me",
            "internet password from {contact}",
        ],
    },
    "appointment": {
        "slots": lambda r, c: {"doc": r.choice(DOCTORS), "day": r.choice(DAYS), "time": _time(r)},
        "msg": [
            "your {doc} appointment is {day} at {time}",
            "booked you the {doc} for {day} {time}",
            "reminder: {doc} {day}, {time}",
            "got the {doc} slot on {day} at {time}",
        ],
        "fix": ["the {doc} moved it to {time2}", "practice called, new time {time2}", "they shifted the {doc} to {time2}"],
        "q": [
            "when is the {doc} appointment {contact} booked?",
            "what time do I see the {doc}",
            "doctor appointment {contact} told me about",
            "which day is my {doc} visit",
        ],
    },
    "money": {
        "slots": lambda r, c: {
            "amount": r.randint(8, 240),
            "what": r.choice(["concert tickets", "the groceries", "the taxi", "dinner", "the gift", "rent share"]),
        },
        "msg": [
            "you owe me {amount}€ for {what}",
            "can you send {amount} euro for {what}?",
            "{what} came to {amount}€ each",
            "paypal me {amount}€ for {what} pls",
        ],
        "fix": ["oops it's {amount2}€ actually", "recalculated, {amount2}€ is right", "sorry, make it {amount2}€"],
        "q": [
            "how much do I owe {contact}?",
            "what did {contact} say I need to pay",
            "money I have to send {contact}",
            "do I still owe {contact} anything",
        ],
    },
    "shopping": {
        "slots": lambda r, c: {"a": r.choice(ITEMS), "b": r.choice(ITEMS)},
        "msg": ["can you grab {a} and {b} on the way home", "we're out of {a} and {b}", "pls buy {a}, {b}", "need {a} + {b} from the store"],
        "fix": ["forget the {a}, I found some", "also {b2} please", "scratch that, just {b}"],
        "q": [
            "what did {contact} ask me to buy?",
            "shopping list from {contact}",
            "groceries {contact} needs",
            "what should I pick up for {contact}",
        ],
    },
    "party": {
        "slots": lambda r, c: {"event": r.choice(EVENTS), "day": r.choice(DAYS), "place": r.choice(PLACES), "time": _time(r)},
        "msg": [
            "{event} on {day} at {place}, starts {time}",
            "you're invited! my {event}, {day} {time}, {place}",
            "{event} is happening {day} from {time} at {place}",
            "save the date: {event}, {place}, {day} at {time}",
        ],
        "fix": ["{event} now starts at {time2}", "we moved the {event} to {time2}", "new start time for the {event}: {time2}"],
        "q": [
            "when is {contact}'s {event}?",
            "where is the party {contact} invited me to",
            "what time does {contact}'s celebration start",
            "details of {contact}'s {event}",
        ],
    },
}


@dataclass
class Message:
    id: str
    user: str
    contact: str
    text: str
    sent_at: datetime
    kind: str | None = None  # fact kind, None for small talk


@dataclass
class Query:
    id: str
    user: str
    text: str
    relevance: dict[str, int] = field(default_factory=dict)  # message id -> grade (2 = best answer, 1 = partly)
    kind: str = ""


@dataclass
class Corpus:
    messages: dict[str, list[Message]]  # user -> messages
    queries: list[Query]


def doc_text(contact: str, text: str) -> str:
    """What gets indexed: the sender is part of the document, as it is in any chat UI."""
    return f"{contact}: {text}"


def _fmt(t: str, slots: dict) -> str:
    return t.format(**slots)


def generate(
    n_users: int, templates: tuple[int, ...], seed: int, facts_per_user: int = 14, small_talk_per_user: int = 220, user_prefix: str = "u"
) -> Corpus:
    """`templates` selects which phrasings (by index) messages and questions are written with."""
    rng = random.Random(seed)
    t0 = datetime(2026, 9, 1, 8, 0)
    messages: dict[str, list[Message]] = {}
    queries: list[Query] = []
    for u in range(n_users):
        user = f"{user_prefix}{seed}-{u}"
        contacts = rng.sample(NAMES, 10)
        msgs: list[Message] = []
        mid = 0

        def add(contact: str, text: str, minutes: int, kind: str | None = None, _user: str = user, _msgs: list[Message] = msgs) -> Message:
            nonlocal mid
            m = Message(f"{_user}-m{mid}", _user, contact, text, t0 + timedelta(minutes=minutes), kind)
            mid += 1
            _msgs.append(m)
            return m

        for _ in range(small_talk_per_user):
            add(rng.choice(contacts), rng.choice(SMALL_TALK), rng.randint(0, 40_000))
        for f in range(facts_per_user):
            kind = rng.choice(list(KINDS))
            spec = KINDS[kind]
            contact = contacts[f % len(contacts)]
            slots = spec["slots"](rng, contact) | {"contact": contact}
            msg_t = [spec["msg"][i] for i in templates]
            q_t = [spec["q"][i] for i in templates]
            when = rng.randint(0, 36_000)
            original = add(contact, _fmt(rng.choice(msg_t), slots), when, kind)
            rel = {original.id: 2}
            # Look-alike: the same kind of fact with a different contact (not relevant).
            other = rng.choice([c for c in contacts if c != contact])
            add(other, _fmt(rng.choice(msg_t), spec["slots"](rng, other) | {"contact": other}), rng.randint(0, 40_000), kind)
            if rng.random() < 0.35:  # a later correction becomes the best answer
                fix_slots = slots | {
                    "time2": _time(rng),
                    "no2": rng.randint(1, 180),
                    "code2": "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10)),
                    "amount2": rng.randint(8, 240),
                    "b2": rng.choice(ITEMS),
                }
                fix = add(contact, _fmt(rng.choice(spec["fix"]), fix_slots), when + rng.randint(30, 3000), kind)
                rel = {original.id: 1, fix.id: 2}
            queries.append(Query(f"{user}-q{f}", user, _fmt(rng.choice(q_t), slots), rel, kind))
        msgs.sort(key=lambda m: m.sent_at)
        messages[user] = msgs
    return Corpus(messages, queries)


ENCODER, RERANK, TEST = (0, 2), (1,), (3,)


def splits(seed: int = 7) -> dict[str, Corpus]:
    """Disjoint users per split; phrasing groups as described in the module docstring."""
    return {
        "encoder": generate(400, ENCODER, seed=seed),
        "rerank": generate(80, RERANK, seed=seed + 1),
        "valid": generate(30, RERANK, seed=seed + 2),
        "test": generate(60, TEST, seed=seed + 3),
    }
