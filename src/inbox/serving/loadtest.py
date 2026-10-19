"""Small concurrent load test for /v1/classify (httpx + asyncio)."""

from __future__ import annotations

import asyncio
import random
import time

import httpx
import numpy as np

SAMPLES = [
    "can you remind me to call the dentist tomorrow",
    "dinner at 8 on friday?",
    "what's the weather like",
    "pay my phone bill",
    "how do you say thank you in german",
    "add milk to my shopping list",
    "I like turtles",
    "set an alarm for 6am",
]


async def _worker(client: httpx.AsyncClient, url: str, n: int, lat: list[float]) -> None:
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.post(f"{url}/v1/classify", json={"messages": [random.choice(SAMPLES)]})
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)


async def _run(url: str, requests: int, concurrency: int) -> dict:
    lat: list[float] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=10) as client:
        await asyncio.gather(*[_worker(client, url, requests // concurrency, lat) for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    a = np.array(lat)
    return {
        "requests": len(a),
        "concurrency": concurrency,
        "throughput_rps": round(len(a) / wall, 1),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
    }


def run(url: str, requests: int = 2000, concurrency: int = 16) -> dict:
    return asyncio.run(_run(url, requests, concurrency))
