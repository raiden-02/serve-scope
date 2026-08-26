"""Deterministic short interactive prompts for the P1 sweep."""

from __future__ import annotations

import random

# Ordinary short questions, similar length. Enough variants that a long
# sweep is not one identical prefix-cache hit.
PROMPTS = [
    "In one short paragraph, what is a hash table used for?",
    "In one short paragraph, what does a mutex protect in a process?",
    "In one short paragraph, why do TCP connections use a handshake?",
    "In one short paragraph, what problem does virtual memory solve?",
    "In one short paragraph, what is an index for in a database?",
    "In one short paragraph, why do GPUs prefer regular data access?",
    "In one short paragraph, what does a load balancer do for servers?",
    "In one short paragraph, why is cache locality useful in loops?",
    "In one short paragraph, what is a deadlock between two locks?",
    "In one short paragraph, why do kernels separate user and kernel mode?",
    "In one short paragraph, what does DNS translate for a browser?",
    "In one short paragraph, why can tail latency matter more than mean?",
]


def select_prompt(seed: int, request_index: int) -> tuple[str, str]:
    """Return (prompt_id, prompt_text) from a fixed seeded shuffle per request."""
    rng = random.Random(seed + request_index * 9973)
    idx = rng.randrange(len(PROMPTS))
    return f"p1_short_{idx:02d}", PROMPTS[idx]


def scheduled_arrivals(n: int, offered_rps: float, t0: float) -> list[float]:
    if offered_rps <= 0:
        raise ValueError("offered_rps must be positive")
    interval = 1.0 / offered_rps
    return [t0 + i * interval for i in range(n)]
