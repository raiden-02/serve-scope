"""Deterministic prompts for interactive and background workloads."""

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


# Synthetic background documents. They are not production data. Each variant
# is a different topic so the run is not one identical prefix-cache hit.
_BACKGROUND_SENTENCES = [
    "The weekly review opened with a count of open incidents and closed tickets.",
    "Two services were rolled back after a configuration change reached canaries.",
    "Owners listed the remaining work in a numbered backlog rather than a chat thread.",
    "The cache hit rate dropped after the key format changed in the evening deploy.",
    "A follow-up task was to rebuild the search index from last night's snapshot.",
    "Latency plots were attached as figures, not as screenshots in the notes.",
    "The on-call rotation swapped at 17:00 UTC and the handoff named three risks.",
    "Disk usage on the log volume crossed seventy percent during the load test.",
    "The document repeats these facts so a summarizer has enough text to compress.",
    "No customer names, secrets, or real production identifiers appear in this text.",
    "The synthetic inventory listed aisle counts, SKU families, and restock windows.",
    "Indexing notes described title, heading, body, and outgoing-link fields.",
    "A background agent was asked to draft a brief that a human would later edit.",
    "Queue depth rose when batch jobs and interactive queries shared one worker pool.",
    "The meeting captured action items with owners, dates, and a one-line reason.",
    "Prefetch hints helped sequential scans and did nothing for random key lookups.",
    "The draft warned that prefix cache hits can hide how unlike two prompts are.",
    "Recovery steps were written as a checklist: drain, snapshot, restore, verify.",
    "The appendix restated the same events in a later time zone for the second shift.",
    "This paragraph exists only to lengthen a synthetic document for token count.",
]


_BACKGROUND_TOPICS = [
    ("meeting_notes", "synthetic weekly engineering meeting notes"),
    ("index_prep", "synthetic notes for rebuilding a search index"),
    ("warehouse_summary", "synthetic warehouse inventory briefing"),
    ("incident_timeline", "synthetic incident timeline for later summarization"),
    ("agent_research", "synthetic background-research brief for an agent"),
    ("release_checklist", "synthetic release-readiness checklist"),
    ("capacity_review", "synthetic capacity-planning discussion notes"),
    ("docs_digest", "synthetic internal-docs digest for offline summarization"),
]


def select_background_prompt(seed: int, request_index: int) -> tuple[str, str]:
    """Return (prompt_id, prompt_text) for a longer synthetic background job."""
    rng = random.Random(seed + request_index * 7919)
    topic_idx = rng.randrange(len(_BACKGROUND_TOPICS))
    topic_id, topic_label = _BACKGROUND_TOPICS[topic_idx]
    paragraphs: list[str] = []
    sentence_count = 48
    for i in range(sentence_count):
        sentence = _BACKGROUND_SENTENCES[(topic_idx * 3 + i) % len(_BACKGROUND_SENTENCES)]
        paragraphs.append(f"{i + 1}. {sentence}")
    body = "\n".join(paragraphs)
    prompt = (
        f"This is a synthetic {topic_label}. It is not real operational data.\n"
        "Write a structured summary with sections for facts, risks, and next steps. "
        "Use only information from the text. Do not invent names or metrics.\n\n"
        f"{body}"
    )
    return f"p2_bg_{topic_id}_{topic_idx:02d}", prompt


def windowed_arrivals(
    offered_rps: float,
    t0: float,
    start_offset_s: float,
    end_offset_s: float,
) -> list[float]:
    """Open-loop arrivals in [t0+start, t0+end). First request is at the window start."""
    if offered_rps <= 0:
        raise ValueError("offered_rps must be positive")
    if end_offset_s <= start_offset_s:
        raise ValueError("arrival window must have positive duration")
    duration_s = end_offset_s - start_offset_s
    n = int(round(offered_rps * duration_s))
    return [t0 + start_offset_s + i / offered_rps for i in range(n)]
