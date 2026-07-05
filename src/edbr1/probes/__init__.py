"""Evaluation-time speech-leakage probes (Phase 4a).

Independent, stronger-than-training-adversary probes that attack the frozen
encoder's emitted codes to measure empirical *lower bounds* on speech leakage:
speaker identification, ASR (linguistic content) and a learned inverter (raw
acoustic content). The leak guards in :mod:`edbr1.probes.splits` are the load-
bearing discipline -- any speaker/utterance/fold leak between probe-train and
probe-test silently invalidates the privacy claim.
"""
from __future__ import annotations

from edbr1.probes.splits import (
    ProbeItem,
    ProbeSplitResult,
    build_probe_split,
    load_transcripts,
    speaker_utterances_with_transcripts,
)

__all__ = [
    "ProbeItem",
    "ProbeSplitResult",
    "build_probe_split",
    "load_transcripts",
    "speaker_utterances_with_transcripts",
]
