# Evaluation

## Utility metrics

Classification **macro-F1** under the official UrbanSound8K 10-fold CV (leak-
guarded split, train-only normalisation/early-stopping). The no-bottleneck
control (0.748 ± 0.058) is the utility ceiling; see [RESULTS.md](../RESULTS.md).

## Privacy / leakage metrics (Phase 4a)

Privacy is measured by **independent, evaluation-time probes** that attack the
**frozen** encoder's emitted codes — deliberately separate from, and stronger
than, the Phase-3 training-time adversary. Every number is an **empirical lower
bound** on leakage: a stronger future probe can only do better.

**Frozen-encoder protocol.** For each operating point (bitrate × adversarial λ)
the trained encoder+bottleneck is frozen (eval mode, no grad — asserted by a
weight fingerprint), and only its discrete codes are exposed. Probes learn their
own embedding of the transmitted indices, so they are at least as strong as the
Phase-3 adversary (which read the codebook latent directly).

**Leak guards (the load-bearing discipline).** Probe speech is drawn from
**dev-clean**, whose speakers are disjoint from the encoder's `train-clean-100`
overlay speakers — so any success is representational leakage, not memorisation.
Splits are verified at construction (raising on any leak):
* speaker-ID: a closed set of speakers on both sides, **utterances disjoint**;
* ASR / inversion: **speakers disjoint** train/test (unseen-speaker generalisation).
Probe speech is overlaid on held-out (fold-10) UrbanSound8K scenes at the
deployment SNRs before encoding.

**The three probes and their metrics:**
* **Speaker-ID** — a closed-set classifier (larger than the Phase-3 head):
  **top-1 accuracy vs chance** (1/N).
* **ASR** — a from-scratch CTC recogniser (Graves et al. 2006) on codes:
  **WER and CER vs the ~1.0 unintelligible ceiling**. (A pretrained-recogniser
  attacker is a stronger future probe, not run here.)
* **Inverter** — a learned code→log-mel decoder: **log-spectral distance (LSD)
  and MSE** against the clean source speech, vs a **silence floor** (no-information
  baseline) and the high-bitrate point as a loose reference. PESQ/STOI are noted
  where the packages/waveform reconstruction are unavailable.

The headline privacy question (RQ3) is whether adversarial λ reduces probe success
**beyond** what bitrate alone achieves, at matched honest bitrates.

## Compute metrics

Encoder parameter count (on-device budget < 500K), nominal and perplexity-
effective bitrate (`tokens/s · log2(codebook)` and `tokens/s · log2(perplexity)`),
and wall time per run. Reported alongside each sweep in [RESULTS.md](../RESULTS.md).

## Pareto / trade-off analysis (Phase 4b)

The privacy–utility–compute frontier assembles every Phase 2–4a operating point
into a ``(utility macro-F1, per-channel leakage, compute)`` triple. Leakage is
**never collapsed into a single scalar** — it is reported per channel (speaker /
ASR / inverter) because the result is content-specific. Compute is the effective
(perplexity-based) bits/s plus the on-device encoder parameter count. The frontier
data is assembled into the committed ``docs/figures/sweep_data.json`` and every
figure is regenerated from it by ``scripts/make_figures.py`` (see RESULTS.md for
the recommended operating point).

## Robustness and generalisation (Phase 4b)

* **Test-time SNR robustness** — at the 1000 bits/s knee (λ=0 and λ=2), utility and
  the three leakage metrics are re-measured across a −10…+10 dB speech-to-scene
  SNR grid (the "loud argument" condition), leak-guarded as in Phase 4a. Utility
  is robust; acoustic-envelope leakage rises with speech loudness; identity
  suppression by λ holds across SNR.
* **ESC-50 cross-dataset transfer** — a light head on the *frozen* code under
  ESC-50's official 5-fold split (encoder never trained on ESC-50) measures whether
  the low-bitrate representation carries transferable scene information. It does
  (~18× chance). Both in RESULTS.md.

## Failure analysis

Per-class confusion (saved per fold) and the operating points where utility or
privacy behaviour changes qualitatively (codebook collapse at low bitrate before
the anti-collapse fix; probe leakage by content type in Phase 4a).
