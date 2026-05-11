# Research Questions

## Main research question

To what extent can a compute-constrained on-device audio encoder produce
a low-bitrate representation that supports accurate environmental sound
event classification while empirically suppressing recoverable speech
content, and how do design choices (bitrate, bottleneck type, adversarial
objective) shape the privacy–utility–compute trade-off?

## Sub-questions

1. How does the bitrate of the on-device representation trade off against
   downstream classification accuracy on environmental sound events?
2. How much speech-content information (speaker identity, phoneme/word
   content, intelligibility-correlated reconstruction quality) leaks
   through the encoded representation, measured by adversarial probes?
3. Does adding an adversarial speech-suppression objective during encoder
   training meaningfully shift the privacy–utility frontier compared to
   a plain bottleneck baseline?
4. Do the privacy and utility properties generalise across acoustic
   domains and unseen speakers (cross-dataset robustness)?

## Operational definitions

Utility = macro-F1 (and per-class F1) on a held-out environmental sound
classification task (UrbanSound8K official 10-fold CV; ESC-50 5-fold for
cross-dataset).

Privacy = a bundle of empirical adversarial measurements against the
encoded representation:
  (a) closed-set speaker identification accuracy of a probe trained on
      encodings;
  (b) ASR word/character error rate from a probe trained on encodings;
  (c) reconstruction quality (PESQ, STOI, log-spectral distance) from a
      learned inverter network.
Privacy is reported as empirical lower bounds against the specific probes
evaluated, NOT as a formal guarantee.

Compute = encoder parameter count, MACs per second of audio, on-device
latency on a single-thread CPU reference, and bitrate of the emitted
representation in bits/second.
