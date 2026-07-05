# Methodology

## Front end (mel-spectrogram)

Every model consumes the same log-mel spectrogram front end
([`edbr1.features.melspec`](../src/edbr1/features/melspec.py)), driven entirely
by [`FeatureConfig`](../src/edbr1/config.py): 16 kHz mono, 64 mel bands, a 25 ms
analysis window and a 10 ms hop (so 100 frames/second), power spectrogram
converted to dB with an 80 dB floor. Waveforms are down-mixed to mono, resampled
to 16 kHz and fixed to a 4 s clip (zero-pad short, crop long) so batches have a
uniform shape. 16 kHz is canonical: 22.05 kHz was A/B-tested at the full 10-fold
and gave no gain for ~1.4x the compute (see [RESULTS.md](../RESULTS.md)).

Normalisation statistics (global or per-band) and any early-stopping validation
fold are always estimated on the **training folds only**, never the held-out
test fold, and the official UrbanSound8K fold split is used with an explicit
leak guard.

## Encoder

The on-device encoder E ([`edbr1.models.encoder`](../src/edbr1/models/encoder.py))
is a compact, MobileNet-style **depthwise-separable** conv stack (depthwise 3x3
+ pointwise 1x1, batch-norm + ReLU), kept well under 500K parameters so it is a
plausible always-on front end. It maps the `(1, 64, ~401)` log-mel to a latent
grid `(latent_dim, latent_freq, latent_frames)` — the sequence of tokens the
bottleneck quantises.

Downsampling to the target grid is honest: the number of frequency/time halvings
is derived from the target grid so the strided trunk never produces a map
*smaller* than the target, and a final adaptive average pool snaps to the exact
`(latent_freq, latent_frames)`. Every emitted token therefore corresponds to a
real strided-conv receptive field — the token count is never inflated by
upsampling, which matters because the token count sets the bitrate. Across the
swept operating points the conv *design* (block type, channel ramp,
`latent_dim`) is held fixed; only the number of downsampling stages adapts to the
grid, so more aggressive downsampling (lower bitrate) uses a slightly deeper
trunk.

## Bottleneck (VQ vs continuous vs VIB)

The bottleneck B ([`edbr1.models.bottleneck`](../src/edbr1/models/bottleneck.py))
sits between E and the classifier C. Two variants are implemented this stage:

* **Continuous (`type: none`)** — the identity: the latent passes straight
  through. This is the *control*, an ordinary encoder->classifier network that
  must reproduce the canonical baseline; it establishes the utility ceiling.
* **Discrete VQ-VAE (`type: vq`)** — a vector quantiser (van den Oord et al.,
  2017, *Neural Discrete Representation Learning*). Each latent token is snapped
  to its nearest of `codebook_size` entries; a **straight-through estimator**
  copies the gradient from the quantised latent back to the encoder; a codebook
  loss (or EMA codebook update) plus a `commitment_beta`-weighted **commitment
  loss** train the codebook. The classification loss is
  `cross_entropy + codebook_loss + beta * commitment_loss`.

VIB / other continuous stochastic bottlenecks are noted as future variants but
not implemented here.

### Honest bitrate accounting

The bitrate of a VQ operating point is computed from the *declared* latent grid
and codebook, not inferred at runtime
([`edbr1.bitrate`](../src/edbr1/bitrate.py)):

```
tokens_per_second = latent_freq * latent_frames / clip_seconds
bits_per_second   = tokens_per_second * log2(codebook_size)
```

The sweep holds the codebook at 1024 codes and varies the token rate (latent
grid) to span ~80 bits/s to ~16 kbits/s, so codebook **usage is comparable
across operating points**. Per fold we log codebook **perplexity** and the
**fraction of codes used**, accumulated over the held-out test fold. Codebook
collapse (few codes used, low perplexity) at low bitrate is expected and is
reported as measured — it is itself a finding for the later privacy analysis, so
it is never massaged.

## Classifier

The classifier C ([`edbr1.models.classifier`](../src/edbr1/models/classifier.py))
is deliberately minimal — a global average pool over the latent grid, one
dropout, and a linear head — mirroring the head of the original `SmallAudioCNN`.
Keeping representational capacity in the encoder (not the classifier) means the
utility-vs-bitrate curve reflects what the bottleneck preserves rather than extra
downstream modelling.

## Adversarial training objective (Phase 3)

To ask whether the code can be made to *hide speech* without spending the
(cheap) classification utility, a training-time adversary is attached to the
code and fought with a gradient reversal layer.

**Speech-overlay training stream.** The adversary needs speech to attack, so the
training fold mixes LibriSpeech speech into each UrbanSound8K scene -- the "loud
argument in the street" condition ([`edbr1.data.overlay`](../src/edbr1/data/overlay.py)).
A scene gets speech with probability `overlay_prob`, mixed by RMS at an SNR drawn
from a configured grid (speech = signal; e.g. `{0, 5, 10}` dB, so speech is
prominent). The training-time speech attribute is a tractable proxy -- a closed
set: **0 = no speech, 1..N = speaker id** over the `N` most-data
`train-clean-100` speakers ([`edbr1.data.librispeech.SpeechPool`](../src/edbr1/data/librispeech.py)) --
not full ASR (that is a Phase-4 probe). This is deliberately a proxy for "is
someone speaking, and who", stated here and in code.

**Leak-free and train-only.** The overlay and its speaker labels are applied to
the **training fold only**; the validation and held-out **test folds are clean
UrbanSound8K**. So (a) classification utility stays directly comparable to the
non-adversarial curve and to the Phase-2 clean ceiling, and (b) no LibriSpeech
speaker and no UrbanSound8K fold ever crosses the train/test boundary. The closed
speaker set is drawn from a *training* subset (`train-clean-100`) only.

**Gradient reversal + adversary head** ([`edbr1.models.adversary`](../src/edbr1/models/adversary.py)).
A small MLP predicts the speech attribute from the (global-average-pooled)
quantised code, behind a gradient reversal layer (Ganin & Lempitsky, 2015):
identity on the forward pass, negated-and-scaled gradient on the backward pass.
The total training loss is

```
cross_entropy(class) + codebook_loss + beta * commitment_loss + adversary_ce
```

with the GRL's `lambda` (linearly warmed up over `warmup_epochs`) scaling the
*reversed* gradient that reaches the encoder through the straight-through
estimator. The adversary head itself learns at full rate; only the encoder is
pushed to make the code un-predictive of the speaker. The bitrate is fixed at the
Phase-2 knee (1000 bits/s, anti-collapse codebook) and `lambda` is swept.

**The training-adversary accuracy is a sanity signal, not a privacy result.** It
is measured on the (train-only) overlay stream and only shows whether the
adversary is learning and whether the encoder is fighting it. True privacy is
measured in Phase 4 by *separate, stronger* probes; the training-time adversary
is intentionally modest and weaker than those probes, by design.

## Evaluation probes (separate from training-time adversary)

**Implemented in Phase 4a/4b** ([`edbr1.probes`](../src/edbr1/probes/)): independent,
stronger-than-adversary speaker-ID / ASR (CTC) / inverter probes trained against
the **frozen** encoded representation, plus the speech-overlay SNR sweep and the
ESC-50 transfer test. These -- not the training-time adversary above -- produce
the privacy numbers, as empirical lower bounds. The full protocol is in
[docs/04_evaluation_plan.md](04_evaluation_plan.md) and the results (leakage
table, robustness, ESC-50, and the recommended operating point) in
[RESULTS.md](../RESULTS.md).
