"""Speech-overlay tests: SNR construction, overlay behaviour, leak guards."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from edbr1.data.librispeech import SpeechPool, speaker_utterances  # noqa: E402
from edbr1.data.overlay import SpeechOverlay, mix_at_snr  # noqa: E402

LIBRISPEECH = Path("data/raw/librispeech/LibriSpeech")
US8K = Path("data/raw/urbansound8k/UrbanSound8K")
_has_libri = (LIBRISPEECH / "train-clean-100").is_dir()
_has_us8k = (US8K / "metadata" / "UrbanSound8K.csv").is_file()


# --- SNR mixing -------------------------------------------------------------


def test_mix_at_snr_achieves_the_target_ratio():
    torch.manual_seed(0)
    scene = torch.randn(16000) * 0.3
    speech = torch.randn(16000)
    for snr in (-5.0, 0.0, 8.0):
        mixed = mix_at_snr(scene, speech, snr)
        added = mixed - scene  # the scaled-in speech component
        achieved = 10 * math.log10(float(added.pow(2).mean() / scene.pow(2).mean()))
        assert achieved == pytest.approx(snr, abs=1e-3)


def test_mix_at_snr_silent_inputs_return_scene_unchanged():
    scene = torch.randn(1000)
    assert torch.equal(mix_at_snr(scene, torch.zeros(1000), 5.0), scene)  # silent speech
    zeros = torch.zeros(1000)
    assert torch.equal(mix_at_snr(zeros, torch.randn(1000), 5.0), zeros)  # silent scene


# --- Overlay sampling (stub pool, no decode) -------------------------------


class _StubPool:
    """Minimal SpeechPool stand-in so overlay logic tests need no audio."""

    num_classes = 4  # 3 speakers + no-speech

    def __init__(self) -> None:
        self.segments = torch.randn(3, 1000)
        self.labels = torch.tensor([1, 2, 3])

    def sample(self, generator: object = None) -> tuple[torch.Tensor, int]:
        i = int(torch.randint(0, 3, (1,)))
        return self.segments[i], int(self.labels[i])


def test_overlay_prob_zero_is_always_no_speech():
    ov = SpeechOverlay(_StubPool(), overlay_prob=0.0)  # type: ignore[arg-type]
    scene = torch.randn(1000)
    for _ in range(10):
        mixed, label = ov.apply(scene)
        assert label == 0
        assert torch.equal(mixed, scene)


def test_overlay_prob_one_always_labels_a_speaker_and_changes_audio():
    torch.manual_seed(0)
    ov = SpeechOverlay(_StubPool(), overlay_prob=1.0, snr_choices=(5.0,))  # type: ignore[arg-type]
    scene = torch.randn(1000)
    for _ in range(10):
        mixed, label = ov.apply(scene)
        assert 1 <= label <= 3  # a closed-set speaker (0 reserved for no speech)
        assert not torch.equal(mixed, scene)


def test_overlay_rejects_bad_config():
    with pytest.raises(ValueError, match="overlay_prob"):
        SpeechOverlay(_StubPool(), overlay_prob=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="snr_choices"):
        SpeechOverlay(_StubPool(), snr_choices=())  # type: ignore[arg-type]


# --- Leak guards ------------------------------------------------------------


def test_speechpool_rejects_non_permitted_subset():
    # Refuses subsets outside the three permitted ones -- before any decoding.
    with pytest.raises(ValueError, match="subset must be one of"):
        SpeechPool("data/raw/librispeech/LibriSpeech", subset="train-other-500")


@pytest.mark.skipif(not _has_libri, reason="LibriSpeech not present")
def test_speechpool_closed_set_is_from_the_training_subset_only():
    pool = SpeechPool(
        str(LIBRISPEECH), subset="train-clean-100",
        num_speakers=3, segments_per_speaker=2, seed=1337,
    )
    train_speakers = set(speaker_utterances(LIBRISPEECH / "train-clean-100"))
    # The closed adversary speaker set never leaves the training subset.
    assert set(pool.speaker_ids) <= train_speakers
    assert len(pool.speaker_ids) == 3
    assert pool.num_classes == 4  # 3 speakers + no-speech
    assert set(pool.labels.tolist()) == {1, 2, 3}  # labels 1..N, 0 reserved
    assert pool.segments.shape == (6, 64_000)  # 3 speakers x 2 segments, 4 s @ 16 kHz


@pytest.mark.skipif(not (_has_libri and _has_us8k), reason="datasets not present")
def test_overlay_dataset_yields_triple_base_yields_pair():
    from edbr1.config import FeatureConfig
    from edbr1.data import OverlaySpeechDataset, UrbanSound8KDataset, load_metadata

    md = load_metadata(str(US8K)).head(2)
    pool = SpeechPool(str(LIBRISPEECH), num_speakers=2, segments_per_speaker=2, seed=1)
    ov = SpeechOverlay(pool, overlay_prob=1.0, snr_choices=(5.0,))
    # Base dataset (clean, e.g. the test fold) stays a 2-tuple; the overlay
    # training dataset yields the extra speech label.
    assert len(UrbanSound8KDataset(md, FeatureConfig(), 4.0)[0]) == 2
    mel, class_id, speaker = OverlaySpeechDataset(md, FeatureConfig(), 4.0, overlay=ov)[0]
    assert mel.shape[0] == 1 and 0 <= class_id <= 9 and 1 <= speaker <= 2
