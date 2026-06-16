"""Feature-extractor and model shape tests (skipped if torch is absent)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from edbr1.config import FeatureConfig  # noqa: E402
from edbr1.features import LogMelExtractor  # noqa: E402
from edbr1.models import SmallAudioCNN  # noqa: E402


def test_logmel_shape_default_config():
    cfg = FeatureConfig()
    extractor = LogMelExtractor(cfg)
    # 1 s of mono audio already at the target rate -> ~101 frames.
    wav = torch.randn(cfg.sample_rate)
    out = extractor(wav, cfg.sample_rate)
    assert out.shape[0] == cfg.n_mels
    expected_frames = cfg.sample_rate // cfg.hop_length + 1
    assert out.shape[1] == expected_frames


def test_logmel_downmixes_and_resamples():
    cfg = FeatureConfig()
    extractor = LogMelExtractor(cfg)
    # Stereo input at a different sample rate must still yield n_mels rows.
    wav = torch.randn(2, 44_100)
    out = extractor(wav, 44_100)
    assert out.shape[0] == cfg.n_mels
    assert out.dim() == 2


def test_logmel_rejects_bad_rank():
    extractor = LogMelExtractor()
    with pytest.raises(ValueError, match="Expected waveform"):
        extractor(torch.randn(2, 3, 4), 16_000)


def test_cnn_forward_shape():
    model = SmallAudioCNN(num_classes=10)
    x = torch.randn(3, 1, 64, 201)
    logits = model(x)
    assert logits.shape == (3, 10)


def test_cnn_is_compact():
    # Guard the "small CNN" claim: keep it well under ~1M parameters.
    model = SmallAudioCNN(num_classes=10)
    assert model.num_parameters() < 1_000_000
