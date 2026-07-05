"""Frozen-encoder invariance: probe training must never change the encoder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from edbr1.config import (  # noqa: E402
    BottleneckConfig,
    EncoderConfig,
    TrainConfig,
    config_to_dict,
)
from edbr1.models import build_model  # noqa: E402
from edbr1.probes.frozen import FrozenEncoder  # noqa: E402
from edbr1.probes.models import SpeakerProbe  # noqa: E402


def _checkpoint(tmp_path):
    cfg = TrainConfig(
        model="encoder_classifier",
        encoder=EncoderConfig(latent_freq=2, latent_frames=10, latent_dim=16),
        bottleneck=BottleneckConfig(type="vq", codebook_size=64, ema=True, kmeans_init=True),
    )
    model = build_model(cfg, num_classes=10)
    path = tmp_path / "encoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_mean": torch.zeros(1, 1, 1, 1),
            "norm_std": torch.ones(1, 1, 1, 1),
            "config": config_to_dict(cfg),
            "class_names": [str(i) for i in range(10)],
            "test_fold": 10,
        },
        path,
    )
    return path


def test_frozen_encoder_is_not_updated_by_probe_training(tmp_path):
    frozen = FrozenEncoder(_checkpoint(tmp_path), torch.device("cpu"))
    # Every encoder/bottleneck parameter is frozen.
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    assert all(not p.requires_grad for p in frozen.bottleneck.parameters())

    before = frozen.parameter_fingerprint()
    codes = frozen.emit_codes(torch.randn(8, 1, 64, 401))
    assert codes.shape == (8, 2, 10)  # (B, latent_freq, latent_frames)
    assert int(codes.max()) < frozen.codebook_size

    # Train a probe on the emitted codes; the frozen encoder must not move.
    probe = SpeakerProbe(frozen.codebook_size, num_speakers=5)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    targets = torch.randint(0, 5, (8,))
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(probe(codes), targets).backward()
        opt.step()

    assert frozen.parameter_fingerprint() == pytest.approx(before, abs=1e-6)
