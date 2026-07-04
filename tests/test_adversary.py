"""Gradient-reversal + adversarial-wiring tests (skipped without torch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from edbr1.config import (  # noqa: E402
    AdversaryConfig,
    BottleneckConfig,
    EncoderConfig,
    OverlayConfig,
    TrainConfig,
)
from edbr1.models import AdversarialEncoderClassifier, gradient_reversal  # noqa: E402
from edbr1.train import _grl_lambda, _run_epoch_adversarial  # noqa: E402

# --- Gradient reversal layer ------------------------------------------------


def test_grl_forward_is_identity_backward_negates():
    x = torch.randn(6, requires_grad=True)
    y = gradient_reversal(x, 1.0)
    assert torch.equal(y, x)  # forward is the identity
    y.sum().backward()
    # d(sum y)/dx would be +1 without the GRL; the GRL flips it to -1.
    assert torch.allclose(x.grad, torch.full_like(x, -1.0))


def test_grl_lambda_scales_the_reversed_gradient():
    for lam in (0.0, 0.5, 2.0):
        x = torch.randn(4, requires_grad=True)
        gradient_reversal(x, lam).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, -lam))


# --- Adversarial model wiring ----------------------------------------------


def _adv_model(adversary_classes: int = 3) -> AdversarialEncoderClassifier:
    return AdversarialEncoderClassifier(
        EncoderConfig(latent_freq=2, latent_frames=10, latent_dim=16),
        BottleneckConfig(type="vq", codebook_size=64, ema=True, kmeans_init=True),
        num_classes=5,
        adversary_classes=adversary_classes,
    )


def test_adversarial_forward_returns_triple_with_right_shapes():
    model = _adv_model()
    model.grl.lambda_ = 1.0
    logits, bottleneck, adv_logits = model(torch.randn(4, 1, 64, 401))
    assert logits.shape == (4, 5)
    assert adv_logits.shape == (4, 3)
    assert bottleneck.indices is not None  # VQ bottleneck still produces codes


def test_adversarial_loss_backprops_to_the_encoder():
    # The combined loss must reach the encoder (via the straight-through path),
    # so the adversary can actually shape the encoder representation.
    model = _adv_model()
    model.grl.lambda_ = 1.0
    logits, bottleneck, adv_logits = model(torch.randn(3, 1, 64, 401))
    (logits.sum() + bottleneck.loss + adv_logits.sum()).backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(float(g.abs().sum()) > 0 for g in enc_grads)


def test_grl_lambda_warmup_schedule():
    def _cfg(grl_lambda: float, warmup: int) -> TrainConfig:
        return TrainConfig(
            overlay=OverlayConfig(enabled=True),
            adversary=AdversaryConfig(enabled=True, grl_lambda=grl_lambda, warmup_epochs=warmup),
        )

    cfg = _cfg(2.0, 4)
    # Linear warmup 0 -> grl_lambda over warmup_epochs, then constant.
    assert _grl_lambda(1, cfg) == pytest.approx(0.5)
    assert _grl_lambda(4, cfg) == pytest.approx(2.0)
    assert _grl_lambda(10, cfg) == pytest.approx(2.0)
    assert _grl_lambda(1, _cfg(1.0, 0)) == pytest.approx(1.0)  # no warmup


def test_run_epoch_adversarial_tracks_accuracy_and_codes():
    # Synthetic overlaid stream (mel, class, speech_label): the adversarial pass
    # must run, populate adversary accuracy, and accumulate codebook usage.
    torch.manual_seed(0)
    n, adv_classes = 16, 3
    mels = torch.randn(n, 1, 64, 401)
    classes = torch.randint(0, 5, (n,))
    speech = torch.randint(0, adv_classes, (n,))
    loader: DataLoader = DataLoader(TensorDataset(mels, classes, speech), batch_size=8)

    model = _adv_model(adv_classes)
    model.grl.lambda_ = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mean = torch.zeros(1, 1, 1, 1)
    std = torch.ones(1, 1, 1, 1)

    result = _run_epoch_adversarial(
        model, loader, torch.device("cpu"), mean, std,
        optimizer=optimizer, adv_criterion=torch.nn.CrossEntropyLoss(),
    )
    assert result.adv_acc is not None and 0.0 <= result.adv_acc <= 1.0
    assert result.adv_loss > 0.0
    assert result.code_counts is not None and int(result.code_counts.sum()) > 0
