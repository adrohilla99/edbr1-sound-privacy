"""Encoder, bottleneck and EncoderClassifier tests (skipped without torch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from edbr1.config import BottleneckConfig, EncoderConfig, TrainConfig  # noqa: E402
from edbr1.models import (  # noqa: E402
    AudioEncoder,
    EncoderClassifier,
    IdentityBottleneck,
    VectorQuantizer,
    build_model,
)

# --- Encoder ---------------------------------------------------------------


def test_encoder_emits_declared_latent_grid():
    cfg = EncoderConfig(latent_dim=128, latent_freq=4, latent_frames=50)
    enc = AudioEncoder(cfg, n_mels=64, nominal_frames=401)
    x = torch.randn(2, 1, 64, 401)
    z = enc(x)
    # The token grid is exactly the declared (latent_dim, latent_freq, latent_frames).
    assert z.shape == (2, 128, 4, 50)


def test_encoder_under_500k_params_across_operating_points():
    # The on-device encoder E must stay well under 500K parameters at every grid.
    for f, t in [(1, 32), (2, 50), (4, 100), (8, 100), (8, 200), (32, 200), (8, 50)]:
        enc = AudioEncoder(
            EncoderConfig(latent_freq=f, latent_frames=t), n_mels=64, nominal_frames=401
        )
        assert enc.num_parameters() < 500_000, (f, t, enc.num_parameters())


# --- Identity bottleneck (the control) -------------------------------------


def test_identity_bottleneck_passes_through_with_zero_loss():
    z = torch.randn(2, 16, 3, 7)
    out = IdentityBottleneck()(z)
    assert torch.equal(out.latent, z)
    assert float(out.loss) == 0.0
    assert out.indices is None
    assert out.codebook_size == 0


# --- Vector quantiser ------------------------------------------------------


def test_vq_output_shapes_and_index_range():
    b, d, f, t, k = 3, 32, 4, 8, 64
    vq = VectorQuantizer(k, d)
    z = torch.randn(b, d, f, t)
    out = vq(z)
    assert out.latent.shape == z.shape
    assert out.indices is not None
    assert out.indices.shape == (b, f, t)
    # All code indices are valid entries of the codebook.
    assert int(out.indices.min()) >= 0
    assert int(out.indices.max()) < k
    assert out.codebook_size == k


def test_vq_straight_through_gradient_is_identity():
    # The straight-through estimator must copy the gradient from the quantised
    # latent straight back to the encoder output (d latent / d z == 1).
    vq = VectorQuantizer(16, 8)
    z = torch.randn(2, 8, 2, 3, requires_grad=True)
    out = vq(z)
    out.latent.sum().backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_vq_commitment_loss_nonnegative_and_finite():
    vq = VectorQuantizer(32, 16, commitment_beta=0.25)
    z = torch.randn(4, 16, 2, 5)
    out = vq(z)
    assert float(out.loss.detach()) >= 0.0
    assert torch.isfinite(out.loss)


def test_vq_perplexity_within_codebook_bounds():
    k = 64
    vq = VectorQuantizer(k, 8)
    z = torch.randn(8, 8, 4, 8)
    out = vq(z)
    assert out.perplexity is not None
    # Perplexity is bounded above by the codebook size (uniform usage).
    assert 1.0 <= float(out.perplexity) <= k + 1e-4


def test_vq_indices_select_the_true_nearest_code():
    # A tiny hand-checkable case: two orthogonal codes, one clearly-closer input.
    vq = VectorQuantizer(2, 2)
    with torch.no_grad():
        vq.embedding.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    z = torch.tensor([[0.9, 0.1]]).reshape(1, 2, 1, 1)  # nearest to code 0
    out = vq(z)
    assert int(out.indices.reshape(-1)[0]) == 0


def test_vq_ema_updates_codebook_and_has_no_codebook_gradient():
    vq = VectorQuantizer(16, 8, ema=True)
    before = vq.embedding.clone()
    vq.train()
    z = torch.randn(4, 8, 2, 4)
    out = vq(z)
    # EMA codebook is a buffer, updated in-place, and never carries a gradient.
    assert not vq.embedding.requires_grad
    assert not torch.equal(vq.embedding, before)
    # Commitment loss still flows to the encoder.
    assert float(out.loss) >= 0.0


# --- anti-collapse: k-means init + dead-code revival -----------------------


def test_vq_kmeans_init_moves_codebook_into_the_data():
    # The tiny uniform init sits at ~0; k-means init must move every code into
    # the (here, far-from-zero) encoder-output distribution on the first batch.
    torch.manual_seed(0)
    vq = VectorQuantizer(8, 4, kmeans_init=True)
    vq.train()
    before = vq.embedding.detach().clone()
    z = torch.randn(4, 4, 2, 4) + 10.0  # data centred at 10, nowhere near init
    vq(z)
    after = vq.embedding.detach()
    assert bool(vq._initted)
    assert not torch.allclose(after, before)
    assert float(after.mean()) > 5.0  # was ~0, now in the data region


def test_vq_kmeans_init_runs_only_once():
    # _initted gates the data-dependent init to the first training batch only.
    torch.manual_seed(0)
    vq = VectorQuantizer(8, 4, kmeans_init=True)
    vq.train()
    vq(torch.randn(4, 4, 2, 4) + 10.0)
    codebook = vq.embedding.detach().clone()  # centred ~ +10
    vq(torch.randn(4, 4, 2, 4) - 10.0)  # a very different second batch
    # No re-init (and no optimiser step), so the codebook is unchanged.
    assert torch.equal(vq.embedding.detach(), codebook)


def test_vq_dead_code_revival_reseeds_unused_codes():
    # Restart every step with an enormous threshold so all under-used codes are
    # revived to random batch vectors (a constant vector here) -> codebook lands
    # on the data and off the uniform init.
    torch.manual_seed(0)
    vq = VectorQuantizer(
        16, 3, restart_dead_codes=True, restart_interval=1,
        dead_code_threshold=1e9, usage_decay=0.5,
    )
    vq.train()
    before = vq.embedding.detach().clone()
    z = torch.zeros(2, 3, 2, 4) + 5.0  # constant batch vector [5, 5, 5]
    vq(z)
    after = vq.embedding.detach()
    assert not torch.allclose(after, before)
    assert torch.allclose(after, torch.full_like(after, 5.0), atol=1e-4)
    assert int(vq._steps) == 1


def test_vq_dead_code_revival_off_by_default_leaves_codebook_static():
    # Without the flag, a forward pass never mutates the (Parameter) codebook.
    vq = VectorQuantizer(16, 3)
    vq.train()
    before = vq.embedding.detach().clone()
    vq(torch.randn(2, 3, 2, 4))
    assert torch.equal(vq.embedding.detach(), before)
    assert int(vq._steps) == 0  # step counter only advances under revival


def test_perplexity_from_counts_uniform_single_and_empty():
    from edbr1.train import _perplexity_from_counts

    k = 8
    assert abs(_perplexity_from_counts(torch.ones(k)) - k) < 1e-6  # uniform -> K
    degenerate = torch.zeros(k)
    degenerate[3] = 100.0
    assert abs(_perplexity_from_counts(degenerate) - 1.0) < 1e-6  # one code -> 1
    assert _perplexity_from_counts(torch.zeros(k)) == 0.0  # empty -> 0


# --- EncoderClassifier wiring ----------------------------------------------


def test_encoder_classifier_control_has_zero_aux_loss():
    cfg = TrainConfig(
        model="encoder_classifier",
        encoder=EncoderConfig(latent_freq=8, latent_frames=50),
        bottleneck=BottleneckConfig(type="none"),
    )
    model = build_model(cfg, num_classes=10)
    assert isinstance(model, EncoderClassifier)
    logits, out = model(torch.randn(3, 1, 64, 401))
    assert logits.shape == (3, 10)
    assert float(out.loss) == 0.0
    assert out.indices is None


def test_encoder_classifier_vq_returns_indices_and_positive_loss():
    cfg = TrainConfig(
        model="encoder_classifier",
        encoder=EncoderConfig(latent_freq=4, latent_frames=50, latent_dim=64),
        bottleneck=BottleneckConfig(type="vq", codebook_size=256),
        clip_seconds=4.0,
    )
    model = build_model(cfg, num_classes=10)
    logits, out = model(torch.randn(2, 1, 64, 401))
    assert logits.shape == (2, 10)
    assert out.indices is not None
    assert out.indices.shape == (2, 4, 50)
    assert int(out.indices.max()) < 256
    assert torch.isfinite(out.loss) and float(out.loss.detach()) > 0.0


def test_build_model_cnn_path_is_legacy_smallcnn():
    from edbr1.models import SmallAudioCNN

    model = build_model(TrainConfig(model="cnn"), num_classes=10)
    assert isinstance(model, SmallAudioCNN)
    # Legacy path still returns bare logits (no bottleneck tuple).
    logits = model(torch.randn(2, 1, 64, 201))
    assert logits.shape == (2, 10)
