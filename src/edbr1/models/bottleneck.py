"""
Discrete bottleneck B between the encoder and the classifier.

Two implementations, selected by :class:`edbr1.config.BottleneckConfig`:

* :class:`IdentityBottleneck` -- the ``type='none'`` control. The continuous
  latent passes straight through with zero auxiliary loss, so the encoder ->
  classifier network behaves like an ordinary (un-quantised) model.

* :class:`VectorQuantizer` -- the ``type='vq'`` VQ-VAE bottleneck of van den Oord
  et al. (2017), "Neural Discrete Representation Learning". Each latent token is
  snapped to its nearest codebook entry; a straight-through estimator copies the
  gradient from the quantised latent back to the encoder, and the codebook is
  trained either by a codebook loss (default) or by EMA updates. A commitment
  loss keeps the encoder outputs close to the codebook. Codebook usage is
  reported honestly via per-forward perplexity and the code indices (so the
  trainer can accumulate fold-level usage and detect collapse).

Both return a :class:`BottleneckOutput` so the classifier and trainer have one
uniform interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from edbr1.config import BottleneckConfig


@dataclass
class BottleneckOutput:
    """Result of passing a latent grid through a bottleneck.

    Attributes:
        latent: ``(B, D, F', T')`` latent to feed the classifier. For VQ this is
            the quantised latent with the straight-through estimator applied.
        loss: scalar auxiliary loss to add to the classification loss during
            training (a zero scalar for the identity control).
        indices: ``(B, F', T')`` long code indices, or ``None`` for identity.
        perplexity: scalar codebook perplexity for this batch, or ``None``.
        codebook_size: number of codes (0 for the identity control).
    """

    latent: Tensor
    loss: Tensor
    indices: Tensor | None
    perplexity: Tensor | None
    codebook_size: int


class IdentityBottleneck(nn.Module):
    """The ``type='none'`` control: pass the latent through unchanged."""

    codebook_size = 0

    def forward(self, z: Tensor) -> BottleneckOutput:
        return BottleneckOutput(
            latent=z,
            loss=z.new_zeros(()),
            indices=None,
            perplexity=None,
            codebook_size=0,
        )


class VectorQuantizer(nn.Module):
    """VQ-VAE vector quantiser over the latent grid.

    Args:
        codebook_size: number of codes ``K``.
        dim: codebook vector dimension ``D`` (the encoder ``latent_dim``).
        commitment_beta: weight ``beta`` on the commitment loss.
        ema: use EMA codebook updates instead of a codebook loss.
        ema_decay, ema_epsilon: EMA momentum and Laplace-smoothing constant.
        chunk: max number of token vectors to score against the codebook at once
            (bounds the ``(N, K)`` distance memory at high token rates).
    """

    # Lloyd iterations used by data-dependent (k-means) init on the first batch.
    _kmeans_iters = 10

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        *,
        commitment_beta: float = 0.25,
        ema: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        kmeans_init: bool = False,
        restart_dead_codes: bool = False,
        restart_interval: int = 250,
        dead_code_threshold: float = 1.0,
        usage_decay: float = 0.99,
        chunk: int = 65_536,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_beta = commitment_beta
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.kmeans_init = kmeans_init
        self.restart_dead_codes = restart_dead_codes
        self.restart_interval = restart_interval
        self.dead_code_threshold = dead_code_threshold
        self.usage_decay = usage_decay
        self.chunk = chunk

        init = torch.empty(codebook_size, dim)
        nn.init.uniform_(init, -1.0 / codebook_size, 1.0 / codebook_size)
        if ema:
            # EMA codebook lives in buffers (no gradient); it is updated in-place
            # from assigned encoder vectors on the training forward pass.
            self.register_buffer("embedding", init.clone())
            self.register_buffer("cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_w", init.clone())
        else:
            self.embedding = nn.Parameter(init)

        # Anti-collapse bookkeeping (used only when the levers are enabled):
        #   _initted   -- whether data-dependent init has run (first batch)
        #   _steps     -- training-step counter, drives the restart schedule
        #   code_usage -- EMA of per-code selection counts, drives revival
        self.register_buffer("_initted", torch.tensor(False))
        self.register_buffer("_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("code_usage", torch.zeros(codebook_size))

    def _codebook(self) -> Tensor:
        emb = self.embedding
        assert isinstance(emb, Tensor)
        return emb

    def _nearest(self, flat_z: Tensor, codebook: Tensor | None = None) -> Tensor:
        """Nearest-code indices for ``(N, D)`` vectors, computed in chunks.

        ``argmin_k ||z - e_k||^2 = argmin_k ||e_k||^2 - 2 z . e_k`` (the ``||z||^2``
        term is constant per row and dropped). Chunking over ``N`` bounds the
        ``(chunk, K)`` distance tensor at high token rates. ``codebook`` defaults
        to the module codebook but may be passed explicitly (used by k-means init).
        """
        if codebook is None:
            codebook = self._codebook()
        code_sq = (codebook * codebook).sum(dim=1)  # (K,)
        indices = torch.empty(flat_z.shape[0], dtype=torch.long, device=flat_z.device)
        for start in range(0, flat_z.shape[0], self.chunk):
            chunk = flat_z[start : start + self.chunk]
            # (chunk, K): ||e||^2 - 2 z.e  (monotone in the true distance)
            dist = code_sq.unsqueeze(0) - 2.0 * chunk @ codebook.t()
            indices[start : start + chunk.shape[0]] = dist.argmin(dim=1)
        return indices

    @torch.no_grad()
    def _kmeans_init(self, flat_z: Tensor) -> None:
        """Data-dependent codebook init from the first training batch.

        Seeds the codebook with ``K`` encoder vectors and refines them with a few
        Lloyd iterations, so every code starts inside the (post-ReLU) latent
        distribution instead of the tiny uniform blob at the origin. When there
        are fewer tokens than codes, samples with replacement (no refinement).
        """
        n = flat_z.shape[0]
        if n < self.codebook_size:
            seed = torch.randint(0, n, (self.codebook_size,), device=flat_z.device)
            centroids = flat_z[seed].clone()
        else:
            seed = torch.randperm(n, device=flat_z.device)[: self.codebook_size]
            centroids = flat_z[seed].clone()
            for _ in range(self._kmeans_iters):
                assign = self._nearest(flat_z, centroids)
                new = torch.zeros_like(centroids)
                new.index_add_(0, assign, flat_z)
                counts = torch.bincount(assign, minlength=self.codebook_size).clamp_min(1)
                centroids = new / counts.unsqueeze(1).to(new.dtype)

        self._codebook().copy_(centroids)
        if self.ema:
            cast(Tensor, self.ema_w).copy_(centroids)
            cast(Tensor, self.cluster_size).fill_(1.0)

    def _update_usage(self, counts: Tensor) -> None:
        """EMA-accumulate per-code selection counts (drives dead-code revival)."""
        usage = cast(Tensor, self.code_usage)
        usage.mul_(self.usage_decay).add_(counts, alpha=1.0 - self.usage_decay)

    @torch.no_grad()
    def _revive_dead_codes(self, flat_z: Tensor) -> int:
        """Re-seed codes whose usage EMA is below threshold to random batch vectors.

        Returns the number of codes revived. For the EMA codebook, the revived
        entries' EMA state is reset too, so the next EMA step does not immediately
        pull them back; their usage EMA is bumped to threshold so they are not
        re-killed before they have had a chance to be selected.
        """
        usage = cast(Tensor, self.code_usage)
        dead = usage < self.dead_code_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        replacement = flat_z[torch.randint(0, flat_z.shape[0], (n_dead,), device=flat_z.device)]
        emb = self._codebook()
        emb[dead] = replacement
        if self.ema:
            cast(Tensor, self.ema_w)[dead] = replacement
            cast(Tensor, self.cluster_size)[dead] = 1.0
        usage[dead] = float(self.dead_code_threshold)
        return n_dead

    def _ema_update(self, flat_z: Tensor, indices: Tensor) -> None:
        """In-place EMA update of the codebook from assigned encoder vectors."""
        # Buffers are typed Tensor|Module by nn.Module.__getattr__; narrow them.
        cluster_size = cast(Tensor, self.cluster_size)
        ema_w = cast(Tensor, self.ema_w)
        embedding = cast(Tensor, self.embedding)

        counts = torch.bincount(indices, minlength=self.codebook_size).to(flat_z.dtype)
        dw = torch.zeros_like(ema_w)
        dw.index_add_(0, indices, flat_z)

        cluster_size.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        ema_w.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)

        n = cluster_size.sum()
        # Laplace smoothing so empty clusters do not divide by zero.
        smoothed = (
            (cluster_size + self.ema_epsilon)
            / (n + self.codebook_size * self.ema_epsilon)
            * n
        )
        embedding.copy_(ema_w / smoothed.unsqueeze(1))

    def forward(self, z: Tensor) -> BottleneckOutput:
        """z: ``(B, D, F', T')`` continuous latent -> quantised BottleneckOutput."""
        b, d, f, t = z.shape
        # (B, D, F, T) -> (B, F, T, D) -> (N, D)
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, d)

        # Data-dependent codebook init on the first training batch, before the
        # nearest-code lookup, so this batch already quantises against real data.
        if self.kmeans_init and self.training and not bool(self._initted):
            self._kmeans_init(flat_z.detach())
            cast(Tensor, self._initted).fill_(True)

        indices = self._nearest(flat_z)
        quantized = F.embedding(indices, self._codebook())  # (N, D)

        # EMA codebook update happens before the loss (which is commitment-only).
        if self.ema and self.training:
            self._ema_update(flat_z.detach(), indices)

        commitment = F.mse_loss(flat_z, quantized.detach())
        if self.ema:
            loss = self.commitment_beta * commitment
        else:
            codebook_loss = F.mse_loss(quantized, flat_z.detach())
            loss = codebook_loss + self.commitment_beta * commitment

        # Straight-through: identity gradient from quantized latent to encoder.
        quantized_st = flat_z + (quantized - flat_z).detach()
        latent = quantized_st.reshape(b, f, t, d).permute(0, 3, 1, 2).contiguous()

        counts = torch.bincount(indices, minlength=self.codebook_size).to(z.dtype)

        # Dead-code revival: track usage and, on schedule, re-seed unused codes to
        # random encoder vectors from this batch. Runs after quantisation so it
        # only affects subsequent batches.
        if self.restart_dead_codes and self.training:
            self._update_usage(counts)
            steps = cast(Tensor, self._steps)
            steps += 1
            if int(steps) % self.restart_interval == 0:
                self._revive_dead_codes(flat_z.detach())

        probs = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return BottleneckOutput(
            latent=latent,
            loss=loss,
            indices=indices.reshape(b, f, t),
            perplexity=perplexity,
            codebook_size=self.codebook_size,
        )


def build_bottleneck(config: BottleneckConfig, latent_dim: int) -> nn.Module:
    """Construct the bottleneck named by ``config`` (identity or VQ)."""
    if config.type == "none":
        return IdentityBottleneck()
    if config.type == "vq":
        return VectorQuantizer(
            config.codebook_size,
            latent_dim,
            commitment_beta=config.commitment_beta,
            ema=config.ema,
            ema_decay=config.ema_decay,
            ema_epsilon=config.ema_epsilon,
            kmeans_init=config.kmeans_init,
            restart_dead_codes=config.restart_dead_codes,
            restart_interval=config.restart_interval,
            dead_code_threshold=config.dead_code_threshold,
            usage_decay=config.usage_decay,
        )
    raise ValueError(f"Unknown bottleneck type: {config.type!r}")
