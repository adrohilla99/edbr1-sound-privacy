"""
UrbanSound8K small-CNN baseline trainer.

Runs the **official 10-fold cross-validation** protocol: for each held-out
test fold, train on the remaining nine and evaluate. Reports macro-F1 per
fold and averaged across folds, writes per-fold confusion matrices and a
machine-readable results JSON, and logs the exact config used.

Reproducibility: a single seed (offset per fold so folds differ but the
whole run is deterministic), the resolved config, and all metrics are
written under a timestamped directory in ``results/`` (gitignored).

Honest-reporting note: this script reports whatever macro-F1 the model
achieves. If it lands below the published ~73-76% band, that is surfaced
in the summary rather than silently tuned away -- the likely culprits to
investigate first are feature normalisation, clip length/cropping, and
making sure fold handling has not leaked.

Usage:
    python -m edbr1.train \
        --root data/raw/urbansound8k/UrbanSound8K \
        --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from edbr1 import bitrate
from edbr1.config import ScheduleConfig, TrainConfig, config_to_dict, load_train_config
from edbr1.data.librispeech import SpeechPool
from edbr1.data.overlay import SpeechOverlay
from edbr1.data.urbansound8k import (
    URBANSOUND8K_CLASSES,
    OverlaySpeechDataset,
    UrbanSound8KDataset,
    carve_validation_fold,
    load_metadata,
    train_test_fold_split,
)
from edbr1.evaluate import classification_metrics, save_confusion_matrix
from edbr1.models import (
    AdversarialEncoderClassifier,
    EncoderClassifier,
    build_model,
    nominal_frames_for,
)
from edbr1.models.bottleneck import BottleneckOutput
from edbr1.utils import seed_everything, seed_worker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(
    dataset: UrbanSound8KDataset,
    *,
    shuffle: bool,
    config: TrainConfig,
    device: torch.device,
    num_workers: int | None = None,
    persistent: bool = False,
) -> DataLoader[tuple[Tensor, int]]:
    """Build a DataLoader with CUDA-friendly, reproducible settings.

    ``num_workers`` defaults to ``config.num_workers``; pass ``0`` to force a
    single-process loader (used for the small validation/test passes to keep
    peak worker-process memory bounded). With workers enabled, ``seed_worker``
    makes per-worker augmentation reproducible and ``pin_memory`` is set on the
    CUDA path. ``persistent`` keeps workers alive across epochs (worth it for
    the repeatedly-iterated training loader; avoids Windows re-spawn overhead).
    """
    nw = config.num_workers if num_workers is None else num_workers
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": nw,
        "pin_memory": device.type == "cuda",
    }
    if nw > 0:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["persistent_workers"] = persistent
    return DataLoader(dataset, **kwargs)


def compute_norm_stats(
    loader: DataLoader[tuple[Tensor, int]], *, per_band: bool
) -> tuple[Tensor, Tensor]:
    """Mean/std of the log-mel features over a (training) loader.

    Standardising features with statistics estimated on the *training* folds
    only (never the test or validation fold) is important for reaching the
    published baseline and avoids leaking held-out statistics.

    Returns broadcastable tensors shaped ``(1, 1, n_mels, 1)`` when
    ``per_band`` (a mean/std per mel band) or ``(1, 1, 1, 1)`` for a single
    global scalar. The global path reproduces the original scalar computation
    exactly so the plain baseline is unchanged.
    """
    if not per_band:
        total = 0.0
        total_sq = 0.0
        count = 0
        for x, _ in tqdm(loader, desc="  norm-stats", leave=False):
            total += float(x.sum())
            total_sq += float((x * x).sum())
            count += x.numel()
        mean = total / count
        std = max(total_sq / count - mean * mean, 1e-12) ** 0.5
        return (
            torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 1),
        )

    # Per-band: reduce over batch, channel and time, keeping the mel axis.
    # Accumulate in float64 for numerical stability across the dataset.
    sum_b: Tensor | None = None
    sumsq_b: Tensor | None = None
    count_b = 0
    for x, _ in tqdm(loader, desc="  norm-stats", leave=False):
        xd = x.double()
        s = xd.sum(dim=(0, 1, 3))
        sq = (xd * xd).sum(dim=(0, 1, 3))
        sum_b = s if sum_b is None else sum_b + s
        sumsq_b = sq if sumsq_b is None else sumsq_b + sq
        count_b += x.shape[0] * x.shape[1] * x.shape[3]
    assert sum_b is not None and sumsq_b is not None
    mean_b = sum_b / count_b
    var_b = (sumsq_b / count_b - mean_b * mean_b).clamp_min(1e-12)
    std_b = var_b.sqrt()
    return (
        mean_b.to(torch.float32).view(1, 1, -1, 1),
        std_b.to(torch.float32).view(1, 1, -1, 1),
    )


def _forward_model(model: nn.Module, x: Tensor) -> tuple[Tensor, BottleneckOutput]:
    """Call ``model`` and return ``(logits, bottleneck_output)`` uniformly.

    ``EncoderClassifier`` already returns that tuple. The legacy ``SmallAudioCNN``
    returns bare logits; those are wrapped in a zero-loss identity output so the
    training loop is identical for both (``ce + 0`` leaves the baseline path
    numerically unchanged).
    """
    out = model(x)
    if isinstance(out, tuple):
        # (logits, bottleneck) for EncoderClassifier, or
        # (logits, bottleneck, adv_logits) for the adversarial model -- take the first two.
        return out[0], out[1]
    return out, BottleneckOutput(
        latent=out, loss=out.new_zeros(()), indices=None, perplexity=None, codebook_size=0
    )


def _codebook_size(model: nn.Module) -> int:
    """Codebook size of the model's bottleneck (0 if it has none)."""
    bottleneck = getattr(model, "bottleneck", None)
    return int(getattr(bottleneck, "codebook_size", 0))


def _perplexity_from_counts(code_counts: Tensor) -> float:
    """Codebook perplexity ``exp(entropy)`` from a ``(K,)`` usage histogram."""
    total = float(code_counts.sum())
    if total <= 0:
        return 0.0
    probs = code_counts.to(torch.float64) / total
    nz = probs[probs > 0]
    return float(torch.exp(-(nz * nz.log()).sum()))


@dataclasses.dataclass
class EpochResult:
    """Outcome of one epoch/eval pass."""

    ce_loss: float          # mean cross-entropy (comparable to the old baseline)
    vq_loss: float          # mean auxiliary VQ loss (0 without a VQ bottleneck)
    y_true: list[int]
    y_pred: list[int]
    code_counts: Tensor | None  # (K,) accumulated code usage, or None
    adv_loss: float = 0.0   # mean adversary cross-entropy (0 without an adversary)
    adv_acc: float | None = None  # adversary speech-attribute accuracy (sanity only)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> EpochResult:
    """One pass. Train if ``optimizer`` is given, else evaluate.

    The optimised loss is ``cross_entropy + bottleneck.loss`` (the VQ codebook +
    commitment loss); ``bottleneck.loss`` is a zero scalar for the no-bottleneck
    control and the legacy CNN, so the baseline path is unchanged. Code usage is
    accumulated into a ``(K,)`` histogram when a VQ bottleneck is present.
    """
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    running_ce = 0.0
    running_vq = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    codebook_size = _codebook_size(model)
    code_counts = (
        torch.zeros(codebook_size, dtype=torch.long) if codebook_size > 0 else None
    )

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = (x.to(device) - mean) / std
            y = y.to(device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
            logits, bottleneck = _forward_model(model, x)
            ce = criterion(logits, y)
            loss = ce + bottleneck.loss
            if training:
                loss.backward()
                assert optimizer is not None
                optimizer.step()
            running_ce += float(ce.detach()) * y.size(0)
            running_vq += float(bottleneck.loss.detach()) * y.size(0)
            seen += y.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())
            if code_counts is not None and bottleneck.indices is not None:
                idx = bottleneck.indices.reshape(-1).to("cpu")
                code_counts += torch.bincount(idx, minlength=codebook_size)

    denom = max(seen, 1)
    return EpochResult(running_ce / denom, running_vq / denom, y_true, y_pred, code_counts)


def _grl_lambda(epoch: int, config: TrainConfig) -> float:
    """GRL reversal strength with linear warmup over the first ``warmup_epochs``."""
    adv = config.adversary
    if adv.warmup_epochs <= 0:
        return adv.grl_lambda
    return adv.grl_lambda * min(1.0, epoch / adv.warmup_epochs)


def _run_epoch_adversarial(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    adv_criterion: nn.Module,
) -> EpochResult:
    """One adversarial training pass over the overlaid ``(x, y, speech_label)`` stream.

    Optimises ``cross_entropy + bottleneck.loss + adversary_ce``; the gradient
    reversal layer (its ``lambda_`` set by the caller for warmup) makes the
    encoder fight the adversary while the adversary head learns at full rate. The
    adversary's accuracy is tracked as an internal sanity signal only.
    """
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_ce = running_vq = running_adv = 0.0
    adv_correct = 0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    codebook_size = _codebook_size(model)
    code_counts = (
        torch.zeros(codebook_size, dtype=torch.long) if codebook_size > 0 else None
    )

    with torch.set_grad_enabled(True):
        for x, y, s in loader:
            x = (x.to(device) - mean) / std
            y = y.to(device)
            s = s.to(device)
            optimizer.zero_grad()
            logits, bottleneck, adv_logits = model(x)
            ce = ce_criterion(logits, y)
            adv = adv_criterion(adv_logits, s)
            loss = ce + bottleneck.loss + adv
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_ce += float(ce.detach()) * bs
            running_vq += float(bottleneck.loss.detach()) * bs
            running_adv += float(adv.detach()) * bs
            adv_correct += int((adv_logits.argmax(dim=1) == s).sum())
            seen += bs
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())
            if code_counts is not None and bottleneck.indices is not None:
                code_counts += torch.bincount(
                    bottleneck.indices.reshape(-1).to("cpu"), minlength=codebook_size
                )

    denom = max(seen, 1)
    return EpochResult(
        running_ce / denom, running_vq / denom, y_true, y_pred, code_counts,
        adv_loss=running_adv / denom, adv_acc=adv_correct / denom,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, schedule: ScheduleConfig, epochs: int
) -> Any:
    """Construct the LR scheduler named by ``schedule`` (or ``None``)."""
    if schedule.scheduler == "none":
        return None
    if schedule.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if schedule.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=schedule.plateau_factor,
            patience=schedule.plateau_patience,
        )
    raise ValueError(f"Unknown scheduler: {schedule.scheduler!r}")


def _pick_val_fold(train_df: Any, schedule: ScheduleConfig) -> int:
    """Choose which training fold to hold out for validation (deterministic)."""
    folds = sorted(set(train_df["fold"].unique()))
    if schedule.val_fold is not None:
        return int(schedule.val_fold)
    return int(folds[-1])  # highest-numbered training fold


def train_one_fold(
    metadata: object,
    test_fold: int,
    config: TrainConfig,
    device: torch.device,
    class_names: Sequence[str],
    cache_dir: str | Path | None = None,
    overlay: SpeechOverlay | None = None,
    save_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train on all folds except ``test_fold`` and evaluate on it.

    When early stopping (or a plateau scheduler) is configured, one training
    fold is carved off as a validation set; the best-by-validation checkpoint
    is restored before the held-out test fold is scored. Normalisation stats
    are always estimated on the (inner) training folds only.

    ``cache_dir`` (if given) enables the dataset's on-disk waveform cache, which
    is bit-identical to recomputing and only removes the per-epoch decode cost.

    When ``config.adversary.enabled`` and an ``overlay`` is supplied, the training
    fold overlays speech and an :class:`AdversarialEncoderClassifier` is trained
    with the gradient-reversal speech adversary. The validation and **test** folds
    are always clean UrbanSound8K (no overlay, no adversary), so utility stays
    comparable to the non-adversarial curve and no speaker leaks into the test set.
    """
    import pandas as pd

    assert isinstance(metadata, pd.DataFrame)
    seed_everything(config.seed + test_fold)
    schedule = config.schedule
    needs_val = schedule.early_stopping or schedule.scheduler == "plateau"

    train_df, test_df = train_test_fold_split(metadata, test_fold)
    val_fold: int | None = None
    val_df = None
    if needs_val:
        val_fold = _pick_val_fold(train_df, schedule)
        train_df, val_df = carve_validation_fold(train_df, val_fold)

    adversary_on = config.adversary.enabled and overlay is not None
    augment = config.augment if config.augment.enabled else None
    train_ds: UrbanSound8KDataset
    if adversary_on:
        assert overlay is not None
        # Train fold overlays speech and yields (mel, class, speech_label).
        train_ds = OverlaySpeechDataset(
            train_df, config.features, config.clip_seconds, overlay=overlay,
            train=True, augment=augment, cache_dir=cache_dir,
        )
    else:
        train_ds = UrbanSound8KDataset(
            train_df, config.features, config.clip_seconds, train=True, augment=augment,
            cache_dir=cache_dir,
        )
    test_ds = UrbanSound8KDataset(
        test_df, config.features, config.clip_seconds, train=False, cache_dir=cache_dir
    )

    # Training loader: workers (parallel CPU decode/resample/augment) kept
    # persistent across epochs to avoid Windows re-spawn cost. The validation
    # and test passes are small and run single-process, so peak worker-process
    # memory stays bounded to the training loader's workers.
    train_loader = _make_loader(
        train_ds, shuffle=True, config=config, device=device, persistent=True
    )
    test_loader = _make_loader(
        test_ds, shuffle=False, config=config, device=device, num_workers=0
    )
    val_loader: DataLoader[tuple[Tensor, int]] | None = None
    if val_df is not None:
        val_ds = UrbanSound8KDataset(
            val_df, config.features, config.clip_seconds, train=False, cache_dir=cache_dir
        )
        val_loader = _make_loader(
            val_ds, shuffle=False, config=config, device=device, num_workers=0
        )

    # Normalisation stats must come from clean (un-augmented) training features.
    # When augmentation is off the train loader already yields clean features,
    # so it is reused -- keeping the plain baseline path unchanged.
    if augment is not None:
        norm_ds = UrbanSound8KDataset(
            train_df, config.features, config.clip_seconds, train=False, cache_dir=cache_dir
        )
        norm_loader: DataLoader[tuple[Tensor, int]] = _make_loader(
            norm_ds, shuffle=False, config=config, device=device
        )
    else:
        norm_loader = train_loader

    mean, std = compute_norm_stats(norm_loader, per_band=config.norm == "per_band")
    mean, std = mean.to(device), std.to(device)

    model: nn.Module
    adv_model: AdversarialEncoderClassifier | None = None
    adv_criterion: nn.Module | None = None
    if adversary_on:
        assert overlay is not None
        adv_model = AdversarialEncoderClassifier(
            config.encoder, config.bottleneck, len(class_names), overlay.num_classes,
            adversary_hidden=config.adversary.hidden_dim,
            n_mels=config.features.n_mels, nominal_frames=nominal_frames_for(config),
        ).to(device)
        model = adv_model
        adv_criterion = nn.CrossEntropyLoss()
    else:
        model = build_model(config, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, schedule, config.epochs)

    best_val = -1.0
    best_state: dict[str, Tensor] | None = None
    best_adv_acc: float | None = None
    epochs_no_improve = 0
    epochs_trained = 0
    early_stopped = False

    for epoch in range(1, config.epochs + 1):
        epochs_trained = epoch
        if adv_model is not None:
            assert adv_criterion is not None
            adv_model.grl.lambda_ = _grl_lambda(epoch, config)
            train_res = _run_epoch_adversarial(
                adv_model, train_loader, device, mean, std,
                optimizer=optimizer, adv_criterion=adv_criterion,
            )
        else:
            train_res = _run_epoch(
                model, train_loader, device, mean, std, optimizer=optimizer
            )
        msg = (
            f"    fold {test_fold} epoch {epoch:>3}/{config.epochs}  "
            f"loss={train_res.ce_loss:.4f}"
        )
        if train_res.vq_loss:
            msg += f" vq={train_res.vq_loss:.4f}"
        if train_res.code_counts is not None:
            # Per-epoch train-set perplexity makes codebook collapse visible as it
            # happens (out of _codebook_size(model) codes), not just at fold end.
            msg += f" ppl={_perplexity_from_counts(train_res.code_counts):.1f}"
        if train_res.adv_acc is not None:
            # Adversary sanity signal (NOT a privacy result): loss/accuracy + lambda.
            msg += (
                f" adv={train_res.adv_loss:.3f} advacc={train_res.adv_acc:.2f}"
                f" lam={_grl_lambda(epoch, config):.2f}"
            )

        val_f1: float | None = None
        if val_loader is not None:
            val_res = _run_epoch(model, val_loader, device, mean, std, optimizer=None)
            val_f1 = float(
                classification_metrics(val_res.y_true, val_res.y_pred, class_names)["macro_f1"]
            )
            msg += f"  val_f1={val_f1:.4f}"

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                assert val_f1 is not None
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if schedule.early_stopping:
            assert val_f1 is not None
            if val_f1 > best_val + schedule.min_delta:
                best_val = val_f1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_adv_acc = train_res.adv_acc  # adversary acc at the restored checkpoint
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print(msg)
            if epochs_no_improve >= schedule.patience:
                print(
                    f"    fold {test_fold} early stop at epoch {epoch} "
                    f"(best val_f1={best_val:.4f})"
                )
                early_stopped = True
                break
        else:
            print(msg)

    if schedule.early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    if save_checkpoint is not None:
        # Frozen-encoder bundle for the Phase-4 probes: weights + the norm stats
        # and config needed to reconstruct the model and re-emit codes.
        save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "norm_mean": mean.detach().cpu(),
                "norm_std": std.detach().cpu(),
                "config": config_to_dict(config),
                "class_names": list(class_names),
                "test_fold": test_fold,
            },
            save_checkpoint,
        )

    test_res = _run_epoch(model, test_loader, device, mean, std, optimizer=None)
    metrics = classification_metrics(test_res.y_true, test_res.y_pred, class_names)
    metrics["test_fold"] = test_fold
    metrics["norm_mean"] = mean.flatten().tolist()
    metrics["norm_std"] = std.flatten().tolist()
    metrics["val_fold"] = val_fold
    metrics["best_val_macro_f1"] = best_val if schedule.early_stopping else None
    metrics["epochs_trained"] = epochs_trained
    metrics["early_stopped"] = early_stopped
    metrics.update(_bottleneck_metrics(config, test_res.code_counts))
    if adversary_on:
        # Internal sanity signal only (NOT a privacy result): the training-time
        # adversary's speech-attribute accuracy at the restored checkpoint.
        metrics["adversary_train_acc"] = (
            best_adv_acc if schedule.early_stopping else train_res.adv_acc
        )
    return metrics


def _bottleneck_metrics(
    config: TrainConfig, code_counts: Tensor | None
) -> dict[str, Any]:
    """Bitrate accounting and codebook-usage stats for a VQ fold (empty if none).

    Bitrate is computed honestly from the *declared* latent grid and codebook
    size (``bits_per_second = tokens_per_second * log2(codebook_size)``).
    Perplexity and the fraction of codes used come from the code-usage histogram
    accumulated over the held-out test fold, so codebook collapse is reported as
    measured -- never massaged.
    """
    if config.model != "encoder_classifier" or config.bottleneck.type != "vq":
        return {}
    codebook_size = config.bottleneck.codebook_size
    tokens_per_clip = config.encoder.tokens_per_clip()
    tps = bitrate.tokens_per_second(tokens_per_clip, config.clip_seconds)
    out: dict[str, Any] = {
        "codebook_size": codebook_size,
        "tokens_per_clip": tokens_per_clip,
        "tokens_per_second": tps,
        "bits_per_token": bitrate.bits_per_token(codebook_size),
        "bits_per_second": bitrate.bits_per_second(tps, codebook_size),
    }
    if code_counts is not None:
        used = int((code_counts > 0).sum())
        out["codebook_perplexity"] = _perplexity_from_counts(code_counts)
        out["codebook_used"] = used
        out["codebook_fraction_used"] = used / codebook_size
    return out


def _operating_point(config: TrainConfig) -> dict[str, float | int]:
    """Resolved bitrate operating point for a VQ config (see edbr1.bitrate)."""
    return bitrate.OperatingPoint(
        latent_freq=config.encoder.latent_freq,
        latent_frames=config.encoder.latent_frames,
        codebook_size=config.bottleneck.codebook_size,
        clip_seconds=config.clip_seconds,
    ).as_dict()


def _run_prefix(config: TrainConfig) -> str:
    """Timestamped-run directory prefix reflecting the model/bottleneck."""
    if config.model != "encoder_classifier":
        return "us8k_baseline"
    if config.bottleneck.type == "vq":
        return "us8k_vq"
    return "us8k_encoder"


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, std


def _summarise_bottleneck(
    config: TrainConfig, fold_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run-level bitrate + codebook-usage summary for a VQ run (empty if none)."""
    if config.model != "encoder_classifier" or config.bottleneck.type != "vq":
        return {}
    op = _operating_point(config)
    ppl_mean, ppl_std = _mean_std(
        [float(m["codebook_perplexity"]) for m in fold_metrics if "codebook_perplexity" in m]
    )
    frac_mean, frac_std = _mean_std(
        [
            float(m["codebook_fraction_used"])
            for m in fold_metrics
            if "codebook_fraction_used" in m
        ]
    )
    return {
        **op,
        "codebook_perplexity_mean": ppl_mean,
        "codebook_perplexity_std": ppl_std,
        "codebook_fraction_used_mean": frac_mean,
        "codebook_fraction_used_std": frac_std,
    }


def _summarise_adversary(
    config: TrainConfig, fold_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Adversary provenance + mean training accuracy (empty if no adversary).

    The accuracy is the training-time adversary's own speech-attribute accuracy,
    an internal sanity signal only -- privacy is measured in Phase 4 by separate,
    stronger probes.
    """
    if not config.adversary.enabled:
        return {}
    accs = [
        float(m["adversary_train_acc"])
        for m in fold_metrics
        if m.get("adversary_train_acc") is not None
    ]
    if not accs:
        return {}
    mean, std = _mean_std(accs)
    return {
        "grl_lambda": config.adversary.grl_lambda,
        "warmup_epochs": config.adversary.warmup_epochs,
        "adversary_classes": config.overlay.num_speakers + 1,
        "num_speakers": config.overlay.num_speakers,
        "overlay_prob": config.overlay.overlay_prob,
        "snr_db": list(config.overlay.snr_db),
        "adversary_train_acc_mean": mean,
        "adversary_train_acc_std": std,
    }


def _build_overlay(
    config: TrainConfig, cache_dir: str | Path | None
) -> SpeechOverlay | None:
    """Build the train-only speech overlay once per run (``None`` if disabled).

    The closed speaker set is fold-independent, so the pool is decoded once (and
    disk-cached under ``cache_dir``) and reused across every fold.
    """
    if not config.overlay.enabled:
        return None
    ov = config.overlay
    pool = SpeechPool(
        ov.librispeech_root,
        subset=ov.subset,
        num_speakers=ov.num_speakers,
        segments_per_speaker=ov.segments_per_speaker,
        segment_seconds=config.clip_seconds,
        sample_rate=config.features.sample_rate,
        seed=ov.seed,
        cache_dir=cache_dir,
    )
    return SpeechOverlay(pool, overlay_prob=ov.overlay_prob, snr_choices=ov.snr_db)


def run_training(
    config: TrainConfig,
    *,
    root: Path,
    results_dir: Path,
    device: torch.device,
    test_folds: Sequence[int] | None = None,
    cache_dir: str | Path | None = None,
    save_checkpoints: bool = False,
) -> dict[str, Any]:
    """Run the CV protocol for ``config``, write its artifact dir, return the summary.

    The returned dict is exactly what is written to ``results.json`` plus a
    ``run_dir`` key pointing at the timestamped artifact directory. Shared by the
    CLI (:func:`main`) and the bitrate sweep runner so both produce identical
    per-run artifacts. ``cache_dir`` enables the (results-preserving) waveform
    cache.
    """
    folds = tuple(test_folds) if test_folds else config.test_folds
    metadata = load_metadata(root)
    class_names = URBANSOUND8K_CLASSES

    run_dir = results_dir / time.strftime(f"{_run_prefix(config)}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(config), fh, sort_keys=False)

    probe_model = build_model(config, num_classes=len(class_names))
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in probe_model.parameters()):,}")
    if isinstance(probe_model, EncoderClassifier):
        print(f"Encoder params (E, on-device): {probe_model.encoder_parameters():,}")
    if config.model == "encoder_classifier" and config.bottleneck.type == "vq":
        op = _operating_point(config)
        print(
            f"VQ operating point: {op['tokens_per_second']:.1f} tokens/s x "
            f"{op['bits_per_token']:.1f} bits/token = {op['bits_per_second']:.0f} bits/s "
            f"(codebook {config.bottleneck.codebook_size}, "
            f"grid {config.encoder.latent_freq}x{config.encoder.latent_frames})"
        )
    overlay = _build_overlay(config, cache_dir)
    if overlay is not None:
        print(
            f"Speech overlay: {config.overlay.num_speakers} speakers "
            f"({config.overlay.subset}), prob {config.overlay.overlay_prob}, "
            f"SNR {list(config.overlay.snr_db)} dB -> adversary {overlay.num_classes}-way"
        )
    if config.adversary.enabled:
        print(
            f"Adversary: GRL lambda {config.adversary.grl_lambda} "
            f"(warmup {config.adversary.warmup_epochs} ep) -- training-time only, "
            "NOT a Phase-4 privacy probe"
        )
    print(f"Evaluating folds: {folds}")

    fold_metrics: list[dict[str, Any]] = []
    for test_fold in folds:
        print(f"\n=== Fold {test_fold} ===")
        checkpoint = run_dir / f"encoder_fold{test_fold}.pt" if save_checkpoints else None
        metrics = train_one_fold(
            metadata, test_fold, config, device, class_names, cache_dir, overlay,
            save_checkpoint=checkpoint,
        )
        fold_metrics.append(metrics)
        print(f"  fold {test_fold} macro-F1 = {metrics['macro_f1']:.4f}")
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            run_dir / f"confusion_fold{test_fold}.png",
            title=f"UrbanSound8K fold {test_fold}",
        )

    macro_f1s = [float(m["macro_f1"]) for m in fold_metrics]
    mean_f1 = sum(macro_f1s) / len(macro_f1s)
    std_f1 = (sum((f - mean_f1) ** 2 for f in macro_f1s) / len(macro_f1s)) ** 0.5
    spread = (max(macro_f1s) - min(macro_f1s)) if macro_f1s else 0.0

    regularisers: list[str] = []
    if config.augment.enabled:
        regularisers.append("augmentation")
    if config.norm == "per_band":
        regularisers.append("per-band norm")
    if config.schedule.scheduler != "none":
        regularisers.append(f"{config.schedule.scheduler} LR")
    if config.schedule.early_stopping:
        regularisers.append("early stopping")
    reg_label = ", ".join(regularisers) if regularisers else "none (plain baseline)"

    summary: dict[str, Any] = {
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "spread_macro_f1": spread,
        "active_regularisers": regularisers,
        "per_fold_macro_f1": {
            int(m["test_fold"]): float(m["macro_f1"]) for m in fold_metrics
        },
        "folds": fold_metrics,
        "config": config_to_dict(config),
    }
    bottleneck_summary = _summarise_bottleneck(config, fold_metrics)
    if bottleneck_summary:
        summary["bottleneck"] = bottleneck_summary
    adversary_summary = _summarise_adversary(config, fold_metrics)
    if adversary_summary:
        summary["adversary"] = adversary_summary
    with (run_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"Active regularisers: {reg_label}")
    print("Per-fold macro-F1:")
    for m in fold_metrics:
        line = f"  fold {int(m['test_fold']):>2}: {float(m['macro_f1']):.4f}"
        if m.get("early_stopped"):
            line += f"  (early stop @ epoch {m['epochs_trained']})"
        print(line)
    print(
        f"Mean macro-F1 over {len(macro_f1s)} fold(s): {mean_f1:.4f} "
        f"(+/- {std_f1:.4f}); per-fold spread {spread:.4f}"
    )
    if bottleneck_summary:
        print(
            f"VQ bitrate: {bottleneck_summary['bits_per_second']:.0f} bits/s "
            f"({bottleneck_summary['tokens_per_second']:.1f} tokens/s x "
            f"{bottleneck_summary['bits_per_token']:.1f} bits/token)"
        )
        print(
            "Codebook usage: perplexity "
            f"{bottleneck_summary['codebook_perplexity_mean']:.1f}"
            f" (+/- {bottleneck_summary['codebook_perplexity_std']:.1f}) of "
            f"{bottleneck_summary['codebook_size']}; fraction used "
            f"{bottleneck_summary['codebook_fraction_used_mean']:.3f}"
        )
    if adversary_summary:
        print(
            "Adversary (SANITY ONLY, not a privacy result): train speech-attribute "
            f"acc {adversary_summary['adversary_train_acc_mean']:.3f} "
            f"(+/- {adversary_summary['adversary_train_acc_std']:.3f}) of "
            f"{adversary_summary['adversary_classes']}-way at lambda "
            f"{adversary_summary['grl_lambda']}"
        )
    print("Published small-CNN reference band: ~0.73-0.76 macro-F1")
    # The below-band advice targets the un-bottlenecked baseline/control. Under a
    # VQ bottleneck a sub-band score can be the honest cost of a low bitrate, not
    # a tuning failure, so the note is suppressed there.
    is_vq = config.model == "encoder_classifier" and config.bottleneck.type == "vq"
    if mean_f1 < 0.73 and not is_vq:
        print(
            "NOTE: still below the published band. If normalisation, clip "
            "length and fold-split integrity are confirmed, the next levers "
            "(without touching the test fold) are: stronger/weaker SpecAugment, "
            "22.05 kHz sample rate, or longer patience. Do NOT tune against the "
            "test folds."
        )
    print(f"Results written to: {run_dir}")
    summary["run_dir"] = str(run_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K",
        help="Extracted UrbanSound8K directory (contains metadata/ and audio/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
        help="Training config YAML.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root for run outputs (gitignored).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument(
        "--test-folds",
        type=int,
        nargs="+",
        default=None,
        help="Subset of folds to evaluate (default: all from config).",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument(
        "--wav-cache",
        type=Path,
        default=None,
        help="Directory for the on-disk resampled-waveform cache (decode once, "
        "reuse across epochs/folds; results are bit-identical). Off by default.",
    )
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if args.epochs is not None:
        config = dataclasses.replace(config, epochs=args.epochs)

    device = _pick_device(args.device)
    run_training(
        config,
        root=args.root,
        results_dir=args.results_dir,
        device=device,
        test_folds=args.test_folds,
        cache_dir=args.wav_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
