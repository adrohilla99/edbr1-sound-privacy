"""
Log-mel spectrogram feature extractor.

A thin, config-driven wrapper around torchaudio transforms:

    waveform (any sample rate, mono/multi-channel)
        -> downmix to mono
        -> resample to FeatureConfig.sample_rate
        -> power mel-spectrogram (n_mels bands, win/hop from config)
        -> amplitude-to-dB (log compression)
        -> (n_mels, n_frames) float tensor

All numeric parameters come from :class:`edbr1.config.FeatureConfig`; there
are no magic numbers in this module.
"""
from __future__ import annotations

import torch
import torchaudio.transforms as T
from torch import Tensor, nn

from edbr1.config import FeatureConfig


class LogMelExtractor(nn.Module):
    """Turn a raw waveform into a log-mel spectrogram.

    Implemented as an ``nn.Module`` so it can live on a device and, if ever
    wanted, be folded into a model graph. The resampler is created lazily
    per input sample rate and cached, because UrbanSound8K clips arrive at a
    mix of native sample rates.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        super().__init__()
        self.config = config or FeatureConfig()
        self.mel = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.resolved_n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.resolved_f_max,
            n_mels=self.config.n_mels,
            power=self.config.power,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=self.config.top_db)
        self._resamplers: dict[int, T.Resample] = {}

    def _resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(
                orig_freq=orig_sr, new_freq=self.config.sample_rate
            )
        return self._resamplers[orig_sr]

    @staticmethod
    def _to_mono(waveform: Tensor) -> Tensor:
        """Collapse a (channels, samples) or (samples,) tensor to (samples,)."""
        if waveform.dim() == 1:
            return waveform
        if waveform.dim() == 2:
            return waveform.mean(dim=0)
        raise ValueError(
            f"Expected waveform of shape (samples,) or (channels, samples); "
            f"got shape {tuple(waveform.shape)}"
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Return a (n_mels, n_frames) log-mel spectrogram for ``waveform``.

        ``waveform`` may be 1-D (mono) or 2-D (channels, samples); it is
        downmixed to mono and resampled to the configured rate first.
        """
        wav = self._to_mono(waveform.to(torch.float32))
        if sample_rate != self.config.sample_rate:
            wav = self._resampler(sample_rate)(wav)
        mel = self.mel(wav)  # (n_mels, n_frames)
        return self.to_db(mel)

    # Convenience alias so callers can write extractor(wav, sr) -- nn.Module
    # __call__ already routes to forward(); this is just for type clarity.
    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        return super().__call__(waveform, sample_rate)
