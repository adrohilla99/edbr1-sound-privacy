"""Model architectures for EDBR.1."""
from __future__ import annotations

from edbr1.models.adversary import (
    AdversarialEncoderClassifier,
    AdversaryHead,
    GradientReversalLayer,
    gradient_reversal,
)
from edbr1.models.bottleneck import (
    BottleneckOutput,
    IdentityBottleneck,
    VectorQuantizer,
    build_bottleneck,
)
from edbr1.models.classifier import LatentClassifier
from edbr1.models.cnn import SmallAudioCNN
from edbr1.models.encoder import AudioEncoder
from edbr1.models.encoder_classifier import (
    EncoderClassifier,
    build_model,
    nominal_frames_for,
)

__all__ = [
    "SmallAudioCNN",
    "AudioEncoder",
    "LatentClassifier",
    "IdentityBottleneck",
    "VectorQuantizer",
    "BottleneckOutput",
    "build_bottleneck",
    "EncoderClassifier",
    "build_model",
    "nominal_frames_for",
    "AdversarialEncoderClassifier",
    "AdversaryHead",
    "GradientReversalLayer",
    "gradient_reversal",
]
