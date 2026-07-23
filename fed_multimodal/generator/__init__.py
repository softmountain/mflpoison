"""Legacy teacher-guided feature generator checkpoint compatibility."""

from .gan_generator import MultimodalFeatureGAN, FeatureGANConfig
from .gan_generator import AudioFeatureGenerator, VideoFeatureGenerator
from .gan_generator import AudioFeatureDiscriminator, VideoFeatureDiscriminator

__all__ = [
    "MultimodalFeatureGAN",
    "FeatureGANConfig",
    "AudioFeatureGenerator",
    "VideoFeatureGenerator",
    "AudioFeatureDiscriminator",
    "VideoFeatureDiscriminator",
]
