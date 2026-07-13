"""
生成器模块

本模块包含用于联邦学习的数据生成和攻击实现：

- gan_generator: 多模态特征 GAN 生成器
- eval_gan_quality: GAN 特征质量评估工具
- label_flip_attack: 标签翻转攻击实现
"""

from .gan_generator import MultimodalFeatureGAN, FeatureGANConfig
from .gan_generator import AudioFeatureGenerator, VideoFeatureGenerator
from .gan_generator import AudioFeatureDiscriminator, VideoFeatureDiscriminator
from .label_flip_attack import UCILabelFlipAttack

__all__ = [
    'MultimodalFeatureGAN',
    'FeatureGANConfig', 
    'AudioFeatureGenerator',
    'VideoFeatureGenerator',
    'AudioFeatureDiscriminator',
    'VideoFeatureDiscriminator',
    'UCILabelFlipAttack',
]
