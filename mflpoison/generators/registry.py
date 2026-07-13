from mflpoison.core.registry import Registry

from .kplus1 import DTMBackend, LegacyKPlusOneBackend, TemporalAdaptiveBackend
from .teacher_guided import TeacherGuidedBackend


GENERATOR_REGISTRY = Registry("generator backend")
GENERATOR_REGISTRY.register("teacher_guided", TeacherGuidedBackend)
GENERATOR_REGISTRY.register("legacy", LegacyKPlusOneBackend)
GENERATOR_REGISTRY.register("kplus1_legacy", LegacyKPlusOneBackend)
GENERATOR_REGISTRY.register("temporal_adaptive", TemporalAdaptiveBackend)
GENERATOR_REGISTRY.register("dtm", DTMBackend)


def load_generator_backend(name, checkpoint_path, device="cpu"):
    return GENERATOR_REGISTRY.create(
        name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
