from collections.abc import Mapping
from pathlib import Path

import torch


_SCHEMAS = {
    "teacher_guided": {
        "required": ("audio_generator", "video_generator"),
        "gan_type": None,
    },
    "kplus1_legacy": {
        "required": (
            "config",
            "generator_state_dict",
            "discriminator_state_dict",
        ),
        "gan_type": None,
    },
    "dtm": {
        "required": (
            "config",
            "generator_state_dict",
            "discriminator_state_dict",
        ),
        "gan_type": "dtm_gan",
    },
    "temporal_adaptive": {
        "required": (
            "config",
            "generator_state_dict",
            "discriminator_state_dict",
        ),
        "gan_type": None,
    },
}


def validate_legacy_checkpoint(checkpoint, variant):
    if variant not in _SCHEMAS:
        raise ValueError(f"unknown legacy checkpoint variant: {variant}")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("legacy checkpoint must contain a mapping")
    schema = _SCHEMAS[variant]
    missing = [key for key in schema["required"] if key not in checkpoint]
    if missing:
        raise ValueError(
            f"{variant} checkpoint is missing keys: {', '.join(missing)}"
        )
    if "config" in checkpoint and not isinstance(checkpoint["config"], Mapping):
        raise TypeError(f"{variant} checkpoint config must be a mapping")
    expected_gan_type = schema["gan_type"]
    if (
        expected_gan_type is not None
        and checkpoint.get("gan_type") != expected_gan_type
    ):
        raise ValueError(
            f"{variant} checkpoint gan_type must be {expected_gan_type!r}"
        )
    state_keys = [key for key in schema["required"] if key != "config"]
    for key in state_keys:
        if not isinstance(checkpoint[key], Mapping):
            raise TypeError(f"checkpoint field {key} must be a state mapping")
        if not checkpoint[key]:
            raise ValueError(f"checkpoint field {key} cannot be empty")
    if "joint_discriminator" in checkpoint and not isinstance(
        checkpoint["joint_discriminator"], Mapping
    ):
        raise TypeError("checkpoint field joint_discriminator must be a state mapping")
    if (
        "joint_discriminator" in checkpoint
        and not checkpoint["joint_discriminator"]
    ):
        raise ValueError("checkpoint field joint_discriminator cannot be empty")
    return checkpoint


def validate_module_state(module, state, field_name):
    """Reject partial legacy states that would leave random model parameters."""
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if expected[key].shape != state[key].shape
        or expected[key].dtype != state[key].dtype
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if mismatched:
            details.append("shape_or_dtype=" + ",".join(mismatched))
        raise ValueError(
            f"{field_name} is incompatible with the evaluator model: "
            + "; ".join(details)
        )
    return state


def load_legacy_checkpoint(path, variant, map_location="cpu"):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"legacy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return validate_legacy_checkpoint(checkpoint, variant)
