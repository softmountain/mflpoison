"""Build the production runner from strict scenario configuration."""

from pathlib import Path
from typing import Mapping, Optional

import torch

from mflpoison.attacks import AttackSpec
from mflpoison.core.config import ScenarioConfig
from mflpoison.core.types import ModelSpec

from .scenario import ScenarioRunner


def _checkpoint_state_from_payload(payload) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("model checkpoint must contain a mapping")
    for key in ("model_state_dict", "state_dict", "state", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("model checkpoint does not contain a model state")


def build_default_runner(
    config: ScenarioConfig,
    *,
    artifact_dir: Optional[Path] = None,
) -> ScenarioRunner:
    """Build the production UCF101/FedMM scenario from strict configuration."""

    if config.dataset.name.lower() != "ucf101":
        raise ValueError("the first scenario runner release supports only UCF101")
    if config.model.name != "MMActionClassifier":
        raise ValueError(
            "the first scenario runner release supports only MMActionClassifier"
        )
    supported_constructor = "fed_multimodal.model.mm_models:MMActionClassifier"
    if config.model.constructor not in {None, supported_constructor}:
        raise ValueError(
            "unsupported model.constructor: " + str(config.model.constructor)
        )
    if not config.dataset.root:
        raise ValueError("dataset.root is required for the UCF101 adapter")
    if config.dataset.partition_path is not None:
        raise ValueError(
            "dataset.partition_path is not supported; UCF101 uses FedMM paths under dataset.root"
        )
    generator_family = config.generator.family.lower()
    generator_variant = config.generator.variant.lower()
    if generator_family != "kplus1":
        raise ValueError("generative poisoning currently supports generator.family=kplus1")
    if generator_variant not in {"dtm", "temporal_adaptive"}:
        raise ValueError("generator.variant must be dtm or temporal_adaptive")
    if config.attack.strategy != "generative_feature_poisoning":
        raise ValueError("only generative_feature_poisoning is implemented")
    if config.defense.policy != "reject_if_both_clip_if_one":
        raise ValueError("unsupported defense.policy: " + config.defense.policy)
    allowed_metrics = {
        "accuracy",
        "acc",
        "uar",
        "f1",
        "loss",
        "top5_acc",
        "attack_success_rate",
    }
    unknown_metrics = sorted(set(config.evaluation.metrics) - allowed_metrics)
    if unknown_metrics:
        raise ValueError(
            "unsupported evaluation metric(s): " + ", ".join(unknown_metrics)
        )
    if config.evaluation.options:
        raise ValueError("evaluation.options are not implemented in the first release")
    allowed_dataset_options = {
        "missing_modality",
        "missing_modality_rate",
        "missing_label",
        "missing_label_rate",
        "label_noisy",
        "label_noise_level",
        "audio_seq_len",
        "video_seq_len",
    }
    unknown_dataset_options = sorted(
        set(config.dataset.options) - allowed_dataset_options
    )
    if unknown_dataset_options:
        raise ValueError(
            "unsupported dataset.options: " + ", ".join(unknown_dataset_options)
        )

    allowed_federation_options = {"device", "mu"}
    unknown_federation_options = sorted(
        set(config.federation.options) - allowed_federation_options
    )
    if unknown_federation_options:
        raise ValueError(
            "unsupported federation.options: "
            + ", ".join(unknown_federation_options)
        )
    if config.model.options:
        raise ValueError("model.options are not implemented")
    allowed_model_kwargs = {
        "hid_size",
        "attention",
        "attention_name",
        "att",
        "att_name",
    }
    unknown_model_kwargs = sorted(set(config.model.kwargs) - allowed_model_kwargs)
    if unknown_model_kwargs:
        raise ValueError(
            "unsupported model.kwargs: " + ", ".join(unknown_model_kwargs)
        )
    if (
        "attention" in config.model.kwargs
        and "att" in config.model.kwargs
        and bool(config.model.kwargs["attention"])
        != bool(config.model.kwargs["att"])
    ):
        raise ValueError("model attention and att aliases conflict")
    if (
        "attention_name" in config.model.kwargs
        and "att_name" in config.model.kwargs
        and str(config.model.kwargs["attention_name"])
        != str(config.model.kwargs["att_name"])
    ):
        raise ValueError("model attention_name and att_name aliases conflict")

    allowed_attack_options = {"generation_batch_size"}
    unknown_attack_options = sorted(
        set(config.attack.options) - allowed_attack_options
    )
    if unknown_attack_options:
        raise ValueError(
            "unsupported attack.options: " + ", ".join(unknown_attack_options)
        )
    allowed_defense_options = {"minimum_reputation", "initial_reputation"}
    unknown_defense_options = sorted(
        set(config.defense.options) - allowed_defense_options
    )
    if unknown_defense_options:
        raise ValueError(
            "unsupported defense.options: " + ", ".join(unknown_defense_options)
        )
    for field_name in (
        "condition_class",
        "assigned_train_label",
        "victim_eval_class",
        "goal_prediction_class",
    ):
        label = getattr(config.attack, field_name)
        if label is not None and int(label) >= int(config.dataset.num_classes):
            raise ValueError(
                f"attack.{field_name} must be in [0, "
                f"{int(config.dataset.num_classes) - 1}]"
            )
    if config.attack.enabled and not config.generator.enabled:
        raise ValueError("generative poisoning requires generator.enabled=true")

    generator_options = dict(config.generator.options)
    max_batches = generator_options.pop("max_batches", None)
    log_interval = int(generator_options.pop("log_interval", 0))
    generator_overrides = dict(config.generator.loss)
    conflicts = sorted(set(generator_overrides) & set(generator_options))
    if conflicts:
        raise ValueError(
            "generator.loss and generator.options overlap: " + ", ".join(conflicts)
        )
    generator_overrides.update(generator_options)
    protected_generator_fields = {
        "num_classes",
        "fake_class",
        "audio_seq_len",
        "audio_feat_dim",
        "video_seq_len",
        "video_feat_dim",
        "seed",
        "lr_g",
        "lr_d",
        "g_steps",
        "d_steps",
    }
    protected_overrides = sorted(
        set(generator_overrides) & protected_generator_fields
    )
    if protected_overrides:
        raise ValueError(
            "generator options cannot override scenario-owned field(s): "
            + ", ".join(protected_overrides)
        )
    generator_overrides["lr_g"] = config.generator.learning_rate
    generator_overrides["lr_d"] = (
        config.generator.learning_rate
        if config.generator.discriminator_learning_rate is None
        else config.generator.discriminator_learning_rate
    )
    generator_overrides["g_steps"] = config.generator.generator_steps_per_batch
    generator_overrides["d_steps"] = (
        config.generator.discriminator_steps_per_batch
    )
    if generator_variant == "dtm":
        from fed_multimodal.dtm_poison_gan import DTMGANConfig

        allowed_generator_fields = set(DTMGANConfig.__dataclass_fields__)
    else:
        from fed_multimodal.temporal_adaptive_gan import TemporalAdaptiveGANConfig

        allowed_generator_fields = set(TemporalAdaptiveGANConfig.__dataclass_fields__)
    unknown_generator_options = sorted(
        set(generator_overrides) - allowed_generator_fields
    )
    if unknown_generator_options:
        raise ValueError(
            "unsupported generator option(s): "
            + ", ".join(unknown_generator_options)
        )

    # Lazy imports keep lightweight config/help commands independent of the
    # legacy FedMM modules until a concrete runner is built.
    from mflpoison.adapters.fedmm import (
        FedAvgClientTrainer,
        FedMMGeneratorTrainer,
        UCF101FedMMAdapter,
    )
    from mflpoison.attacks import GenerativeFeaturePoisoningStrategy
    from mflpoison.defenses import (
        CosineMADDetector,
        DefensePipeline,
        EWMAReputation,
        NormMADDetector,
    )
    from mflpoison.defenses.registry import AGGREGATOR_REGISTRY
    from mflpoison.defenses.update_filter import NormClipper
    from mflpoison.generators import GeneratorLifecycleManager, load_generator_backend

    model_kwargs = dict(config.model.kwargs)
    adapter = UCF101FedMMAdapter(
        data_dir=config.dataset.root,
        alpha=1.0 if config.dataset.alpha is None else config.dataset.alpha,
        fold=config.dataset.fold,
        batch_size=config.federation.batch_size,
        hid_size=int(model_kwargs.get("hid_size", 64)),
        attention=bool(model_kwargs.get("attention", model_kwargs.get("att", False))),
        attention_name=str(
            model_kwargs.get("attention_name", model_kwargs.get("att_name", "base"))
        ),
        **dict(config.dataset.options),
    )
    if int(config.dataset.num_classes) != int(adapter.num_classes):
        raise ValueError(
            "configured dataset.num_classes does not match UCF101: "
            f"{config.dataset.num_classes} != {adapter.num_classes}"
        )
    device = str(config.federation.options.get("device", "cpu"))
    client_trainer = FedAvgClientTrainer(
        adapter.build_model,
        device=device,
        learning_rate=config.federation.learning_rate,
        local_epochs=config.federation.local_epochs,
        mu=float(config.federation.options.get("mu", 0.0)),
        seed=config.federation.seed,
    )
    clean_aggregator = AGGREGATOR_REGISTRY.create("weighted_mean")
    artifact_root = Path(
        config.artifact.root_dir if artifact_dir is None else artifact_dir
    )

    lifecycle_factory = None
    attack_strategy = None
    if config.attack.enabled:
        if config.generator.checkpoint_dir is None:
            checkpoint_root = artifact_root / "checkpoints" / "generators"
        else:
            checkpoint_root = Path(config.generator.checkpoint_dir)
            if not checkpoint_root.is_absolute():
                checkpoint_root = artifact_root / checkpoint_root

        def lifecycle_factory(phase: str):
            output_dir = checkpoint_root / phase

            def trainer_factory(client_id: str):
                del client_id
                trainer = FedMMGeneratorTrainer(
                    variant=generator_variant,
                    output_dir=output_dir,
                    model_metadata=adapter.model_metadata(),
                    modality_shapes=adapter.modality_shapes,
                    num_classes=adapter.num_classes,
                    epochs=config.generator.epochs,
                    max_batches=max_batches,
                    log_interval=log_interval,
                    device=device,
                    config_overrides=generator_overrides,
                    batch_size=config.generator.batch_size,
                )
                return trainer

            return GeneratorLifecycleManager(
                trainer_factory=trainer_factory,
                variant=generator_variant,
                mode=config.generator.lifecycle,
                refresh_every=config.generator.refresh_interval,
                seed=config.generator.seed,
            )

        attack_spec = AttackSpec(
            condition_class=config.attack.condition_class,
            assigned_train_label=config.attack.assigned_train_label,
            victim_eval_class=config.attack.victim_eval_class,
            goal_prediction_class=config.attack.goal_prediction_class,
            poison_ratio=config.attack.poison_ratio,
            poison_count=config.attack.poison_count,
            injection_mode=config.attack.injection_mode,
            start_round=config.attack.start_round,
            end_round=config.attack.end_round,
            every=config.attack.every,
            seed=config.generator.seed,
            metadata=dict(config.attack.options),
        )
        attack_options = dict(config.attack.options)
        generation_batch_size = int(attack_options.pop("generation_batch_size", 64))
        attack_strategy = GenerativeFeaturePoisoningStrategy(
            attack_spec,
            seed=config.generator.seed,
            generation_batch_size=generation_batch_size,
            backend_factory=lambda artifact: load_generator_backend(
                artifact.variant,
                artifact.checkpoint_path,
                device=device,
            ),
        )

    defense_pipeline = None
    if config.defense.enabled:
        detector_specs = config.defense.detectors or (
            {"name": "norm_mad"},
            {"name": "cosine_mad"},
        )
        detectors = []
        detector_types = {
            "norm_mad": NormMADDetector,
            "cosine_mad": CosineMADDetector,
            "cosine_center": CosineMADDetector,
        }
        for detector_spec in detector_specs:
            values = dict(detector_spec)
            name = str(values.pop("name")).lower()
            try:
                detector_type = detector_types[name]
            except KeyError as exc:
                raise KeyError("unknown defense detector: " + name) from exc
            detectors.append(detector_type(**values))
        sanitizer_values = dict(config.defense.sanitizer)
        sanitizer_name = str(sanitizer_values.pop("name", "norm_clipping"))
        if sanitizer_name != "norm_clipping":
            raise KeyError("unknown defense sanitizer: " + sanitizer_name)
        sanitizer = NormClipper(**{"max_norm": None, **sanitizer_values})
        aggregator_values = dict(config.defense.aggregator)
        aggregator_name = str(aggregator_values.pop("name", "weighted_mean"))
        defended_aggregator = AGGREGATOR_REGISTRY.create(
            aggregator_name, **aggregator_values
        )
        reputation = None
        if config.defense.ewma_decay is not None:
            reputation = EWMAReputation(
                decay=config.defense.ewma_decay,
                minimum_reputation=float(
                    config.defense.options.get("minimum_reputation", 0.5)
                ),
                initial_reputation=float(
                    config.defense.options.get("initial_reputation", 1.0)
                ),
            )
        defense_pipeline = DefensePipeline(
            detectors=detectors,
            sanitizer=sanitizer,
            reputation=reputation,
            aggregator=defended_aggregator,
        )

    initial_state = None
    if config.model.checkpoint_path:
        checkpoint_path = Path(config.model.checkpoint_path)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        initial_state = _checkpoint_state_from_payload(checkpoint_payload)
        legacy_args = (
            dict(checkpoint_payload.get("args", {}))
            if isinstance(checkpoint_payload, Mapping)
            and isinstance(checkpoint_payload.get("args", {}), Mapping)
            else {}
        )
        expected_legacy_args = {
            "hid_size": adapter.hid_size,
            "att": adapter.attention,
            "att_name": adapter.attention_name,
        }
        for name, expected in expected_legacy_args.items():
            if name in legacy_args and legacy_args[name] != expected:
                raise ValueError(
                    f"legacy checkpoint {name} does not match model configuration"
                )
    return ScenarioRunner(
        config,
        adapter=adapter,
        client_trainer=client_trainer,
        aggregator=clean_aggregator,
        initial_state=initial_state,
        model_spec=ModelSpec(
            name=config.model.name,
            constructor=config.model.constructor,
            kwargs=dict(config.model.kwargs),
            metadata=adapter.model_metadata(),
        ),
        generator_lifecycle_factory=lifecycle_factory,
        attack_strategy=attack_strategy,
        defense_pipeline=defense_pipeline,
        artifact_dir=artifact_root,
    )
