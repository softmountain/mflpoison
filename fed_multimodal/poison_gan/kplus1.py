import torch
import torch.nn as nn

from fed_multimodal.model.mm_models import MMActionClassifier


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _checkpoint_args(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {}) or {}
    return {}


def load_teacher_model(
    model_path,
    num_classes,
    audio_input_dim=80,
    video_input_dim=1280,
    hid_size=128,
    att=False,
    att_name="",
    device="cpu",
):
    checkpoint = torch.load(model_path, map_location=device)
    saved_args = _checkpoint_args(checkpoint)
    model = MMActionClassifier(
        num_classes=num_classes,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        d_hid=saved_args.get("hid_size", hid_size),
        en_att=saved_args.get("att", saved_args.get("en_att", att)),
        att_name=saved_args.get("att_name", att_name),
    ).to(device)
    model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    return model, checkpoint


def build_kplus1_discriminator(
    model_path,
    num_classes,
    audio_input_dim=80,
    video_input_dim=1280,
    hid_size=128,
    att=False,
    att_name="",
    freeze="none",
    device="cpu",
):
    teacher, checkpoint = load_teacher_model(
        model_path=model_path,
        num_classes=num_classes,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        hid_size=hid_size,
        att=att,
        att_name=att_name,
        device=device,
    )
    saved_args = _checkpoint_args(checkpoint)
    model = MMActionClassifier(
        num_classes=num_classes + 1,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        d_hid=saved_args.get("hid_size", hid_size),
        en_att=saved_args.get("att", saved_args.get("en_att", att)),
        att_name=saved_args.get("att_name", att_name),
    ).to(device)

    teacher_state = teacher.state_dict()
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in teacher_state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)

    old_head = teacher.classifier[-1]
    new_head = model.classifier[-1]
    with torch.no_grad():
        new_head.weight[:num_classes].copy_(old_head.weight)
        new_head.bias[:num_classes].copy_(old_head.bias)
        new_head.weight[num_classes].copy_(old_head.weight.mean(dim=0) + 0.01 * torch.randn_like(old_head.weight[0]))
        new_head.bias[num_classes].copy_(old_head.bias.mean())

    apply_freeze(model, freeze)
    return model, checkpoint


def apply_freeze(model, freeze):
    freeze = (freeze or "none").lower()
    if freeze == "none":
        for p in model.parameters():
            p.requires_grad = True
    elif freeze == "backbone":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif freeze == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze mode: {freeze}")
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
