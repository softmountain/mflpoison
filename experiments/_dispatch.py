import json
import os
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mflpoison.core.config import load_config


EVAL_SCRIPTS = {
    "teacher_guided": "fed_multimodal/Local/eval_local_gan_quality.py",
    "legacy": "fed_multimodal/Local/eval_poison_gan.py",
    "kplus1_legacy": "fed_multimodal/Local/eval_poison_gan.py",
    "temporal_adaptive": "fed_multimodal/Local/eval_temporal_adaptive_gan.py",
    "dtm": "fed_multimodal/Local/eval_dtm_poison_gan.py",
}


def load_experiment_config(path):
    if path is None:
        return {}
    config = load_config(path)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a mapping")
    return config


def config_to_cli_arguments(config):
    """Flatten supported config sections into legacy argparse flags."""
    values = {}
    for key, value in config.items():
        if key in {"generator", "dataset", "name", "description"}:
            continue
        if isinstance(value, dict):
            values.update(value)
        else:
            values[key] = value

    arguments = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = "--" + key
        if value is True:
            arguments.append(flag)
        elif isinstance(value, (dict, list)):
            arguments.extend([flag, json.dumps(value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def dispatch(script, arguments):
    command = [sys.executable, str(ROOT / script)] + list(arguments)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_path if existing_path else ""
    )
    return subprocess.call(command, cwd=str(ROOT), env=environment)
