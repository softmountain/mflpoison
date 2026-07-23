import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


EVAL_SCRIPTS = {
    "teacher_guided": "fed_multimodal/Local/eval_local_gan_quality.py",
    "legacy": "fed_multimodal/Local/eval_poison_gan.py",
    "kplus1_legacy": "fed_multimodal/Local/eval_poison_gan.py",
    "temporal_adaptive": "fed_multimodal/Local/eval_temporal_adaptive_gan.py",
    "dtm": "fed_multimodal/Local/eval_dtm_poison_gan.py",
}


def dispatch(script, arguments):
    command = [sys.executable, str(ROOT / script)] + list(arguments)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_path if existing_path else ""
    )
    return subprocess.call(command, cwd=str(ROOT), env=environment)
