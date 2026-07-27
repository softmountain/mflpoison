import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EVAL_MODULES = {
    "teacher_guided": "fed_multimodal.legacy_evaluation.teacher_guided",
    "legacy": "fed_multimodal.legacy_evaluation.kplus1",
    "kplus1_legacy": "fed_multimodal.legacy_evaluation.kplus1",
    "temporal_adaptive": "fed_multimodal.legacy_evaluation.temporal_adaptive",
    "dtm": "fed_multimodal.legacy_evaluation.dtm",
}

# Historical import name retained for external experiment launchers.
EVAL_SCRIPTS = EVAL_MODULES


def dispatch(module, arguments):
    command = [sys.executable, "-m", module] + list(arguments)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_path if existing_path else ""
    )
    return subprocess.call(command, cwd=str(ROOT), env=environment)
