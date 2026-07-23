import unittest
from pathlib import Path

from experiments._dispatch import EVAL_SCRIPTS

ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTest(unittest.TestCase):
    def test_legacy_evaluator_dispatch_targets_are_retained(self):
        self.assertEqual(
            set(EVAL_SCRIPTS),
            {"teacher_guided", "legacy", "kplus1_legacy", "temporal_adaptive", "dtm"},
        )
        for script in EVAL_SCRIPTS.values():
            with self.subTest(script=script):
                self.assertTrue((ROOT / script).is_file())


if __name__ == "__main__":
    unittest.main()
