import importlib.util
import unittest
from unittest import mock

from experiments._dispatch import EVAL_MODULES, EVAL_SCRIPTS, dispatch

class ExperimentConfigTest(unittest.TestCase):
    def test_legacy_evaluator_dispatch_targets_are_retained(self):
        self.assertEqual(
            set(EVAL_SCRIPTS),
            {"teacher_guided", "legacy", "kplus1_legacy", "temporal_adaptive", "dtm"},
        )
        self.assertIs(EVAL_SCRIPTS, EVAL_MODULES)
        for module in EVAL_MODULES.values():
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(module))

    @mock.patch("experiments._dispatch.subprocess.call", return_value=0)
    def test_dispatch_uses_python_module(self, call):
        self.assertEqual(dispatch("package.module", ["--help"]), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[1:], ["-m", "package.module", "--help"])


if __name__ == "__main__":
    unittest.main()
