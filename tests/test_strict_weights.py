from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch
from safetensors.torch import save_file
from task2_pi05_eval_node import strict_pi05_load, Task2ActionSafetyFilter, RIGHT_GRIPPER_DRIVER, LEFT_JOINTS, RIGHT_JOINTS


class SmallPolicy(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)

    def _fix_pytorch_state_dict_keys(self, state, cfg):
        return state


class StrictWeightsTest(unittest.TestCase):
    def test_loaded_values_and_missing_shape_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            weights = {"model.weight": torch.full((2, 2), 3.), "model.bias": torch.full((2,), 4.)}
            save_file(weights, path)
            cfg = SimpleNamespace(device="cpu")
            policy = strict_pi05_load(SmallPolicy, cfg, path)
            self.assertTrue(torch.equal(policy.model(torch.ones(2)), torch.full((2,), 10.)))
            self.assertEqual(cfg.device, "cpu")
            save_file({"model.weight": weights["model.weight"]}, path)
            with self.assertRaisesRegex(RuntimeError, "keys differ"):
                strict_pi05_load(SmallPolicy, cfg, path)
            save_file({**weights, "model.weight": torch.zeros(3, 2)}, path)
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                strict_pi05_load(SmallPolicy, cfg, path)

    def test_measured_gripper_hold_converts_angle_without_changing_state(self):
        joints = {name: 0 for name in LEFT_JOINTS + RIGHT_JOINTS}
        joints[RIGHT_GRIPPER_DRIVER] = .789
        filt = Task2ActionSafetyFilter(spine_min=434, spine_max=434, contract="real42_17")
        self.assertAlmostEqual(filt._measured_action(joints)[15], 0)
        joints[RIGHT_GRIPPER_DRIVER] = 0
        self.assertAlmostEqual(filt._measured_action(joints)[15], 1)


if __name__ == "__main__":
    unittest.main()
