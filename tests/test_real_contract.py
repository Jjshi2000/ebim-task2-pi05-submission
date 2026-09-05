import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "app" / "task2_pi05_eval_node.py"
SPEC = importlib.util.spec_from_file_location("task2_pi05_eval_node", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RealContractTest(unittest.TestCase):
    def test_bgra_camera_stride_preserves_rgb_colors(self):
        packet = dict(height=2, width=1, encoding="bgra8", step=8,
                      data=bytes([0, 0, 255, 255, 77, 77, 77, 77,
                                  255, 0, 0, 255, 77, 77, 77, 77]))
        image = MODULE.decode_image(packet)
        self.assertEqual(image.shape, (2, 1, 3))
        self.assertEqual(image[:, 0].tolist(), [[255, 0, 0], [0, 0, 255]])

    def make_joints(self):
        joints = {
            name: float(index + 1)
            for index, name in enumerate(MODULE.LEFT_JOINTS + MODULE.RIGHT_JOINTS)
        }
        joints[MODULE.LEFT_GRIPPER_DRIVER] = 0.25
        joints[MODULE.RIGHT_GRIPPER_DRIVER] = 0.75
        return joints

    def test_real_state_matches_official_42d_order(self):
        state = MODULE.build_real_state(
            self.make_joints(),
            {"left": [101, 102, 103, 104, 105, 106], "right": [201, 202, 203, 204, 205, 206]},
        )
        self.assertEqual(len(state), 42)
        self.assertEqual(state[0:7], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(state[7:15], [0.0] * 8)
        self.assertEqual(state[15:21], [101, 102, 103, 104, 105, 106])
        self.assertEqual(state[21:28], [8, 9, 10, 11, 12, 13, 14])
        self.assertEqual(state[28], 0.75)
        self.assertEqual(state[29:36], [0.0] * 7)
        self.assertEqual(state[36:42], [201, 202, 203, 204, 205, 206])

    def test_real_state_rejects_missing_or_nonfinite_input(self):
        self.assertIsNone(MODULE.build_real_state(self.make_joints(), {"left": [0] * 6, "right": None}))
        joints = self.make_joints()
        joints[MODULE.LEFT_JOINTS[0]] = math.nan
        self.assertIsNone(MODULE.build_real_state(joints, {"left": [0] * 6, "right": [0] * 6}))

    def test_real_action_filter_uses_17d_layout_and_training_spine(self):
        joints = self.make_joints()
        for index, name in enumerate(MODULE.LEFT_JOINTS):
            joints[name] = MODULE.REAL_DEMO_LEFT_ARM_MIN[index] + 0.1
        for index, name in enumerate(MODULE.RIGHT_JOINTS):
            joints[name] = MODULE.REAL_DEMO_RIGHT_ARM_MIN[index] + 0.1
        safety = MODULE.Task2ActionSafetyFilter(
            spine_min=MODULE.REAL_DEMO_SPINE_HEIGHT,
            spine_max=MODULE.REAL_DEMO_SPINE_HEIGHT,
            contract="real42_17",
        )
        requested = [10.0] * 17
        filtered, report = safety.filter(requested, joints)
        self.assertEqual(len(filtered), 17)
        self.assertEqual(filtered[16], MODULE.REAL_DEMO_SPINE_HEIGHT)
        self.assertEqual(filtered[7], 1.0)
        self.assertEqual(filtered[15], 1.0)
        self.assertTrue(report["limited"])
        self.assertLessEqual(
            filtered[0] - joints[MODULE.LEFT_JOINTS[0]],
            MODULE.REAL_DEMO_LEFT_ARM_MAX_STEP[0] + 1e-12,
        )

    def test_real_action_filter_rejects_old_20d_layout(self):
        safety = MODULE.Task2ActionSafetyFilter(
            spine_min=MODULE.REAL_DEMO_SPINE_HEIGHT,
            spine_max=MODULE.REAL_DEMO_SPINE_HEIGHT,
            contract="real42_17",
        )
        with self.assertRaisesRegex(ValueError, "17D"):
            safety.filter([0.0] * 20, self.make_joints())

    def test_real_action_split_uses_official_17d_order(self):
        parts = MODULE.split_task2_action(list(range(17)), real=True, real_spine_height=434.0)
        self.assertEqual(parts["left_arm"], list(range(7)))
        self.assertEqual(parts["left_gripper"], 1.0)
        self.assertEqual(parts["right_arm"], list(range(8, 15)))
        self.assertEqual(parts["right_gripper"], 1.0)
        self.assertEqual(parts["spine"], 434.0)

    def test_legacy_isaac_action_split_remains_compatible(self):
        action = [float(value) for value in range(20)]
        parts = MODULE.split_task2_action(action, real=False, spine_min=0.0, spine_max=0.54)
        self.assertEqual(parts["left_arm"], action[3:10])
        self.assertEqual(parts["right_arm"], action[10:17])
        self.assertEqual(parts["left_gripper"], 1.0)
        self.assertEqual(parts["right_gripper"], 1.0)
        self.assertEqual(parts["spine"], 0.54)

    def test_legacy_isaac_spine_slew_limit_remains_enabled(self):
        joints = self.make_joints()
        joints[MODULE.SPINE_JOINT] = 0.0
        safety = MODULE.Task2ActionSafetyFilter(spine_min=0.0, spine_max=0.54)
        measured = safety._measured_action(joints)
        requested = list(measured)
        requested[19] = 0.54
        filtered, report = safety.filter(requested, joints)
        self.assertEqual(filtered[19], MODULE.DEMO_SPINE_MAX_STEP)
        self.assertIn(19, report["rate_limited"])


if __name__ == "__main__":
    unittest.main()
