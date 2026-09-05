import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "scripts")]
from navigation.base_motion import BaseMotion
from navigation.config import load_config
from navigation.coordinator import Coordinator, CoordinateSearch
from navigation.safety import SafetyMonitor, scan_points, direction_clearance
from navigation.types import Pose, SafeStop, State, Evaluation
from replay_visual_alignment import SimIO, SyntheticBank
from real_interfaces import SourceClock, arm_sample, gripper_sample, resolve_channels
from timed_chunk import TimedChunk


class NavigationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config/task2.real.yaml")
        self.nav = self.cfg["navigation"]

    def test_primitives_both_signs_and_rotated_frame(self):
        for yaw in (0, 1.5):
            for delta in ((.02, 0, 0), (-.02, 0, 0), (0, .02, 0), (0, -.02, 0), (0, 0, -.025)):
                with self.subTest(yaw=yaw, delta=delta):
                    io = SimIO(Pose(.1, .2, yaw))
                    target = io.pose.offset(*delta)
                    motion = BaseMotion(io, self.nav)
                    motion.execute_base_delta(*delta)
                    self.assertLess(math.hypot(io.pose.x - target.x, io.pose.y - target.y), .0021)
                    self.assertLess(abs(io.pose.yaw - target.yaw), .0041)
                    self.assertEqual(io.commands[-1], (0, 0, 0))

    def test_full_search_converges_with_noise_and_reversed_offsets(self):
        for offset in ((.06, -.04, 0), (-.07, .05, 0), (.02, .07, 0), (-.04, -.06, .06)):
            for noise in (0, .015):
                nav = copy.deepcopy(self.nav)
                nav["fine"]["enable_yaw"] = offset[2] != 0
                io = SimIO(Pose(*offset))
                coordinator = Coordinator(io, SyntheticBank(noise=noise), nav)
                self.assertTrue(coordinator.run())
                self.assertEqual(coordinator.state, State.READY_FOR_PI05)
                self.assertLess(math.hypot(io.pose.x, io.pose.y), .008)
                self.assertLess(coordinator.motion.safety.path_m, nav["fine"]["max_path_m"])
                self.assertEqual(io.commands[-1], (0, 0, 0))

    def test_stale_missing_nonfinite_inputs_latch_stop(self):
        for field, value in (("odom_stamp", None), ("lidar_stamp", None),
                             ("camera_stamp", -1), ("pose", Pose(math.nan, 0, 0)),
                             ("velocity", (math.nan, 0, 0)), ("clearance", lambda _: .1)):
            monitor = SafetyMonitor(self.nav)
            io = SimIO()
            snap = io.snapshot()
            snap[field] = value
            with self.assertRaises(SafeStop):
                monitor.check(snap, io.clock())
            with self.assertRaises(SafeStop):
                monitor.check(io.snapshot(), io.clock())

    def test_odom_jump(self):
        io, monitor = SimIO(), SafetyMonitor(self.nav)
        monitor.check(io.snapshot(), io.clock())
        io.sleep(.05)
        io.pose = Pose(.2, 0, 0)
        with self.assertRaisesRegex(SafeStop, "discontinuity"):
            monitor.check(io.snapshot(), io.clock())

    def test_primitive_timeout_finishes_zero(self):
        io = SimIO()
        io.sleep = lambda seconds: setattr(io, "t", io.t + seconds)
        with self.assertRaisesRegex(SafeStop, "timeout"):
            BaseMotion(io, self.nav).execute_base_delta(.02, 0, 0)
        self.assertEqual(io.commands[-1], (0, 0, 0))

    def test_final_stop_repeats_despite_sleep_failure(self):
        io = SimIO()
        def failure(_):
            raise SafeStop("already faulted")
        io.sleep = failure
        BaseMotion(io, self.nav).final_stop()
        self.assertEqual(len(io.commands), self.nav["safety"]["repeated_zero_cmd_count"])

    def test_dry_run_never_moves_or_hands_off(self):
        io = SimIO(Pose(.1, .1, 0), dry_run=True)
        self.assertFalse(Coordinator(io, SyntheticBank(), self.nav).run())
        self.assertTrue(all(not any(cmd) for cmd in io.commands))

    def test_iteration_limit_and_shrink(self):
        io = SimIO()
        motion = BaseMotion(io, self.nav)
        baseline = Evaluation(0, False, 0)
        optimizer = CoordinateSearch(motion, lambda _: baseline, self.nav["fine"], lambda *a, **k: None)
        optimizer.sweep(baseline)
        self.assertAlmostEqual(optimizer.steps[0], self.nav["fine"]["step_x_m"] * .5)
        self.assertLess(math.hypot(io.pose.x, io.pose.y), .0021)
        nav = copy.deepcopy(self.nav)
        nav["fine"]["max_iterations"] = 1
        io = SimIO(Pose(.1, .1, 0))
        coordinator = Coordinator(io, SyntheticBank(), nav)
        with self.assertRaisesRegex(SafeStop, "iterations"):
            coordinator.run()
        self.assertEqual(coordinator.state, State.SAFE_STOP)

    def test_final_handoff_recheck_cannot_be_skipped(self):
        io = SimIO()
        coordinator = Coordinator(io, SyntheticBank(), self.nav)
        original = coordinator.evaluate
        def evaluate(reference_id=None):
            result = original(reference_id)
            if coordinator.state == State.FINAL_SETTLE_AND_VERIFY:
                result.ready = False
            return result
        coordinator.evaluate = evaluate
        with self.assertRaises(SafeStop):
            coordinator.run(verify_only=True)
        self.assertEqual(coordinator.state, State.SAFE_STOP)

    def test_coarse_pins_a_reference_for_consecutive_frames(self):
        bank = SyntheticBank()
        calls = []
        original = bank.evaluate
        def evaluate(image, reference_id=None):
            calls.append(reference_id)
            result = original(image, reference_id)
            result.reference_id = len(calls) % 2 if reference_id is None else reference_id
            return result
        bank.evaluate = evaluate
        self.assertTrue(Coordinator(SimIO(), bank, self.nav).run())
        self.assertIsNone(calls[0])
        self.assertEqual(calls[1:4], [1, 1, 1])

    def test_valid_distant_registration_does_not_enter_fine_search(self):
        bank = SyntheticBank()
        original = bank.evaluate
        def evaluate(image, reference_id=None):
            result = original(image, reference_id)
            result.details = {"geometry_distance": 100, "distance_limit": 1}
            return result
        bank.evaluate = evaluate
        nav = copy.deepcopy(self.nav)
        nav["coarse"]["max_travel_m"] = .01
        events = []
        io = SimIO()
        coordinator = Coordinator(io, bank, nav, lambda event, **fields: events.append(fields))
        with self.assertRaisesRegex(SafeStop, "coarse cumulative"):
            coordinator.run()
        self.assertFalse(any(e.get("state") == State.FINE_VISUAL_ALIGNMENT.value for e in events))
        self.assertEqual(io.commands[-1], (0, 0, 0))

    def test_lidar_unknown_sectors_and_offsets(self):
        scan = scan_points([2.] * 181, -math.pi / 2, math.pi / 180, .05, 10)
        self.assertAlmostEqual(direction_clearance([scan], 0, .6, .8), 2)
        with self.assertRaises(SafeStop):
            direction_clearance([scan], math.pi, .6, .8)
        bad = scan_points([math.nan] * 181, -math.pi / 2, math.pi / 180, .05, 10)
        with self.assertRaises(SafeStop):
            direction_clearance([bad], 0, .6, .8)
        with self.assertRaises(SafeStop):
            scan_points([1, 1, 1], 0, 0, .05, 10)


class InterfaceTest(unittest.TestCase):
    def test_named_joint_order_and_incomplete_rejection(self):
        names = [f"fr3_joint{i}" for i in range(7, 0, -1)]
        ordered, values = arm_sample(names, list(range(7, 0, -1)))
        self.assertEqual(values, list(range(1, 8)))
        self.assertEqual(ordered[0], "fr3_joint1")
        for n, q in ((names[:-1], [0] * 6), (names + names[:1], [0] * 8), (names, [math.nan] * 7)):
            with self.assertRaises(SafeStop):
                arm_sample(n, q)

    def test_driver_knuckle(self):
        self.assertEqual(gripper_sample(["robotiq_85_left_knuckle_joint"], [.789]), .789)
        with self.assertRaises(SafeStop):
            gripper_sample(["inner_left_knuckle_joint"], [.2])

    def test_source_clock_frozen_and_host_offset(self):
        clock = SourceClock()
        self.assertTrue(clock.observe(1000, 10))
        self.assertFalse(clock.observe(1000, 10.1))
        self.assertAlmostEqual(clock.remote_time(10.2, .1), 1000.3)
        with self.assertRaises(SafeStop):
            SourceClock().observe(0, 1)

    def test_typed_discovery_prefers_follower(self):
        topics = [("/right/gello/joint_states", ["sensor_msgs/msg/JointState"]),
                  ("/right/follower/gello/joint_states", ["sensor_msgs/msg/JointState"]),
                  ("/swerve_drive_controller/cmd_vel", ["geometry_msgs/msg/Twist"])]
        channels, missing = resolve_channels(topics)
        self.assertEqual(channels["right_cmd"][0], topics[1][0])
        self.assertEqual(channels["base_cmd"][1], "geometry_msgs/msg/Twist")
        self.assertTrue(missing)


class ChunkTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config/task2.real.yaml")["policy"]
        self.actions = np.repeat(np.arange(50)[:, None], 17, axis=1)

    def test_elapsed_steps_are_not_replayed(self):
        chunk = TimedChunk(self.cfg)
        chunk.install(self.actions, 10, 10.3)
        self.assertGreaterEqual(chunk.action(10.3)[0], 6)
        self.assertTrue(chunk.replan_due(10.6))
        self.assertGreater(chunk.action(11)[0], chunk.action(10.6)[0])

    def test_slow_inference_and_bounded_hold(self):
        chunk = TimedChunk(self.cfg)
        chunk.install(self.actions, 10, 12)
        self.assertLess(chunk.speed, 1)
        end = 10 + 50 / (20 * chunk.speed)
        self.assertEqual(chunk.action(end + .1)[0], 49)
        with self.assertRaisesRegex(SafeStop, "hold"):
            chunk.action(end + .6)
        with self.assertRaises(SafeStop):
            chunk.install(self.actions, 0, 9)
        with self.assertRaises(SafeStop):
            chunk.install(np.full((50, 17), np.nan), 1, 2)


if __name__ == "__main__":
    unittest.main()
