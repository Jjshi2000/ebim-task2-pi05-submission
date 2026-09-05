"""Service/packet/timing regression checks; no robot or motion simulator."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from command_stream import packet_fault
from navigation.config import load_config
from navigation.types import SafeStop
from real_interfaces import SourceClock, resolve_channels
from rig_lifecycle import RigLifecycle, controller_plan, select_controller
from timed_chunk import TimedChunk


class DeploymentContractTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config/task2.real.yaml")

    def test_impedance_lifecycle_states_and_missing_type(self):
        cases = [
            ("active", "type", ["deactivate", "unload", "load", "configure", "activate"]),
            ("inactive", "type", ["unload", "load", "configure", "activate"]),
            ("unconfigured", "type", ["configure", "activate"]),
            (None, "type", ["load", "configure", "activate"]),
            ("active", None, ["deactivate", "activate"]),
            ("inactive", None, ["activate"]),
            ("unconfigured", None, ["configure", "activate"]),
        ]
        for state, typ, expected in cases:
            with self.subTest(state=state, typ=typ):
                self.assertEqual(controller_plan(state, typ, True), expected)
        for state in (None, "finalized", "error"):
            with self.assertRaises(SafeStop):
                controller_plan(state, None, True)

    def test_controller_selection_never_chooses_arbitrary_candidate(self):
        candidates = self.cfg["rig"]["controller_names"]
        follower, impedance = candidates
        self.assertEqual(select_controller({follower: "inactive", impedance: "active"}, {}, candidates), impedance)
        self.assertEqual(select_controller({}, {impedance: "type"}, candidates), impedance)
        with self.assertRaises(SafeStop):
            select_controller(dict.fromkeys(candidates, "active"), {}, candidates)
        with self.assertRaises(SafeStop):
            select_controller(dict.fromkeys(candidates, "inactive"), {}, candidates)
        with self.assertRaises(SafeStop):
            select_controller({impedance: "active"}, {}, candidates, follower)

    def test_arm_hold_occurs_after_configure_before_activation(self):
        lifecycle = RigLifecycle.__new__(RigLifecycle)
        events = []
        lifecycle.io = NS(cfg=self.cfg, log=lambda *a, **k: None,
                          hold_current=lambda side: events.append((side, "hold")))
        lifecycle.controllers = lambda manager: ({"joint_impedance_controller": "active"}, [])
        lifecycle.parameter = lambda manager, name: "known-type" if name.startswith("joint_impedance") else None
        lifecycle.step = lambda manager, name, step: events.append((manager.split("/")[1], step))
        lifecycle.arms()
        expected = ["deactivate", "unload", "load", "configure", "hold", "activate"]
        for side in ("right", "left"):
            self.assertEqual([step for s, step in events if s == side], expected)

    def test_switch_request_matches_hkust_and_false_success_is_rejected(self):
        class Request:
            BEST_EFFORT = 1
            def __init__(self):
                self.timeout = NS(sec=0)
        lifecycle = RigLifecycle.__new__(RigLifecycle)
        lifecycle.cms = NS(SwitchController=NS(Request=Request))
        requests = []
        def call(srv, name, request, **kwargs):
            requests.append((name, request))
            return NS(ok=True)
        lifecycle.io = NS(call=call)
        lifecycle.controllers = lambda _: ({"joint_impedance_controller": "inactive"}, [])
        with self.assertRaisesRegex(SafeStop, "not confirmed"):
            lifecycle.step("/right/controller_manager", "joint_impedance_controller", "activate")
        name, request = requests[0]
        self.assertEqual(name, "/right/controller_manager/switch_controller")
        self.assertEqual(request.activate_controllers, ["joint_impedance_controller"])
        self.assertEqual(request.deactivate_controllers, [])
        self.assertEqual(request.strictness, Request.BEST_EFFORT)
        self.assertTrue(request.activate_asap)

    def test_zero_stamps_and_frozen_positive_stamps(self):
        clock = SourceClock()
        self.assertTrue(clock.observe(0, 1, required=False))
        self.assertTrue(clock.observe(0, 2, required=False))
        with patch("real_interfaces.time.time", return_value=1000):
            self.assertEqual(clock.remote_time(2, .1), 1000.1)
        self.assertTrue(clock.observe(1001, 3, required=False))
        self.assertFalse(clock.observe(1001, 4, required=False))
        self.assertFalse(clock.observe(0, 5, required=False))
        self.assertEqual(clock.arrival, 3)

    def test_bootstrap_requires_observed_namespace_or_explicit_override(self):
        grip = "/right/follower/gripper/gripper_client/target_gripper_width_percent"
        topics = [(grip, ["std_msgs/msg/Float32"])]
        channels, _ = resolve_channels(topics, bootstrap=True)
        self.assertEqual(channels["right_cmd"], ("/right/follower/gello/joint_states", "sensor_msgs/msg/JointState"))
        self.assertNotIn("left_cmd", channels)
        channels, _ = resolve_channels(topics)
        self.assertNotIn("right_cmd", channels)
        topics.append(("/right/follower/gello/joint_states", ["std_msgs/msg/String"]))
        channels, _ = resolve_channels(topics, bootstrap=True)
        self.assertNotIn("right_cmd", channels)

    def test_independent_leases_expire_despite_other_activity(self):
        packet = dict(base=(.08, 0, 0), base_until=1.2, policy_active=False, policy_until=0, fault=None)
        self.assertIsNone(packet_fault(packet, 1.1, 1.1, .3))
        self.assertIn("base", packet_fault(packet, 1.21, 1.21, .3))
        packet["base"] = (0, 0, 0)
        self.assertIn("heartbeat", packet_fault(packet, 1.5, 1, .3))
        packet.update(policy_active=True, policy_until=1.6)
        self.assertIn("policy", packet_fault(packet, 1.7, 1.7, .3))
        packet["fault"] = "source failed"
        self.assertEqual(packet_fault(packet, 1.1, 1.1, .3), "source failed")

    def test_two_slow_inferences_fit_before_chunk_exhaustion(self):
        actions = np.zeros((50, 17), np.float32)
        for latency in (.3, 2., 4.):
            chunk = TimedChunk(self.cfg["policy"])
            observed = 0.
            for _ in range(5):
                now = observed + latency
                chunk.install(actions, observed, now)
                request_at = max(now, chunk.refresh_at) + .05
                result_at = request_at + latency * 1.05
                # Includes one scheduling tick and 5% next-request jitter.
                self.assertLess(chunk.index(result_at), 50)
                chunk.action(result_at)
                observed = request_at
        with self.assertRaisesRegex(SafeStop, "sustain"):
            chunk.install(actions, 0, 5)

    def test_front_range_uses_sensor_origin_side_range_uses_footprint(self):
        from real_io import RealIO
        import threading
        from navigation.safety import scan_points
        io = RealIO.__new__(RealIO)
        io.lock = threading.RLock()
        io.cfg = self.cfg
        scan = scan_points([1.] * 361, -np.pi, np.pi/180, .05, 10)
        io.data = dict(lidar_front=scan, lidar_rear=scan, front_range_m=.5)
        io.stamps = dict(lidar_front=1, lidar_rear=1)
        io.camera_seq, io.fault = 0, None
        snap = io.snapshot()
        self.assertAlmostEqual(snap["clearance"](0), .5)
        self.assertAlmostEqual(snap["clearance"](np.pi/2), .6)

    def test_lidar_transform_supports_inverted_horizontal_mount(self):
        from navigation.safety import scan_points, transform_scan
        scan = scan_points([1., 1., 1.], 0, np.pi/4, .05, 10)
        angles, distances, valid = transform_scan(scan, 0., 0., (1., 0., 0., 0.))
        np.testing.assert_allclose(angles, [0, -np.pi/4, -np.pi/2], atol=1e-6)
        np.testing.assert_allclose(distances, 1.)
        with self.assertRaisesRegex(SafeStop, "horizontal"):
            transform_scan(scan, 0., 0., (np.sin(np.pi/8), 0., 0., np.cos(np.pi/8)))

    def test_base_activation_selects_only_claimed_hardware_and_checks_state(self):
        class State(NS):
            PRIMARY_STATE_ACTIVE = 3
        for confirmed in (True, False):
            lifecycle = RigLifecycle.__new__(RigLifecycle)
            lifecycle.cms = NS(ListHardwareComponents=NS(Request=NS),
                               SetHardwareComponentState=NS(Request=NS))
            components = [NS(name="base", state=NS(label="inactive"), command_interfaces=[NS(name="wheel/velocity")]),
                          NS(name="unrelated", state=NS(label="inactive"), command_interfaces=[NS(name="spine/position")])]
            calls = []
            def call(srv, name, request, **kwargs):
                if name.endswith("set_hardware_component_state"):
                    calls.append(request)
                    if confirmed:
                        components[0].state.label = "active"
                    return NS(ok=True)
                return NS(component=components)
            lifecycle.io = NS(cfg=self.cfg, call=call, log=lambda *a, **k: None)
            base_name = self.cfg["rig"]["base_controller"]
            lifecycle.controllers = lambda _: ({base_name: "active"}, [NS(
                name=base_name, required_command_interfaces=["wheel/velocity"], claimed_interfaces=[])])
            with patch.dict(sys.modules, {"lifecycle_msgs.msg": NS(State=State)}):
                if confirmed:
                    lifecycle.base()
                else:
                    with self.assertRaisesRegex(SafeStop, "not confirmed"):
                        lifecycle.base()
            self.assertEqual([r.name for r in calls], ["base"])
            self.assertEqual(calls[0].target_state.id, 3)
            self.assertEqual(components[1].state.label, "inactive")


if __name__ == "__main__":
    unittest.main()
