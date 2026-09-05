from __future__ import annotations

from dataclasses import asdict
import statistics

from .base_motion import BaseMotion
from .types import Evaluation, SafeStop, State


class CoordinateSearch:
    def __init__(self, motion: BaseMotion, evaluate, cfg: dict, log):
        self.motion, self.evaluate, self.cfg, self.log = motion, evaluate, cfg, log
        self.steps = [cfg["step_x_m"], cfg["step_y_m"], cfg["step_yaw_rad"]]
        self.minimum = [cfg["min_step_x_m"], cfg["min_step_y_m"], cfg["min_step_yaw_rad"]]

    def sweep(self, baseline: Evaluation) -> tuple[Evaluation, bool]:
        improved = False
        for axis in range(3 if self.cfg["enable_yaw"] else 2):
            origin = self.motion.snapshot()["pose"]
            before = baseline.score
            accepted = False
            for sign in (1, -1):
                delta = [0.0, 0.0, 0.0]
                delta[axis] = sign * self.steps[axis]
                # Absolute odometry target prevents rollback error accumulating across probes.
                self.motion.move_to(origin.offset(*delta))
                trial = self.evaluate(baseline.reference_id)
                accepted = trial.score >= before + self.cfg["score_improvement_min"]
                self.log("probe", axis=axis, sign=sign, steps=self.steps.copy(), before=before,
                         after=trial.score, accepted=accepted, reference_id=trial.reference_id)
                if accepted:
                    baseline, improved = trial, True
                    break
            if not accepted:
                self.motion.move_to(origin)
                baseline = self.evaluate(baseline.reference_id)
                if baseline.score < before - self.cfg["score_improvement_min"] * 3:
                    raise SafeStop("visual baseline did not recover after odometry rollback")
            if baseline.ready:
                return baseline, improved
        if not improved:
            self.steps = [max(m, s * self.cfg["step_shrink_factor"]) for s, m in zip(self.steps, self.minimum)]
        return baseline, improved


class Coordinator:
    def __init__(self, io, bank, nav: dict, log=lambda *a, **k: None):
        self.io, self.bank, self.nav, self.log = io, bank, nav, log
        self.motion = BaseMotion(io, nav)
        self.state = State.BOOT

    def transition(self, state: State):
        self.state = state
        self.log("state", state=state.value)

    def evaluate(self, reference_id=None) -> Evaluation:
        f = self.nav["fine"]
        self.motion.settle(f["settle_s"])
        last_seq = self.motion.snapshot()["camera_seq"]
        end = min(self.motion.deadline, self.io.clock() + f["observation_timeout_s"])
        evaluations = []
        while self.io.clock() < end and len(evaluations) < f["frames_per_evaluation"]:
            self.motion.stop()
            snap = self.motion.snapshot()
            if snap["camera_seq"] != last_seq:
                last_seq = snap["camera_seq"]
                e = self.bank.evaluate(snap["image"], reference_id)
                self.motion.snapshot()  # Vision compute time cannot bypass a freshness/deadline gate.
                if e.registration is not None and e.registration.valid and e.score > -1e6:
                    if reference_id is None:
                        reference_id = e.reference_id
                    evaluations.append(e)
            self.io.sleep(1.0 / self.nav["rate_hz"])
        if len(evaluations) < f["min_valid_frames"]:
            raise SafeStop("visual registration did not recover within observation deadline")
        representative = sorted(evaluations, key=lambda e: e.score)[len(evaluations) // 2]
        representative.score = float(statistics.median(e.score for e in evaluations))
        representative.ready = (len(evaluations) >= f["min_valid_frames"] and
                                sum(e.ready for e in evaluations) >= f["min_valid_frames"])
        representative.valid_frames = len(evaluations)
        r = representative.registration
        self.log("evaluation", state=self.state.value, score=representative.score, ready=representative.ready,
                 reference_id=reference_id, registration=asdict(r), pose=asdict(self.motion.snapshot()["pose"]),
                 travel_m=self.motion.safety.path_m, lidar_front_m=self.motion.snapshot()["clearance"](0.0),
                 safety="ok", **representative.details)
        return representative

    def run(self, verify_only=False) -> bool:
        try:
            self.transition(State.WAIT_FOR_SENSORS)
            self.motion.snapshot()
            c = self.nav["coarse"]
            self.motion.deadline = self.io.clock() + c["timeout_s"]
            if not verify_only:
                self.transition(State.COARSE_APPROACH)
                valid, last_seq = 0, None
                last_reference = None
                while True:
                    snap = self.motion.snapshot()
                    # Stop while registering so expensive perception never extends a command.
                    self.motion.stop()
                    if snap["camera_seq"] != last_seq:
                        last_seq = snap["camera_seq"]
                        e = self.bank.evaluate(snap["image"], last_reference)
                        self.motion.snapshot()
                        self.log("coarse_observation", pose=asdict(snap["pose"]),
                                 lidar_front_m=snap["clearance"](0.0), travel_m=self.motion.safety.path_m,
                                 score=e.score, ready=e.ready, registration=asdict(e.registration))
                        good = (e.registration is not None and e.registration.valid and e.score > -1e6 and
                                e.details.get("geometry_distance", 0) <=
                                e.details.get("distance_limit", float("inf")) * c["visual_capture_limit_multiplier"])
                        valid = valid + 1 if good and e.reference_id == last_reference else int(good)
                        last_reference = e.reference_id if good else None
                        if valid >= c["visual_valid_frames"]:
                            break
                    if getattr(self.io, "dry_run", False):
                        self.log("would_move", primitive="VERIFY_VISUAL" if e.ready else "FINE_PROBE" if good else "COARSE_FORWARD",
                                 vx=0 if good else c["cruise_vx"], score=e.score, reference_id=e.reference_id)
                        return False
                    speed = c["slow_vx"] if snap["clearance"](0.0) < c["lidar_slow_m"] else c["cruise_vx"]
                    until = self.io.clock() + c["vision_period_s"]
                    while self.io.clock() < until:
                        self.motion.command(speed, 0.0)
                        self.io.sleep(1.0 / self.nav["rate_hz"])
            self.motion.stop()
            self.transition(State.VISUAL_SEARCH_INIT)
            self.motion.fine_origin = self.motion.snapshot()["pose"]
            self.motion.fine_path_start = self.motion.safety.path_m
            self.motion.deadline = self.io.clock() + self.nav["fine"]["timeout_s"]
            baseline = self.evaluate()
            optimizer = CoordinateSearch(self.motion, self.evaluate, self.nav["fine"], self.log)
            passes = 0
            for _ in range(self.nav["fine"]["max_iterations"]):
                self.transition(State.FINE_VISUAL_ALIGNMENT)
                if baseline.ready:
                    passes += 1
                    if passes >= self.nav["handoff"]["consecutive_passes"]:
                        self.transition(State.FINAL_SETTLE_AND_VERIFY)
                        self.motion.settle(self.nav["handoff"]["post_stop_settle_s"])
                        baseline = self.evaluate(baseline.reference_id)
                        if baseline.ready:
                            self.transition(State.READY_FOR_PI05)
                            return True
                        passes = 0
                    else:
                        baseline = self.evaluate(baseline.reference_id)
                else:
                    passes = 0
                    if verify_only:
                        raise SafeStop("post-docking visual gate failed")
                    if getattr(self.io, "dry_run", False):
                        self.log("would_move", primitive="MOVE_X", delta=self.nav["fine"]["step_x_m"])
                        return False
                    baseline, _ = optimizer.sweep(baseline)
            raise SafeStop("maximum visual search iterations")
        except BaseException as exc:
            self.transition(State.SAFE_STOP)
            self.log("safe_stop", reason=str(exc))
            raise
        finally:
            self.motion.final_stop()
