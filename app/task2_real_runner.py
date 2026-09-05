"""Bounded docking and PI0.5 handoff with an isolated ROS command stream."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import signal
import socket
import time

import cv2
import numpy as np

from navigation.config import load_config
from navigation.coordinator import Coordinator
from navigation.demo_bank import DemoBank
from navigation.safety import stopped
from navigation.types import SafeStop
from task2_pi05_eval_node import (recv_obj, send_obj, camera_logical_from_feature,
                                  Task2ActionSafetyFilter)
from timed_chunk import TimedChunk


class EventLog:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.stream = (Path(directory) / f"real-{time.time_ns()}.jsonl").open("w", buffering=1)

    def __call__(self, event, **fields):
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if is_dataclass(value):
                return asdict(value)
            raise TypeError(type(value).__name__)
        row = dict(time=time.time(), monotonic=time.monotonic(), event=event, **fields)
        self.stream.write(json.dumps(row, default=convert, allow_nan=False) + "\n")
        if event in ("state", "safe_stop", "topics", "policy_started", "dry_run_complete"):
            print(json.dumps(row, default=convert), flush=True)


class PolicyClient:
    def __init__(self, host, port, cfg):
        self.cfg = cfg
        self.sock = None
        end = time.monotonic() + cfg["connect_timeout_s"]
        while time.monotonic() < end:
            candidate = None
            try:
                candidate = socket.create_connection((host, port), timeout=2)
                candidate.settimeout(cfg["request_timeout_s"])
                hello = recv_obj(candidate)
                self.validate_hello(hello)
                self.sock, self.hello = candidate, hello
                return
            except (OSError, EOFError):
                if candidate is not None:
                    candidate.close()
                time.sleep(0.2)
            except BaseException:
                if candidate is not None:
                    candidate.close()
                raise
        raise SafeStop("policy server connection timeout")

    @staticmethod
    def validate_hello(hello):
        if (not hello.get("ok") or hello.get("policy") != "pi05" or
                hello.get("state_shape") != [42] or hello.get("action_shape") != [17] or
                hello.get("chunk_size") != 50 or hello.get("replan_interval") != 10 or
                not hello.get("full_chunk_protocol")):
            raise SafeStop("incompatible PI0.5 server contract")
        keys = hello.get("image_keys", [])
        if {camera_logical_from_feature(k) for k in keys} != {"head", "wrist_left", "wrist_right"}:
            raise SafeStop("real policy must have exactly head and both wrist cameras")

    def request(self, observation):
        state, images, _ = observation
        packets = {}
        for key in self.hello["image_keys"]:
            rgb = images[camera_logical_from_feature(key)]
            shape = self.hello["expected_shapes"][key]
            if len(shape) != 3 or shape[0] != 3:
                raise SafeStop("invalid checkpoint image dimensions")
            h, w = shape[1:]
            if rgb.shape != (h, w, 3):
                actual = rgb.shape[1] / rgb.shape[0]
                if not self.cfg["image_resize"] or abs(actual / (w / h) - 1) > self.cfg["max_aspect_error"]:
                    raise SafeStop(f"camera aspect ratio does not match training: {key}")
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            packets[key] = dict(height=h, width=w, encoding="rgb8", step=w * 3,
                                data=np.ascontiguousarray(rgb).tobytes())
        send_obj(self.sock, dict(op="chunk", state=state, images=packets))
        reply = recv_obj(self.sock)
        if not reply.get("ok"):
            raise SafeStop("policy inference failed: " + str(reply.get("error")))
        return reply["chunk"]

    def close(self):
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()


def policy_loop(io, client, cfg, log, bank):
    pc = cfg["policy"]
    buffer = TimedChunk(pc)
    action_filter = Task2ActionSafetyFilter(spine_min=434, spine_max=434, contract="real42_17")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-request")
    pending = None
    start = time.monotonic()
    try:
        observed = time.monotonic()
        pending = worker.submit(client.request, io.observation())
        while not pending.done():
            io.command(0, 0, 0)
            io.sleep(1.0 / pc["rate_hz"])
            if time.monotonic() - observed > pc["max_inference_s"]:
                raise SafeStop("first action inference timed out")
        first_chunk = pending.result()
        pending = None
        io.verify_start_pose(bank.metadata["start_pose"])
        if not bank.evaluate(io.snapshot()["image"]).ready:
            raise SafeStop("fresh visual gate failed after first action inference")
        buffer.install(first_chunk, observed, time.monotonic())
        # Inputs stayed fresh during inference; the physical base must still be at rest.
        if not stopped(io.snapshot(), cfg["navigation"]["handoff"]):
            raise SafeStop("base moved during first action inference")
        with io.lock:
            io.policy_until = time.monotonic() + pc["max_hold_s"]
            io.policy_active = True
        log("policy_started", playback_rate=buffer.speed)
        next_tick = time.monotonic()
        steps = 0
        while not pc["max_episode_s"] or time.monotonic() - start < pc["max_episode_s"]:
            now = time.monotonic()
            observation = io.observation()
            io.command(0, 0, 0)
            if not stopped(io.snapshot(), cfg["navigation"]["handoff"]):
                raise SafeStop("base moved during manipulation")
            if pending is not None:
                if now - observed > pc["max_inference_s"]:
                    raise SafeStop("policy request deadline exceeded")
                if pending.done():
                    buffer.install(pending.result(), observed, now)
                    log("policy_chunk", latency_s=now - observed, playback_rate=buffer.speed)
                    pending = None
            if pending is None and buffer.replan_due(now):
                observed = now
                pending = worker.submit(client.request, observation)
            action, info = action_filter.filter(buffer.action(now).tolist(), observation[2])
            # Physical FR3 limits intersect the demonstrated policy envelope.
            for offset in (0, 8):
                action[offset:offset + 7] = np.clip(action[offset:offset + 7], cfg["rig"]["joint_lower_rad"], cfg["rig"]["joint_upper_rad"]).tolist()
            io.set_arms(action[:7], action[8:15], grips=(action[7], action[15]))
            if steps % 20 == 0:
                log("policy_action", action=action, filter=info, chunk_index=buffer.index(now))
            steps += 1
            next_tick += 1.0 / pc["rate_hz"]
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
            io.sleep(max(0, next_tick - time.monotonic()))
    finally:
        io.fault = io.fault or "policy loop ended"
        io.command(0, 0, 0)
        client.close()
        worker.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    if not (root / "config/task2.real.yaml").is_file():
        root = root.parent
    ap.add_argument("--config", default=str(root / "config/task2.real.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--navigation-only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cv2.setNumThreads(2)
    bank = DemoBank(cfg["navigation"]["visual"]["bank_path"], cfg["navigation"]["visual"])
    log = EventLog(cfg["log"]["directory"])
    io, client = None, None
    def interrupted(signum, frame):
        raise SafeStop(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if not (args.navigation_only or args.preflight or args.dry_run):
            client = PolicyClient(args.host, args.port, cfg["policy"])
        from real_io import RealIO
        io = RealIO(cfg, dry_run=args.dry_run or args.preflight, log=log)
        if args.dry_run or args.preflight:
            io.wait_ready()
        else:
            io.prepare_controllers()
        if args.preflight:
            log("preflight", status="sensors and command graph ready")
            return 0
        if client is not None:
            for _ in range(cfg["policy"]["warmup_chunks"]):
                start = time.monotonic()
                chunk = client.request(io.observation())
                TimedChunk(cfg["policy"]).install(chunk, start, time.monotonic())
        io.prepare(bank.metadata["start_pose"])
        coordinator = Coordinator(io, bank, cfg["navigation"], log)
        ready = coordinator.run(verify_only=not cfg["navigation"]["enabled"])
        if args.dry_run:
            log("dry_run_complete", currently_ready=ready)
            return 0
        if not ready:
            raise SafeStop("navigation did not reach the handoff gate")
        if args.navigation_only:
            return 0
        policy_loop(io, client, cfg, log, bank)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if io is not None:
            io.fault = str(exc) or "interrupted"
        log("safe_stop", reason=str(exc) or "interrupted")
        return 1
    finally:
        if io is not None:
            io.close()
        if client is not None:
            client.close()
        log.stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
