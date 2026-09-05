"""ROS command process isolated from image callbacks, vision, and model inference."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import uuid

from navigation.types import SafeStop


def packet_fault(packet, now, last_receive, heartbeat_s):
    if packet.get("fault"):
        return packet["fault"]
    if now - last_receive > heartbeat_s:
        return "command parent heartbeat expired"
    if any(packet["base"]) and now > packet["base_until"]:
        return "base command lease expired"
    if packet["policy_active"] and now > packet["policy_until"]:
        return "policy command lease expired"
    return None


def stream_loop(conn, updates, shutdown, channels, name, rate_hz, heartbeat_s, zero_count):
    import rclpy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Twist, TwistStamped
    from std_msgs.msg import Float32
    from builtin_interfaces.msg import Time
    rclpy.init()
    node = rclpy.create_node(name)
    types = {"sensor_msgs/msg/JointState": JointState, "std_msgs/msg/Float32": Float32,
             "geometry_msgs/msg/Twist": Twist, "geometry_msgs/msg/TwistStamped": TwistStamped}
    pubs = {key: node.create_publisher(types[typ], topic, 1)
            for key, (topic, typ) in channels.items() if key.endswith("_cmd")}
    conn.send(("ready", os.getpid()))
    packet = None
    last_receive = time.monotonic()
    fault = None
    frozen = None
    published_sides = set()
    period = 1 / rate_hz

    def base(command):
        msg = types[channels["base_cmd"][1]]()
        twist = msg.twist if hasattr(msg, "twist") else msg
        twist.linear.x, twist.linear.y, twist.angular.z = command
        pubs["base_cmd"].publish(msg)

    def latch(reason):
        nonlocal fault, frozen
        if fault is None:
            fault = reason
            frozen = dict(packet.get("arms", {})) if packet else {}
            for side, value in list(frozen.items()):
                measured = packet.get("measured", {}).get(side)
                if measured and time.monotonic() - measured[1] <= packet["state_max_age_s"]:
                    frozen[side] = (measured[0], value[1], value[2])
            try:
                conn.send(("fault", reason))
            except OSError:
                pass

    try:
        next_tick = time.monotonic()
        while not shutdown.is_set() and rclpy.ok():
            shutdown.wait(max(0, next_tick - time.monotonic()))
            if shutdown.is_set():
                break
            # Bounded drain: a busy parent can never starve command publication.
            for _ in range(2):
                try:
                    packet = updates.get_nowait()
                    last_receive = packet["sent_at"]
                except queue.Empty:
                    break
            now = time.monotonic()
            if packet is not None:
                reason = packet_fault(packet, now, last_receive, heartbeat_s)
                if reason:
                    latch(reason)
                base((0., 0., 0.) if fault else packet["base"])
                for side, (q, names, offset) in (frozen if fault else packet["arms"]).items():
                    msg = JointState()
                    stamp = now + offset + packet["stamp_bias_s"]
                    sec = int(stamp)
                    msg.header.stamp = Time(sec=sec, nanosec=int((stamp - sec) * 1e9))
                    msg.name, msg.position = names, [float(v) for v in q]
                    pubs[side + "_cmd"].publish(msg)
                    if side not in published_sides:
                        conn.send(("published", side))
                        published_sides.add(side)
                if not fault:
                    for side, width in packet["grips"].items():
                        pubs[side + "_gripper_cmd"].publish(Float32(data=float(width)))
                    if packet["spine"] is not None:
                        pubs["spine_cmd"].publish(Float32(data=float(packet["spine"])))
            else:
                base((0., 0., 0.))
            next_tick = max(next_tick + period, time.monotonic() + .001)
    except (EOFError, OSError):
        pass
    finally:
        for _ in range(zero_count):
            try:
                base((0., 0., 0.))
            except Exception:
                pass
            time.sleep(period)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class CommandStream:
    def __init__(self, channels, cfg):
        self.name = "task2_command_stream_" + uuid.uuid4().hex[:10]
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe(duplex=False)
        self.updates = ctx.Queue(maxsize=2)
        self.shutdown = ctx.Event()
        self.published_sides = set()
        self.process = ctx.Process(target=stream_loop, args=(child, self.updates, self.shutdown, channels, self.name,
            cfg["policy"]["rate_hz"], cfg["rig"]["stream_heartbeat_s"],
            cfg["navigation"]["safety"]["repeated_zero_cmd_count"]), daemon=True)
        self.process.start()
        child.close()
        if not self.conn.poll(cfg["rig"]["stream_start_timeout_s"]) or self.conn.recv()[0] != "ready":
            self.close()
            raise SafeStop("command process failed to start")

    def update(self, packet):
        if not self.process.is_alive():
            raise SafeStop("command process exited")
        while self.conn.poll():
            kind, reason = self.conn.recv()
            if kind == "fault":
                raise SafeStop(reason)
            if kind == "published":
                self.published_sides.add(reason)
        packet["sent_at"] = time.monotonic()
        try:
            self.updates.put_nowait(packet)
        except queue.Full:
            # Never block the safety watchdog on IPC. The child independently
            # rejects old packets and stops if the parent cannot deliver fresh ones.
            pass

    def close(self):
        self.shutdown.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.conn.close()
        self.updates.cancel_join_thread()
        self.updates.close()
