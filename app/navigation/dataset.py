from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


class RealDataset:
    """Released LeRobot v2 episodes; video PTS and table frame indices are distinct."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        ids = [int(e["episode_index"]) for e in self.episodes]
        if len(ids) != len(set(ids)) or len(ids) != self.info["total_episodes"]:
            raise ValueError("episode metadata count or identities inconsistent")

    def paths(self, episode: int):
        values = dict(episode_index=episode, episode_chunk=episode // self.info["chunks_size"],
                      video_key="observation.images.head")
        return (self.root / self.info["data_path"].format(**values),
                self.root / self.info["video_path"].format(**values))

    def table(self, episode: int):
        return pq.read_table(self.paths(episode)[0]).to_pydict()

    def frames(self, episode: int, seconds: list[float]):
        wanted = sorted(set(float(t) for t in seconds))
        if not wanted or wanted[0] < 0:
            raise ValueError("nonnegative frame times required")
        output = {}
        with av.open(str(self.paths(episode)[1])) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            previous = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("video frame has no presentation timestamp")
                stamp = float(frame.pts * frame.time_base)
                while wanted and stamp >= wanted[0]:
                    target = wanted.pop(0)
                    selected = previous if previous is not None and abs(previous[0] - target) < abs(stamp - target) else (stamp, frame)
                    if abs(selected[0] - target) > 0.15:
                        raise ValueError(f"episode {episode}: no video frame near {target}s")
                    output[target] = (selected[1].to_ndarray(format="rgb24"), selected[0])
                if not wanted:
                    break
                previous = (stamp, frame)
        if wanted:
            raise ValueError(f"episode {episode}: video ended before {wanted}")
        return output

    def start_samples(self, episode: int, window_s: float, count: int):
        table = self.table(episode)
        stamps = np.asarray(table["timestamp"], float).reshape(-1)
        indices = np.flatnonzero(stamps <= stamps[0] + window_s)
        if len(indices) < count + 2:
            raise ValueError(f"episode {episode}: early window too short")
        q = np.asarray(table["observation.state.franka_robot_right_measured_joint_states"], float)
        motion = np.r_[np.linalg.norm(np.diff(q, axis=0), axis=1), 0]
        # Select low-motion samples from different temporal bins; not necessarily frame zero.
        selected = [int(part[np.argmin(motion[part])]) for part in np.array_split(indices[1:-1], count)]
        decoded = self.frames(episode, [stamps[i] for i in selected])
        return [(dict(episode_index=episode, frame_index=int(table["frame_index"][i]),
                      timestamp=float(stamps[i]), video_timestamp=decoded[stamps[i]][1],
                      joint_motion=float(motion[i])), decoded[stamps[i]][0]) for i in selected]
