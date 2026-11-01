"""Lossless, opt-in client recording. No ROS imports or robot commands."""
from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import threading
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported record type: {type(value)}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, default=json_value, indent=2)
        stream.flush()
    temporary.replace(path)


class SessionRecorder:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ('observations', 'requests'):
            (self.root/name).mkdir()
        self.lock = threading.Condition(threading.RLock())
        self.stream = (self.root/'events.jsonl').open('x', buffering=1)
        self.steps = 0
        self.requests = 0
        self.pending = set()
        self.closed = False
        atomic_json(self.root/'metadata.json', metadata)

    def event(self, event, **fields):
        with self.lock:
            if self.closed:
                raise RuntimeError('Recording already closed')
            self.stream.write(json.dumps(dict(event=event, timestamp_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(), **fields), default=json_value) + '\n')

    def packet(self, relative, value):
        """Store arrays without lossy image conversion; JSON preserves the full tree."""
        arrays = {}
        def extract(x):
            if isinstance(x, np.ndarray):
                key = f'array_{len(arrays):04d}'
                arrays[key] = np.array(x, copy=True)
                return {'npz_key': key, 'shape': list(x.shape), 'dtype': str(x.dtype)}
            if isinstance(x, dict):
                return {k: extract(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return [extract(v) for v in x]
            return x
        tree = extract(value)
        path = self.root/relative
        # Uncompressed NPZ avoids expensive image encoding in the control loop.
        # Completed packets are renamed atomically; partial files remain identifiable.
        with path.with_suffix('.npz.partial').open('xb') as stream:
            np.savez(stream, **arrays)
        path.with_suffix('.npz.partial').replace(path.with_suffix('.npz'))
        atomic_json(path.with_suffix('.json'), tree)
        return str(relative)

    def close(self, reason):
        with self.lock:
            if self.closed:
                return
            deadline = time.monotonic() + 3
            while self.pending and time.monotonic() < deadline:
                self.lock.wait(timeout=max(0, deadline-time.monotonic()))
            self.event('recording_end', reason=reason, pending_requests=sorted(self.pending))
            atomic_json(self.root/'summary.json', dict(reason=reason, observations=self.steps,
                requests=self.requests, unfinished_requests=sorted(self.pending),
                closed_ns=time.time_ns(), images='lossless exact client input arrays',
                action_semantics='submission is not controller execution acknowledgement'))
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.closed = True


class RecordedPolicy:
    def __init__(self, policy, recorder):
        self.policy, self.recorder = policy, recorder

    def get_server_metadata(self):
        method = getattr(self.policy, 'get_server_metadata', None)
        return method() if callable(method) else {}

    def reset(self):
        return self.policy.reset()

    def infer(self, obs):
        obs = copy.deepcopy(obs)
        r = self.recorder
        with r.lock:
            i = r.requests
            r.requests += 1
            r.pending.add(i)
        try:
            path = r.packet(f'requests/{i:06d}_input', obs)
            r.event('inference_start', request=i, input=path)
            result = self.policy.infer(obs)
            stored = dict(result)
            if self.get_server_metadata().get('action_representation') == 'delta_joint_current_v1':
                delta = np.array(result['actions'], dtype=np.float32, copy=True)
                joints = [0,1,2,3,4,5,7,8,9,10,11,12]
                delta[:, joints] -= np.asarray(obs['state'])[None, joints]
                stored['_diagnostic_reconstructed_delta_joint'] = delta
            if self.get_server_metadata().get('action_representation') == 'delta_eef_current_rpy_v1':
                delta = np.array(result['actions'], dtype=np.float32, copy=True)
                pose = [0,1,2,3,4,5,7,8,9,10,11,12]
                angles = [3,4,5,10,11,12]
                delta[:, pose] -= np.asarray(obs['state'])[None, pose]
                delta[:, angles] = (delta[:, angles]+np.pi) % (2*np.pi)-np.pi
                stored['_diagnostic_reconstructed_delta'] = delta
            path = r.packet(f'requests/{i:06d}_output', stored)
            r.event('inference_result', request=i, output=path)
            # Extra diagnostic metadata never changes action tensors or server inputs.
            return {**result, '_record_request_id': i}
        except BaseException as exc:
            if not r.closed:
                r.event('inference_error', request=i, error=repr(exc))
            raise
        finally:
            with r.lock:
                r.pending.discard(i)
                r.lock.notify_all()


class RecordedEnvironment:
    def __init__(self, environment, recorder):
        self.environment, self.recorder = environment, recorder
        self.observation_id = None

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def get_observation(self):
        r = self.recorder
        try:
            obs = self.environment.get_observation()
            i = r.steps
            r.steps += 1
            self.observation_id = i
            start = time.monotonic_ns()
            path = r.packet(f'observations/{i:06d}', obs)
            r.event('observation', observation=i, packet=path,
                write_duration_ms=(time.monotonic_ns()-start)/1e6)
            return obs
        except BaseException as exc:
            r.event('observation_error', error=repr(exc))
            raise

    def apply_action(self, action):
        r = self.recorder
        r.event('action_attempt', observation=self.observation_id, action=action)
        try:
            result = self.environment.apply_action(action)
        except BaseException as exc:
            r.event('action_rejected', observation=self.observation_id, error=repr(exc))
            raise
        r.event('action_submitted', observation=self.observation_id,
            request=action.get('_record_request_id'), chunk=action.get('_chunk_trace'))
        return result
