"""Experimental joint-space TOPP-RA policy. No ROS/CAN operations in this module."""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import Future

import numpy as np


def load_toppra():
    if importlib.util.find_spec('toppra') is None:
        target = Path(__file__).resolve().parents[2] / 'third_party/toppra_deps'
        if target.is_dir():
            sys.path.insert(0, str(target))
    try:
        import toppra
        return toppra
    except ImportError as exc:
        raise RuntimeError('TOPP entrypoint requires toppra==0.6.3; see TOPP_README.md') from exc


def vector(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError('Expected 14 finite values: left six rad + gripper m, then right')
    return result.copy()


def limits(joint, gripper):
    result = np.full(14, joint, dtype=float)
    result[[6, 13]] = gripper
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError('TOPP velocity and acceleration limits must be finite and positive')
    return result


class Planner:
    def __init__(self, velocity, acceleration, fps=15, gridpoints=200, path_fps=15):
        self.ta = load_toppra()
        self.velocity, self.acceleration = vector(velocity), vector(acceleration)
        if (self.velocity <= 0).any() or (self.acceleration <= 0).any():
            raise ValueError('Limits must be positive')
        if not np.isfinite(path_fps) or path_fps <= 0 or fps <= 0 or gridpoints < 20:
            raise ValueError('fps and path_fps must be positive and gridpoints >= 20')
        self.fps, self.gridpoints = fps, gridpoints
        self.path_fps = path_fps

    def plan(self, q, dq, actions):
        q, dq = vector(q), vector(dq)
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 1 or not np.isfinite(actions).all():
            raise ValueError('Policy actions must have shape (N,14) and be finite')
        points = [q]
        for row in actions:
            if np.max(np.abs(row-points[-1])/self.velocity) > 1e-8:
                points.append(row.copy())
        if len(points) == 1:
            return None
        moving = np.max(np.abs(dq)) > 1e-8
        # q'(0)*sd(0)=dq: match the full joint tangent, not merely scalar speed.
        bc = ((1, dq), (2, np.zeros(14))) if moving else 'natural'
        knots = np.arange(len(points), dtype=float)/self.path_fps
        path = self.ta.SplineInterpolator(knots, np.asarray(points), bc_type=bc)
        constraints = [self.ta.constraint.JointVelocityConstraint(self.velocity * .98),
                       self.ta.constraint.JointAccelerationConstraint(self.acceleration * .95,
                           discretization_scheme=self.ta.constraint.DiscretizationType.Interpolation)]
        grid = np.unique(np.r_[np.linspace(knots[0], knots[-1], self.gridpoints), knots])
        algorithm = self.ta.algorithm.TOPPRA(constraints, path, gridpoints=grid,
                                            solver_wrapper='seidel', parametrizer='ParametrizeConstAccel')
        trajectory = algorithm.compute_trajectory(1.0 if moving else 0.0, 0.0)
        if trajectory is None or not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
            raise RuntimeError('TOPP path infeasible from current planned velocity')
        ts = np.linspace(0, trajectory.duration, max(1000, len(grid)*5))
        for order, bound in ((1, self.velocity), (2, self.acceleration)):
            values = trajectory(ts, order)
            if not np.isfinite(values).all() or np.any(np.abs(values) > bound*1.01 + 1e-7):
                raise RuntimeError(f'TOPP dense check failed for derivative {order}')
        if not np.allclose(trajectory(0), q, atol=1e-7) or not np.allclose(trajectory(0, 1), dq, atol=1e-5):
            raise RuntimeError('TOPP handoff does not preserve position/velocity')
        return trajectory


class ToppPolicy:
    """One inference worker; immutable chunk handoff; never emits unplanned targets.

    Time advances by one output tick, not wall-time catch-up. Slow solves stretch
    execution rather than skipping trajectory samples. Planning is synchronous;
    latency is measured explicitly. Terminal zero velocity provides a stop if
    the next chunk is late. Acceleration/jerk continuity is NOT guaranteed.
    """
    def __init__(self, policy, planner, *, horizon=22, prefetch_seconds=.4,
                 asynchronous=True, metrics_path=None, clock=time.monotonic):
        if horizon < 1 or not np.isfinite(prefetch_seconds) or prefetch_seconds < 0:
            raise ValueError('Invalid horizon or prefetch time')
        self.policy, self.planner = policy, planner
        self.horizon, self.prefetch_seconds = horizon, prefetch_seconds
        self.asynchronous, self.clock = asynchronous, clock
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.future = None
        self.thread = None
        self.closed = False
        self.trajectory = None
        self.elapsed = 0.0
        self.q = None
        self.chunk_id = 0
        self.last_requested = False
        self.last_plan_ms = 0.0

    def get_server_metadata(self):
        getter = getattr(self.policy, 'get_server_metadata', None)
        return getter() if callable(getter) else {}


    def _log(self, **event):
        logging.info('TOPP %s', event)
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open('a') as stream:
                stream.write(json.dumps(dict(time_ns=time.time_ns(), **event))+'\n')

    def _request(self, observation):
        if self.future is not None:
            return
        future = Future()
        obs = copy.deepcopy(observation)
        self.future = future
        self.last_requested = True
        def work():
            start = self.clock()
            try:
                result = self.policy.infer(obs)
                future.set_result((result, (self.clock()-start)*1000))
            except BaseException as exc:
                future.set_exception(exc)
        if self.asynchronous:
            self.thread = threading.Thread(target=work, daemon=True, name='topp-inference')
            self.thread.start()
        else:
            work()

    def infer(self, observation):
        if self.closed:
            raise RuntimeError('TOPP policy closed')
        if self.q is None:
            self.q = vector(observation['state'])
        dt = 1/self.planner.fps
        if self.trajectory is not None:
            self.elapsed = min(self.elapsed+dt, self.trajectory.duration)
            self.q = np.asarray(self.trajectory(self.elapsed)).reshape(14)
            dq = np.asarray(self.trajectory(self.elapsed, 1)).reshape(14)
            if self.elapsed >= self.trajectory.duration:
                dq = np.zeros(14)
        else:
            dq = np.zeros(14)
        remaining = 0 if self.trajectory is None else self.trajectory.duration-self.elapsed
        if not self.last_requested and remaining <= self.prefetch_seconds:
            self._request(observation)
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            result, inference_ms = future.result()  # propagate server errors, never raw fallback
            start = self.clock()
            try:
                chunk = np.asarray(result['actions'], dtype=float)[:self.horizon]
                trajectory = self.planner.plan(self.q, dq, chunk)
            except (ValueError, RuntimeError, KeyError) as exc:
                # Retain the previous feasible stopping trajectory.
                self._log(event='rejected', reason=str(exc), inference_ms=inference_ms)
                self.last_requested = False
            else:
                self.last_plan_ms = (self.clock()-start)*1000
                if trajectory is not None:
                    self.trajectory, self.elapsed = trajectory, 0.0
                self.chunk_id += 1
                self.last_requested = False
                self._log(event='adopted', chunk_id=self.chunk_id, plan_ms=self.last_plan_ms,
                          inference_ms=inference_ms, duration_s=0 if trajectory is None else trajectory.duration,
                          nominal_s=len(chunk)/self.planner.path_fps,
                          start_velocity_max=float(np.max(np.abs(dq))))
        return {'actions': self.q.astype(np.float32),
                '_topp_trace': {'chunk_id': self.chunk_id, 'plan_ms': self.last_plan_ms,
                                'trajectory_time_s': self.elapsed}}

    def reset(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError('Previous inference still active; refusing concurrent episode reset')
        self.future = None
        self.policy.reset()
        self.trajectory, self.q = None, None
        self.elapsed, self.chunk_id = 0.0, 0
        self.last_requested = False

    def close(self):
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=1)
