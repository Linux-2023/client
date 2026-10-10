"""Experimental RTC conditioning on a time-resampled TOPP plan.

Wire contract follows local pi0 RTC: H-s is the soft prefix length, d the
committed prefix. Model time is independent of output ticks. No robot I/O.
"""
from pathlib import Path
from concurrent.futures import Future
import copy
import json
import logging
import math
import threading
import time
import numpy as np
from topp_policy import Planner, limits, vector

class ToppRtcPolicy:
    """One inference worker; immutable chunk handoff; never emits unplanned targets.

    Time advances by one output tick, not wall-time catch-up. Slow solves stretch
    execution rather than skipping trajectory samples. Planning is synchronous;
    latency is measured explicitly. Terminal zero velocity provides a stop if
    the next chunk is late. Acceleration/jerk continuity is NOT guaranteed.
    """
    def __init__(self, policy, planner, *, horizon=22, prefetch_seconds=.4,
                 asynchronous=True, metrics_path=None, clock=time.monotonic,
                 model_horizon=50, actions_during_latency=5):
        if horizon < 1 or not np.isfinite(prefetch_seconds) or prefetch_seconds < 0:
            raise ValueError('Invalid horizon or prefetch time')
        if not asynchronous:
            raise ValueError('RTC TOPP requires asynchronous inference')
        if not 0 < horizon < model_horizon or not 0 <= actions_during_latency < model_horizon-horizon:
            raise ValueError('Require 0 < action horizon < model horizon and 0 <= delay < H-s')
        self.model_horizon = model_horizon
        self.initial_delay = actions_during_latency
        self.latency_s = None
        self.tick = 0
        self.request_tick = 0
        self.request_has_plan = False
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
        self.request_tick = self.tick
        self.request_has_plan = self.trajectory is not None
        # Prefix index zero corresponds to this request's planned position.
        times = self.elapsed + np.arange(self.model_horizon)/self.planner.path_fps
        if self.trajectory is None:
            prefix = np.repeat(self.q[None, :], self.model_horizon, axis=0)
        else:
            prefix = np.asarray(self.trajectory(np.minimum(times, self.trajectory.duration)))
        delay = self.initial_delay if self.latency_s is None else math.ceil(self.latency_s*self.planner.path_fps)
        delay = min(delay, self.model_horizon-self.horizon-1)
        obs['rtc_obs'] = {'prev_action': prefix.astype(np.float32),
                          'prev_state': vector(observation['state']).astype(np.float32),
                          's': self.horizon, 'd': delay}
        self._log(event='request', rtc_s=self.horizon, rtc_d=delay,
                  model_fps=self.planner.path_fps, output_fps=self.planner.fps)
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
        self.tick += 1
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
                raw = np.asarray(result['actions'], dtype=float)
                if raw.shape != (self.model_horizon, 14) or not np.isfinite(raw).all():
                    raise ValueError(f'RTC server actions must have shape ({self.model_horizon},14) and be finite')
                self.latency_s = inference_ms/1000
                # Logical execution time, not fps-dependent raw action indices.
                waited = (self.tick-self.request_tick)/self.planner.fps
                skip = math.floor(waited*self.planner.path_fps + 1e-9) if self.request_has_plan else 0
                if skip >= self.model_horizon:
                    raise ValueError('RTC result fully expired; retain old stopping trajectory')
                chunk = raw[skip:skip+self.horizon]
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
                self._log(event='adopted', skipped_steps=skip, chunk_id=self.chunk_id, plan_ms=self.last_plan_ms,
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
        self.tick, self.request_tick = 0, 0
        self.request_has_plan = False
        self.latency_s = None
        self.last_requested = False

    def close(self):
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=1)
