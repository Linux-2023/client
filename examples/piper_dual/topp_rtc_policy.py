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
from topp_policy import limits, vector
from topp_adaptive_planner import AdaptivePlanner as Planner

class SplicedTrajectory:
    """Keep an exact prefix of the old plan, then follow a C1-matched new plan."""
    def __init__(self, old, offset, prefix_duration, tail):
        while isinstance(old, SplicedTrajectory) and offset >= old.prefix_duration:
            offset -= old.prefix_duration
            old = old.tail
        self.old, self.offset = old, offset
        self.prefix_duration, self.tail = prefix_duration, tail
        self.duration = prefix_duration + tail.duration

    def __call__(self, times, order=0):
        values = np.asarray(times)
        flat = values.reshape(-1)
        result = np.empty((len(flat), 14))
        before = flat < self.prefix_duration
        if np.any(before):
            result[before] = self.old(self.offset+flat[before], order)
        if np.any(~before):
            result[~before] = self.tail(flat[~before]-self.prefix_duration, order)
        return result.reshape(values.shape+(14,))


class ToppRtcPolicy:
    """One inference worker; immutable chunk handoff; never emits unplanned targets.

    Time advances by one output tick, not wall-time catch-up. Slow solves stretch
    execution rather than skipping trajectory samples. Planning is synchronous;
    latency is measured explicitly. Terminal zero velocity provides a stop if
    the next chunk is late. Acceleration/jerk continuity is NOT guaranteed.
    """
    def __init__(self, policy, planner, *, horizon=22, prefetch_seconds=.4,
                 asynchronous=True, metrics_path=None, clock=time.monotonic,
                 model_horizon=50, actions_during_latency=5, rolling=True,
                 commit_seconds=.1, planning_margin=.05):
        if horizon < 1 or not np.isfinite(prefetch_seconds) or prefetch_seconds < 0:
            raise ValueError('Invalid horizon or prefetch time')
        if not asynchronous:
            raise ValueError('RTC TOPP requires asynchronous inference')
        if not 0 < horizon < model_horizon or not 0 <= actions_during_latency < model_horizon-horizon:
            raise ValueError('Require 0 < action horizon < model horizon and 0 <= delay < H-s')
        if not np.isfinite(commit_seconds) or commit_seconds < 0 or not np.isfinite(planning_margin) or planning_margin < 0:
            raise ValueError('Commit time and planning margin must be finite and nonnegative')
        self.rolling = rolling
        self.commit_seconds, self.planning_margin = commit_seconds, planning_margin
        self.request_at = 0.0
        self.retry_after_tick = 0
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
        self.request_has_plan = self.trajectory is not None and self.elapsed < self.trajectory.duration
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

    def _schedule(self):
        if self.trajectory is None:
            self.request_at = 0.0
            return
        # Conservative heuristic: last near-peak normalized speed precedes
        # terminal deceleration. This is not an exact braking-time detector.
        ts = np.linspace(0, self.trajectory.duration, 400)
        speed = np.max(np.abs(self.trajectory(ts, 1))/self.planner.velocity, axis=1)
        peak = float(np.max(speed))
        brake_at = float(ts[np.flatnonzero(speed >= .98*peak)[-1]]) if peak > 1e-8 else 0.0
        latency = self.initial_delay/self.planner.path_fps if self.latency_s is None else self.latency_s
        lead = max(self.prefetch_seconds, latency+self.last_plan_ms/1000+self.planning_margin+self.commit_seconds)
        self.request_at = max(1/self.planner.fps, min(self.horizon/self.planner.path_fps,
                                                    brake_at-lead))
        self._log(event='schedule', request_at_s=self.request_at, brake_estimate_s=brake_at, lead_s=lead)

    def _plan_rolling(self, raw, skip, dq):
        # Consume the full available prediction, placing the stopping endpoint
        # beyond the shorter execution/replanning horizon.
        remaining = 0 if self.trajectory is None else self.trajectory.duration-self.elapsed
        count = min(math.ceil(self.commit_seconds*self.planner.path_fps),
                    max(0, len(raw)-skip-2), math.floor(remaining*self.planner.path_fps))
        buffer_s = count/self.planner.path_fps
        if count:
            join = self.elapsed+buffer_s
            q, v = self.trajectory(join), self.trajectory(join, 1)
            tail = self.planner.plan(q, v, raw[skip+count:])
            if tail is None:
                return None
            return SplicedTrajectory(self.trajectory, self.elapsed, buffer_s, tail)
        return self.planner.plan(self.q, dq, raw[skip:])

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
        due = self.trajectory is None or (self.elapsed >= self.request_at if self.rolling else remaining <= self.prefetch_seconds)
        if not self.last_requested and due and self.tick >= self.retry_after_tick:
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
                trajectory = self._plan_rolling(raw, skip, dq) if self.rolling else self.planner.plan(self.q, dq, chunk)
            except (ValueError, RuntimeError, KeyError) as exc:
                # Retain the previous feasible stopping trajectory.
                self._log(event='rejected', reason=str(exc), inference_ms=inference_ms)
                if self.rolling:
                    if ('path infeasible' in str(exc) and self.trajectory is not None
                            and self.elapsed < self.trajectory.duration):
                        # Keep executing the valid deceleration, then request a
                        # fresh observation at rest rather than retrying every tick.
                        self.request_at = self.trajectory.duration
                        self._log(event='recover_at_rest', remaining_s=self.trajectory.duration-self.elapsed)
                    else:
                        # elapsed saturates at trajectory.duration: a deadline
                        # expressed as elapsed + .1 can become unreachable.
                        self.request_at = self.elapsed
                    self.retry_after_tick = self.tick + max(1, math.ceil(.1*self.planner.fps))
                self.last_requested = False
            else:
                self.last_plan_ms = (self.clock()-start)*1000
                if trajectory is not None:
                    self.trajectory, self.elapsed = trajectory, 0.0
                    if self.rolling:
                        self._schedule()
                self.chunk_id += 1
                self.last_requested = False
                self._log(event='adopted', skipped_steps=skip, chunk_id=self.chunk_id, plan_ms=self.last_plan_ms,
                          inference_ms=inference_ms, duration_s=0 if trajectory is None else trajectory.duration,
                          nominal_s=(len(raw)-skip if self.rolling else len(chunk))/self.planner.path_fps,
                          rolling=self.rolling,
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
        self.request_at = 0.0
        self.retry_after_tick = 0
        self.last_requested = False

    def close(self):
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=1)
