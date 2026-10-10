"""RTC with an immutable execution prefix and background TOPP planning.

One worker owns inference and planning. Main thread alone owns execution state.
A returned plan may only start at the pre-announced output tick, never late.
"""
from concurrent.futures import Future
from collections import deque
from fractions import Fraction
from pathlib import Path
import copy
import json
import logging
import math
import threading
import time

import numpy as np
from topp_policy import vector


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported log value: {type(value).__name__}')


class CommittedRtcPolicy:
    def __init__(self, policy, planner, *, horizon=22, model_horizon=50,
                 actions_during_latency=5, prefetch_seconds=.4,
                 commit_seconds=.1, planning_margin=.05, plan_budget=.2,
                 soft_prefix_steps=3, metrics_path=None, clock=time.monotonic):
        if not 0 < horizon < model_horizon:
            raise ValueError('Require 0 < action horizon < model horizon')
        if not 1 <= soft_prefix_steps < model_horizon-2:
            raise ValueError('RTC soft prefix must leave room for committed and new actions')
        if not 0 <= actions_during_latency < model_horizon-soft_prefix_steps-1:
            raise ValueError('RTC initial delay is too long for the model horizon')
        if any(not np.isfinite(x) or x < 0 for x in
               (prefetch_seconds, commit_seconds, planning_margin, plan_budget)):
            raise ValueError('Timing parameters must be finite and nonnegative')
        self.policy, self.planner = policy, planner
        self.horizon, self.model_horizon = horizon, model_horizon
        self.initial_delay = actions_during_latency
        self.prefetch_seconds, self.commit_seconds = prefetch_seconds, commit_seconds
        self.planning_margin, self.plan_budget = planning_margin, plan_budget
        self.soft_prefix_steps = soft_prefix_steps
        self.clock = clock
        self.metrics_path = Path(metrics_path) if metrics_path else None
        # Join exactly on BOTH model and output grids (including rational rates).
        output_dt = 1/Fraction(str(planner.fps))
        model_dt = 1/Fraction(str(planner.path_fps))
        self.common_dt = Fraction(math.lcm(output_dt.numerator, model_dt.numerator),
                                  math.gcd(output_dt.denominator, model_dt.denominator))
        self.model_stride = int(self.common_dt/model_dt)
        self.output_stride = int(self.common_dt/output_dt)
        if self.model_stride >= model_horizon-soft_prefix_steps-1:
            raise ValueError('Model/output frequencies have no usable common handoff tick')
        self.latencies = deque(maxlen=10)
        self.plan_times = deque(maxlen=10)
        self.thread = self.future = None
        self.closed = False
        self._clear()

    def _clear(self):
        self.tick = self.origin_tick = self.request_at_tick = 0
        self.trajectory = self.q = self.pending = self.job = None
        self.chunk_id = 0
        self.moving_retries = 0
        self.last_plan_ms = 0.0
        self.last_call_wall = None

    def _log(self, **event):
        logging.info('TOPP committed %s', event)
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(dict(time_ns=time.time_ns(), **event), default=_json_scalar)
            with self.metrics_path.open('a') as f:
                f.write(line+'\n')

    def _state_at(self, tick):
        if self.trajectory is None:
            return self.q.copy(), np.zeros(14)
        t = min(max(0, (tick-self.origin_tick)/self.planner.fps), self.trajectory.duration)
        q = np.asarray(self.trajectory(t)).reshape(14)
        v = np.asarray(self.trajectory(t, 1)).reshape(14) if t < self.trajectory.duration else np.zeros(14)
        return q, v

    def _commit_window(self):
        inference = max(self.latencies, default=self.initial_delay/self.planner.path_fps)
        planning = max(self.plan_budget, max(self.plan_times, default=0))
        seconds = max(self.commit_seconds, inference+planning+self.planning_margin)
        multiples = max(1, math.ceil(seconds/float(self.common_dt)))
        maximum = (self.model_horizon-self.soft_prefix_steps-2)//self.model_stride
        saturated = multiples > maximum
        multiples = min(multiples, maximum)
        return multiples*self.output_stride, multiples*self.model_stride, saturated

    def _request(self, observation):
        ticks, d, saturated = self._commit_window()
        target_tick = self.tick+ticks
        q, v = self._state_at(target_tick)
        if self.trajectory is None:
            prefix = np.repeat(self.q[None, :], self.model_horizon, axis=0)
        else:
            times = (self.tick-self.origin_tick)/self.planner.fps + np.arange(self.model_horizon)/self.planner.path_fps
            prefix = self.trajectory(np.clip(times, 0, self.trajectory.duration))
        # s controls H-s soft-prefix length in the local RTC server; it is NOT
        # the number of points we retain for the new TOPP plan.
        s = self.model_horizon-(d+self.soft_prefix_steps)
        obs = copy.deepcopy(observation)
        obs['rtc_obs'] = {'prev_action': np.asarray(prefix, dtype=np.float32),
                          'prev_state': vector(observation['state']).astype(np.float32),
                          's': s, 'd': d}
        self.job = dict(target_tick=target_tick, request_tick=self.tick, d=d, q=q, v=v)
        self._log(event='request', rtc_s=s, rtc_d=d, target_tick=target_tick,
                  committed_s=ticks/self.planner.fps, budget_saturated=saturated,
                  output_fps=self.planner.fps, model_fps=self.planner.path_fps)
        future = self.future = Future()

        def work():
            begin = self.clock()
            try:
                # Server/network errors remain fatal, distinct from infeasible plans.
                result = self.policy.infer(obs)
                inference_s = self.clock()-begin
            except BaseException as exc:
                future.set_exception(exc)
                return
            start = self.clock()
            try:
                raw = np.asarray(result['actions'], dtype=float)
                if raw.shape != (self.model_horizon, 14) or not np.isfinite(raw).all():
                    raise ValueError(f'RTC actions must have shape ({self.model_horizon},14) and be finite')
                # Positions before d are committed; q at d is the exact anchor.
                # The first new target is d+1, avoiding a duplicate seam waypoint.
                chunk = raw[d+1:d+1+self.horizon]
                trajectory = self.planner.plan(q, v, chunk)
                brake_at = 0.0
                if trajectory is not None:
                    ts = np.linspace(0, trajectory.duration, 300)
                    speeds = np.max(np.abs(trajectory(ts,1))/self.planner.velocity, axis=1)
                    peak = float(np.max(speeds))
                    if peak > 1e-8:
                        brake_at = float(ts[np.flatnonzero(speeds >= .98*peak)[-1]])
                future.set_result(dict(trajectory=trajectory, brake_at=brake_at,
                                       inference_s=inference_s, plan_s=self.clock()-start,
                                       points=len(chunk), error=None,
                                       transition=copy.deepcopy(self.planner.last_transition)
                                       if isinstance(getattr(self.planner, 'last_transition', None), dict) else {}))
            except (ValueError, RuntimeError, KeyError) as exc:
                diagnostic_dir = getattr(self.planner, 'diagnostic_dir', None)
                if isinstance(diagnostic_dir, (str, Path)) and 'chunk' in locals():
                    try:
                        destination = Path(diagnostic_dir)
                        destination.mkdir(parents=True, exist_ok=True)
                        np.savez(destination/'last_committed_failure.npz', q=q, dq=v, actions=chunk,
                                 velocity=self.planner.velocity, acceleration=self.planner.acceleration,
                                 fps=self.planner.fps, path_fps=self.planner.path_fps,
                                 reason=str(exc))
                    except OSError:
                        logging.exception('Could not save failed committed plan')
                future.set_result(dict(error=str(exc), inference_s=inference_s,
                                       plan_s=self.clock()-start,
                                       transition=copy.deepcopy(self.planner.last_transition)
                                       if isinstance(getattr(self.planner,'last_transition',None),dict) else {}))
            except BaseException as exc:
                future.set_exception(exc)

        self.thread = threading.Thread(target=work, daemon=True, name='rtc-inference-topp')
        self.thread.start()

    def _recover(self, reason, at_rest=False):
        retry_tick = self.tick+max(1, math.ceil(.1*self.planner.fps))
        # A failed result cannot revoke the already announced old prefix.
        if self.job is not None:
            retry_tick = max(retry_tick, self.job['target_tick']+1)
        self.request_at_tick = retry_tick
        retry_moving = False
        if at_rest and self.trajectory is not None:
            stop_tick = self.origin_tick+math.ceil(self.trajectory.duration*self.planner.fps)
            budget_ticks, _, saturated = self._commit_window()
            retry_target = retry_tick+budget_ticks
            recoverable = 'infeasible' in reason or 'No bounded velocity transition' in reason
            # One further request per adopted plan, with a full pipeline budget
            # before the old endpoint and a genuinely moving future join.
            if recoverable and not saturated and self.moving_retries < 1 and retry_target < stop_tick:
                _, future_v = self._state_at(retry_target)
                retry_moving = bool(np.max(np.abs(future_v)) > 1e-4)
            if retry_moving:
                self.moving_retries += 1
            else:
                self.request_at_tick = max(retry_tick, stop_tick)
        self._log(event='rejected', reason=reason, at_rest=at_rest and not retry_moving,
                  retry_moving=retry_moving, retry_at_tick=self.request_at_tick)
        self.pending = self.job = None

    def infer(self, observation):
        begin = self.clock()
        if self.closed:
            raise RuntimeError('RTC TOPP policy closed')
        if self.q is None:
            self.q = vector(observation['state'])
        self.tick += 1
        self.q, _ = self._state_at(self.tick)
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            result = future.result()
            self.latencies.append(result['inference_s'])
            self.plan_times.append(result['plan_s'])
            self.last_plan_ms = result['plan_s']*1000
            self._log(event='worker_done', inference_ms=result['inference_s']*1000,
                      plan_ms=self.last_plan_ms, target_tick=self.job['target_tick'], tick=self.tick,
                      transition=result.get('transition', {}))
            if result['error'] is not None:
                self._recover(result['error'], at_rest=True)
            elif self.tick > self.job['target_tick']:
                self._recover('missed committed handoff deadline; old plan retained')
            else:
                self.pending = result
        if self.pending is not None and self.tick == self.job['target_tick']:
            result, job = self.pending, self.job
            trajectory = result['trajectory']
            if trajectory is not None:
                if (not np.allclose(trajectory(0), self.q, atol=1e-7)
                        or not np.allclose(trajectory(0,1), job['v'], atol=1e-5)):
                    raise RuntimeError('Committed handoff position/velocity mismatch')
                self.trajectory, self.origin_tick = trajectory, self.tick
                self.moving_retries = 0
            self.chunk_id += 1
            ticks, _, _ = self._commit_window()
            lead = max(self.prefetch_seconds, ticks/self.planner.fps)
            duration = 0 if trajectory is None else trajectory.duration
            delay = max(1/self.planner.fps, min(duration*.5, result['brake_at']-lead))
            self.request_at_tick = self.tick+max(1, math.ceil(delay*self.planner.fps))
            self._log(event='adopted', chunk_id=self.chunk_id, target_tick=self.tick,
                      start_velocity_max=float(np.max(np.abs(job['v']))), points=result['points'],
                      duration_s=duration, request_after_s=delay, plan_ms=self.last_plan_ms,
                      transition=result.get('transition', {}))
            self.pending = self.job = None
        if self.future is None and self.pending is None and self.tick >= self.request_at_tick:
            self._request(observation)
        elapsed = 0 if self.trajectory is None else min((self.tick-self.origin_tick)/self.planner.fps, self.trajectory.duration)
        wall_dt = 0 if self.last_call_wall is None else begin-self.last_call_wall
        self.last_call_wall = begin
        return {'actions': self.q.astype(np.float32), '_topp_trace': {
            'chunk_id': self.chunk_id, 'plan_ms': self.last_plan_ms, 'trajectory_time_s': elapsed,
            'policy_ms': (self.clock()-begin)*1000, 'call_interval_ms': wall_dt*1000,
            'committed': True}}

    def reset(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError('Worker still active; refusing concurrent reset')
        self.policy.reset()
        self.future = self.thread = None
        self.latencies.clear()
        self.plan_times.clear()
        self._clear()

    def close(self):
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=1)
