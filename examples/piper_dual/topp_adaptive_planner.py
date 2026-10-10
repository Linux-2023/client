"""RTC-only adaptive TOPP grid refinement; retains configured limits and C1 boundary."""
import logging
from pathlib import Path
import numpy as np
from topp_policy import load_toppra, vector

class AdaptivePlanner:
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
        last_error = 'No feasible trajectory'
        self.attempts = 0
        for factor in (1, 4, 16):
            self.attempts += 1
            grid = np.unique(np.r_[np.linspace(knots[0], knots[-1], self.gridpoints*factor), knots])
            algorithm = self.ta.algorithm.TOPPRA(constraints, path, gridpoints=grid,
                                                solver_wrapper='seidel', parametrizer='ParametrizeConstAccel')
            trajectory = algorithm.compute_trajectory(1.0 if moving else 0.0, 0.0)
            if trajectory is None or not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
                last_error = 'TOPP path infeasible from current planned velocity'
                continue
            # Include both sides of each constant-acceleration interval, not
            # just uniform wall-time samples which can miss short intervals.
            boundaries = np.asarray(trajectory._ts)  # pinned toppra 0.6.3
            fractions = np.linspace(1e-7, 1-1e-7, 9)
            interior = (boundaries[:-1,None] + np.diff(boundaries)[:,None]*fractions).ravel()
            ts = np.unique(np.r_[0, trajectory.duration, interior,
                                 np.linspace(0, trajectory.duration, max(1000,len(grid)*5))])
            failure = None
            for order, bound in ((1, self.velocity), (2, self.acceleration)):
                values = trajectory(ts, order)
                if not np.isfinite(values).all():
                    raise RuntimeError(f'Nonfinite TOPP derivative {order}')
                ratios = np.abs(values)/bound
                index = np.unravel_index(np.argmax(ratios), ratios.shape)
                peak = float(ratios[index])
                if peak > 1.01 + 1e-7/float(bound[index[1]]):
                    dim = int(index[1])
                    arm = 'left' if dim < 7 else 'right'
                    joint = 'gripper' if dim % 7 == 6 else f'joint{dim%7+1}'
                    failure = (f'TOPP derivative {order}: {arm}.{joint} '
                               f'actual={values[index]:.6g}, limit={bound[dim]:.6g}, '
                               f'ratio={peak:.6g}, t={ts[index[0]]:.6g}s, grid={len(grid)}')
                    break
            if failure is None:
                if not np.allclose(trajectory(0), q, atol=1e-7) or not np.allclose(trajectory(0,1), dq, atol=1e-5):
                    raise RuntimeError('TOPP handoff does not preserve position/velocity')
                if self.attempts > 1:
                    logging.info('TOPP refined solution accepted: attempts=%s grid=%s', self.attempts, len(grid))
                return trajectory
            last_error = failure
            logging.warning('%s; retry with finer grid', failure)
        if getattr(self, 'diagnostic_dir', None) is not None:
            destination = Path(self.diagnostic_dir)
            destination.mkdir(parents=True, exist_ok=True)
            np.savez(destination/'last_rejected_plan.npz', q=q, dq=dq, actions=actions,
                     velocity=self.velocity, acceleration=self.acceleration,
                     fps=self.fps, path_fps=self.path_fps, gridpoints=self.gridpoints)
        raise RuntimeError(last_error + '; adaptive retries exhausted')
