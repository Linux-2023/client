"""Bounded velocity-matching transition after the committed RTC prefix.

Try exact-path TOPP first. If infeasible, blend a finite-duration velocity
correction into a time-scaled, rest-start TOPP reference. Never clear dq while
moving or modify the already committed prefix.
"""
import json
from collections import Counter
from pathlib import Path
import logging
import time
import numpy as np
from topp_adaptive_planner import AdaptivePlanner
from topp_policy import vector, limits


class VelocityTransition:
    def __init__(self, reference, velocity, blend_s, scale):
        self.reference, self.velocity = reference, velocity.copy()
        self.blend_s, self.scale = blend_s, scale
        self.duration = reference.duration*scale

    def __call__(self, times, order=0):
        t = np.asarray(times, dtype=float)
        u = np.clip(t/self.blend_s, 0, 1)
        active = (t >= 0) & (t < self.blend_s)
        if order == 0:
            h = self.blend_s*(u-6*u**3+8*u**4-3*u**5)
        elif order == 1:
            h = 1-18*u**2+32*u**3-15*u**4
        elif order == 2:
            h = (-36*u+96*u**2-60*u**3)/self.blend_s
        else:
            raise ValueError('Only derivatives 0, 1, 2 supported')
        correction = np.where(active, h, 0)[..., None]*self.velocity
        return self.reference(t/self.scale, order)/self.scale**order + correction


class TransitionPlanner(AdaptivePlanner):
    def __init__(self, *args, joint_deviation=.03, gripper_deviation=.002,
                 max_blend_seconds=.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.deviation = limits(joint_deviation, gripper_deviation)
        if not np.isfinite(max_blend_seconds) or max_blend_seconds <= 0:
            raise ValueError('Transition duration must be finite and positive')
        self.max_blend_seconds = max_blend_seconds
        self.last_transition = {}

    def _initial_acceleration_feasible(self, q, v, actions):
        points = [q]
        for row in actions:
            if np.max(np.abs(row-points[-1])/self.velocity)>1e-8:
                points.append(row.copy())
        if len(points)<2:
            return False
        path=self.ta.SplineInterpolator(np.arange(len(points))/self.path_fps,
                    np.asarray(points), bc_type=((1,v),(2,np.zeros(14))))
        curvature=np.asarray(path(0,2))
        # At sd=1, qdd = curvature + v*sdd. A common scalar sdd must
        # satisfy every joint's acceleration interval. Empty => no grid can fix it.
        lower, upper = -np.inf, np.inf
        bounds=.95*self.acceleration
        for tangent, c, bound in zip(v,curvature,bounds):
            if abs(tangent)<1e-12:
                if abs(c)>bound+1e-8:
                    return False
            else:
                ends=sorted(((-bound-c)/tangent,(bound-c)/tangent))
                lower,upper=max(lower,ends[0]),min(upper,ends[1])
        return lower <= upper

    @staticmethod
    def _axis(index):
        return ('left' if index < 7 else 'right') + '.' + ('gripper' if index % 7 == 6 else f'joint{index%7+1}')

    def _violations(self, constraint, values, bound, ts=None, tolerance=0):
        values=np.asarray(values)
        matrix=values.reshape(-1,14)
        issues=[]
        for axis in range(14):
            column=matrix[:,axis]
            finite=np.isfinite(column)
            if not finite.all():
                row=int(np.flatnonzero(~finite)[0])
                issues.append(dict(constraint=constraint, axis=self._axis(axis),
                    kind='nonfinite', actual=None, limit=float(bound[axis]), ratio=None,
                    time_s=None if ts is None else float(ts[row])))
                continue
            row=int(np.argmax(np.abs(column)))
            actual=float(column[row]);limit=float(bound[axis])
            if abs(actual)>limit+tolerance:
                unit=('m' if axis%7==6 else 'rad')
                if constraint=='velocity': unit+='/s'
                elif constraint=='acceleration': unit+='/s^2'
                issues.append(dict(constraint=constraint,axis=self._axis(axis),kind='exceeded',
                    actual=actual, magnitude=abs(actual), limit=limit, ratio=abs(actual)/limit,
                    unit=unit,time_s=None if ts is None else float(ts[row])))
        return issues

    def _finish_diagnostics(self):
        report=self.last_diagnostics
        counts=Counter(v['constraint'] for c in report['candidates'] for v in c['violations'])
        axes=Counter(v.get('axis','trajectory') for c in report['candidates'] for v in c['violations'])
        report['summary']={'evaluated_candidates':len(report['candidates']),
            'rejected_candidates':sum(c['status']=='rejected' for c in report['candidates']),
            'violation_counts':dict(counts),'axis_counts':dict(axes)}
        worst={}
        for candidate in report['candidates']:
            for violation in candidate['violations']:
                name=violation['constraint']
                if name not in worst or (violation.get('ratio') or 0) > (worst[name].get('ratio') or 0):
                    worst[name]=dict(violation,blend_s=candidate['blend_s'],time_scale=candidate['time_scale'])
        report['summary']['worst_by_constraint']=worst
        summary=dict(report['summary'], outcome=report['outcome'], diagnostic_id=report['diagnostic_id'])
        # Summaries travel with success AND failure; detailed candidates stay on disk.
        self.last_transition['diagnostics']=summary
        destination=getattr(self,'diagnostic_dir',None)
        if destination is not None and (report['candidates'] or report['outcome']=='rejected'):
            try:
                directory=Path(destination);directory.mkdir(parents=True,exist_ok=True)
                with (directory/'transition_candidates.jsonl').open('a') as f:
                    f.write(json.dumps(report,allow_nan=False)+'\n')
                if report['outcome']=='rejected':
                    (directory/'last_transition_failure.json').write_text(json.dumps(report,indent=2,allow_nan=False))
                summary['details_file']=str(directory/'transition_candidates.jsonl')
            except (OSError,ValueError):
                logging.exception('Failed to save transition diagnostic details')
        if report['candidates']:
            logging.info('TOPP transition candidate diagnostics: %s',summary)

    def plan(self, q, dq, actions):
        self.last_transition={'used':False}
        self.last_diagnostics={'diagnostic_id':time.time_ns(), 'outcome':'pending',
            'search':{'max_blend_seconds':self.max_blend_seconds,
                      'time_scales':[1.,1.15,1.35,1.6],
                      'velocity_limits':self.velocity.tolist(),
                      'acceleration_limits':self.acceleration.tolist(),
                      'extra_deviation_limits':self.deviation.tolist(),
                      'path_fps':self.path_fps,
                      'duration_note':'Candidates outside the configured search are not tested; failure does not prove a longer transition is necessary.'},
            'candidates':[]}
        try:
            result=self._plan(q,dq,actions)
            self.last_diagnostics['outcome']='accepted'
            return result
        except Exception as exc:
            self.last_diagnostics['outcome']='rejected'
            self.last_diagnostics['error']=str(exc)
            raise
        finally:
            self._finish_diagnostics()

    def _plan(self, q, dq, actions):
        begin=time.monotonic()
        q,dq=vector(q),vector(dq)
        actions=np.asarray(actions,dtype=float)
        self.last_transition={'used':False}
        if actions.ndim!=2 or actions.shape[1]!=14 or len(actions)<1 or not np.isfinite(actions).all():
            raise ValueError('Policy actions must have shape (N,14) and be finite')
        if np.max(np.abs(dq))<1e-8:
            return super().plan(q,dq,actions)
        if np.any(np.abs(dq)>self.velocity+1e-7):
            raise RuntimeError('Initial velocity exceeds configured bound')
        direct_error='Initial acceleration intervals do not intersect'
        if self._initial_acceleration_feasible(q,dq,actions):
            try:
                result=super().plan(q,dq,actions)
                if result is not None:
                    return result
            except RuntimeError as exc:
                direct_error=str(exc)
        self.last_diagnostics['direct_failure']=direct_error
        # Use a separate planner so no changed constraints leak into direct mode.
        reference_planner=AdaptivePlanner(self.velocity,self.acceleration,
            fps=self.fps,path_fps=self.path_fps,gridpoints=self.gridpoints)
        reference=reference_planner.plan(q,np.zeros(14),actions)
        if reference is None:
            raise RuntimeError('No moving reference for bounded transition; retain old stopping plan')
        # The only path change is the bounded quintic correction. Try the
        # fastest reference first, then modest slowing to reserve acceleration.
        for scale in (1.,1.15,1.35,1.6):
            boundaries=np.asarray(reference._ts)*scale  # pinned toppra 0.6.3
            for blend in np.linspace(min(.1,self.max_blend_seconds),self.max_blend_seconds,6):
                record={'blend_s':float(blend),'time_scale':scale,
                        'reference_duration_s':reference.duration*scale,
                        'status':'rejected','violations':[], 'checked':[], 'not_checked':[]}
                self.last_diagnostics['candidates'].append(record)
                if blend>reference.duration*scale or blend<=0:
                    record['checked']=['duration']
                    record['violations']=[dict(constraint='duration', axis='trajectory',
                        actual=float(blend),limit=reference.duration*scale,
                        ratio=float(blend)/(reference.duration*scale),unit='s',
                        reason='blend exceeds available reference duration' if blend>0 else 'nonpositive blend')]
                    record['not_checked']=['deviation','velocity','acceleration']
                    continue
                record['checked']=['duration','deviation']
                candidate=VelocityTransition(reference,dq,float(blend),scale)
                # h(u) peaks at u=1/3: exact offset bound, per dimension.
                peak_offset=np.abs(dq)*blend*(16/81)
                record['violations'].extend(self._violations('deviation',peak_offset,self.deviation))
                if record['violations']:
                    record['not_checked']=['velocity','acceleration']
                    continue
                fractions=np.linspace(1e-7,1-1e-7,9)
                interior=(boundaries[:-1,None]+np.diff(boundaries)[:,None]*fractions).ravel()
                ts=np.unique(np.r_[0,candidate.duration,blend,interior,
                        np.linspace(0,blend,401),np.linspace(0,candidate.duration,2000)])
                valid=True
                for order,bound in ((1,self.velocity),(2,self.acceleration)):
                    values=candidate(ts,order)
                    constraint='velocity' if order==1 else 'acceleration'
                    record['checked'].append(constraint)
                    violations=self._violations(constraint,values,bound,ts,tolerance=1e-7)
                    record['violations'].extend(violations)
                    if violations:
                        valid=False
                if not valid:
                    continue
                if not np.allclose(candidate(0),q,atol=1e-7) or not np.allclose(candidate(0,1),dq,atol=1e-6):
                    raise RuntimeError('Transition changed the committed handoff boundary')
                record['status']='accepted'
                self.last_transition={'used':True,'blend_s':float(blend),'time_scale':scale,
                    'max_joint_offset_rad':float(np.max(np.delete(peak_offset,[6,13]))),
                    'max_gripper_offset_m':float(np.max(peak_offset[[6,13]])),
                    'direct_failure':direct_error,'total_ms':(time.monotonic()-begin)*1000}
                logging.info('TOPP bounded transition accepted: %s',self.last_transition)
                return candidate
        raise RuntimeError('No bounded velocity transition satisfies velocity/acceleration/deviation limits; '+direct_error)
