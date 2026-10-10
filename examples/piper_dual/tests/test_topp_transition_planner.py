from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from topp_transition_planner import TransitionPlanner, VelocityTransition
from topp_adaptive_planner import AdaptivePlanner
from topp_policy import limits

FIXTURE=Path(__file__).parent/'fixtures/rtc_nonzero_handoff_failure.npz'


def load_case(**kwargs):
    z=np.load(FIXTURE)
    p=TransitionPlanner(z['velocity'],z['acceleration'],
                        fps=float(z['fps']),path_fps=float(z['path_fps']),**kwargs)
    return p,z


def test_real_rejected_handoff_recovers_without_zeroing_velocity():
    p,z=load_case()
    with pytest.raises(RuntimeError,match='infeasible'):
        AdaptivePlanner(p.velocity,p.acceleration,path_fps=p.path_fps).plan(z['q'],z['dq'],z['actions'])
    t=p.plan(z['q'],z['dq'],z['actions'])
    assert isinstance(t,VelocityTransition)
    np.testing.assert_allclose(t(0),z['q'],atol=1e-7)
    np.testing.assert_allclose(t(0,1),z['dq'],atol=1e-7)
    ts=np.linspace(0,t.duration,50000)
    for order,bound in ((1,p.velocity),(2,p.acceleration)):
        assert np.all(np.abs(t(ts,order))<=bound+1e-7)
    offset=t(ts)-t.reference(ts/t.scale)
    assert np.all(np.abs(offset)<=p.deviation+1e-10)
    after=np.linspace(t.blend_s,t.duration,300)
    for order in (0,1,2):
        np.testing.assert_allclose(t(after,order),t.reference(after/t.scale,order)/t.scale**order,atol=1e-8)
    np.testing.assert_allclose(t(t.duration),z['actions'][-1],atol=1e-7)
    np.testing.assert_allclose(t(t.duration,1),0,atol=1e-7)
    # No simultaneous zero-speed stop inside the tested handoff transition.
    assert np.min(np.max(np.abs(t(np.linspace(0,t.blend_s,500),1)),axis=1))>1e-3


def test_tiny_deviation_budget_rejects_instead_of_ignoring_bound():
    p,z=load_case(joint_deviation=1e-10,gripper_deviation=1e-10)
    with pytest.raises(RuntimeError,match='No bounded velocity transition'):
        p.plan(z['q'],z['dq'],z['actions'])


def test_rest_start_keeps_existing_toppr_a_path():
    p,z=load_case()
    a=p.plan(z['q'],np.zeros(14),z['actions'])
    b=AdaptivePlanner(p.velocity,p.acceleration,path_fps=p.path_fps).plan(z['q'],np.zeros(14),z['actions'])
    ts=np.linspace(0,b.duration,200)
    np.testing.assert_allclose(a(ts),b(ts))
    assert not p.last_transition['used']


def test_analytic_correction_peak_and_derivative():
    p,z=load_case()
    t=p.plan(z['q'],z['dq'],z['actions'])
    ts=np.linspace(0,t.blend_s,10001)
    maximum=np.max(np.abs(t(ts)-t.reference(ts/t.scale)),axis=0)
    np.testing.assert_allclose(maximum,np.abs(z['dq'])*t.blend_s*16/81,atol=1e-9)
    x=t.blend_s*.4;eps=1e-6
    np.testing.assert_allclose((t(x+eps)-t(x-eps))/(2*eps),t(x,1),atol=1e-6)
    np.testing.assert_allclose((t(x+eps,1)-t(x-eps,1))/(2*eps),t(x,2),atol=1e-5)


@pytest.mark.parametrize('kwargs',[{'joint_deviation':0},{'gripper_deviation':-1},
                                  {'max_blend_seconds':float('nan')}])
def test_invalid_transition_settings(kwargs):
    with pytest.raises(ValueError): load_case(**kwargs)


def test_above_velocity_bound_not_silently_clamped():
    p,z=load_case()
    v=z['dq'].copy();v[6]=p.velocity[6]*2
    with pytest.raises(RuntimeError,match='Initial velocity exceeds'):
        p.plan(z['q'],v,z['actions'])


def test_feasible_direct_handoff_is_not_blended():
    p=TransitionPlanner(limits(.5,.05),limits(1,.2))
    baseline=AdaptivePlanner(p.velocity,p.acceleration)
    actions=np.zeros((22,14));actions[:,0]=np.linspace(.01,.4,22)
    old=baseline.plan(np.zeros(14),np.zeros(14),actions)
    q,v=old(.2),old(.2,1)
    actions[:,0]=np.linspace(q[0]+v[0]/15,.5,22)
    expected=baseline.plan(q,v,actions)
    actual=p.plan(q,v,actions)
    assert not p.last_transition['used']
    ts=np.linspace(0,expected.duration,500)
    np.testing.assert_allclose(actual(ts),expected(ts),atol=1e-9)


def test_failed_candidates_saved_with_deviation_axis_units_and_skips(tmp_path):
    import json
    p,z=load_case(joint_deviation=1e-10,gripper_deviation=1e-10)
    p.diagnostic_dir=tmp_path
    with pytest.raises(RuntimeError): p.plan(z['q'],z['dq'],z['actions'])
    report=json.loads((tmp_path/'last_transition_failure.json').read_text())
    assert report['outcome']=='rejected'
    assert report['summary']['rejected_candidates']==24
    candidate=report['candidates'][0]
    assert candidate['not_checked']==['velocity','acceleration']
    issues=candidate['violations']
    assert any(v['axis']=='left.gripper' and v['unit']=='m' for v in issues)
    assert all(v['constraint']=='deviation' and v['ratio']>1 for v in issues)
    assert all(v['actual']>v['limit'] for v in issues)
    assert p.last_transition['diagnostics']['worst_by_constraint']['deviation']['ratio']>1
    assert json.loads((tmp_path/'transition_candidates.jsonl').read_text())['diagnostic_id']==report['diagnostic_id']


def test_gripper_velocity_and_joint_acceleration_attribution():
    p,_=load_case()
    values=np.zeros((3,14));values[1,6]=.06
    v=p._violations('velocity',values,p.velocity,np.array([0,.1,.2]),tolerance=1e-7)[0]
    assert v['axis']=='left.gripper' and v['unit']=='m/s'
    assert v['time_s']==.1 and v['ratio']==pytest.approx(1.2)
    values=np.zeros((3,14));values[2,10]=-3
    a=p._violations('acceleration',values,p.acceleration,np.array([0,.1,.2]))[0]
    assert a['axis']=='right.joint4' and a['actual']==-3
    assert a['unit']=='rad/s^2' and a['ratio']==pytest.approx(1.5)


def test_duration_rejection_distinguished_from_unsearched_longer_blends(monkeypatch):
    from types import SimpleNamespace
    p,z=load_case()
    monkeypatch.setattr(p,'_initial_acceleration_feasible',lambda *args:False)
    monkeypatch.setattr(AdaptivePlanner,'plan',lambda *args:SimpleNamespace(duration=.01,_ts=np.array([0,.01])))
    with pytest.raises(RuntimeError): p.plan(z['q'],z['dq'],z['actions'])
    report=p.last_diagnostics
    assert report['summary']['violation_counts']=={'duration':24}
    assert report['search']['max_blend_seconds']==.6
    for c in report['candidates']:
        assert c['checked']==['duration']
        assert c['violations'][0]['reason']=='blend exceeds available reference duration'


def test_success_preserves_failed_candidate_diagnostics(tmp_path):
    import json
    p,z=load_case();p.diagnostic_dir=tmp_path
    p.plan(z['q'],z['dq'],z['actions'])
    report=json.loads((tmp_path/'transition_candidates.jsonl').read_text())
    assert report['outcome']=='accepted'
    assert report['candidates'][-1]['status']=='accepted'
    assert report['summary']['violation_counts'].get('acceleration',0)>0
    assert not (tmp_path/'last_transition_failure.json').exists()
