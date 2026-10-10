"""Offline dispatch benchmark; no server, ROS or CAN. Not a task-success test."""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from topp_adaptive_planner import AdaptivePlanner
from topp_transition_planner import TransitionPlanner
from topp_committed_rtc_policy import CommittedRtcPolicy
from topp_policy import limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--transition', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seconds',type=float,default=6)
    parser.add_argument('--out',type=Path,default=Path('data/topp_rtc_dispatch_benchmark.json'))
    args=parser.parse_args()
    if args.input:
        z=np.load(args.input)
        q=z['q']; raw=z['actions']
        velocity,acceleration=z['velocity'],z['acceleration']
    else:
        q=np.zeros(14);raw=np.zeros((50,14));raw[:,0]=np.linspace(.001,.2,50)
        velocity,acceleration=limits(.5,.05),limits(1,.2)
    raw=np.concatenate([raw,np.repeat(raw[-1:],max(0,50-len(raw)),axis=0)])[:50]
    class Replay:
        def infer(self,obs):
            time.sleep(.25)  # simulated inference, not an actual model
            return {'actions':raw.copy()}
        def reset(self): pass
    planner_class=TransitionPlanner if args.transition else AdaptivePlanner
    planner=planner_class(velocity,acceleration,fps=15,path_fps=15)
    policy=CommittedRtcPolicy(Replay(),planner)
    events=[]
    policy._log=lambda **event: events.append(event)
    costs=[];intervals=[];previous=None
    start=time.monotonic()
    try:
        while time.monotonic()-start<args.seconds:
            tick=time.monotonic()
            if previous is not None: intervals.append((tick-previous)*1000)
            previous=tick
            output=policy.infer({'state':q})
            q=output['actions']
            costs.append((time.monotonic()-tick)*1000)
            time.sleep(max(0,1/15-(time.monotonic()-tick)))
    finally:
        policy.close()
    report={'source':str(args.input) if args.input else 'synthetic',
            'transition':args.transition, 'simulated_inference_ms':250,'seconds':args.seconds,'samples':len(costs),
            'policy_ms':{'p50':float(np.percentile(costs,50)),'p95':float(np.percentile(costs,95)),
                         'max':float(max(costs))},
            'interval_ms_p95':float(np.percentile(intervals,95)),
            'events':events}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))
    print('adopted',sum(e['event']=='adopted' for e in events),
          'rejected',sum(e['event']=='rejected' for e in events))


if __name__=='__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
