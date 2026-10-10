#!/usr/bin/env python3
"""No ROS/server: benchmark synthetic or .npy joint chunks with TOPP-RA."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from topp_policy import Planner, limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Optional .npy array (chunks,points,14), SI units')
    parser.add_argument('--output',type=Path,default=Path('data/topp_benchmark.json'))
    args=parser.parse_args()
    if args.input:
        chunks=np.load(args.input,allow_pickle=False)
        if chunks.ndim != 3 or chunks.shape[-1] != 14:
            raise ValueError('Expected (chunks,points,14)')
    else:
        rng=np.random.default_rng(42)
        chunks=[]
        for _ in range(40):
            end=rng.uniform(-.25,.25,14);end[[6,13]]=rng.uniform(0,.03,2)
            chunk=np.linspace(np.zeros(14),end,23)[1:]
            chunks.append(chunk)
    p=Planner(limits(.5,.05),limits(1,.2))
    rows=[]
    for chunk in chunks:
        begin=time.perf_counter()
        try:
            trajectory=p.plan(np.zeros(14),np.zeros(14),chunk)
            rows.append({'plan_ms':(time.perf_counter()-begin)*1000,
                         'duration_s':0 if trajectory is None else trajectory.duration,
                         'nominal_s':len(chunk)/15})
        except RuntimeError as exc:
            rows.append({'error':str(exc),'plan_ms':(time.perf_counter()-begin)*1000})
    times=[r['plan_ms'] for r in rows]
    report={'input':str(args.input) if args.input else 'synthetic, rest-start; not robot latency',
            'count':len(rows),'failures':sum('error' in r for r in rows),
            'plan_ms_p50':float(np.percentile(times,50)),
            'plan_ms_p95':float(np.percentile(times,95)),
            'plan_ms_max':max(times),'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
