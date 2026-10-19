"""Finish all pre-specified controls serially after the registered driver exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-driver-pid',type=int);args=parser.parse_args()
    if args.wait_for_driver_pid:
        while True:
            try:os.kill(args.wait_for_driver_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    assert all(json.loads(Path(f'runs/decisive-{mode}-{seed}-test/summary.json').read_text())['completed'] for seed in [1201,1202] for mode in ['text','memory'])
    steps=[]
    if not Path('runs/decisive-graph-reference/summary.json').exists():steps.append('run_decisive_graph_reference.py')
    steps+=['run_decisive_depth_tests.py','run_decisive_trained_diagnostic.py']
    if not Path('runs/decisive-latency/summary.json').exists():steps.append('benchmark_decisive_latency.py')
    steps+=['report_decisive.py','publish_decisive_artifacts.py']
    for script in steps:
        print('RUN',script,flush=True)
        subprocess.run([sys.executable,'-u','scripts/'+script],check=True)
    print('All model work and artifact export completed',flush=True)


if __name__=='__main__':main()
