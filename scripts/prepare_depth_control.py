"""Freeze deeper test worlds without increasing bank size or record-ID vocabulary."""
import json
from pathlib import Path
from engram_poc.decisive_data import world,LABELS


def main():
    cases=[world(f'depth-only-{i:04d}',41000+i,[3,4][(i//4)%2],LABELS[i%4],10) for i in range(96)]
    original=json.loads(Path('data/processed/decisive-v1/dataset.json').read_text())
    for case,other in zip(cases,[x for x in original['splits']['structure'] if x['kind']=='reason']):assert case['query']==other['query']
    data=dict(cases=cases,stage='depth_only',purpose='Isolate unseen depth 3–4 while preserving the training bank size and record-ID vocabulary. Same core world seeds and queries as the original 12-record structural stratum; independent record IDs/order.',frozen_before_heldout=True)
    target=Path('data/processed/decisive-v1/depth-only.json');target.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
