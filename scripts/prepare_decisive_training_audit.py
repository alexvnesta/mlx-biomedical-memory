"""Freeze a causal readout audit on 16 training banks, separate from held-out scores."""
import copy
import json
from pathlib import Path


def main():
    corpus=json.loads(Path('data/processed/decisive-v1/dataset.json').read_text());cases=[]
    for original in [c for c in corpus['splits']['train'] if c['kind']=='atomic'][:16]:
        for condition in ['normal','masked','changed']:
            case=copy.deepcopy(original);case.update(id=original['id']+'-audit-'+condition,parent_id=original['id'],condition=condition)
            if condition=='changed':
                record=next(r for r in case['records'] if r['id']==case['record_id'])
                for field in ['subject','object']:record[field]=(record[field]+1)%16
                for field in ['species','variant']:record[field]^=1
                record['sign']*=-1;record['source']=(record['source']+1)%4
            cases.append(case)
    data=dict(stage='training_readout_audit',cases=cases,purpose='Training-bank causal diagnostic only; excluded from held-out scores. Change requested record values with question/slots fixed, or disable the codec. No additional fitting.')
    Path('data/processed/decisive-v1/training-audit.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
