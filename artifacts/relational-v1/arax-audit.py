"""Verify every blinded ARAX projection against its retained source edge."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true',help='Reuse the published audit for the identical dataset; do not re-read raw sources')
    args=parser.parse_args()
    dataset=Path('runs/relational-v1/dataset.json');corpus=json.loads(dataset.read_text())
    if args.replay:
        artifact=Path('artifacts/relational-v1/arax-audit.json')
        registered=json.loads(Path('artifacts/SHA256SUMS.json').read_text())
        assert sha(artifact.read_bytes())==registered[str(artifact)]
        audit=json.loads(artifact.read_text());assert audit['dataset_sha256']==sha(dataset.read_bytes())
        Path('runs/relational-arax-audit.json').write_bytes(artifact.read_bytes())
        print('Replayed published projection audit; source snapshots were not re-read.');return
    records={r['id']:r for r in corpus['records']};results=[]
    for path in sorted(Path('data/processed/arax-anchored').glob('*.snapshot.json')):
        snapshot=json.loads(path.read_text())
        edges={sha(json.dumps(e,sort_keys=True).encode()):e for e in snapshot['edges']}
        receipts=[r for r in corpus['arax_receipts'] if r['disease']==path.stem]
        subjects={};objects={}
        for receipt in receipts:
            assert receipt['snapshot_sha256']==sha(path.read_bytes())
            edge=edges[receipt['original_edge_sha256']];record=records[receipt['record_id']]
            qualifiers={q['qualifier_type_id'].split(':')[-1]:q['qualifier_value'] for q in edge.get('qualifiers') or []}
            assert edge['predicate']==receipt['original_predicate']
            assert record['relation']==edge['predicate'].split(':')[-1]
            assert record['sign']=={'increased':'UP','decreased':'DOWN'}[qualifiers['object_direction_qualifier']]
            context={key:qualifiers[name] for key,name in [('species','species_context_qualifier'),
                ('aspect','object_aspect_qualifier'),('form','subject_form_or_variant_qualifier')] if name in qualifiers}
            assert record['context']==context
            for field,role in [('primary_sources','primary_knowledge_source'),('aggregators','aggregator_knowledge_source')]:
                assert record.get(field,[])==[s['resource_id'] for s in edge.get('sources',[]) if s['resource_role']==role]
            assert subjects.setdefault(edge['subject'],record['subject'])==record['subject']
            assert objects.setdefault(edge['object'],record['object'])==record['object']
        assert len(set(subjects.values()))==len(subjects),'Distinct source subjects merged'
        assert len(objects)==len(set(objects.values()))==1,'Distinct gene objects merged'
        results.append(dict(snapshot=path.stem,snapshot_sha256=sha(path.read_bytes()),verified_records=len(receipts),
            distinct_original_subjects=len(subjects),distinct_original_objects=len(objects),
            direction_qualifiers_sources_and_alias_bijection_verified=True))
    assert len(results)==3 and sum(r['verified_records'] for r in results)==36
    out=Path('runs/relational-arax-audit.json')
    out.write_text(json.dumps(dict(dataset_sha256=sha(dataset.read_bytes()),projections=results,
        audit_source_sha256=sha(Path(__file__).read_bytes())),indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':main()
