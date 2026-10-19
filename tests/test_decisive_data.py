from engram_poc.decisive_data import fields,grade,solution,world


def test_signed_worlds_have_correct_executable_witnesses_and_reject_foreign_context():
    for label in ['UP','DOWN','BOTH','UNKNOWN']:
        for depth in [1,2,3,4]:
            case=world('fixture',123+depth,depth,label,12)
            expected=solution(case)
            assert expected['answer']==label
            assert grade(case,expected)['proof_valid']
    case=world('fixture',125,2,'UP',10)
    trap=next(r for r in case['records'] if r['subject']==case['query']['root'] and r['object']==case['query']['target'])
    forged=dict(answer='UP',paths=[dict(edges=[trap['id']],sign='UP')],evidence=[trap['id']])
    assert not grade(case,forged)['proof_valid']


def test_memory_encoder_input_cannot_access_question_or_solution():
    a=world('world-a',123,2,'DOWN',10)
    bank=fields(a['records']);a['query']['root']=15
    assert fields(a['records'])==bank
    assert all(len(row)==7 for row in bank)


def test_generated_splits_and_single_field_counterfactuals_are_independent(tmp_path):
    from engram_poc.decisive_data import build
    corpus=build(tmp_path/'worlds.json')
    namespaces={name:{c['namespace'] for c in cases} for name,cases in corpus['splits'].items()}
    names=list(namespaces)
    for i,name in enumerate(names):
        for other in names[i+1:]:assert not namespaces[name]&namespaces[other]
    for split in ['train','dev','iid','structure']:
        depth={c['depth'] for c in corpus['splits'][split]}
        assert depth==({3,4} if split=='structure' else {1,2})
    originals={c['id']:c for name in ['iid','structure'] for c in corpus['splits'][name]}
    for changed in corpus['interventions']:
        original=originals[changed['parent_id']]
        assert original['query']==changed['query']
        differences=[(a['id'],key) for a,b in zip(original['records'],changed['records']) for key in a if a[key]!=b[key]]
        assert differences==[(changed['changed_record'],changed['changed_field'])]
        assert solution(changed)['answer']!=solution(original)['answer']


def test_right_label_requires_both_signed_paths_and_complete_unknown_evidence():
    import copy
    case=world('both',143,3,'BOTH',12);expected=solution(case)
    incomplete=copy.deepcopy(expected);incomplete['paths']=incomplete['paths'][:1]
    assert grade(case,incomplete)['correct'] and not grade(case,incomplete)['proof_valid']
    forged=copy.deepcopy(expected);forged['paths'][0]['sign']='DOWN' if forged['paths'][0]['sign']=='UP' else 'UP'
    assert not grade(case,forged)['proof_valid']
    case=world('unknown',144,4,'UNKNOWN',12);expected=solution(case)
    expected['evidence'].pop()
    assert grade(case,expected)['correct'] and not grade(case,expected)['proof_valid']
