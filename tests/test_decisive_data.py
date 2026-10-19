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
