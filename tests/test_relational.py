from engram_poc.relational import EvidenceGraph,grade,infer,parse_response
from engram_poc.relational_data import signature


def edge(identifier,subject,relation,obj,sign=None,context=None,source='S1'):
    return dict(id=identifier,subject=subject,relation=relation,object=obj,sign=sign,
        context=context or {},primary_sources=[source])


def test_signed_paths_require_direction_and_matching_context():
    rows=[edge('a','C','regulates','G','DOWN',{'species':'human'}),
        edge('b','G','regulates','P','DOWN',{'species':'human'}),
        edge('c','C','regulates','P','DOWN',{'species':'mouse'})]
    case=dict(family='sign',params=dict(root='C',target='P',context={'species':'human'}))
    assert infer(case,rows)==('UP',[{'a','b'}])
    rows.append(edge('d','C','regulates','P','DOWN',{'species':'human'}))
    assert infer(case,rows)==('BOTH',[{'a','b','d'}])


def test_provenance_deduplicates_aggregators_and_preserves_both_signs():
    case=dict(family='provenance',params=dict(root='C',context={}))
    rows=[edge('a','C','claim','T','POS'),edge('b','C','claim','T','POS')]
    assert infer(case,rows)[0]=='UNKNOWN'
    rows.append(edge('c','C','claim','T','POS',source='S2'))
    assert infer(case,rows)[0]=='SUPPORTED'
    rows.extend([edge('d','C','claim','T','NEG'),edge('e','C','claim','T','NEG',source='S3')])
    assert infer(case,rows)[0]=='BOTH'


def test_join_requires_shared_pathway_and_abduction_does_not_assert_options():
    rows=[edge('a','C','candidate_gene','G'),edge('b','G','gene_pathway','P'),
        edge('c','C','candidate_phenotype','H')]
    missing=edge('hyp','H','phenotype_pathway','P')
    case=dict(family='join',params=dict(root='C'))
    assert infer(case,rows)[0]=='UNKNOWN'
    abduct=dict(family='abduction',params=dict(root='C',options={'A':missing,'B':{**missing,'id':'other','object':'Q'}}))
    assert infer(abduct,rows)==('A',[{'a','b','c'}])
    assert infer(case,rows)[0]=='UNKNOWN'
    assert infer(case,[*rows,missing])[0]=='SUPPORTED'


def test_graph_uses_only_caller_parameters_and_scores_citations():
    rows=[edge('a','C','claim','G','UP',{'species':'human'}),edge('b','C','claim','G','DOWN',{'species':'mouse'})]
    graph=EvidenceGraph(rows)
    assert [r['id'] for r in graph.lookup('G',context={'species':'human'},direction='in')['records']]==['a']
    assert graph.lookup('G',direction='out')['records']==[]
    case=dict(answer='SUPPORTED',proofs=[['a']],core_ids=['a','b'])
    assert grade(case,'SUPPORTED',['a'],['a'])['proof_valid']
    assert not grade(case,'SUPPORTED',['a'],[])['proof_valid']
    unknown=dict(answer='UNKNOWN',proofs=[],core_ids=['a','b'])
    assert not grade(unknown,'UNKNOWN',[],[])['proof_valid']
    assert grade(unknown,'UNKNOWN',['a','b'],['a','b'])['proof_valid']


def test_scoped_addresses_and_json_parser():
    assert signature('C','r',{'species':'human','variant':'lof'})==signature('C','r',{'variant':'lof','species':'human'})
    assert signature('C','r',{'species':'human'})!=signature('C','r',{'species':'mouse'})
    assert parse_response('```json\n{"answer":"BOTH","evidence":["a"]}\n```')['answer']=='BOTH'
    assert parse_response('{"lookups":[') is None


def test_complete_absence_certificate_cannot_use_wrong_entity_or_extra_filter():
    case=dict(family='context',params=dict(root='C',relation='effect',context={'species':'human'}),
        answer='UNKNOWN',proofs=[],core_ids=['mouse-record'])
    cert=dict(entity='C',relation=None,context={'species':'human'},direction='out',complete=True)
    assert grade(case,'UNKNOWN',[],[],certificates=[cert])['proof_valid']
    assert not grade(case,'UNKNOWN',[],[],certificates=[{**cert,'entity':'Other'}])['proof_valid']
    assert not grade(case,'UNKNOWN',[],[],certificates=[{**cert,'context':{'species':'human','variant':'gof'}}])['proof_valid']
    assert not grade(case,'UNKNOWN',[],[],certificates=[{**cert,'complete':False}])['proof_valid']


def test_missing_signed_path_needs_exhaustive_frontier_and_record_citations():
    rows=[edge('a','C','regulates','G','UP',{'species':'human'})]
    case=dict(family='sign',params=dict(root='C',target='P',context={'species':'human'}),answer='UNKNOWN',proofs=[],core_ids=['a','other'])
    certs=[dict(entity=e,relation=None,context={},direction='both',complete=True) for e in ['C','G']]
    assert grade(case,'UNKNOWN',['a'],['a'],certificates=certs,records=rows)['proof_valid']
    assert not grade(case,'UNKNOWN',['a'],['a'],certificates=certs[:1],records=rows)['proof_valid']
    assert not grade(case,'UNKNOWN',[],['a'],certificates=certs,records=rows)['proof_valid']
