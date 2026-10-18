import numpy as np

from engram_poc.retrieval import Retriever


def corpus():
    return dict(facts=[
        dict(id='a', subject='Chemical Vela', target='Gene Mira', text='Chemical Vela selects Gene Mira.'),
        dict(id='b', subject='Gene Mira', target='Disease Cedar', text='Gene Mira selects Disease Cedar.'),
    ], distractors=[dict(id='c', subject='Chemical Decoy', target='Gene Decoy', text='Chemical Decoy selects Gene Decoy.')])


def test_graph_follows_hidden_intermediate_and_absent_entity_stays_absent():
    retriever = Retriever(corpus())
    question = dict(entity='Chemical Vela', text='Follow two links from Chemical Vela.', hops=2, needed=['a', 'b'])
    assert [d['id'] for d in retriever.retrieve(question, 'kg')] == ['a', 'b']
    assert retriever.kg(dict(entity='Unknown'), depth=2) == []
    assert [d['id'] for d in retriever.retrieve(question, 'oracle')] == ['a', 'b']


def test_fts_quotes_user_text_and_dense_and_hybrid_use_supplied_vectors():
    vectors = np.eye(3, dtype=np.float32)
    retriever = Retriever(corpus(), vectors, {'q': vectors[1]})
    q = dict(id='q', text='Gene Mira " OR *')
    scores = retriever.bm25_scores(q['text'])
    assert np.isfinite(scores).all()
    assert scores[0] > scores[2] and scores[1] > scores[2]
    assert retriever.retrieve(q, 'dense', k=1)[0]['id'] == 'b'
    q['text'] = 'Disease Cedar'
    assert retriever.retrieve(q, 'hybrid', k=1)[0]['id'] == 'b'
