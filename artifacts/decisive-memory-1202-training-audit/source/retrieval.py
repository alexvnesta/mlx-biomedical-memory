"""Explicit, bounded retrieval baselines over the exact same fact corpus."""

from collections import defaultdict
import re
import sqlite3

import numpy as np


def words(text):
    return re.findall(r"[\w]+", text.casefold())


class Retriever:
    def __init__(self, corpus, document_vectors=None, query_vectors=None):
        self.docs = corpus['facts']+corpus['distractors']
        self.by_id = {d['id']: d for d in self.docs}
        self.adjacency = defaultdict(list)
        for doc in self.docs:
            self.adjacency[doc['subject']].append(doc)
        self.document_vectors = document_vectors
        self.query_vectors = query_vectors
        self.database = sqlite3.connect(':memory:')
        self.database.execute('CREATE VIRTUAL TABLE documents USING fts5(text)')
        self.database.executemany('INSERT INTO documents(text) VALUES (?)', [(d['text'],) for d in self.docs])

    def kg(self, question, depth=2):
        # Oracle entity normalization is explicit: the benchmark supplies its exact alias.
        frontier = [question['entity']]
        found, seen = [], set()
        for _ in range(depth):
            following = []
            for entity in frontier:
                for doc in self.adjacency.get(entity, []):
                    if doc['id'] not in seen:
                        found.append(doc); seen.add(doc['id']); following.append(doc['target'])
            frontier = following
        return found

    def bm25_scores(self, question):
        result = np.zeros(len(self.docs), dtype=np.float32)
        terms = sorted(set(words(question)))
        if not terms:
            return result
        expression = ' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
        for row_id, rank in self.database.execute(
                'SELECT rowid, bm25(documents) FROM documents WHERE documents MATCH ?', (expression,)):
            result[row_id-1] = -rank
        return result

    def retrieve(self, question, method, k=4):
        if method == 'oracle':
            return [self.by_id[i] for i in question['needed']]
        if method == 'kg':
            return self.kg(question, depth=question.get('hops', 1))
        lexical = self.bm25_scores(question['text'])
        if method == 'bm25':
            rank = np.argsort(-lexical)
        elif method in ('dense', 'hybrid'):
            vector = self.query_vectors[question['id']]
            dense = self.document_vectors@vector
            if method == 'dense':
                rank = np.argsort(-dense)
            else:
                rrf = np.zeros(len(self.docs))
                for ordering in (np.argsort(-lexical), np.argsort(-dense)):
                    for position, index in enumerate(ordering):
                        rrf[index] += 1/(60+position+1)
                rank = np.argsort(-rrf)
        else:
            raise ValueError(method)
        return [self.docs[i] for i in rank[:k]]
