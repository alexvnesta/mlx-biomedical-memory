"""Disjoint pair partitions and independent banks for atomic memory readout."""
import random
import json


def pair_split(subject, object):
    value=(5*subject+3*object)%8
    return 'dev' if value==0 else 'test' if value==1 else 'train'


def make_case(seed, count, split='train'):
    rng=random.Random(seed);ids=rng.sample(range(10),count)
    records=[dict(id=f'F{i}',subject=rng.randrange(16),object=rng.randrange(16),
        sign=rng.choice([-1,1]),species=rng.randrange(2),variant=rng.randrange(2),source=rng.randrange(4)) for i in ids]
    for r in records:
        while pair_split(r['subject'],r['object'])!=split:
            r['subject']=rng.randrange(16);r['object']=rng.randrange(16)
    return dict(id=f'fresh-{seed}-{count}',kind='atomic',records=records,record_id=rng.choice(records)['id'])


def noisy_prefix(values, tokenize, seed, probability):
    """Corrupt past target values, never the true bank or next-token labels.

    Preserve token count exactly so causal next-token positions stay aligned.
    This helper only builds training inputs; it is never used for inference.
    """
    rng=random.Random(seed);changed=dict(values)
    encode=lambda obj:tokenize(json.dumps(obj,separators=(',',':')))
    size=len(encode(values))
    domains=dict(subject=[f'N{i}' for i in range(16)],object=[f'N{i}' for i in range(16)],
        sign=['UP','DOWN'],species=['human','mouse'],variant=['lof','gof'],source=[f'S{i}' for i in range(4)])
    for field in domains:
        if field not in values or rng.random()>=probability:continue
        choices=[v for v in domains[field] if v!=values[field]];rng.shuffle(choices)
        for value in choices:
            proposal=dict(changed);proposal[field]=value
            if len(encode(proposal))==size:changed=proposal;break
    return changed,encode(changed)
