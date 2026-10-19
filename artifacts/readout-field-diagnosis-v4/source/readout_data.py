"""Disjoint pair partitions and independent banks for atomic memory readout."""
import random


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
