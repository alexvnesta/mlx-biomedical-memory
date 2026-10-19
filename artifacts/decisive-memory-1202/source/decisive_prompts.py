"""Matched information and supervision for text and latent fact banks."""

import json


INSTRUCTION=('Formal graph task, not a biological claim. Records are the complete local world. '
    'Each directed edge has UP (+1) or DOWN (-1), species, variant and primary source. '
    'Use only edges matching BOTH requested species and variant. Traverse subject to object only. '
    'Multiply signs along a path: DOWN followed by DOWN is UP. '
    'Return BOTH if matching paths yield both signs; UNKNOWN if no matching path reaches the target. '
    'List one valid path per resulting sign. For UNKNOWN list no paths and cite every record ID. '
    'Output JSON: {"paths":[{"edges":["F1","F2"],"sign":"UP"}],"answer":"UP","evidence":["F1","F2"]}. '
    'These example IDs are format examples only. '
    'If asked to READ a record, return its record, subject, object, sign, species, variant and source fields as JSON.\n')


def record_text(r):
    return (f"{r['id']}: N{r['subject']} -> N{r['object']}; "
        f"{'UP' if r['sign']==1 else 'DOWN'}; {'human' if r['species']==0 else 'mouse'}; "
        f"{'lof' if r['variant']==0 else 'gof'}; S{r['source']}")


def question(case):
    if case['kind']=='atomic':return 'READ '+case['record_id']+'. Return only the record JSON.'
    q=case['query']
    return (f"Trace N{q['root']} to N{q['target']} in "
        f"{'human' if q['species']==0 else 'mouse'} {'lof' if q['variant']==0 else 'gof'}. Return the path JSON.")


def encode_prompt(tokenizer,case,mode):
    text=INSTRUCTION+'Complete records:\n'
    placements=[]
    for record in case['records']:
        if mode=='text':text+=record_text(record)+'\n'
        else:text+=record['id']+': @\n'
    prefix=text+'\n'
    content=prefix+'Question: '+question(case)
    prompt=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if mode=='memory':
        cursor=0
        for record in case['records']:
            needle=record['id']+': @';end=prompt.index(needle,cursor)+len(needle);cursor=end
            placements.append(len(tokenizer.encode(prompt[:end],add_special_tokens=False))-1)
    # Leave three boundary tokens unprimed so BPE merges cannot invalidate reuse.
    start=prompt.index('Question: ')
    cut=max(0,len(tokenizer.encode(prompt[:start],add_special_tokens=False))-3)
    return dict(prompt=prompt,ids=ids,placements=placements,prefix_cut=cut)


def target_text(case):
    from .decisive_data import solution,atomic_solution
    return json.dumps(atomic_solution(case) if case['kind']=='atomic' else solution(case),separators=(',',':'))
