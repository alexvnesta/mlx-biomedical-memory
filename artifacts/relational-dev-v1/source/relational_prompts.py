"""Shared, explicit prompts for the scoped-memory experiment."""

from .relational import record_text


HEADER=('You are a controlled evidence reasoning engine. All synthetic rules are stipulated, not biological laws. '
    'Use only the supplied or internally stored rule and retrieved records. Do not use world knowledge. '
    'The rule determines the answer. Preserve conflicts; missing evidence is not refutation. '
    'Return exactly one JSON object with answer and evidence (an array of existing record IDs). '
    'Allowed labels: SUPPORTED, REFUTED, BOTH, UNKNOWN, UP, DOWN, A, B, MULTIPLE, NONE. '
    'For UNKNOWN or NONE, cite all inspected records so absence can be checked. '
    'For other labels, cite records sufficient to prove the answer. '
    'Example output format: {"answer":"SUPPORTED","evidence":["DemoF1","DemoF2"]}. '
    'The example IDs are not evidence for your actual question.\n')


def rule_prefix(request,textual):
    return ('Stored rule lookup:\n'+request['prompt'].format(request['subject'])+
        (' '+request['target']+'\n' if textual else '\n[Value is stored in internal latent memory.]\n'))


def direct_prompt(case,request,records,textual):
    return HEADER+rule_prefix(request,textual)+'Evidence records:\n'+('\n'.join(record_text(r) for r in records) or 'No records retrieved.')+'\nQuestion: '+case['question']+'\nFinal JSON:'


AGENT_TOOLS=('You can inspect the global graph. You are NOT given a gold path or hop count. '
    'To retrieve, return {"lookups":[{"entity":"EXACT_ENTITY_ID","direction":"both","relation":null,"context":{}}]}. '
    'direction is out, in, or both. relation=null returns all predicates. context filters require exact field/value matches. '
    'Use context={} to inspect all contexts. Entity IDs come from the question or previous tool results. '
    'You may request at most two lookups in a response, six total, over four responses. '
    'Each lookup returns at most 16 records and states whether enumeration is complete. '
    'Do not request a lookup once you can answer. The final response is {"answer":"LABEL","evidence":["ID",...]}.\n')


def agent_prompt(case,request,textual,history):
    return HEADER+AGENT_TOOLS+rule_prefix(request,textual)+'Question: '+case['question']+'\n'+history+'\nNext JSON:'


def subject_position(tokenizer,prompt,request):
    needle=request['prompt'].split('{}')[0]+request['subject']
    pos=prompt.index(needle)+len(needle)
    return len(tokenizer.encode(prompt[:pos],add_special_tokens=False))-1
