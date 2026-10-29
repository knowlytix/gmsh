# SPDX-License-Identifier: Apache-2.0
"""Source-grounded summary bases for native OntoDisc assertion snapshots.

These bases return reference evidence, not canonical generated summaries. Use a
source-grounded coverage/faithfulness evaluator, not exact-string answer scoring.
No inference is made from internal projection paths or absent graph edges.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from knowlytix.benchmark.generators.base import GeneratedQuestion
from knowlytix.knowledge.assertions.contract import OPERATORS

BASES = frozenset({'assertion_summary', 'assertion_scope'})
SAFE_LEVELS = {
    'length': {'Medium', 'Long'},
    'style': {'Formal', 'Casual', 'Technical'},
    'role': {'Employee', 'Manager', 'Executive', 'Auditor', 'Regulator'},
    'answer_format': {'Paragraph', 'Citation'},
    'clarity': {'Clear'},
}
# Every operator that scopes a claim (modality, negation, condition, qualification...):
# all but the structural ones, so a new contract operator counts by default.
SCOPED_OPS = frozenset(OPERATORS - {'event', 'and', 'or'})


def assertion_snapshot(source):
    """Recognize a native snapshot, a store path, or GMSExpertStore.

    Loading/validation errors propagate: a broken native store must never fall
    back to treating its structural projection as a domain-fact graph.
    """
    if isinstance(source, dict) and {'graph', 'index', 'generation'} <= source.keys():
        return source
    if isinstance(source, (str, Path)):
        if not (Path(source) / 'assertions' / 'CURRENT').exists():
            return None                   # a legacy store path: not native
        from knowlytix.knowledge.assertions.store import load_snapshot
        return load_snapshot(source)
    getter = getattr(source, 'assertion_snapshot', None)
    if callable(getter):
        return getter()
    graph = getattr(source, 'doc_graph', source)
    if any(r == 'has_operator' and re.match(r'^[0-9a-f]{64}:', str(h))
           for h, r, _ in getattr(graph, 'triples', [])):
        raise ValueError('Assertion projection requires its native store/snapshot, including provenance')
    return None


def validate_enrichment(assignment, *, method='template'):
    if method != 'template':
        raise ValueError('Assertion bases currently require template enrichment; LLM rewrites need semantic validation')
    for factor, level in assignment.items():
        if level not in SAFE_LEVELS.get(factor, set()):
            raise ValueError(f'Unsupported assertion enrichment {factor}={level}; supported levels: {SAFE_LEVELS}')


def _reference(snapshot, unit_id, ids):
    from knowlytix.knowledge.assertions.store import resolve_query
    units = {u['id']: u for u in snapshot['index']['units']}
    result = resolve_query(snapshot, assertion_ids=list(ids))
    if result['state'] != 'supported':
        raise ValueError(f'Assertion reference is {result["state"]}, not supported')
    return units[unit_id]['source']['text']


def _quoted(text: str) -> str:
    """``text`` in double quotes, single quotes if it holds a double quote, and
    typographic quotes if it holds both: always the whole text, so unique stays unique."""
    if '"' not in text:
        return f'"{text}"'
    if "'" not in text:
        return f"'{text}'"
    return f'\u201c{text}\u201d'


def _locators(index: dict, start_words: int = 10, step: int = 5) -> dict[str, tuple[str, str]]:
    """Per unit, a passage opening unique within its document, and a numbering suffix.

    Openings start at ``start_words`` words and grow only where two passages open
    alike; passages with identical text are numbered by their position.
    """
    texts = {d['id']: d['text'] for d in index['documents']}
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for u in index['units']:
        by_doc[u['source']['document']].append(u)
    out: dict[str, tuple[str, str]] = {}
    for doc, group in by_doc.items():
        group.sort(key=lambda u: u['source']['start'])
        words = {u['id']: texts[doc][u['source']['start']:u['source']['end']].split() for u in group}
        n = dict.fromkeys(words, start_words)
        while True:
            seen: dict[str, list[str]] = defaultdict(list)
            for uid in words:
                seen[' '.join(words[uid][:n[uid]])].append(uid)
            growing = [i for ids in seen.values() if len(ids) > 1 for i in ids
                       if n[i] < len(words[i])]
            if not growing:
                break
            for i in growing:
                n[i] += step
        for lead, ids in seen.items():
            for k, uid in enumerate(ids, 1):
                out[uid] = (lead, f' (occurrence {k} of {len(ids)})' if len(ids) > 1 else '')
    return out


def generate_assertion_questions(source, bases=None, max_per_category=10, seed=42):
    """Generate source-unit summaries, optionally restricted to scoped assertions.

    Unit grouping keeps sibling qualifications together. Incomplete, ambiguous,
    unresolved or unverified groups are excluded, never flattened to positive
    triples. Reference functions re-read the supplied store and pin generation.
    """
    snapshot = assertion_snapshot(source)
    if snapshot is None:
        raise ValueError('Assertion bases require a native assertion store or snapshot')
    names = list(bases) if bases is not None else sorted(BASES)
    if set(names) - BASES:
        raise ValueError(f'Native assertion stores support explicit bases {sorted(BASES)}; got {names}')
    if max_per_category < 0:
        raise ValueError('max_per_category must be nonnegative')
    graph, index = snapshot['graph'], snapshot['index']
    units = {u['id']: u for u in index['units']}
    documents = {d['id']: d for d in index['documents']}
    coverage = {c['unit']: c for c in graph['coverage']}
    groups = {}
    for a in graph['assertions']:
        groups.setdefault(a['unit'], []).append(a)
    locators = _locators(index)
    output = []
    for name in names:
        candidates = []
        for uid, assertions in groups.items():
            if not coverage.get(uid, {}).get('complete'):
                continue
            if any(a['state'] != 'verified' for a in assertions):
                continue
            if name == 'assertion_scope' and not any(
                    n['op'] in SCOPED_OPS for a in assertions for n in a['nodes']):
                continue
            ids = tuple(sorted(a['id'] for a in assertions))
            try:
                answer = _reference(snapshot, uid, ids)
            except ValueError:
                continue
            anchor = units[uid]['source']
            document = documents[anchor['document']]
            if document['text'][anchor['start']:anchor['end']] != answer:
                raise ValueError('Reference source offsets do not match original document')
            lead, occurrence = locators[uid]
            query = (f'In {_quoted(document["name"])}, summarize the passage beginning '
                     f'{_quoted(lead)}{occurrence}. '
                     'Preserve its actors, conditions, exceptions, negation and qualifications.')
            generation = snapshot['generation']

            def recover(current, _uid=uid, _ids=ids, _generation=generation):
                fresh = assertion_snapshot(current)
                if fresh is None or fresh['generation'] != _generation:
                    raise ValueError('Reference requires the original assertion generation')
                return _reference(fresh, _uid, _ids)

            evidence = {}
            for a in assertions:
                for e in a['evidence'] + [units[u]['source'] for u in a['context']]:
                    evidence[(e['document'], e['start'], e['end'])] = e
            candidates.append(GeneratedQuestion(
                qid=f'{name}:{uid}', category=name, natural_language=query,
                graph_answer_fn=recover, ground_truth=answer,
                llm_prompt=query + '\nAnswer from the supplied source evidence only.',
                answer_type='str', metadata={
                    'assertion_format': 'assertions/1', 'generation': generation,
                    'unit_ids': [uid], 'assertion_ids': list(ids),
                    'source_documents': [document['name']],
                    'evidence': deepcopy(list(evidence.values())),
                    'assertions': deepcopy(assertions),
                    'reference_kind': 'source_evidence',
                    'scoring': 'source_grounded_summary',
                    'split_group': uid,
                }))
        output.extend(random.Random(seed).sample(candidates, min(max_per_category, len(candidates))))
    return output
