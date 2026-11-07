# SPDX-License-Identifier: Apache-2.0
"""Full-structure multiset scoring against separately authored source annotations.

Predictions cannot act as their own gold annotations. Callers supply frozen gold
structures using the public nested shape below and record their provenance.
Exact scoring intentionally does not use a model judge or match selected edges.
"""
from __future__ import annotations

from collections import Counter

from .contract import validate_graph
from .source import digest, role_structure


def assertion_structure(assertion):
    """Canonical nested interpretation, preserving every role/operator/quantity."""
    nodes = {n['id']: n for n in assertion['nodes']}

    def visit(key):
        node = nodes[key]
        if node['op'] == 'event':
            return {'op': 'event', 'predicate': node['predicate']['text'],
                    'roles': {k: role_structure(v) if v is not None else None for k, v in node['roles'].items()},
                    'quantities': [{k: (v['text'] if isinstance(v, dict) else v)
                                    for k, v in q.items() if k != 'source'} for q in node['quantities']],
                    **({'qualifiers': [{k: role_structure(v) if k == 'value' else v['text'] for k, v in q.items()} for q in node['qualifiers']]}
                       if node.get('qualifiers') else {})}
        return {'op': node['op'], 'children': [visit(c) for c in node['children']]}
    return {'kind': assertion['kind'], 'structure': visit(assertion['root'])}


def score_assertions(index, graph, annotations, *, annotation_provenance):
    """Score all verified predictions, including unexpected extras and duplicates.

    Each annotation is {unit, kind, structure}. Structures follow
    assertion_structure's nested format, but must be authored from source before
    inspecting predictions. The caller owns annotation quality and adjudication.
    """
    errors = validate_graph(graph, index)
    if errors:
        raise ValueError(f'invalid prediction graph: {errors[:8]}')
    if not isinstance(annotation_provenance, str) or not annotation_provenance.strip():
        raise ValueError('annotation provenance required')
    units = {u['id'] for u in index['units']}
    expected = Counter()
    for annotation in annotations:
        if set(annotation) != {'unit', 'kind', 'structure'} or annotation['unit'] not in units:
            raise ValueError('invalid gold annotation')
        expected[digest(annotation)] += 1
    predicted = Counter(digest({'unit': a['unit'], **assertion_structure(a)})
                        for a in graph['assertions'] if a['state'] in {'verified', 'ambiguous'})
    tp = sum((expected & predicted).values())
    fp, fn = sum((predicted - expected).values()), sum((expected - predicted).values())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'precision': precision, 'recall': recall, 'predicted': sum(predicted.values()),
            'expected': sum(expected.values()), 'annotation_provenance': annotation_provenance,
            'gold_digest': digest(annotations), 'graph_digest': digest(graph),
            'metric': 'exact_complete_assertion_multiset'}
