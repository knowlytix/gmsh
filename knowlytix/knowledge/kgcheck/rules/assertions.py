# SPDX-License-Identifier: Apache-2.0
"""Native checks for source-assertion stores; no semantic accuracy claim."""
from knowlytix.knowledge.kgcheck.models import GraphView, RawRef, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleSpec, builtin, ratio


@builtin(RuleSpec('ASR001', '1', 'artifacts', 'Assertion snapshot integrity', Severity.ERROR,
    prerequisites=('artifact:assertion_graph.json',),
    basis='immutable snapshot digests, source anchors, graph contract, decision identity and projection equality'))
def integrity(ctx):
    problems = [d for d in ctx.snapshot.diagnostics if d.code == 'assertion_contract']
    return Outcome(eligible=1, findings=tuple(ctx.finding(d.message,
        locations=(d.location,), discriminator=str(i)) for i, d in enumerate(problems)))


@builtin(RuleSpec('ASR002', '1', 'coverage', 'Source-unit completion', Severity.ERROR,
    prerequisites=('artifact:assertion_graph.json',), views=(GraphView.STORED,),
    basis='every indexed nonstructural source unit must have a complete disposition and no unresolved dependencies'))
def completeness(ctx):
    if not ctx.snapshot.metadata.get('contract_valid'):
        return Outcome.unavailable('valid_assertion_contract')
    graph = ctx.snapshot.parsed['assertion_graph.json']
    rows = [r for r in graph['coverage'] if r['state'] != 'structural']
    resolved = {'verified', 'context_only', 'non_assertive', 'ambiguous'}
    pending = [(i, r) for i, r in enumerate(graph['coverage']) if r['state'] != 'structural'
               and (r['state'] not in resolved or not r['complete'] or r.get('dependencies'))]
    findings = tuple(ctx.finding('source unit remains unresolved', affected=(r['unit'],),
        locations=(RawRef('assertion_graph.json', f'/coverage/{i}'),),
        witness={'state': r['state'], 'reason': r.get('reason', ''),
                 'dependencies': r.get('dependencies', ())},
        suggestion='repair this source unit using retained source and candidate feedback') for i, r in pending)
    return Outcome(eligible=len(rows), findings=findings, metrics={
        'source_units_complete': ratio(len(rows)-len(pending), len(rows)),
        'unresolved_units': [r['unit'] for _, r in pending],
        'semantic_accuracy_measured': False})


@builtin(RuleSpec('ASR003', '1', 'evidence', 'Unresolved interpretations', Severity.WARNING,
    prerequisites=('artifact:assertion_graph.json',), views=(GraphView.STORED,),
    basis='alternative interpretations remain explicit and cannot produce trusted projection edges'))
def ambiguity(ctx):
    if not ctx.snapshot.metadata.get('contract_valid'):
        return Outcome.unavailable('valid_assertion_contract')
    assertions = ctx.snapshot.parsed['assertion_graph.json']['assertions']
    return Outcome(eligible=len(assertions), findings=tuple(ctx.finding(
        'assertion has unresolved alternative interpretations', affected=(a['id'],),
        locations=(RawRef('assertion_graph.json', f'/assertions/{i}'),))
        for i, a in enumerate(assertions) if a['state'] == 'ambiguous'))


# These checks read the retained assertion representation directly. Enabling
# legacy triple capabilities would certify an empty triple view, not this KG.
def _native_data(ctx):
    from knowlytix.knowledge.kgcheck.frozen import thaw
    return (thaw(ctx.snapshot.parsed['source_index.json']),
            thaw(ctx.snapshot.parsed['assertion_graph.json']))


@builtin(RuleSpec('ASR004', '1', 'evidence', 'Assertion evidence coverage', Severity.ERROR,
    prerequisites=('artifact:source_index.json', 'artifact:assertion_graph.json'),
    basis='each retained assertion has at least one citation resolving exactly in the retained source; not entailment'))
def evidence_coverage(ctx):
    from knowlytix.knowledge.assertions.contract import valid_anchor
    index, graph = _native_data(ctx)
    docs = {d['id']: d for d in index.get('documents', [])}
    rows = graph.get('assertions', [])
    findings = tuple(ctx.finding('assertion has no valid source evidence', affected=(a['id'],),
        locations=(RawRef('assertion_graph.json', f'/assertions/{i}/evidence'),))
        for i, a in enumerate(rows)
        if not any(valid_anchor(e, docs) for e in a.get('evidence', [])))
    return Outcome(eligible=len(rows), findings=findings,
        metrics={'evidence_coverage': ratio(len(rows)-len(findings), len(rows)),
                 'semantic_accuracy_measured': False})


def _anchors(value, pointer=''):
    if isinstance(value, dict):
        if 'document' in value and ('start' in value or 'end' in value):
            yield pointer, value
        else:
            for key, child in value.items():
                token = str(key).replace('~', '~0').replace('/', '~1')
                yield from _anchors(child, pointer + '/' + token)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _anchors(child, pointer + '/' + str(i))


@builtin(RuleSpec('ASR005', '1', 'evidence', 'Exact source citation validity', Severity.ERROR,
    prerequisites=('artifact:source_index.json', 'artifact:assertion_graph.json'),
    basis='all source anchors, including roles, quantities and verification quotes, resolve by document ID and Unicode offsets'))
def citation_validity(ctx):
    from knowlytix.knowledge.assertions.contract import valid_anchor
    index, graph = _native_data(ctx)
    docs = {d['id']: d for d in index.get('documents', [])}
    anchors = [(name, pointer, value) for name, data in
               (('source_index.json', index), ('assertion_graph.json', graph))
               for pointer, value in _anchors(data)]
    findings = tuple(ctx.finding('citation does not match its retained document and offsets',
        locations=(RawRef(name, pointer),), discriminator=name + pointer)
        for name, pointer, value in anchors if not valid_anchor(value, docs))
    return Outcome(eligible=len(anchors), findings=findings,
                   metrics={'anchors_checked': len(anchors), 'documents': len(docs)})


@builtin(RuleSpec('ASR006', '1', 'coverage', 'Retained source documents cited', Severity.INFO,
    prerequisites=('artifact:source_index.json', 'artifact:assertion_graph.json'),
    basis='documents with substantive units overlapped by assertion evidence; separate from source-unit disposition and semantic recall'))
def document_coverage(ctx):
    from knowlytix.knowledge.assertions.contract import valid_anchor
    index, graph = _native_data(ctx)
    docs = {d['id']: d for d in index.get('documents', [])}
    units = [u for u in index.get('units', []) if not u.get('structural')]
    substantive = {u['source']['document'] for u in units}
    cited = set()
    for a in graph.get('assertions', []):
        for e in a.get('evidence', []):
            if valid_anchor(e, docs) and any(u['source']['document'] == e['document']
                    and u['source']['start'] < e['end'] and e['start'] < u['source']['end'] for u in units):
                cited.add(e['document'])
    findings = tuple(ctx.finding('source document has no cited substantive assertion evidence',
        affected=(doc_id,), locations=(RawRef('source_index.json', f'/documents/{i}'),),
        witness={'document': d['name']}) for i, (doc_id, d) in enumerate(docs.items())
        if doc_id in substantive - cited)
    return Outcome(eligible=len(substantive), findings=findings,
        metrics={'documents_with_cited_assertions': ratio(len(substantive & cited), len(substantive)),
                 'semantic_accuracy_measured': False})


@builtin(RuleSpec('ASR007', '1', 'numeric', 'Assertion quantities against quoted source', Severity.WARNING,
    prerequisites=('artifact:source_index.json', 'artifact:assertion_graph.json'),
    basis='localized quantity values and scales, recognized units and bound direction compared with the exact quoted phrase; unsupported phrases are untested, not certified'))
def quantity_evidence(ctx):
    import re
    from decimal import Decimal

    from knowlytix.knowledge.kgcheck.quantities import compatible_unit, quantities
    if not ctx.snapshot.metadata.get('contract_valid'):
        return Outcome.unavailable('valid_assertion_contract')
    _, graph = _native_data(ctx)
    findings, eligible, tested = [], 0, 0
    aliases = {'$': 'dollar', 'dollars': 'dollar', 'days': 'day', 'years': 'year',
               'months': 'month', 'weeks': 'week', '%': 'percent'}
    number_words = dict(zip(
        'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),
        range(20)))
    unknown_units = unknown_bounds = 0
    for i, assertion in enumerate(graph['assertions']):
        for j, node in enumerate(assertion['nodes']):
            for k, q in enumerate(node.get('quantities', [])):
                eligible += 1
                text = re.sub(r'\b(' + '|'.join(number_words) + r')\b',
                    lambda m: str(number_words[m[0].lower()]), q['source']['text'], flags=re.I)
                observed = quantities(text)
                # Multiple numbers or number words need a richer interpretation.
                if len(observed) != 1:
                    continue
                tested += 1
                seen = observed[0]
                issues = []
                if Decimal(q['value']) != seen.scaled:
                    issues.append('value_or_scale')
                unit = aliases.get(q['unit'].lower(), q['unit'])
                if seen.unit and not compatible_unit(unit, seen.unit):
                    issues.append('unit')
                unknown_units += seen.unit is None
                allowed = {'minimum': {'gt', 'gte'}, 'maximum': {'lt', 'lte'}, 'exact': {'eq'}}
                if seen.qualifier and q['comparison'] not in allowed[seen.qualifier]:
                    issues.append('comparison_direction')
                unknown_bounds += seen.qualifier is None
                if issues:
                    findings.append(ctx.finding('quantity disagrees with its quoted source',
                        affected=(assertion['id'],),
                        locations=(RawRef('assertion_graph.json', f'/assertions/{i}/nodes/{j}/quantities/{k}'),),
                        witness={'issues': issues, 'source': q['source']['text'],
                                 'value': q['value'], 'unit': q['unit'], 'comparison': q['comparison']}))
    return Outcome(eligible=eligible, tested=tested, findings=tuple(findings),
        partial='some quantity phrases could not be interpreted' if tested < eligible else None,
        metrics={'quantities_checked': tested, 'unparsed_quantities': eligible-tested,
                 'unrecognized_units': unknown_units, 'unstated_bound_direction': unknown_bounds,
                 'limits': 'Checks only explicit recognized units and bound direction; does not certify period, strictness, aggregation or semantic scope.'})
