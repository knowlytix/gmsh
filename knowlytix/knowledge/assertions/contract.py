# SPDX-License-Identifier: Apache-2.0
"""Independent JSON contract validation. No extraction, training, or LLM imports."""
from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from functools import wraps

from .source import VERSION, bibliographic_reference, digest, unresolved_references

ROLES = {'subject', 'agent', 'patient', 'recipient', 'instrument', 'authority', 'location'}
OPERATORS = {'event', 'and', 'or', 'alternative', 'not', 'obligation', 'prohibition', 'recommendation',
             'recommendation_against', 'permission', 'not_required', 'possibility',
             'prediction', 'condition', 'exception', 'purpose', 'temporal_anchor', 'qualification', 'cause'}
STATES = {'candidate', 'verified', 'ambiguous', 'rejected', 'needs_context', 'needs_review', 'unsupported', 'execution_error'}
KINDS = {'fact', 'observation', 'definition', 'event', 'procedure', 'measurement', 'rule'}


def _json_boundary(function):
    """Malformed JSON must produce diagnostics, never escape the audit boundary."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (TypeError, KeyError, ValueError, AttributeError, RecursionError, OverflowError):
            return ['malformed_contract:' + function.__name__]
    return checked


def claim_digest(assertion):
    return digest({k: v for k, v in assertion.items() if k not in {'state', 'reasons'}})


def assertion_dependencies(assertion, decision=None):
    """Missing citations are dependencies only when the claim needs their content.

    Legacy decisions without an explicit grounded review retain conservative
    blocking. A supported review can establish that a claim is self-contained.
    """
    decision = decision or {}
    independent = {c['reference'] for c in decision.get('reference_checks', [])
                   if not c['requires_content']} if decision.get('verdict') == 'supported' else set()
    return [r for i, r in enumerate(assertion['references'])
            if i not in independent and unresolved_references([r])]


def validate_reference_checks(checks, assertion, index):
    if not isinstance(checks, list):
        return ['invalid_reference_checks']
    docs = {d['id']: d for d in index['documents']}
    seen = set()
    for c in checks:
        if (not isinstance(c, dict) or set(c) != {'reference', 'requires_content', 'reason', 'quotes'}
                or type(c['reference']) is not int or not 0 <= c['reference'] < len(assertion['references'])
                or c['reference'] in seen or type(c['requires_content']) is not bool
                or not isinstance(c['reason'], str) or not c['reason'].strip()
                or not isinstance(c['quotes'], list) or not c['quotes']
                or any(not valid_anchor(q, docs) or not any(q['document'] == e['document']
                    and e['start'] <= q['start'] < q['end'] <= e['end'] for e in assertion['evidence']) for q in c['quotes'])):
            return ['invalid_reference_checks']
        seen.add(c['reference'])
    return []


@_json_boundary
def validate_index(index):
    errors = []
    if not isinstance(index, dict) or index.get('version') != VERSION:
        return ['invalid_source_version']
    docs, units = index.get('documents'), index.get('units')
    if not isinstance(docs, list) or not isinstance(units, list):
        return ['invalid_source_collections']
    by_doc, by_unit = {}, {}
    for doc in docs:
        if not isinstance(doc, dict) or not isinstance(doc.get('text'), str) or not isinstance(doc.get('name'), str):
            errors.append('invalid_document')
            continue
        if (doc.get('id') != digest([doc['name'], doc['text']]) or doc.get('id') in by_doc
                or doc.get('sha256') != hashlib.sha256(doc['text'].encode()).hexdigest()):
            errors.append('document_identity')
        by_doc[doc.get('id')] = doc
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get('id'), str):
            errors.append('invalid_unit')
            continue
        if unit['id'] in by_unit:
            errors.append('duplicate_unit')
        by_unit[unit['id']] = unit
        if not valid_anchor(unit.get('source'), by_doc):
            errors.append('unit_source')
        elif unit['id'] != digest([unit['source']['document'], unit.get('kind'), unit['source']['start'], unit['source']['end']]):
            errors.append('unit_identity')
        if type(unit.get('structural')) is not bool or not isinstance(unit.get('links'), list):
            errors.append('unit_shape')
    for unit in by_unit.values():
        parent, seen = unit.get('parent'), {unit['id']}
        while parent is not None:
            if not isinstance(parent, str) or parent not in by_unit or parent in seen:
                errors.append('invalid_parent_or_cycle')
                break
            seen.add(parent)
            parent = by_unit[parent].get('parent')
    for unit in by_unit.values():
        parent = unit.get('parent')
        if parent in by_unit and by_unit[parent]['source']['document'] != unit['source']['document']:
            errors.append('cross_document_parent')
        for link in unit.get('links', []):
            if not isinstance(link, dict) or not isinstance(link.get('kind'), str):
                errors.append('invalid_source_link')
            elif link['kind'] == 'reference':
                targets = link.get('targets')
                if ('usage' in link and (link['usage'] != 'citation'
                        or not bibliographic_reference(unit['source']['text'], link.get('mention', '')))):
                    errors.append('invalid_reference_usage')
                if (not isinstance(targets, list) or any(t not in by_unit for t in targets)
                        or link.get('state') != ('resolved' if len(targets) == 1 else 'missing' if not targets else 'ambiguous')):
                    errors.append('invalid_reference_link')
            elif link.get('target') not in by_unit:
                errors.append('missing_source_link_target')
    fragment_ids = set()
    for fragment in index.get('fragments', []):
        if (not isinstance(fragment, dict) or fragment.get('parent') not in by_unit
                or not valid_anchor(fragment.get('source'), by_doc) or fragment.get('id') in fragment_ids):
            errors.append('invalid_source_fragment')
            continue
        fragment_ids.add(fragment['id'])
        parent_source, source = by_unit[fragment['parent']]['source'], fragment['source']
        if (source['document'] != parent_source['document']
                or not parent_source['start'] <= source['start'] < source['end'] <= parent_source['end']):
            errors.append('fragment_outside_parent')
    # All non-whitespace source characters must be inventoried exactly once.
    for doc_id, doc in by_doc.items():
        intervals = sorted((u['source']['start'], u['source']['end']) for u in by_unit.values()
                           if valid_anchor(u.get('source'), by_doc) and u['source']['document'] == doc_id)
        cursor = 0
        for start, end in intervals:
            if start < cursor:
                errors.append('overlapping_source_units')
            if doc['text'][cursor:start].strip():
                errors.append('uninventoried_source')
            cursor = max(cursor, end)
        if doc['text'][cursor:].strip():
            errors.append('uninventoried_source')
    return errors


def valid_anchor(value, documents):
    if not isinstance(value, dict) or set(value) != {'document', 'start', 'end', 'text'}:
        return False
    doc_id = value.get('document')
    if not isinstance(doc_id, str) or doc_id not in documents:
        return False
    lo, hi = value.get('start'), value.get('end')
    return (type(lo) is int and type(hi) is int and 0 <= lo < hi <= len(documents[doc_id]['text'])
            and documents[doc_id]['text'][lo:hi] == value.get('text'))


@_json_boundary
def validate_assertion(assertion, index):
    errors = []
    if not isinstance(assertion, dict):
        return ['assertion_not_object']
    required = {'id', 'unit', 'context', 'kind', 'nodes', 'root', 'evidence', 'references', 'state', 'reasons'}
    if set(assertion) != required:
        errors.append('assertion_fields')
    index_errors = validate_index(index)
    if index_errors:
        return index_errors
    docs = {d['id']: d for d in index['documents']}
    units = {u['id']: u for u in index['units']}
    if (assertion.get('unit') not in units or assertion.get('kind') not in KINDS
            or assertion.get('state') not in STATES or not isinstance(assertion.get('id'), str)
            or not isinstance(assertion.get('reasons'), list)):
        return [*errors, 'assertion_identity_or_state']
    identity = {k: v for k, v in assertion.items() if k != 'id'}
    identity.update(state='candidate', reasons=[])
    if assertion['id'] != digest(identity):
        errors.append('assertion_content_identity')
    evidence = assertion.get('evidence')
    if not isinstance(evidence, list) or not evidence or not all(valid_anchor(a, docs) for a in evidence):
        return [*errors, 'assertion_evidence']
    context = assertion.get('context')
    if not isinstance(context, list) or assertion['unit'] not in context or any(not isinstance(u, str) or u not in units for u in context):
        return [*errors, 'assertion_context']
    allowed = [units[u]['source'] for u in context]
    if evidence != allowed:
        errors.append('incomplete_context_evidence')
    for key in context:
        parent = units[key]['parent']
        if parent is not None and parent not in context:
            errors.append('missing_structural_context')

    def supported(a):
        return valid_anchor(a, docs) and any(a['document'] == e['document']
            and e['start'] <= a['start'] < a['end'] <= e['end'] for e in allowed)

    def supported_role(a):
        if isinstance(a, dict) and set(a) == {'phrase', 'bindings'}:
            phrase, bindings = a['phrase'], a['bindings']
            if not supported(phrase) or not isinstance(bindings, list) or not 1 <= len(bindings) <= 16:
                return False
            for binding in bindings:
                if (not isinstance(binding, dict) or set(binding) != {'mention', 'relation', 'target'}
                        or binding['relation'] not in ('identity', 'domain')):
                    return False
                mention, target = binding['mention'], binding['target']
                if (not supported(mention) or not supported(target)
                        or mention['document'] != phrase['document'] or target['document'] != phrase['document']
                        or not phrase['start'] <= mention['start'] < mention['end'] <= phrase['end']
                        or max(mention['start'], target['start']) < min(mention['end'], target['end'])):
                    return False
            # A single mention cannot silently select competing identities/domains.
            # Represent competing readings as alternative propositions instead.
            keys = [(digest(b['mention']), b['relation']) for b in bindings]
            return len(set(keys)) == len(keys)
        if isinstance(a, dict) and set(a) == {'mention', 'referent', 'modifiers'}:
            modifiers = a['modifiers']
            return (supported(a['mention']) and supported(a['referent'])
                    and a['mention']['document'] == a['referent']['document']
                    and (a['referent']['end'] <= a['mention']['start']
                         or a['mention']['end'] <= a['referent']['start'])
                    and isinstance(modifiers, list) and len(modifiers) <= 16
                    and all(supported(m) and m['document'] == a['referent']['document'] for m in modifiers)
                    and len({digest(m) for m in modifiers}) == len(modifiers))
        if isinstance(a, dict) and set(a) == {'parts'}:
            parts = a['parts']
            return (isinstance(parts, list) and 2 <= len(parts) <= 16
                    and all(supported(p) for p in parts)
                    and all(x['document'] == y['document'] and x['end'] <= y['start']
                            for x, y in zip(parts, parts[1:])))
        return supported(a)

    if not all(supported(a) for a in evidence):
        errors.append('evidence_outside_context')
    nodes = assertion.get('nodes')
    if not isinstance(nodes, list) or not nodes or len(nodes) > 128:
        return [*errors, 'invalid_nodes']
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or node['id'] in by_id:
            errors.append('node_identity')
            continue
        by_id[node['id']] = node
        if node.get('op') not in OPERATORS or not supported(node.get('source')):
            errors.append('node_operator_or_source')
        if node.get('op') == 'event':
            if set(node) - {'qualifiers'} != {'id', 'op', 'source', 'predicate', 'roles', 'quantities'}:
                errors.append('event_fields')
            qualifiers = node.get('qualifiers', [])
            if (not isinstance(qualifiers, list) or any(not isinstance(q, dict)
                    or set(q) not in ({'relation', 'value'}, {'text'})
                    or not all((supported_role(v) if k == 'value' else supported(v)) for k, v in q.items())
                    for q in qualifiers)):
                errors.append('invalid_event_qualifier')
            if not supported(node.get('predicate')):
                errors.append('predicate_source')
            roles = node.get('roles')
            if not isinstance(roles, dict) or set(roles) - ROLES or any(v is not None and not supported_role(v) for v in roles.values()):
                errors.append('participant_source')
            quantities = node.get('quantities')
            if not isinstance(quantities, list):
                errors.append('quantities_not_array')
                continue
            for q in quantities:
                if (not isinstance(q, dict) or set(q) != {'source', 'value', 'unit', 'comparison', 'aggregation'}
                        or not supported(q.get('source')) or not isinstance(q.get('unit'), str)
                        or q.get('comparison') not in {'eq', 'lt', 'lte', 'gt', 'gte', 'approximately'}
                        or (q.get('aggregation') is not None and not supported(q['aggregation']))):
                    errors.append('invalid_quantity')
                    continue
                try:
                    if not isinstance(q.get('value'), str) or not Decimal(q['value']).is_finite():
                        errors.append('quantity_value')
                except (InvalidOperation, ValueError):
                    errors.append('quantity_value')
        else:
            if set(node) != {'id', 'op', 'source', 'children'}:
                errors.append('operator_fields')
            children = node.get('children')
            count = 2 if node.get('op') in {'condition', 'exception', 'purpose', 'temporal_anchor', 'qualification', 'cause'} else 1
            if (not isinstance(children, list) or not all(isinstance(c, str) for c in children)
                    or (len(children) < 2 if node.get('op') in {'and', 'or', 'alternative'} else len(children) != count)
                    or len(set(children)) != len(children)):
                errors.append('operator_arity')
    root = assertion.get('root')
    if not isinstance(root, str) or root not in by_id:
        return [*errors, 'missing_root']
    visiting, visited = set(), set()

    def walk(key):
        if key in visiting:
            errors.append('operator_cycle')
            return
        if key not in by_id:
            errors.append('missing_child')
            return
        if key in visited:
            return
        visiting.add(key)
        children = by_id[key].get('children', [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, str):
                    walk(child)
        visiting.remove(key)
        visited.add(key)
    walk(root)
    if set(by_id) != visited:
        errors.append('unreachable_nodes')
    refs = assertion.get('references')
    expected_refs = [link for u in context for link in units[u]['links'] if link['kind'] == 'reference']
    if not isinstance(refs, list) or refs != expected_refs:
        errors.append('invalid_reference')
    return errors


@_json_boundary
def validate_graph(graph, index):
    errors = validate_index(index)
    if errors:
        return errors
    if not isinstance(graph, dict) or graph.get('version') != VERSION or graph.get('source_digest') != digest(index):
        return [*errors, 'graph_source_version']
    assertions, decisions, coverage = graph.get('assertions'), graph.get('decisions'), graph.get('coverage')
    if not all(isinstance(v, list) for v in [assertions, decisions, coverage]):
        return [*errors, 'graph_collections']
    ids, accepted = set(), {}
    for decision in decisions:
        if (not isinstance(decision, dict) or not isinstance(decision.get('assertion'), str)
                or decision.get('verdict') not in {'supported', 'contradicted', 'insufficient_evidence'}
                or not isinstance(decision.get('reason'), str) or not decision['reason']
                or not isinstance(decision.get('method'), str)):
            errors.append('invalid_decision')
            continue
        if decision['assertion'] in accepted:
            errors.append('duplicate_decision')
        accepted[decision['assertion']] = decision
    for assertion in assertions:
        issues = validate_assertion(assertion, index)
        errors.extend(issues)
        if not isinstance(assertion, dict) or not isinstance(assertion.get('id'), str):
            continue
        if assertion['id'] in ids:
            errors.append('duplicate_assertion')
        ids.add(assertion['id'])
        decision = accepted.get(assertion['id'])
        if decision is not None:
            errors.extend(validate_reference_checks(decision.get('reference_checks', []), assertion, index))
            docs = {d['id']: d for d in index['documents']}
            quotes = decision.get('quotes')
            allowed = assertion.get('evidence', [])
            if (decision.get('digest') != claim_digest(assertion)
                    or sorted(decision.get('checked_nodes', [])) != sorted(n['id'] for n in assertion['nodes'])
                    or not isinstance(quotes, list)
                    or (decision['verdict'] != 'insufficient_evidence' and not quotes)
                    or any(not valid_anchor(q, docs) or not any(
                        q['document'] == e['document'] and e['start'] <= q['start'] < q['end'] <= e['end']
                        for e in allowed) for q in (quotes or []))):
                errors.append('invalid_decision_evidence')
        has_alternatives = any(n.get('op') == 'alternative' for n in assertion.get('nodes', []))
        if ((assertion.get('state') == 'verified' and has_alternatives)
                or (assertion.get('state') == 'ambiguous' and not has_alternatives)):
            errors.append('unresolved_interpretation_choice')
        if assertion.get('state') in {'verified', 'ambiguous'}:
            decision = accepted.get(assertion['id'], {})
            if (issues or decision.get('verdict') != 'supported'
                    or decision.get('digest') != claim_digest(assertion)):
                errors.append('unverified_assertion')
    if set(accepted) - ids:
        errors.append('orphan_decision')
    units = {u['id']: u for u in index['units']}
    by_assertion = {a['id']: a for a in assertions}
    unit_decisions = {}
    for decision in graph.get('unit_decisions', []):
        if (not isinstance(decision, dict) or decision.get('unit') not in units
                or decision['unit'] in unit_decisions
                or decision.get('classification') not in {'context_only', 'non_assertive', 'incomplete'}
                or not isinstance(decision.get('reason'), str) or not decision['reason'].strip()
                or not isinstance(decision.get('method'), str) or not decision['method']
                or decision.get('quote') != units[decision['unit']]['source']):
            errors.append('invalid_unit_decision')
            continue
        unit_decisions[decision['unit']] = decision
    coverage_by_id = {r['unit']: r for r in coverage}
    seen = set()
    for row in coverage:
        if not isinstance(row, dict) or row.get('unit') not in units or row['unit'] in seen:
            errors.append('coverage_unit')
            continue
        seen.add(row['unit'])
        if row.get('state') not in {'structural', 'verified', 'needs_review', 'needs_context', 'execution_error', 'unsupported', 'context_only', 'non_assertive', 'ambiguous'}:
            errors.append('coverage_state')
        linked = row.get('assertions')
        if not isinstance(linked, list) or any(a not in ids for a in linked):
            errors.append('coverage_assertion')
        elif len(set(linked)) != len(linked) or any(by_assertion[a]['unit'] != row['unit'] for a in linked):
            errors.append('coverage_wrong_owner')
        if row.get('state') == 'structural' and not units[row['unit']]['structural']:
            errors.append('nonstructural_unit_silenced')
        if row.get('state') in {'context_only', 'non_assertive'}:
            decision = unit_decisions.get(row['unit'], {})
            if (decision.get('classification') != row['state'] or row.get('complete') is not True
                    or row.get('dependencies') or linked
                    or decision.get('context_digest') != row.get('context_digest')):
                errors.append('unverified_empty_unit')
            if row['state'] == 'context_only':
                dependents = [u for u in units.values() if not u['structural'] and (u['parent'] == row['unit']
                              or any(link.get('kind') == 'table_header' and link.get('target') == row['unit'] for link in u['links']))]
                if not dependents or any(coverage_by_id[u['id']]['state'] not in {'verified', 'ambiguous'} for u in dependents):
                    errors.append('unverified_context_dependents')
        if row.get('state') in {'verified', 'ambiguous'}:
            selected = [a for a in assertions if a.get('id') in (linked or [])]
            if not selected or any(a.get('state') not in ({'verified'} if row['state'] == 'verified' else {'verified', 'ambiguous'}) for a in selected) or row.get('complete') is not True or row.get('dependencies'):
                errors.append('coverage_false_readiness')
    if seen != set(units):
        errors.append('incomplete_inventory')
    if {aid for row in coverage for aid in row.get('assertions', [])} != ids:
        errors.append('unaccounted_assertion')
    return errors
