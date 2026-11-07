# SPDX-License-Identifier: Apache-2.0
"""Atomic immutable assertion snapshots, safe projections and source-bound query."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

from .contract import assertion_dependencies, validate_graph
from .source import VERSION, anchored_text, digest


def compile_projections(index, graph, *, candidates=False):
    """Compile trusted projections, or a separate unverified training projection.

    Candidate projection never changes the graph, decisions, or trusted snapshot.
    """
    errors = validate_graph(graph, index)
    if errors:
        raise ValueError(f'invalid assertion graph: {errors[:8]}')
    triples, literals, numeric, dependencies = [], [], [], []
    def project_reference(owner, value, aid):
        if 'phrase' in value:
            triples.append({'triple': [owner, 'has_phrase_text', value['phrase']['text']], 'assertion': aid})
            for i, binding in enumerate(value['bindings']):
                bkey = f'{owner}:binding:{i}'
                triples.append({'triple': [owner, 'has_binding', bkey], 'assertion': aid})
                triples.append({'triple': [bkey, 'has_relation', binding['relation']], 'assertion': aid})
                for field in ('mention', 'target'):
                    triples.append({'triple': [bkey, f'has_{field}_text', binding[field]['text']], 'assertion': aid})
            return
        if 'mention' not in value:
            return
        # The reference is attached to its role/qualifier, never to the event.
        triples.append({'triple': [owner, 'has_mention_text', value['mention']['text']], 'assertion': aid})
        triples.append({'triple': [owner, 'has_referent_text', value['referent']['text']], 'assertion': aid})
        for i, modifier in enumerate(value['modifiers']):
            triples.append({'triple': [owner, f'has_modifier_{i}_text', modifier['text']], 'assertion': aid})
    decisions = {d['assertion']: d for d in graph['decisions']}
    for assertion in graph['assertions']:
        aid = assertion['id']
        unresolved = assertion_dependencies(assertion, decisions.get(aid))
        literals.append({'assertion': aid, 'state': assertion['state'], 'evidence': assertion['evidence'],
                         'dependencies': unresolved})
        dependencies.extend({'assertion': aid, **r} for r in unresolved)
        if candidates:
            if assertion['state'] not in {'candidate', 'verified'}:
                continue
        elif assertion['state'] != 'verified' or unresolved:
            continue
        by_id = {n['id']: n for n in assertion['nodes']}
        for node in assertion['nodes']:
            key = f"{aid}:{node['id']}"
            triples.append({'triple': [key, 'has_operator', node['op']], 'assertion': aid})
            if node['op'] == 'event':
                triples.append({'triple': [key, 'has_predicate_text', node['predicate']['text']], 'assertion': aid})
                for role, value in node['roles'].items():
                    if value is not None:
                        triples.append({'triple': [key, f'has_{role}', anchored_text(value)], 'assertion': aid})
                        if 'mention' in value or 'phrase' in value:
                            rkey = f'{key}:role:{role}'
                            triples.append({'triple': [key, f'has_{role}_reference', rkey], 'assertion': aid})
                            project_reference(rkey, value, aid)
                for i, qualifier in enumerate(node.get('qualifiers', [])):
                    qkey = f'{key}:qualifier:{i}'
                    triples.append({'triple': [key, 'has_qualifier', qkey], 'assertion': aid})
                    for field, value in qualifier.items():
                        relation = 'has_text' if field == 'text' else f'has_{field}_text'
                        triples.append({'triple': [qkey, relation, anchored_text(value)], 'assertion': aid})
                        project_reference(qkey, value, aid)
                for quantity in node['quantities']:
                    triples.append({'triple': [key, 'has_quantity_text', quantity['source']['text']], 'assertion': aid})
            else:
                for position, child in enumerate(node['children']):
                    triples.append({'triple': [key, f'has_operand_{position}', f'{aid}:{child}'], 'assertion': aid})
        root = by_id[assertion['root']]
        # Only a directly asserted measurement can form a numeric slot. Rules,
        # negations, conjunctions, temporal/conditional groups never qualify.
        if not candidates and root['op'] == 'event' and assertion['kind'] == 'measurement' and len(root['quantities']) == 1 and not root.get('qualifiers'):
            quantity = root['quantities'][0]
            subject = root['roles'].get('subject')
            if subject and 'phrase' not in subject and quantity['comparison'] == 'eq' and quantity['aggregation'] is None:
                numeric.append({'assertion': aid, 'subject': anchored_text(subject),
                                'predicate': root['predicate']['text'], **quantity})
    units = {u['id']: u for u in index['units']}
    for row in graph['coverage']:
        if not row['assertions'] and row['state'] != 'structural':
            literals.append({'assertion': row['unit'], 'state': row['state'],
                             'evidence': [units[row['unit']]['source']],
                             'dependencies': row.get('dependencies', [])})
    return {'version': VERSION, 'graph_digest': digest(graph), 'triples': triples,
            'literals': literals, 'numeric': numeric, 'dependencies': dependencies}


def publish_snapshot(root, index, graph, *, mode='shadow', expected_generation=None):
    base = Path(root) / 'assertions'
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if expected_generation is not None:
            if not (base / 'CURRENT').exists() or (base / 'CURRENT').read_text().strip() != expected_generation:
                raise ValueError('concurrent update: stale generation')
        return _publish_locked(root, index, graph, mode=mode)


def _publish_locked(root, index, graph, *, mode):
    """Write a complete generation, then atomically change a tiny CURRENT pointer.

    Previous generations remain readable. A failed publication cannot replace
    the previous pointer. Caller-owned legacy artifacts are never overwritten.
    """
    if mode not in {'shadow', 'enforce'}:
        raise ValueError('mode must be shadow or enforce')
    projections = compile_projections(index, graph)
    base = Path(root) / 'assertions'
    base.mkdir(parents=True, exist_ok=True)
    payloads = {'source_index.json': index, 'assertion_graph.json': graph,
                'projection_manifest.json': projections, 'coverage.json': graph['coverage']}
    manifest = {'version': VERSION, 'mode': mode,
                'files': {name: digest(value) for name, value in payloads.items()}}
    generation = digest(manifest)
    target = base / generation
    staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=base))
    try:
        for name, value in payloads.items():
            (staging / name).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (staging / 'assertion_decisions.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=True) + '\n' for d in graph['decisions']))
        for path in staging.iterdir():
            with path.open('rb') as artifact:
                os.fsync(artifact.fileno())
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not target.exists():
            os.rename(staging, target)
        else:
            existing = json.loads((target / 'manifest.json').read_text())
            if existing != manifest:
                raise ValueError('generation collision')
            load_snapshot(root, generation=generation)
        handle, temp = tempfile.mkstemp(prefix='.CURRENT-', dir=base)
        try:
            with os.fdopen(handle, 'w') as output:
                output.write(generation + '\n')
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, base / 'CURRENT')
            descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            Path(temp).unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return generation


def _manifest(root: str | Path, generation: str | None) -> tuple[str, Path, dict]:
    """Resolve a generation and return its digest-checked manifest and folder."""
    base = Path(root) / 'assertions'
    if generation is None:
        if (base / 'CURRENT').is_symlink():
            raise ValueError('generation pointer is a symlink')
        generation = (base / 'CURRENT').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', generation):
        raise ValueError('invalid generation pointer')
    folder = base / generation
    if folder.is_symlink() or folder.resolve().parent != base.resolve():
        raise ValueError('snapshot path escapes store')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(manifest) != generation or manifest.get('version') != VERSION or manifest.get('mode') not in {'shadow', 'enforce'}:
        raise ValueError('invalid snapshot manifest')
    return generation, folder, manifest


def snapshot_identity(root: str | Path) -> tuple[str, str] | None:
    """Return ``(generation, mode)`` of the current snapshot, or None without one.

    Only the pointer and manifest are read; the manifest's digest is the
    generation name, so neither can be altered without detection. Artifacts are
    not validated, which keeps this cheap enough to call on every lookup.
    """
    if not (Path(root) / 'assertions' / 'CURRENT').exists():
        return None
    generation, _, manifest = _manifest(root, None)
    return generation, manifest['mode']


def snapshot_mode(root: str | Path) -> str | None:
    """Return the current generation's mode, or None without a generation.

    Shadow artifacts are not validated: they are never authoritative, so damage
    to them must not block a legacy store.
    """
    identity = snapshot_identity(root)
    return identity[1] if identity else None


def load_snapshot(root, *, generation=None):
    generation, folder, manifest = _manifest(root, generation)
    expected = {'source_index.json', 'assertion_graph.json', 'projection_manifest.json', 'coverage.json'}
    if set(manifest.get('files', {})) != expected:
        raise ValueError('incomplete snapshot manifest')
    payloads = {}
    for name in expected:
        path = folder / name
        if path.is_symlink():
            raise ValueError('snapshot artifact is a symlink')
        value = json.loads(path.read_text())
        if digest(value) != manifest['files'][name]:
            raise ValueError(f'snapshot digest mismatch: {name}')
        payloads[name] = value
    index, graph, projections = (payloads[k] for k in ['source_index.json', 'assertion_graph.json', 'projection_manifest.json'])
    decision_path = folder / 'assertion_decisions.jsonl'
    if decision_path.is_symlink() or [json.loads(line) for line in decision_path.read_text().splitlines()] != graph['decisions']:
        raise ValueError('decision log mismatch')
    if compile_projections(index, graph) != projections or graph['coverage'] != payloads['coverage.json']:
        raise ValueError('stale projection or coverage')
    return {'generation': generation, 'mode': manifest['mode'], 'index': index,
            'graph': graph, 'projections': projections}


def resolve_query(snapshot, *, assertion_ids=None, subject=None, predicate=None, role=None, numeric=False):
    """Select verified interpretations, never infer an applicability condition.

    Explicit assertion IDs disambiguate a source occurrence. Free text is only a
    candidate search; callers must not confuse retrieval with semantic approval.
    """
    graph = snapshot['graph']
    wanted = set(assertion_ids) if assertion_ids is not None else None
    matches = []
    for a in graph['assertions']:
        if wanted is not None and a['id'] not in wanted:
            continue
        events = [n for n in a['nodes'] if n['op'] == 'event']
        if subject and not any(subject.casefold() == anchored_text(v).casefold() for e in events for v in e['roles'].values() if v):
            continue
        if predicate and not any(predicate.casefold() == e['predicate']['text'].casefold() for e in events):
            continue
        matches.append(a)
    if wanted is not None and wanted - {a['id'] for a in matches}:
        return {'state': 'needs_context', 'values': [], 'assertions': sorted(wanted), 'reason': 'unknown assertion selection'}
    if not matches:
        return {'state': 'no_match', 'values': [], 'assertions': []}
    ids = [a['id'] for a in matches]
    if any(a['state'] not in {'verified', 'ambiguous'} for a in matches):
        return {'state': 'needs_review', 'values': [], 'assertions': ids}
    decisions = {d['assertion']: d for d in graph['decisions']}
    if any(assertion_dependencies(a, decisions.get(a['id'])) for a in matches):
        return {'state': 'needs_context', 'values': [], 'assertions': ids}
    if any(a['state'] == 'ambiguous' for a in matches):
        return {'state': 'ambiguous', 'values': matches, 'assertions': ids}
    if numeric:
        slots = [q for q in snapshot['projections']['numeric'] if q['assertion'] in ids]
        if len(slots) != len(matches):
            return {'state': 'needs_context', 'values': [], 'assertions': ids}
        values = {(Decimal(q['value']), q['unit']) for q in slots}
        return {'state': 'supported' if len(values) == 1 else 'conflict',
                'values': slots, 'assertions': ids}
    if role:
        values = [e['roles'].get(role) for a in matches for e in a['nodes'] if e['op'] == 'event']
        if not values or any(v is None for v in values):
            return {'state': 'unspecified', 'values': [], 'assertions': ids}
        return {'state': 'supported', 'values': values, 'assertions': ids}
    return {'state': 'supported', 'values': matches, 'assertions': ids}


def retrieve_literals(snapshot, question, *, limit=5):
    """Return source quotations with state, without free-text semantic synthesis."""
    terms = set(re.findall(r'\w+', question.casefold())) - {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'must', 'should'}
    rows = []
    for row in snapshot['projections']['literals']:
        text = ' '.join(s['text'] for s in row['evidence'])
        score = len(terms & set(re.findall(r'\w+', text.casefold())))
        if score:
            rows.append((score, row['assertion'], row))
    return [r for _, _, r in sorted(rows, key=lambda x: (-x[0], x[1]))[:limit]]


def audit_assertion_store(path):
    """Independent persisted-state audit; pending work cannot yield ready status."""
    try:
        snapshot = load_snapshot(path)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return {'state': 'invalid', 'errors': [str(exc)], 'counts': {}}
    counts = {}
    for row in snapshot['graph']['coverage']:
        counts[row['state']] = counts.get(row['state'], 0) + 1
    pending = sum(n for state, n in counts.items() if state not in {'verified', 'structural', 'context_only', 'non_assertive', 'ambiguous'})
    return {'state': 'needs_review' if pending else 'ready_with_ambiguity' if counts.get('ambiguous') else 'ready', 'errors': [],
            'counts': counts, 'generation': snapshot['generation'], 'pending_units': pending}


def restore_snapshot(path, generation, *, expected_generation):
    """Validate a retained generation and atomically restore it without deletion."""
    previous = load_snapshot(path, generation=generation)
    publish_snapshot(path, previous['index'], previous['graph'], mode=previous['mode'],
                     expected_generation=expected_generation)
    return load_snapshot(path)
