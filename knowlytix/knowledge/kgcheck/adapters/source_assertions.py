# SPDX-License-Identifier: Apache-2.0
"""Read immutable source-assertion stores without inventing legacy artifacts.

Projection edges describe a proposition's syntax, not unconditional domain
facts. Keep the source graph intact for native checks instead of flattening
conditions into triples. Typed legacy rules therefore remain unavailable.
"""
from pathlib import Path

from knowlytix.knowledge.kgcheck.models import (
    ArtifactRecord,
    AuditSnapshot,
    Capabilities,
    Diagnostic,
    GraphView,
    OntologyDecl,
    RawRef,
)


def assertion_snapshot(index, graph, *, generation=None, errors=(), changed=False):
    """In-memory audit input; producer validation results are explicit evidence."""
    from knowlytix.knowledge.assertions.contract import validate_graph
    from knowlytix.knowledge.assertions.source import digest
    problems = list(errors) or validate_graph(graph, index)
    return AuditSnapshot(
        snapshot_id=generation or digest([index, graph, problems]),
        nodes=(), assertions=(), views={GraphView.STORED: ()}, ontology=OntologyDecl(),
        capabilities=Capabilities(adapter='source_assertions', adapter_version='1'),
        diagnostics=tuple(Diagnostic('assertion_contract', str(e), RawRef('assertion_graph.json'))
                          for e in problems) + ((Diagnostic('changed_during_read',
                          'CURRENT changed while reading the snapshot', RawRef('assertions/CURRENT')),) if changed else ()),
        parsed={'source_index.json': index, 'assertion_graph.json': graph},
        metadata={'representation': 'source_assertions',
                  'semantic_accuracy_measured': False,
                  'contract_valid': not problems})


def read_assertion_store(store):
    """Validate digests, source anchors, decisions, coverage and saved projections."""
    from dataclasses import replace

    from knowlytix.knowledge.assertions.store import load_snapshot
    root = Path(store)
    pointer = root / 'assertions/CURRENT'
    index, graph, generation, errors, changed = {}, {}, None, [], False
    try:
        before = pointer.read_bytes()
        saved = load_snapshot(root)
        index, graph, generation = saved['index'], saved['graph'], saved['generation']
        changed = before != pointer.read_bytes()
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        errors = [f'{type(exc).__name__}: {exc}']
    snapshot = assertion_snapshot(index, graph, generation=generation, errors=errors, changed=changed)
    return replace(snapshot, artifacts=(ArtifactRecord('assertions/CURRENT', pointer.exists(), True),))
