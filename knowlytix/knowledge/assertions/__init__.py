# SPDX-License-Identifier: Apache-2.0
"""Independent source/graph validation, persisted snapshots and scoped queries."""
from .contract import claim_digest, validate_assertion, validate_graph, validate_index
from .evaluation import assertion_structure, score_assertions
from .source import assemble_context, index_documents
from .store import (
    audit_assertion_store,
    compile_projections,
    load_snapshot,
    publish_snapshot,
    resolve_query,
    restore_snapshot,
    retrieve_literals,
)

__all__ = [
    'claim_digest', 'validate_assertion', 'validate_graph', 'validate_index',
    'assertion_structure', 'score_assertions', 'assemble_context', 'index_documents',
    'audit_assertion_store', 'compile_projections', 'load_snapshot', 'publish_snapshot',
    'resolve_query', 'restore_snapshot', 'retrieve_literals',
]
