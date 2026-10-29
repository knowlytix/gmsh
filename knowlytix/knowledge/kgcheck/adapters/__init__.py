# SPDX-License-Identifier: Apache-2.0
"""Input adapters that build an ``AuditSnapshot`` from a source format."""

from knowlytix.knowledge.kgcheck.adapters.ontodisc_store import read_store
from knowlytix.knowledge.kgcheck.adapters.ontology_only import read_ontology
from knowlytix.knowledge.kgcheck.adapters.rdf_input import read_rdf

__all__ = ["read_ontology", "read_rdf", "read_store"]
