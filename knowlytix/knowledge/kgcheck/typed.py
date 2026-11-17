# SPDX-License-Identifier: Apache-2.0
"""Instance typing for the constraint rules: which classes a node belongs to, and why.

A node's direct types are its declared ``types`` (audit schema sidecar) and the
objects of its type-predicate edges in the view. Its types are the direct types
closed under the view's declared subclass edges; that closure is the only
inference made. Domain and range declarations never add a type: they are the
constraints being tested, and a type inferred from a constraint would satisfy
it by construction.

Each type carries a derivation path (the node, its direct type, then each
superclass), so a finding can show how a node came to be an instance of a
class. The closure walks the hierarchy iteratively and visits each class once.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from knowlytix.knowledge.kgcheck.models import AuditSnapshot, DeclKey, GraphView
from knowlytix.knowledge.kgcheck.taxonomy import hierarchy

Paths = dict[str, list[str]]  # class -> derivation path ending at that class


@dataclass
class Typing:
    """Types of the nodes of one view. ``closed`` is the declared closed-world typing."""

    direct: Mapping[str, tuple[str, ...]]
    parents: Mapping[str, list[str]]
    closed: bool
    _memo: dict[str, Paths] = field(default_factory=dict)

    def superclasses(self, cls: str) -> Paths:
        """``cls`` and every class above it, each with the path from ``cls``."""
        if cls not in self._memo:
            paths: Paths = {cls: [cls]}
            queue = deque([cls])
            while queue:
                current = queue.popleft()
                for parent in self.parents.get(current, ()):
                    if parent not in paths:
                        paths[parent] = [*paths[current], parent]
                        queue.append(parent)
            self._memo[cls] = paths
        return self._memo[cls]

    def types(self, node: str) -> Paths:
        """Every class ``node`` is an instance of, with the path from ``node``; shortest first."""
        out: Paths = {}
        for cls in self.direct.get(node, ()):
            for above, path in self.superclasses(cls).items():
                candidate = [node, *path]
                if above not in out or len(candidate) < len(out[above]):
                    out[above] = candidate
        return dict(sorted(out.items()))

    def typed(self) -> list[str]:
        """Nodes with at least one direct type, sorted."""
        return sorted(n for n, ts in self.direct.items() if ts)


def typing(snapshot: AuditSnapshot, view: GraphView) -> Typing:
    """The typing of ``view`` under the accepted sidecar declarations."""
    h = hierarchy(snapshot, view)
    closed = snapshot.semantics.accepted().flag(DeclKey.CLOSED_TYPING, "")
    return Typing(dict(h.types), h.successors(), closed)

