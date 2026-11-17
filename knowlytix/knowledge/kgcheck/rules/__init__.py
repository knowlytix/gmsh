# SPDX-License-Identifier: Apache-2.0
"""The shipped rules. Importing this package registers them with the builtin registry."""

from knowlytix.knowledge.kgcheck.rules import (
    artifacts,  # noqa: F401
    assertions,  # noqa: F401
    constraints,  # noqa: F401
    coverage,  # noqa: F401
    definitions,  # noqa: F401
    evidence,  # noqa: F401
    grounding,  # noqa: F401
    hierarchy,  # noqa: F401
    inheritance,  # noqa: F401
    lexical,  # noqa: F401
    metadata,  # noqa: F401
    modeling,  # noqa: F401
    numeric,  # noqa: F401
    producer,  # noqa: F401
    reasoner,  # noqa: F401
    shacl,  # noqa: F401
    structure,  # noqa: F401
    vocabulary,  # noqa: F401
)
