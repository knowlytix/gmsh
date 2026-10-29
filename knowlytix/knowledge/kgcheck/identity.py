# SPDX-License-Identifier: Apache-2.0
"""Stable identifiers: independent of paths, timestamps, row order and views."""

from __future__ import annotations

import hashlib
import json


def node_id(namespace: str, token: str) -> str:
    """``entity:<token>`` or ``relation:<token>``, the token exactly as stored."""
    return f"{namespace}:{token}"


def assertion_id(head: str, predicate: str, obj: str) -> str:
    """Content id of one (head, predicate, object) statement over raw stored tokens.

    The same statement gets the same id in every view, so views can be compared.
    """
    payload = json.dumps([head, predicate, obj], ensure_ascii=False, separators=(",", ":"))
    return "a:" + hashlib.sha256(_bytes(payload)).hexdigest()[:20]


def finding_id(rule_id: str, rule_version: str, view: str | None, affected: tuple[str, ...],
               locations: tuple[str, ...], discriminator: str = "") -> str:
    """Stable id of one finding across reruns and report orderings.

    Identity is the rule and its version, the view and what the finding is
    about: its sorted affected node/assertion ids or, for a finding about raw
    rows with no graph identity, its sorted artifact locations. Message text,
    witness content, timestamps and absolute paths are excluded; a rule that
    raises two distinct issues on the same target sets ``discriminator``.
    """
    subject = sorted(affected) if affected else sorted(locations)
    payload = json.dumps([rule_id, rule_version, view or "", subject, discriminator],
                         ensure_ascii=False, separators=(",", ":"))
    return "f:" + hashlib.sha256(_bytes(payload)).hexdigest()[:20]


def _bytes(payload: str) -> bytes:
    """UTF-8, passing lone surrogates through so a malformed token still gets an id."""
    return payload.encode("utf-8", errors="surrogatepass")
