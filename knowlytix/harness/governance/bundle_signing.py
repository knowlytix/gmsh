# SPDX-License-Identifier: Apache-2.0
"""Provenance signing for GovernanceBundle.

SHA-256 catches mutation but does not authenticate the producer:
anyone can compute a hash. For high-stakes deployment, the bundle
must carry a signature that proves it came from a deployment-managed
signer. We use HMAC-SHA-256 with a shared key.

Two signing methods, selected per bundle:

  ``sha256``         : plain digest. Integrity-only. Fast,
                       no secret material. Suitable for development.
  ``hmac-sha256``    : keyed HMAC. Integrity + producer auth.
                       Required for high-stakes deployment.

Bundles record their signing method so callers can detect attempts
to downgrade the signing scheme (a tampered bundle re-signed with
sha256 will fail to verify against the deployment key).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Literal


SigningMethod = Literal["sha256", "hmac-sha256"]


SIGNING_KEY_ENV_VAR = "GOVERNED_AGENT_SIGNING_KEY"


class BundleSigningError(RuntimeError):
    """Bundle signature does not authenticate."""


def resolve_signing_key(key: bytes | str | None = None) -> bytes | None:
    """Resolve a signing key from the argument or environment.

    Returns ``None`` when neither is set. Callers receive ``None``
    and should fall back to ``sha256`` integrity-only signing.
    """
    if key is None:
        env = os.environ.get(SIGNING_KEY_ENV_VAR)
        if env is None or env == "":
            return None
        return env.encode("utf-8")
    if isinstance(key, str):
        return key.encode("utf-8")
    return key


def sign(payload: bytes, key: bytes | None = None) -> tuple[str, SigningMethod]:
    """Sign payload bytes; returns ``(signature_hex, method)``."""
    if key is None:
        return hashlib.sha256(payload).hexdigest(), "sha256"
    if len(key) < 16:
        raise BundleSigningError(
            f"Signing key too short ({len(key)} bytes). Use at least "
            f"16 bytes of entropy for HMAC-SHA-256."
        )
    return hmac.new(key, payload, hashlib.sha256).hexdigest(), "hmac-sha256"


def verify(
    payload: bytes,
    signature: str,
    method: SigningMethod,
    key: bytes | None = None,
) -> bool:
    """Verify a signature. Constant-time comparison.

    When ``key`` is provided, the bundle must be signed with
    ``hmac-sha256`` — a key-equipped caller refuses to accept an
    integrity-only ``sha256`` signature, since that defeats producer
    authentication and opens a downgrade attack.
    """
    if method == "sha256":
        if key is not None:
            raise BundleSigningError(
                "Bundle is signed with sha256 (integrity-only), but "
                "the caller provided a signing key — refusing to "
                "accept downgraded bundle. Re-sign with HMAC-SHA-256 "
                "before deploying behind an authenticated gate."
            )
        expected = hashlib.sha256(payload).hexdigest()
        return hmac.compare_digest(signature, expected)
    if method == "hmac-sha256":
        if key is None:
            raise BundleSigningError(
                "Bundle was signed with HMAC-SHA-256 but no signing "
                f"key was provided. Set the {SIGNING_KEY_ENV_VAR} "
                "environment variable or pass key= explicitly."
            )
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    raise BundleSigningError(f"Unknown signing method: {method}")
