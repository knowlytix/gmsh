#!/usr/bin/env bash
# GMS-056: verify the four wheels built by `build_all.sh` install cleanly
# and pass the smoke + audit checks listed in the story's acceptance
# criteria.
#
# Runs in a throwaway venv created under $TMPDIR (default: /tmp) so the
# host environment is untouched. Exits non-zero on the first failure.
#
# Usage:
#   ./packaging/verify_wheels.sh             # uses ./dist
#   DIST_DIR=/path/to/dist ./packaging/verify_wheels.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${DIST_DIR:-${REPO_ROOT}/dist}"

# Pull the four wheel paths (newest of each name) from DIST_DIR.
# GMS-111 renamed dist → wheel-filename prefixes (pip converts
# hyphens to underscores): knowlytix_core, knowlytix_doc,
# knowlytix_bench, knowlytix_harness.
shopt -s nullglob
CORE_WHL=( "${DIST_DIR}"/knowlytix_core-*.whl )
DOC_WHL=( "${DIST_DIR}"/knowlytix_doc-*.whl )
BENCH_WHL=( "${DIST_DIR}"/knowlytix_bench-*.whl )
HARNESS_WHL=( "${DIST_DIR}"/knowlytix_harness-*.whl )
shopt -u nullglob

for name in CORE_WHL DOC_WHL BENCH_WHL HARNESS_WHL; do
    # eval-based indirection — compatible with macOS Bash 3.2 (which
    # lacks `declare -n` namerefs introduced in Bash 4.3).
    eval "count=\${#${name}[@]}"
    if [[ "${count}" -eq 0 ]]; then
        echo "ERROR: no ${name} wheel found under ${DIST_DIR}" >&2
        echo "  run ./packaging/build_all.sh first" >&2
        exit 1
    fi
done

echo "=========================================="
echo "Verifying wheels in ${DIST_DIR}"
echo "=========================================="

# --- Wheel-layout audit (no-exec; just unzip -l) -----------------------------
# The story requires: no .tex/.pdf/private-tree files, and the fixture
# assets are present in each wheel that should carry them.

audit_wheel() {
    local whl="$1" ; shift
    local wheel_name
    wheel_name="$(basename "${whl}")"
    echo ""
    echo "-- auditing ${wheel_name} --"

    # Forbidden patterns from `distribution/gmsh_packaging_plan.md`.
    # `[[:space:]]demo/` catches top-level `demo/` only (preceded by the
    # whitespace that `unzip -l` inserts before the path), so it blocks the
    # repo-root `demo/` tree without colliding with the legitimate shipped
    # `knowlytix/bench/fixtures/demo/` fixtures package added 2026-04-18.
    local forbidden=(
        '\.tex$'
        '\.pdf$'
        '/gmsh_testing_int/'
        '/gmsh_testing_ext/'
        '/gms_wiki/'
        '/gms_llm/'
        '/gms_agent_builder_'
        '[[:space:]]demo/'
    )
    local listing
    listing=$(unzip -l "${whl}")
    for pat in "${forbidden[@]}"; do
        if echo "${listing}" | grep -Eq "${pat}"; then
            echo "ERROR: ${wheel_name} contains forbidden pattern: ${pat}" >&2
            echo "${listing}" | grep -E "${pat}" >&2
            exit 1
        fi
    done

    # Caller-supplied required patterns (fixtures, LICENSE, README).
    for pat in "$@"; do
        if ! echo "${listing}" | grep -Eq "${pat}"; then
            echo "ERROR: ${wheel_name} missing required pattern: ${pat}" >&2
            exit 1
        fi
    done
    echo "ok: ${wheel_name}"
}

audit_wheel "${CORE_WHL[0]}" \
    'LICENSE\.txt' \
    'knowlytix-public\.pem'

# GMS-113: knowlytix-doc wheel ships knowlytix/doc/ namespace slice.
audit_wheel "${DOC_WHL[0]}" \
    'LICENSE' \
    'knowlytix/doc/fixtures/smoke/sample\.md'

# GMS-114: knowlytix-bench wheel ships knowlytix/bench/ namespace slice.
audit_wheel "${BENCH_WHL[0]}" \
    'LICENSE' \
    'knowlytix/bench/fixtures/smoke/questions\.json' \
    'knowlytix/bench/fixtures/smoke/predictions\.json' \
    'knowlytix/bench/instances/model_validation\.md'

# GMS-115: knowlytix-harness wheel ships knowlytix/harness/ namespace slice.
audit_wheel "${HARNESS_WHL[0]}" \
    'LICENSE' \
    'knowlytix/harness/fixtures/smoke/doe_smoke\.json' \
    'knowlytix/harness/fixtures/smoke/ground_truth\.json' \
    'knowlytix/harness/configs/default\.yaml'

# --- Wheel METADATA integrity check ------------------------------------------
# Verifying a README.md file as a wheel member is fragile — PEP 621's
# `project.readme` embeds README content into the wheel's
# `.dist-info/METADATA` as the `Description:` field, not as a standalone
# file. This check inspects METADATA directly (the same surface `pip
# show` and PyPI render) and asserts:
#   - Name / Version match expectations
#   - Summary is non-empty (one-line project description)
#   - Description body is non-empty (README content was embedded)
#   - License SPDX identifier is declared
# Robust against hatchling version drift and monorepo-inner-README
# collisions; it probes the invariant that actually matters to users.

echo ""
echo "-- wheel METADATA integrity check --"
python - "${CORE_WHL[0]}" "${DOC_WHL[0]}" "${BENCH_WHL[0]}" "${HARNESS_WHL[0]}" <<'PY'
import sys
import zipfile
from email import message_from_string
from pathlib import Path

EXPECTED = {
    "knowlytix-core": {"version": "0.1.0", "license": "LicenseRef-Knowlytix-GMS-EULA"},
    "knowlytix-doc": {"version": "0.1.0", "license": "Apache-2.0"},
    "knowlytix-bench": {"version": "0.1.0", "license": "Apache-2.0"},
    "knowlytix-harness": {"version": "0.1.0", "license": "Apache-2.0"},
}

for whl_path in sys.argv[1:]:
    whl_name = Path(whl_path).name
    with zipfile.ZipFile(whl_path) as zf:
        metadata_entries = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metadata_entries) == 1, f"{whl_name}: expected 1 METADATA file, got {metadata_entries}"
        raw = zf.read(metadata_entries[0]).decode("utf-8")

    # METADATA is RFC822-style: headers, blank line, body (the README).
    msg = message_from_string(raw)
    name = msg["Name"]
    version = msg["Version"]
    summary = (msg["Summary"] or "").strip()
    license_expr = (msg["License-Expression"] or msg["License"] or "").strip()
    body = msg.get_payload().strip()

    assert name in EXPECTED, f"{whl_name}: unexpected Name {name!r}"
    exp = EXPECTED[name]
    assert version == exp["version"], (
        f"{whl_name}: Version {version!r} != expected {exp['version']!r}"
    )
    assert summary, f"{whl_name}: Summary is empty"
    assert body, f"{whl_name}: Description body is empty (README content not embedded)"
    assert len(body) > 200, (
        f"{whl_name}: Description body is only {len(body)} chars — README probably didn't embed"
    )
    assert license_expr == exp["license"], (
        f"{whl_name}: License {license_expr!r} != expected {exp['license']!r}"
    )
    print(f"  ok: {name}=={version} [{license_expr}] summary={summary[:60]!r}... desc={len(body)} chars")

print("METADATA integrity check ok")
PY

# --- Fixture size guard (reuses tests/api/test_smoke_fixtures.py rule) -------
# Every shipped fixture must be under 500 KB. Extracted from the wheel so
# we measure what customers actually receive.

echo ""
echo "-- fixture size guard --"
TMPEX="$(mktemp -d)"
trap 'rm -rf "${TMPEX}"' EXIT
for whl in "${DOC_WHL[0]}" "${BENCH_WHL[0]}" "${HARNESS_WHL[0]}"; do
    unzip -qq -o "${whl}" -d "${TMPEX}/$(basename "${whl}" .whl)"
done
# shellcheck disable=SC2044
for f in $(find "${TMPEX}" -path '*/fixtures/smoke/*' -type f ! -name '__init__.py'); do
    size=$(stat -c%s "${f}" 2>/dev/null || stat -f%z "${f}")
    if [[ ${size} -gt 512000 ]]; then
        echo "ERROR: fixture exceeds 500 KB: ${f} (${size} bytes)" >&2
        exit 1
    fi
done
echo "ok: all fixtures under 500 KB"

# --- Fresh-venv install + import smoke ---------------------------------------
# Install the four wheels in dep order and exercise the declared public
# API (GMS-036) plus `importlib.resources` fixture lookups (GMS-037).
#
# Skipped when VERIFY_SKIP_INSTALL=1 is set — useful on machines without
# the torch binary wheels cached, or in environments where pip install is
# gated.

if [[ "${VERIFY_SKIP_INSTALL:-0}" != "1" ]]; then
    echo ""
    echo "-- fresh-venv install smoke --"
    VENV_DIR="$(mktemp -d)/venv"
    python -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet \
        "${CORE_WHL[0]}" \
        "${DOC_WHL[0]}" \
        "${BENCH_WHL[0]}" \
        "${HARNESS_WHL[0]}"

    # CLI entry points wired via pyproject.toml project.scripts.
    gmsh --help >/dev/null
    knowlytix-doe --help >/dev/null

    # Public-API checks — mirror tests/api/test_public_api.py exactly (same
    # REQUIRED sets, same assertions) but against the installed wheels so
    # we catch "source says X but wheel ships Y" drift. Also exercises the
    # `from <pkg> import *` wildcard path the AC calls out explicitly.
    python - <<'PY'
import importlib

REQUIRED = {
    "gms": {
        "GMSCoreSettings", "LLMClient", "LLMConfigError",
        "LLMSettings", "ModelPurpose", "get_llm",
    },
    "knowlytix.doc": {
        "DocGMSSettings", "GMSExpertStore", "QueryEngine",
        "QueryResult", "IngestResult", "ingest_document",
    },
    "knowlytix.bench": {
        "FinStructBenchSettings", "GeneratedQuestion",
        "ScoreResult", "score_answer",
    },
    "knowlytix.harness.doe": {
        "CampaignManager", "DOEGMSBenchmark", "GMSHSettings",
        "HallucinationOracle", "PolicyEngine",
        "SeverityClassifier", "ToolGateway",
    },
}

for pkg, required in REQUIRED.items():
    mod = importlib.import_module(pkg)
    assert hasattr(mod, "__all__"), f"{pkg} missing __all__"
    all_ = set(mod.__all__)
    missing = required - all_
    assert not missing, f"{pkg} __all__ missing required symbols: {missing}"
    # Every declared name must actually resolve.
    for sym in mod.__all__:
        assert hasattr(mod, sym), f"{pkg}.{sym} declared in __all__ but not resolvable"
    # Wildcard import yields exactly __all__ — no leaks, no missing.
    ns: dict[str, object] = {}
    exec(f"from {pkg} import *", ns)
    leaked = {k for k in ns if not k.startswith("_")} - all_
    assert not leaked, f"{pkg} wildcard import leaks non-__all__ symbols: {sorted(leaked)}"
print("public API check ok")
PY

    # Fixture reachability via importlib.resources (README quickstart).
    python - <<'PY'
from importlib.resources import files
for pkg, name in [
    ("knowlytix.doc.fixtures.smoke", "sample.md"),
    ("knowlytix.bench.fixtures.smoke", "questions.json"),
    ("knowlytix.bench.fixtures.smoke", "predictions.json"),
    ("knowlytix.harness.fixtures.smoke", "doe_smoke.json"),
    ("knowlytix.harness.fixtures.smoke", "ground_truth.json"),
]:
    p = files(pkg) / name
    data = p.read_bytes()
    assert len(data) > 0, f"{pkg}/{name} is empty"
print("importlib.resources fixture check ok")
PY

    # End-to-end smoke: instantiate Settings classes + run one real
    # public-API entry point per package (finstructbench.score_answer)
    # + validate harness DOE fixture schema.
    #
    # Calls the installed `knowlytix-smoke` console script (GMS-096) so the
    # verification exercises the same entry point external PyPI users
    # will run — not a repo-local wrapper, which is only reachable to
    # teammates with a full repo clone.
    knowlytix-smoke

    deactivate
    echo "ok: installed wheels pass all smoke checks"
else
    echo ""
    echo "-- fresh-venv install smoke: SKIPPED (VERIFY_SKIP_INSTALL=1) --"
fi

# --- Config centralization audit against installed tree ----------------------
# Re-run the GMS-024 audit to confirm no shipped source in the wheels
# violated the centralization rule.

echo ""
echo "-- LLM centralization audit --"
python "${REPO_ROOT}/scripts/audit_llm_centralization.py"

echo ""
echo "=========================================="
echo "All wheel verifications passed."
echo "=========================================="
