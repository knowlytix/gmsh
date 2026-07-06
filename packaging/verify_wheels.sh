#!/usr/bin/env bash
# verify the four wheels built by `build_all.sh` install cleanly
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
# renamed dist → wheel-filename prefixes (pip converts
# hyphens to underscores): knowlytix_core, knowlytix_knowledge,
# knowlytix_benchmark, knowlytix_harness. (Phase 4.10/e
# renamed `knowlytix_doc` → `knowlytix_knowledge` and
# `knowlytix_bench` → `knowlytix_benchmark`.)
shopt -s nullglob
CORE_WHL=( "${DIST_DIR}"/knowlytix_core-*.whl )
KNOWLEDGE_WHL=( "${DIST_DIR}"/knowlytix_knowledge-*.whl )
BENCHMARK_WHL=( "${DIST_DIR}"/knowlytix_benchmark-*.whl )
HARNESS_WHL=( "${DIST_DIR}"/knowlytix_harness-*.whl )
shopt -u nullglob

for name in CORE_WHL KNOWLEDGE_WHL BENCHMARK_WHL HARNESS_WHL; do
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
    # `knowlytix/benchmark/fixtures/demo/` fixtures package added 2026-04-18.
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
    # Use here-strings instead of `echo | grep` to avoid a SIGPIPE
    # race: with `set -o pipefail`, when `grep -Eq` finds an early
    # match and exits, the upstream `echo` gets SIGPIPE (exit 141),
    # which makes the pipeline non-zero and inverts the if-condition.
    # BSD echo on macOS surfaces this loudly ("write error: Broken
    # pipe"), GNU echo on Linux absorbs it — but the false-positive
    # is platform-agnostic. Fired on the macOS rc9 build when LICENSE
    # ended up early enough in the harness wheel's central directory
    # to lose the race. Here-strings have no pipe, so no SIGPIPE.
    for pat in "${forbidden[@]}"; do
        if grep -Eq "${pat}" <<< "${listing}"; then
            echo "ERROR: ${wheel_name} contains forbidden pattern: ${pat}" >&2
            grep -E "${pat}" <<< "${listing}" >&2
            exit 1
        fi
    done

    # Caller-supplied required patterns (fixtures, LICENSE, README).
    for pat in "$@"; do
        if ! grep -Eq "${pat}" <<< "${listing}"; then
            echo "ERROR: ${wheel_name} missing required pattern: ${pat}" >&2
            exit 1
        fi
    done
    echo "ok: ${wheel_name}"
}

audit_wheel "${CORE_WHL[0]}" \
    'LICENSE\.txt' \
    'knowlytix-public\.pem' \
    'knowlytix/core/_license_keys/'

# knowlytix-knowledge wheel ships knowlytix/knowledge/ namespace slice.
audit_wheel "${KNOWLEDGE_WHL[0]}" \
    'LICENSE' \
    'knowlytix/knowledge/fixtures/smoke/sample\.md'

# knowlytix-benchmark wheel ships knowlytix/benchmark/ namespace slice.
audit_wheel "${BENCHMARK_WHL[0]}" \
    'LICENSE' \
    'knowlytix/benchmark/fixtures/smoke/questions\.json' \
    'knowlytix/benchmark/fixtures/smoke/predictions\.json' \
    'knowlytix/benchmark/instances/model_validation\.md'

# knowlytix-harness wheel ships knowlytix/harness/ namespace slice.
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
# Robust against hatchling version drift and source repository-inner-README
# collisions; it probes the invariant that actually matters to users.

echo ""
echo "-- wheel METADATA integrity check --"
python - "${CORE_WHL[0]}" "${KNOWLEDGE_WHL[0]}" "${BENCHMARK_WHL[0]}" "${HARNESS_WHL[0]}" <<'PY'
import sys
import zipfile
from email import message_from_string
from pathlib import Path

# Versions are derived from the git tag (hatch-vcs) — there is no literal to
# assert against. Verify the invariants that actually matter instead: each
# wheel's license SPDX is correct, every version is valid PEP 440, and all
# four wheels carry the SAME version (lockstep).
from packaging.version import Version

EXPECTED_LICENSE = {
    "knowlytix-core": "LicenseRef-Knowlytix-GMS-EULA",
    "knowlytix-knowledge": "Apache-2.0",
    "knowlytix-benchmark": "Apache-2.0",
    "knowlytix-harness": "Apache-2.0",
}

seen_versions = {}
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

    assert name in EXPECTED_LICENSE, f"{whl_name}: unexpected Name {name!r}"
    Version(version)  # raises InvalidVersion if hatch-vcs emitted garbage
    seen_versions[name] = version
    assert summary, f"{whl_name}: Summary is empty"
    assert body, f"{whl_name}: Description body is empty (README content not embedded)"
    assert len(body) > 200, (
        f"{whl_name}: Description body is only {len(body)} chars — README probably didn't embed"
    )
    assert license_expr == EXPECTED_LICENSE[name], (
        f"{whl_name}: License {license_expr!r} != expected {EXPECTED_LICENSE[name]!r}"
    )
    print(f"  ok: {name}=={version} [{license_expr}] summary={summary[:60]!r}... desc={len(body)} chars")

# Lockstep: every wheel must carry the identical git-derived version.
unique = set(seen_versions.values())
assert len(unique) == 1, f"lockstep version mismatch across wheels: {seen_versions}"
print(f"lockstep version ok: all wheels == {unique.pop()}")
print("METADATA integrity check ok")
PY

# --- Fixture size guard (reuses tests/api/test_smoke_fixtures.py rule) -------
# Every shipped fixture must be under 500 KB. Extracted from the wheel so
# we measure what customers actually receive.

echo ""
echo "-- fixture size guard --"
TMPEX="$(mktemp -d)"
trap 'rm -rf "${TMPEX}"' EXIT
for whl in "${KNOWLEDGE_WHL[0]}" "${BENCHMARK_WHL[0]}" "${HARNESS_WHL[0]}"; do
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
# API plus `importlib.resources` fixture lookups.
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
        "${KNOWLEDGE_WHL[0]}" \
        "${BENCHMARK_WHL[0]}" \
        "${HARNESS_WHL[0]}"

    # CLI entry points wired via pyproject.toml project.scripts.
    knowlytix-harness --help >/dev/null
    knowlytix-testing --help >/dev/null

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
    "knowlytix.knowledge": {
        "DocGMSSettings", "GMSExpertStore", "QueryEngine",
        "QueryResult", "IngestResult", "ingest_document",
    },
    "knowlytix.benchmark": {
        "FinStructBenchSettings", "GeneratedQuestion",
        "ScoreResult", "score_answer",
    },
    "knowlytix.harness.testing": {
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
    ("knowlytix.knowledge.fixtures.smoke", "sample.md"),
    ("knowlytix.benchmark.fixtures.smoke", "questions.json"),
    ("knowlytix.benchmark.fixtures.smoke", "predictions.json"),
    ("knowlytix.harness.fixtures.smoke", "doe_smoke.json"),
    ("knowlytix.harness.fixtures.smoke", "ground_truth.json"),
]:
    p = files(pkg) / name
    data = p.read_bytes()
    assert len(data) > 0, f"{pkg}/{name} is empty"
print("importlib.resources fixture check ok")
PY

    # End-to-end smoke: instantiate Settings classes + run one real
    # public-API entry point per package (knowlytix.benchmark.score_answer)
    # + validate harness DOE fixture schema.
    #
    # Calls the installed `knowlytix-smoke` console script so the
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
# Re-run the audit to confirm no shipped source in the wheels
# violated the centralization rule.

echo ""
echo "-- LLM centralization audit --"
python "${REPO_ROOT}/scripts/audit_llm_centralization.py"

# --- Compiled-core source-leak gate -----------------------------------------
# The commercial knowlytix-core ships the proprietary geometry as a Nuitka
# `.so`, never as readable `.py`. Two checks:
#   1. Self-describing (always): a PLATFORM-tagged core wheel must contain no
#      raw `knowlytix/core/*.py` beyond the re-export stub __init__.py.
#   2. Opt-in (VERIFY_REQUIRE_COMPILED_CORE=1, set by build-internal-rc.yml):
#      the core wheel MUST be compiled — a py3-none-any wheel is a leak.
# ci.yml builds with SKIP_NUITKA=1 and does NOT set the opt-in flag, so its
# legitimate pure-Python py3-none-any wheel still passes.
echo ""
echo "-- compiled-core source-leak gate --"
core_whl="${CORE_WHL[0]}"
core_base="$(basename "${core_whl}")"
# Use bare filenames (namelist), not `zipfile -l` — its tabular output ends
# each line with the size column, so anchoring `\.py$` against it never
# matches (it would silently pass a source-leaking wheel).
core_names="$(python3 -c \
    "import sys, zipfile; print('\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))" \
    "${core_whl}")"
# Both commercial trees in this wheel must ship compiled: knowlytix/core
# and knowlytix/embedding. Any .py beyond each tree's re-export stub
# __init__.py is leaked proprietary source.
core_src_py="$(echo "${core_names}" \
    | grep -E 'knowlytix/(core|embedding)/.*\.py$' \
    | grep -vE 'knowlytix/(core|embedding)/__init__\.py$' || true)"

core_is_pure=0
case "${core_base}" in
    *-py3-none-any.whl) core_is_pure=1 ;;
esac

if [[ "${VERIFY_REQUIRE_COMPILED_CORE:-0}" == "1" && "${core_is_pure}" == "1" ]]; then
    echo "ERROR: knowlytix-core shipped as pure-Python (${core_base}) but" >&2
    echo "  VERIFY_REQUIRE_COMPILED_CORE=1 — the commercial core MUST ship" >&2
    echo "  Nuitka-compiled. The Nuitka output was not consumed by the wheel." >&2
    exit 1
fi
if [[ "${core_is_pure}" == "0" && -n "${core_src_py}" ]]; then
    echo "ERROR: platform-tagged knowlytix-core wheel (${core_base}) contains" >&2
    echo "  raw .py source — proprietary source leaked into a compiled wheel:" >&2
    echo "${core_src_py}" >&2
    exit 1
fi
echo "ok: knowlytix-core passes the source-leak gate (pure=${core_is_pure})"

echo ""
echo "=========================================="
echo "All wheel verifications passed."
echo "=========================================="
