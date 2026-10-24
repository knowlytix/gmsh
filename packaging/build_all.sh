#!/usr/bin/env bash
# orchestrate `python -m build --wheel` across all four scoped wheels.
#
# Dependency order per `distribution/distribution_plan.md` §Packaging
# layout: `gms-core` first (every other wheel depends on it), then
# `docgms` + `finstructbench` in parallel (both depend only on gms-core),
# then `gmsh` last (depends on the other three).
#
# This script intentionally does NOT parallelize — serial builds make
# CI log output readable and sidestep any concurrency issues in the
# build backends. The cost is a few seconds per wheel.
#
# **Wheel-only policy**: we pass `--wheel` to `python -m build`. The default
# (`python -m build` with no flags) builds the sdist first, extracts it to
# a tmpdir, then builds the wheel from that tmpdir — which re-roots the
# project directory at `/tmp/.../` and breaks the `../../gms` etc. paths
# our scoped pyproject files use to reach source repository source. `--wheel`
# bypasses the sdist intermediate and runs the hatchling backend directly
# in the packaging/<pkg>/ directory, so `../../<source>` resolves
# correctly. Sdists aren't published for the source repository wheels today; if
# we need them later they'll require a staging-dir refactor (copy source
# alongside pyproject before building). Tracked alongside.
#
# Usage:
#   ./packaging/build_all.sh          # builds all 4 into ./dist/
#   DIST_DIR=/tmp/out ./packaging/build_all.sh
#
# Exits non-zero on any build failure (set -e).

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${DIST_DIR:-${REPO_ROOT}/dist}"

mkdir -p "${DIST_DIR}"

# Dependency order: core <- benchmark <- knowledge <- harness
# (knowlytix-knowledge depends on knowlytix-benchmark; knowlytix-studio#78).
PACKAGES=(
    "knowlytix-core"
    "knowlytix-benchmark"
    "knowlytix-knowledge"
    "knowlytix-harness"
)

echo "=========================================="
echo "GMS wheel build — dist dir: ${DIST_DIR}"
echo "=========================================="

for pkg in "${PACKAGES[@]}"; do
    pkg_dir="${SCRIPT_DIR}/${pkg}"
    if [[ ! -f "${pkg_dir}/pyproject.toml" ]]; then
        echo "ERROR: missing pyproject.toml at ${pkg_dir}" >&2
        exit 1
    fi
    echo ""
    echo "---- building ${pkg} ----"

    if [[ "${pkg}" == "knowlytix-core" && -x "${pkg_dir}/build.sh" ]]; then
        # knowlytix-core builds via its own compiled-binary build script.
        DIST_DIR="${DIST_DIR}" "${pkg_dir}/build.sh"
    else
        # `--wheel` skips the sdist intermediate; see the header comment for
        # why that matters with cross-directory `force-include` paths.
        python -m build --wheel --outdir "${DIST_DIR}" "${pkg_dir}"
    fi
done

echo ""
echo "=========================================="
echo "Built wheels:"
echo "=========================================="
ls -1 "${DIST_DIR}"/*.whl 2>/dev/null || {
    echo "ERROR: no wheels produced in ${DIST_DIR}" >&2
    exit 1
}
