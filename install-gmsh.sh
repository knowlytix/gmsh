#!/usr/bin/env bash
# One-command installer for the Knowlytix GMS wheels from the
# knowlytix/gmsh release channel.
#
# Does the whole dance for you: verifies Python 3.12 + gh auth, creates a
# fresh virtualenv, downloads the latest (or a pinned) release's 4 wheels,
# verifies checksums, installs them in dependency order (pip pulls the
# heavy transitive deps — torch, scipy, litellm, ... — from PyPI), and
# runs the post-install smoke test.
#
# Usage:
#   ./install-gmsh.sh                 # latest release into ~/.venv/knowlytix
#   ./install-gmsh.sh --tag v0.2.0-rcN
#   ./install-gmsh.sh --venv ./.venv  # custom venv location
#   ./install-gmsh.sh --no-smoke      # skip the smoke step
#
# Requirements:
#   - python3.12 on PATH (the knowlytix-core wheel is compiled for 3.12)
#   - gh (GitHub CLI) authenticated with read access to knowlytix/gmsh
#   - KNOWLYTIX_LICENSE_KEY exported for the smoke step (optional otherwise)
set -euo pipefail

REPO="knowlytix/gmsh"
VENV="${HOME}/.venv/knowlytix"
TAG=""
RUN_SMOKE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)      TAG="$2"; shift 2 ;;
    --venv)     VENV="$2"; shift 2 ;;
    --no-smoke) RUN_SMOKE=0; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Preconditions ----------------------------------------------------------
command -v python3.12 >/dev/null \
  || die "python3.12 not found on PATH. Install it (macOS: brew install python@3.12; else python.org/downloads)."
command -v gh >/dev/null \
  || die "gh (GitHub CLI) not found. Install from https://cli.github.com then run 'gh auth login'."
gh auth status >/dev/null 2>&1 \
  || die "gh is not authenticated. Run 'gh auth login' with an account that has read access to ${REPO}."

# 2. Resolve the release tag ------------------------------------------------
if [[ -z "$TAG" ]]; then
  TAG=$(gh release list --repo "$REPO" --limit 1 --json tagName -q '.[0].tagName') \
    || die "could not list releases on ${REPO} (check your access)."
  [[ -n "$TAG" ]] || die "no releases found on ${REPO}."
fi
echo "==> Release: $TAG"

# 3. Fresh virtualenv -------------------------------------------------------
echo "==> Creating venv at $VENV"
python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null

# 4. Download wheels + checksums -------------------------------------------
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
echo "==> Downloading wheels + checksums"
gh release download "$TAG" --repo "$REPO" \
  --pattern "knowlytix_*.whl" --pattern "SHA256SUMS.txt" \
  --dir "$workdir"

# 5. Verify checksums (shasum on macOS, sha256sum on Linux) -----------------
if [[ -f "$workdir/SHA256SUMS.txt" ]]; then
  echo "==> Verifying checksums"
  ( cd "$workdir" \
    && { shasum -a 256 -c SHA256SUMS.txt 2>/dev/null || sha256sum -c SHA256SUMS.txt; } ) \
    || die "checksum verification failed — re-run to re-download."
fi

# 6. Install via --find-links so pip picks the platform-compatible -------
# knowlytix-core wheel (the release ships one per platform); transitive
# deps (torch/scipy/litellm/...) resolve from PyPI. Installing by name
# (not by globbing the wheel paths) is what lets pip choose the right core.
echo "==> Installing wheels (pip pulls torch/scipy/litellm/... from PyPI; this can take a few minutes)"
pip install --find-links "$workdir" \
  knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness

echo "==> Installed into $VENV"
echo "    Re-activate later with: source $VENV/bin/activate"

# 7. Smoke test -------------------------------------------------------------
if [[ "$RUN_SMOKE" -eq 1 ]]; then
  if [[ -z "${KNOWLYTIX_LICENSE_KEY:-}" ]]; then
    echo "==> Skipping smoke: KNOWLYTIX_LICENSE_KEY not set."
    echo "    Set it, then verify:  export KNOWLYTIX_LICENSE_KEY=gmsh_live_... && knowlytix-smoke"
  else
    echo "==> Running knowlytix-smoke"
    knowlytix-smoke
  fi
fi
