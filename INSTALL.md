# Knowlytix GMS — early-access install

> **Audience**: collaborators on `knowlytix/gmsh` (early-adopter
> channel). The full v0.2.0 GA release ships to PyPI later via
> `pip install knowlytix-harness`; this guide covers the
> early-access install **from this repo's release assets**.

## TL;DR (≈ 2 min)

```bash
# 1. Read the EULA. Installation = acceptance.
# 2. Get a license key (see §Get a license key below)
# 3. Create a fresh venv
python3.12 -m venv ~/.venv/knowlytix
source ~/.venv/knowlytix/bin/activate
pip install --upgrade pip

# 4. Fetch the 4 wheels from the latest release
gh release download --repo knowlytix/gmsh \
    --pattern "knowlytix_*.whl" \
    --pattern "SHA256SUMS.txt" \
    --dir /tmp/knowlytix-rc

# 5. Verify checksums (optional but recommended)
( cd /tmp/knowlytix-rc && shasum -a 256 -c SHA256SUMS.txt )   # macOS
( cd /tmp/knowlytix-rc && sha256sum -c SHA256SUMS.txt )       # Linux

# 6. Install — pip resolves transitive deps from PyPI
pip install \
    /tmp/knowlytix-rc/knowlytix_core-*.whl \
    /tmp/knowlytix-rc/knowlytix_knowledge-*.whl \
    /tmp/knowlytix-rc/knowlytix_benchmark-*.whl \
    /tmp/knowlytix-rc/knowlytix_harness-*.whl

# 7. Set your license key + run smoke
export KNOWLYTIX_LICENSE_KEY=gmsh_live_eyJhbGc...   # from §Get a license key
knowlytix-smoke
```

`knowlytix-smoke` exits 0 with a 5-step PASS banner if the install
is healthy. If it fails, jump to [§Troubleshooting](#troubleshooting).

## Prerequisites

- **Python 3.12** — the commercial `knowlytix-core` wheel is
  Nuitka-compiled and ships per-platform (`macosx_11_0_arm64`,
  `macosx_11_0_x86_64`, `manylinux2014_x86_64`) for 3.12 only.
- **A clean virtualenv.** Strongly recommended — installing into a
  system or project venv can mask dependency conflicts.
- **A license key.** `knowlytix.core` raises `LicenseError`
  immediately on import without a valid `KNOWLYTIX_LICENSE_KEY`.
  See [§Get a license key](#get-a-license-key).
- **Read the EULA.** Installing the commercial wheel constitutes
  acceptance of the [Knowlytix EULA](packaging/knowlytix-core/LICENSE.txt)
  ([markdown version](packaging/knowlytix-core/EULA.md)).

## Path A: install from release wheels (recommended for most users)

This is the TL;DR flow above, expanded.

### 1. Find the latest release

The early-access channel publishes RC releases (`v0.2.0-rcN`) ahead
of the public PyPI release. Browse:
<https://github.com/knowlytix/gmsh/releases> (you must be signed in
as a repo collaborator; incognito → 404).

Pick the newest entry — the "Pre-release" badge means it's an RC,
the absence of that badge means it's a stable cut.

### 2. Fetch the assets

```bash
# Latest release (omit --tag to take whatever the page shows newest)
gh release download --repo knowlytix/gmsh \
    --pattern "knowlytix_*.whl" \
    --pattern "SHA256SUMS.txt" \
    --dir /tmp/knowlytix-rc
```

Or pin to a specific tag:

```bash
gh release download v0.2.0-rcN --repo knowlytix/gmsh \
    --pattern "knowlytix_*.whl" \
    --pattern "SHA256SUMS.txt" \
    --dir /tmp/knowlytix-rc
```

### 3. Verify checksums

```bash
cd /tmp/knowlytix-rc
shasum -a 256 -c SHA256SUMS.txt    # macOS
# or
sha256sum -c SHA256SUMS.txt         # Linux
```

Every line should print `OK`. If any line prints `FAILED`, your
download is corrupt; re-fetch.

### 4. Create the venv

```bash
python3.12 -m venv ~/.venv/knowlytix
source ~/.venv/knowlytix/bin/activate
pip install --upgrade pip
```

If `python3.12` isn't on your PATH, install it via your platform's
package manager (`brew install python@3.12` on macOS;
[python.org/downloads](https://www.python.org/downloads/) elsewhere).

### 5. Install the 4 wheels

Order matters — `knowlytix-core` is a dependency of the other three,
so it must install first.

```bash
pip install \
    /tmp/knowlytix-rc/knowlytix_core-*.whl \
    /tmp/knowlytix-rc/knowlytix_knowledge-*.whl \
    /tmp/knowlytix-rc/knowlytix_benchmark-*.whl \
    /tmp/knowlytix-rc/knowlytix_harness-*.whl
```

pip resolves transitive deps (litellm, torch, pydantic, numpy, …)
from PyPI normally. Do NOT use `--no-index`: that blocks pip from
reaching PyPI for those deps.

### 6. Set your license key

```bash
export KNOWLYTIX_LICENSE_KEY=gmsh_live_eyJhbGc...
```

Persist it to your shell rc file (`~/.zshrc`, `~/.bashrc`) so it
survives shell restarts. **Never commit the key** to a repo —
`.gitignore` it if you put it in a `.env` file.

### 7. Run the post-install smoke

```bash
knowlytix-smoke
```

Expected output:

```
[1/5] imports + __all__ resolved for 4 packages
[2/5] 5 Settings classes instantiate with defaults
[3/5] 5 fixtures reachable via importlib.resources
[4/5] knowlytix.benchmark.score_answer() ran on 5 smoke questions
[5/5] harness DOE smoke fixtures parse and match expected schema
Knowlytix GMS wheels — post-install smoke test PASSED
```

Then explore the bundled exploration notebook:

```bash
jupyter lab "$(knowlytix-smoke --notebook-path)"
```

## Path B: editable install from this clone (for developers)

If you've cloned the gmsh repo locally and want to develop against
the source tree (edit, re-import, no reinstall), use the editable
flow.

```bash
# 1. Fetch knowlytix-core wheel (still needed — it's the
#    commercial obfuscated binary; the source isn't in this repo)
gh release download --repo knowlytix/gmsh \
    --pattern "knowlytix_core-*.whl" \
    --dir /tmp/knowlytix-rc

# 2. From your gmsh clone:
cd path/to/gmsh-clone
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install knowlytix-core from the wheel; install the rest from
#    the local source tree (editable mode).
pip install /tmp/knowlytix-rc/knowlytix_core-*.whl
pip install -e ".[dev]"   # quote `[dev]` so zsh doesn't glob-expand
```

The 3 Apache-2.0 packages (`knowlytix-knowledge`, `-benchmark`,
`-harness`) resolve from the editable clone — edits to any of
`knowlytix/{knowledge,benchmark,harness}/` take effect immediately.

For the deeper internal-team developer flow (against the private
GMS monorepo + local mirror clone), see
[`docs/internal_team_install.md`](https://github.com/knowlytix/GMS/blob/main/docs/internal_team_install.md)
in the source repo (collaborator access required).

## Get a license key

Pick the path that matches your use:

- **Free trial** — 30-day key at
  [knowlytix.ai/signup](https://knowlytix.ai/signup).
- **Classroom / research** — accredited institutions get no-cost
  multi-seat keys at
  [knowlytix.ai/edu](https://knowlytix.ai/edu).
- **Enterprise** — commercial deployment licensing at
  [hello@knowlytix.ai](mailto:hello@knowlytix.ai).
- **No license, just curious?** — set `KNOWLYTIX_DEMO_MODE=1`
  before `import knowlytix.core`. Read-only mode loads bundled
  fixtures, mock agents work, LLM calls return deterministic
  placeholders. No key, no network.

  **Demo mode is for kicking the tires only — not for POC or
  evaluation work.** Caps are deliberately tiny (50 entities,
  10 doc pages, 2 benchmark runs, **0 training epochs**) and
  any write, ingestion, training, or real LLM call raises
  `DemoModeError`. For a real POC, get a free 30-day trial key
  via the link above and unset `KNOWLYTIX_DEMO_MODE`.

Keys arrive over email as a `gmsh_live_…` JWT string. Treat them
like passwords: don't commit, don't share.

## Run the tutorial notebooks

Source-tree-clone variant (Path B) gives you the tutorials
directly:

```bash
# From your gmsh clone, with venv active:
python -m ipykernel install --user --name knowlytix \
    --display-name "Knowlytix"

# Open the tutorials JupyterLab:
jupyter lab tutorials/
```

The `tutorials/` directory contains:

- `tutorials/agentic_testing_book/notebooks/` — agentic testing book
  companions (~18 chapters + appF). Pre-built lab stores at
  `tutorials/agentic_testing_book/lab_stores/`.
- `tutorials/user_guide/notebooks/` — governance user-guide
  companions (~25 chapters)

Each track's `_shared.py` + `_helpers_phase*.py` ship next to the
notebooks so imports resolve without path setup.

If you're on Path A (no clone), grab the tutorials from a clone:

```bash
git clone git@github.com:knowlytix/gmsh.git
cd gmsh/tutorials
```

## Troubleshooting

### `LicenseError: KNOWLYTIX_LICENSE_KEY env var is not set`

Your venv doesn't have the key. Re-run `export
KNOWLYTIX_LICENSE_KEY=gmsh_live_…` and try again. If the key WAS
set but you still see this, the key may have expired or be
malformed — check the trial-key expiry date or request a fresh
one via the channels above.

### `LicenseError: license signature is invalid`

The key string is corrupted (truncated paste, wrong key type, etc.).
Get a fresh key from the channels above.

### `DemoModeError: <feature> is disabled in demo mode`

You're running with `KNOWLYTIX_DEMO_MODE=1`, which blocks
ingestion, training, writes, and outbound LLM calls. Demo mode
only loads bundled fixtures. For real work, get a trial key
(see [§Get a license key](#get-a-license-key)) and:

```bash
unset KNOWLYTIX_DEMO_MODE
export KNOWLYTIX_LICENSE_KEY=gmsh_live_…
```

### `pip install` says "No matching distribution found for knowlytix-core"

You probably ran `pip install knowlytix-harness` directly without
a wheel path. The commercial `knowlytix-core` wheel only lives on
this repo's release assets (and PyPI for stable releases) — pip
needs an explicit path or a pre-installed core. See
[Path A step 5](#5-install-the-4-wheels).

### `knowlytix-smoke` fails at step 1 (imports)

Likely a stale Python version. Confirm:

```bash
python3 -c "import sys; print(sys.version_info[:2])"
```

Must print `(3, 12)`. If not, your venv was created with the
wrong Python; recreate with `python3.12 -m venv …`.

### Wrong-platform wheel

If pip resolves a wheel that doesn't match your platform (e.g.,
macOS arm64 wheel on Linux x86_64), the Nuitka-compiled
`knowlytix-core` import will crash at load time. Confirm the
matching wheel filename pattern:

- macOS Apple Silicon: `…macosx_11_0_arm64.whl`
- macOS Intel: `…macosx_11_0_x86_64.whl`
- Linux x86_64: `…manylinux2014_x86_64.whl`

`gh release download --pattern "knowlytix_core-*.whl"` fetches all
platform variants — let pip pick the right one (it does this
automatically based on `sys.platform`).

### `SHA256SUMS` mismatch

Your download is corrupt. Re-fetch via `gh release download` (it
re-downloads from scratch by default). If the mismatch persists
across multiple fetches, file an issue at
<https://github.com/knowlytix/gmsh/issues> with your `gh --version`
output.

### Other issues

For deeper troubleshooting (Nuitka loader paths, license-clamp
issues, multi-platform install matrix), see the internal-team
install doc at
[`docs/internal_team_install.md`](https://github.com/knowlytix/GMS/blob/main/docs/internal_team_install.md)
in the source repo (collaborator access required).

## Feedback

Early-access bugs / install-friction reports / "this doc is
wrong" notes — file them at
<https://github.com/knowlytix/gmsh/issues>. The whole point of
the early-access channel is to surface install-path bugs before
the public PyPI release; please do report.
