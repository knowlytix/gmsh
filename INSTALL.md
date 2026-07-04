# Knowlytix GMS — early-access install

> **Audience**: collaborators on `knowlytix/gmsh` (early-adopter
> channel). The full v0.2.0 GA release ships to PyPI later via
> `pip install knowlytix-harness`; this guide covers the
> early-access install **from this repo's release assets**.

## Fastest: one command

The [`install-gmsh.sh`](install-gmsh.sh) helper at the repo root runs the
entire flow below for you — checks Python 3.12 + `gh` auth, creates a
venv, resolves the latest release tag, downloads + checksum-verifies the
4 wheels, installs them in dependency order, and runs the smoke test:

```bash
export KNOWLYTIX_LICENSE_KEY=gmsh_live_...   # from §Get a license key
./install-gmsh.sh                            # latest release → ~/.venv/knowlytix
# ./install-gmsh.sh --tag v0.2.0             # or pin a release
# ./install-gmsh.sh --venv ./.venv           # custom venv path
```

Prefer to run the steps yourself? The manual flow follows.

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

# 6. Install by name via --find-links — pip picks the platform-matching
#    knowlytix-core wheel and resolves transitive deps from PyPI
pip install --find-links /tmp/knowlytix-rc \
    knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness

# 7. Set your license key + run smoke
export KNOWLYTIX_LICENSE_KEY=gmsh_live_eyJhbGc...   # from §Get a license key
knowlytix-smoke
```

`knowlytix-smoke` exits 0 with a 5-step PASS banner if the install
is healthy. If it fails, jump to [§Troubleshooting](#troubleshooting).

## Prerequisites

- **Python 3.12** — the commercial `knowlytix-core` wheel is
  Nuitka-compiled and ships per-platform (`manylinux2014_x86_64`,
  `macosx_11_0_arm64`, `win_amd64`) for 3.12 only.
- **A clean virtualenv.** Strongly recommended — installing into a
  system or project venv can mask dependency conflicts.
- **A license key (optional).** With no key, `knowlytix.core` runs in
  keyless **demo** mode (built-in fixtures only) rather than erroring; a
  `developer` or `enterprise` key unlocks your own data. See
  [§Get a license key](#get-a-license-key).
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

### 5. Install the wheels

Install by name with `--find-links` pointed at the download dir. pip
picks the `knowlytix-core` wheel matching your platform (the release
ships one per platform), resolves the dependency order itself, and pulls
transitive deps from PyPI.

```bash
pip install --find-links /tmp/knowlytix-rc \
    knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness
```

Do NOT glob the wheel paths (`knowlytix_core-*.whl`) — that matches every
platform's core wheel and pip errors on the incompatible one. Do NOT use
`--no-index` either: that blocks pip from reaching PyPI for the deps.

**Faster with [uv](https://docs.astral.sh/uv/).** This dep tree is
large; `uv` resolves and downloads it much faster than pip and creates
the venv for you. Substitute steps 4–5 with:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --find-links /tmp/knowlytix-rc \
    knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness
```

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

# 3. Install knowlytix-core from the wheel (--find-links picks your
#    platform's wheel); install the rest from the local source tree.
pip install --find-links /tmp/knowlytix-rc knowlytix-core
pip install -e ".[dev]"   # quote `[dev]` so zsh doesn't glob-expand
```

The 3 Apache-2.0 packages (`knowlytix-knowledge`, `-benchmark`,
`-harness`) resolve from the editable clone — edits to any of
`knowlytix/{knowledge,benchmark,harness}/` take effect immediately.

## Get a license key

Pick the path that matches your use:

- **Developer** — free, 180-day, self-serve, non-commercial key at
  [knowlytix.ai/signup](https://knowlytix.ai/signup). Unlocks your own
  data (formerly the trial/classroom tiers, now one self-serve tier).
- **Enterprise** — commercial deployment licensing at
  [hello@knowlytix.ai](mailto:hello@knowlytix.ai).
- **No license, just curious?** — you don't need to do anything: with no
  key, `knowlytix.core` runs in keyless **demo** mode by default.
  Read-only mode loads bundled fixtures, mock agents work, LLM calls
  return deterministic placeholders. No key, no network. (You can also
  force it explicitly with `KNOWLYTIX_DEMO_MODE=1`.)

  **Demo mode is for kicking the tires only — not for POC or
  evaluation work.** Caps are deliberately tiny (50 entities,
  10 doc pages, 2 benchmark runs, **0 training epochs**) and
  any write, ingestion, training, or real LLM call raises
  `DemoModeError`. For a real POC, get a free developer key
  via the link above.

Keys arrive over email as a `gmsh_live_…` JWT string. Treat them
like passwords: don't commit, don't share.

## Run the tutorial notebooks

> **Tip — silence the license banner in notebooks.** On import,
> `knowlytix.core` prints a license banner (and an EULA reminder) to
> stderr, which Jupyter renders as a red output cell. Set
> `KNOWLYTIX_QUIET=1` to suppress all of that startup output. Put it in a
> cell above the import (`%env KNOWLYTIX_QUIET=1`) or in the kernel
> environment. It only skips the printing — license verification still
> runs, so an expired or tampered key still raises `LicenseError` (a
> *missing* key drops to demo mode, it doesn't error).

Two tracks live in two places — there is no top-level `tutorials/`
directory in this repo.

**Governance USER_GUIDE companions** ship inside the
`knowlytix-harness` wheel and are visible in your clone (Path B) at
`knowlytix/harness/governance/tutorials/notebooks/` — 27 short
chapters accompanying `knowlytix.harness.governance`'s
`USER_GUIDE.md`. Run them either against your clone or against the
installed wheel (Path A or B):

```bash
# From your gmsh clone, with venv active:
python -m ipykernel install --user --name knowlytix \
    --display-name "Knowlytix"

# Open the in-package notebooks JupyterLab:
jupyter lab knowlytix/harness/governance/tutorials/notebooks/
```

Fixtures (`ds_practices_store`, `california_housing_store`) build
on first use:

```bash
python -m knowlytix.harness.governance.tutorials.notebooks.build_fixture
```

(~10 min on GPU, then cached at `~/.knowlytix/governance_fixtures/`.)

**Agentic-testing book companions** for *How to Test and Validate
Agentic AI Systems* (~22 chapters + appendix F, with pre-built lab
stores) live in the licensed-reader repo
[`knowlytix/gmsh-books`](https://github.com/knowlytix/gmsh-books)
under `books/agentic_testing/`. Developer-tier license holders receive
access; see the gms-books README for setup.

## Troubleshooting

### The banner says `tier=demo` but I have a key

With no key present, `knowlytix.core` runs in keyless demo mode by
design — a missing key does **not** error. If you set a key but still
land in demo, your `KNOWLYTIX_LICENSE_KEY` is unset/empty in this shell
(or `~/.knowlytix/license.key` is missing). Re-run `export
KNOWLYTIX_LICENSE_KEY=gmsh_live_…` and re-import. If the key IS set but
import raises `LicenseError`, it's expired or malformed — check the
key's expiry date or request a fresh one via the channels above.

### `LicenseError: license signature is invalid`

The key string is corrupted (truncated paste, wrong key type, etc.).
Get a fresh key from the channels above.

### `DemoModeError: <feature> is disabled in demo mode`

You're running with `KNOWLYTIX_DEMO_MODE=1`, which blocks
ingestion, training, writes, and outbound LLM calls. Demo mode
only loads bundled fixtures. For real work, get a developer key
(see [§Get a license key](#get-a-license-key)) and:

```bash
unset KNOWLYTIX_DEMO_MODE
export KNOWLYTIX_LICENSE_KEY=gmsh_live_…
```

### `pip install` says "No matching distribution found for knowlytix-core"

Almost always a **Python-version mismatch**: `knowlytix-core` ships as a
compiled wheel for **CPython 3.12 only** (`cp312`), so an interpreter on
3.11 or 3.13 finds no matching wheel. Confirm which Python pip is using:

```bash
python --version    # must be 3.12.x
pip --version       # shows the interpreter pip is bound to
```

If it isn't 3.12, create a 3.12 venv (see step 1) and install there. Much
less commonly, a very old pip can't read the `manylinux2014` tag — fix
with `pip install --upgrade pip`.

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
issues, multi-platform install matrix), file an issue at
<https://github.com/knowlytix/gmsh/issues> with your environment
details and we'll route to the right help.

## Feedback

Early-access bugs / install-friction reports / "this doc is
wrong" notes — file them at
<https://github.com/knowlytix/gmsh/issues>. The whole point of
the early-access channel is to surface install-path bugs before
the public PyPI release; please do report.
