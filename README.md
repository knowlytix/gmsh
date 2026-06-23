# knowlytix / gmsh

[![CI](https://github.com/knowlytix/gmsh/actions/workflows/ci.yml/badge.svg)](https://github.com/knowlytix/gmsh/actions/workflows/ci.yml)
[![PyPI — knowlytix-harness](https://pypi.org/project/knowlytix-harness/badge)](https://pypi.org/project/knowlytix-harness/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Agentic-AI testing platform for enterprise model risk.** Source for the
three Apache-2.0 packages in the Knowlytix GMSH stack:
`knowlytix-knowledge`, `knowlytix-benchmark`, `knowlytix-harness`.

## Install

> **Use a Python 3.12 virtualenv.** The stack pulls a large dependency
> tree (torch, scipy, scikit-learn, litellm, …) and `knowlytix-core` is
> a wheel compiled for Python 3.12 — installing into system Python will
> fail with `externally-managed-environment` (PEP 668) or shadow other
> projects' packages. Every snippet below creates one first. Full
> walkthrough + troubleshooting: [`INSTALL.md`](INSTALL.md).

```bash
python3.12 -m venv ~/.venv/knowlytix && source ~/.venv/knowlytix/bin/activate
pip install --upgrade pip
pip install knowlytix-harness     # full stack: pulls knowlytix-core + -knowledge + -benchmark
knowlytix-smoke                   # post-install sanity check (5 pass/fail steps)
jupyter lab $(knowlytix-smoke --notebook-path)   # open the shipped exploration notebook
```

Three ways to install, depending on scope:

```bash
pip install knowlytix            # EVERYTHING — the umbrella meta-package (see note)
pip install knowlytix-harness    # the GMSH testing stack: harness + core + knowledge + benchmark
pip install knowlytix-knowledge  # a single package (knowlytix-core comes along automatically)
```

- **`pip install knowlytix`** is the meta-package: it pulls all **five**
  distributions — the four above plus **`knowlytix-kal`**, the optional
  Postgres/pgvector knowledge store released separately from
  [`knowlytix/KAL`](https://github.com/knowlytix/KAL). Use this if you want
  the whole platform in one command (and you don't mind pulling KAL).
- **`pip install knowlytix-harness`** is the umbrella for just the GMSH
  testing stack — it depends on `knowlytix-core`, `-knowledge`, and
  `-benchmark`, so the one command gives you those four, **without** KAL.
- Individual packages (`knowlytix-knowledge`, `-benchmark`, `-core`) each
  pull `knowlytix-core` automatically — install just the piece you need.

### Install from GitHub releases (alternative channel)

The wheels are also published to this repo's
[Releases](https://github.com/knowlytix/gmsh/releases) page — a standing channel alongside
PyPI, not just a pre-PyPI stopgap. This channel *leads the timeline*: it
carries the release candidates (`-rcN`) before a version is promoted to
PyPI, and every final release lands here too. Both channels ship the
identical lockstep wheels (`knowlytix-core`, `knowlytix-knowledge`,
`knowlytix-benchmark`, `knowlytix-harness`) — use PyPI for the
conservative path, GitHub releases to track the newest builds.

**One command** — the [`install-gmsh.sh`](install-gmsh.sh) helper at the
repo root does the whole dance (checks Python 3.12 + `gh` auth, creates a
venv, resolves the latest tag, downloads + checksum-verifies the wheels,
installs in dependency order, runs the smoke test):

```bash
./install-gmsh.sh                  # latest release into ~/.venv/knowlytix
./install-gmsh.sh --tag v0.2.0     # or pin a specific release
```

**Manual** — if you'd rather run the steps yourself:

```bash
python3.12 -m venv ~/.venv/knowlytix && source ~/.venv/knowlytix/bin/activate
pip install --upgrade pip

# Replace rcN with the latest tag — check https://github.com/knowlytix/gmsh/releases first,
# or resolve it automatically:
tag=$(gh release list --repo knowlytix/gmsh --limit 1 --json tagName -q '.[0].tagName')

gh release download "$tag" --repo knowlytix/gmsh --pattern "knowlytix_*.whl"
# Install by name with --find-links so pip picks the core wheel matching
# your platform (the release ships one knowlytix-core wheel per platform:
# Linux x86_64, macOS arm64, Windows x86_64).
pip install --find-links . \
    knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness
```

The literal form is `gh release download v0.2.0 --repo knowlytix/gmsh
--pattern "knowlytix_*.whl"` — substitute the current tag (a final like
`v0.2.0`, or an `-rcN` release candidate if you're tracking the newest
builds); always confirm the latest on the Releases page. Requires `gh`
authenticated with read access to this repo. Then run `knowlytix-smoke`
as above.

**Faster with [uv](https://docs.astral.sh/uv/)** — `uv` resolves and
downloads this heavy dep tree far quicker than pip and manages the venv
for you:

```bash
uv venv --python 3.12 && source .venv/bin/activate
gh release download "$tag" --repo knowlytix/gmsh --pattern "knowlytix_*.whl"
uv pip install --find-links . \
    knowlytix-core knowlytix-knowledge knowlytix-benchmark knowlytix-harness
```

## Tutorials

Two tracks live in different places — neither is a top-level
`tutorials/` directory in this repo.

- **Governance USER_GUIDE companions** ship inside the
  `knowlytix-harness` wheel and are visible in your clone at
  [`knowlytix/harness/governance/tutorials/notebooks/`](knowlytix/harness/governance/tutorials/notebooks/).
  27 short notebooks accompanying `knowlytix.harness.governance`'s
  `USER_GUIDE.md`. After `pip install knowlytix-harness`, fixtures
  build on first use:
  `python -m knowlytix.harness.governance.tutorials.notebooks.build_fixture`
  (~10 min on GPU, then cached at `~/.knowlytix/governance_fixtures/`).
- **Agentic-testing book companions** (chapters of *How to Test and
  Validate Agentic AI Systems*) live in the licensed-reader repo
  [`knowlytix/gmsh-books`](https://github.com/knowlytix/gmsh-books)
  under `books/agentic_testing/`. Pre-built lab stores included.

## What's commercial?

`knowlytix-harness`, `knowlytix-knowledge`, and `knowlytix-benchmark` are Apache-2.0
and covered by the [`LICENSE`](LICENSE) at the repo root. **They depend on
`knowlytix-core`, which is proprietary** — distributed as an
obfuscated wheel under a commercial EULA. The full EULA text ships
in this repo at
[`packaging/knowlytix-core/LICENSE.txt`](packaging/knowlytix-core/LICENSE.txt)
(plain text) and as
[`packaging/knowlytix-core/EULA.md`](packaging/knowlytix-core/EULA.md)
(markdown for easier reading on github.com). Both are also embedded
in the wheel's `dist-info/licenses/` per PEP 639. **Please read the
EULA before installing the commercial wheel** — installation
constitutes acceptance of those terms.

- **Developer**: free, 180-day, self-serve, non-commercial keys at
  [knowlytix.ai/signup](https://knowlytix.ai/signup) — unlocks your own data.
- **Enterprise**: commercial deployment licensing at
  [hello@knowlytix.ai](mailto:hello@knowlytix.ai).
- **No license, just curious?** Do nothing — with no key, `knowlytix.core`
  runs in keyless **demo** mode by default: bundled fixtures only, mock
  agents, deterministic LLM placeholders, no network. (Force it explicitly
  with `KNOWLYTIX_DEMO_MODE=1`.) Demo is for kicking the tires, not a POC.

## The three Apache-2.0 packages

| Package | What it does | Depends on |
|---|---|---|
| [`knowlytix-knowledge`](packaging/knowlytix-knowledge/) | Document ingestion: PDF → markdown → graph → GMS expert store | `knowlytix-core` |
| [`knowlytix-benchmark`](packaging/knowlytix-benchmark/) | Benchmark question generation, scoring rubrics, FinStructBench suite | `knowlytix-core` |
| [`knowlytix-harness`](packaging/knowlytix-harness/) | DOE-driven agent testing: scenario design, tool-gateway, policy engine, hallucination oracles | `knowlytix-core`, `knowlytix-knowledge`, `knowlytix-benchmark` |

Versions ship lockstep — every release bumps all four packages (the three above plus `knowlytix-core`) together.

## License-key setup

The harness and its dependencies verify a signed JWT license key at
`import knowlytix.core` time. Two accepted sources, checked in order:

1. `KNOWLYTIX_LICENSE_KEY` environment variable (CI / containers)
2. `~/.knowlytix/license.key` file, single-line JWT (workstations)

Minimum install + verify:

```bash
python3.12 -m venv ~/knowlytix-env
source ~/knowlytix-env/bin/activate
pip install knowlytix-harness
export KNOWLYTIX_LICENSE_KEY="<your JWT>"
knowlytix-smoke        # 5-step post-install sanity check
```

On a successful import, `knowlytix.core` prints a one-line license
banner to stderr (plus an EULA reminder until you accept it). To silence
**all** of that startup output — banner, EULA reminder, demo disclaimer —
set `KNOWLYTIX_QUIET=1`. This is most useful in Jupyter notebooks, where
stderr shows up as a red output cell:

```bash
export KNOWLYTIX_QUIET=1          # before importing knowlytix.core
```

Quiet mode only skips the printing — license verification still runs, so
a missing or expired key still raises `LicenseError`. In a notebook, set
it in a cell above the import (`%env KNOWLYTIX_QUIET=1`) or in the kernel
environment.

Full onboarding walkthrough (troubleshooting, air-gap notes, CI
integration examples): [knowlytix.ai/docs/onboarding](https://knowlytix.ai/docs/onboarding).

## Contributing

External contributions welcome. This repo is a release-time mirror of
an internal development trunk, so the contribution model is slightly
different from a normal OSS repo — maintainers cherry-pick accepted PRs
into the source repository, and changes surface here on the next
release. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.

- **Issues**: [https://github.com/knowlytix/gmsh/issues](https://github.com/knowlytix/gmsh/issues) — bug reports, feature requests, tutorials gaps
- **Security disclosures**: [`SECURITY.md`](SECURITY.md) — GitHub Private Vulnerability Reporting or
  [security@knowlytix.ai](mailto:security@knowlytix.ai)
- **Community behavior**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1

## Links

- **PyPI**: [https://pypi.org/project/knowlytix-harness/](https://pypi.org/project/knowlytix-harness/)
- **Product & pricing**: [knowlytix.ai](https://knowlytix.ai)
- **Documentation**: [knowlytix.ai/docs](https://knowlytix.ai/docs)
- **Tutorials**: see the §Tutorials section above (governance notebooks ship inside the `knowlytix-harness` wheel; the agentic-testing book lives in [`knowlytix/gmsh-books`](https://github.com/knowlytix/gmsh-books))

## Supported platforms

Python 3.12, Linux x86_64, macOS arm64, and Windows x86_64. macOS Intel
is deferred to a later release. See the package READMEs under
[`packaging/`](packaging/) for per-package platform notes.
