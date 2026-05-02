# knowlytix / gmsh

[![CI](https://github.com/knowlytix/gmsh/actions/workflows/ci.yml/badge.svg)](https://github.com/knowlytix/gmsh/actions/workflows/ci.yml)
[![PyPI — knowlytix-harness](https://pypi.org/project/knowlytix-harness/badge)](https://pypi.org/project/knowlytix-harness/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Agentic-AI testing platform for enterprise model risk.** Source for the
three Apache-2.0 packages in the Knowlytix GMSH stack:
`knowlytix-knowledge`, `knowlytix-benchmark`, `knowlytix-harness`.

## Install

```bash
pip install knowlytix-harness     # pulls knowlytix-core + -doc + -bench as deps
knowlytix-smoke                   # post-install sanity check (5 pass/fail steps)
jupyter lab $(knowlytix-smoke --notebook-path)   # open the shipped exploration notebook
```

See [`tutorials/`](tutorials/) for chapter-aligned exercises.

## What's commercial?

`knowlytix-harness`, `knowlytix-knowledge`, and `knowlytix-benchmark` are Apache-2.0
and covered by the [`LICENSE`](LICENSE) at the repo root. **They depend on
`knowlytix-core`, which is proprietary** — distributed as an
obfuscated wheel via PyPI under a commercial EULA, not in this
repository.

- **Free trial**: 30-day trial keys at [knowlytix.ai/signup](https://knowlytix.ai/signup).
- **Classroom / research**: accredited educational institutions qualify for
  no-cost multi-seat keys — see [knowlytix.ai/edu](https://knowlytix.ai/edu).
- **Enterprise**: commercial deployment licensing at
  [hello@knowlytix.ai](mailto:hello@knowlytix.ai).
- **No license, just curious?** Set `KNOWLYTIX_DEMO_MODE=1` before
  `import knowlytix.core` — read-only mode loads bundled fixtures, mock
  agents work, LLM calls return deterministic placeholders. No key, no
  network. See the package README for details.

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

Full onboarding walkthrough (troubleshooting, air-gap notes, CI
integration examples): [knowlytix.ai/docs/onboarding](https://knowlytix.ai/docs/onboarding).

## Contributing

External contributions welcome. This repo is a release-time mirror of
an internal development trunk, so the contribution model is slightly
different from a normal OSS repo — maintainers cherry-pick accepted PRs
into the internal monorepo, and changes surface here on the next
release. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.

- **Issues**: [https://github.com/knowlytix/gmsh/issues](https://github.com/knowlytix/gmsh/issues) — bug reports, feature requests, tutorials gaps
- **Security disclosures**: [`SECURITY.md`](SECURITY.md) — GitHub Private Vulnerability Reporting or
  [security@knowlytix.ai](mailto:security@knowlytix.ai)
- **Community behavior**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1

## Links

- **PyPI**: [https://pypi.org/project/knowlytix-harness/](https://pypi.org/project/knowlytix-harness/)
- **Product & pricing**: [knowlytix.ai](https://knowlytix.ai)
- **Documentation**: [knowlytix.ai/docs](https://knowlytix.ai/docs)
- **Tutorials**: [`tutorials/`](tutorials/) in this repo

## Supported platforms

Python 3.12, Linux x86_64 and macOS arm64. macOS Intel and Windows are
deferred to a later release. See the package READMEs under
[`packaging/`](packaging/) for per-package platform notes.
