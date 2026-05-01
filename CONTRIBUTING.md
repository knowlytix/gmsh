# Contributing to knowlytix / gmsh

Thanks for wanting to help. A few things worth knowing up front — this
repo is structured slightly differently from a typical OSS project.

## This repo is a release-time mirror

`knowlytix/gmsh` is a read-only public mirror of Knowlytix's internal
development trunk. The mirror is force-pushed once per release (single
commit + matching tag). What this means for you as a contributor:

- **Normal PR review happens here.** Open PRs against `main` in this
  repo — maintainers read, comment, approve.
- **Accepted PRs are cherry-picked into the internal monorepo.** The
  change surfaces in this mirror on the next release, not immediately.
- **PRs are never merged directly into this repo's `main`.** The mirror
  script overwrites `main` on each release; merging here would get
  erased. This is the design.

Practical upshot: you'll see your commit message disappear (replaced by
a single "Release gmsh vX.Y.Z" mirror commit), but your change will be
present in the next release and attribution is preserved via
`Co-authored-by` in the internal monorepo's commit.

## Before you open a PR

### Sign off your commits (DCO)

Every commit needs a `Signed-off-by:` trailer. Configure once:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Then sign each commit:

```bash
git commit --signoff -m "Your commit message"
```

This adds a line like `Signed-off-by: Your Name <you@example.com>` to
the commit message — your certification that you wrote the change (or
have the right to submit it) per the
[Developer Certificate of Origin](https://developercertificate.org/).

### Run tests locally

```bash
pip install -e .[dev]
pytest                          # full suite (some tests require a license — see note below)
pytest -m "not needs_license"  # subset that works in demo mode
```

Running the full suite requires a valid `KNOWLYTIX_LICENSE_KEY` (or
`KNOWLYTIX_DEMO_MODE=1` for the filtered subset — see the package README).
A trial key is fine.

### Write tests for new behavior

Changes that add or modify public API need tests under the relevant
`tests/` subtree. Changes that fix bugs need a regression test that
fails before the fix and passes after.

## What kinds of contributions we welcome

- **Bug fixes**: anywhere in `knowlytix/{doc,bench,harness}/`.
- **New tutorials** under `tutorials/notebooks/` — hands-on exercises
  teach better than docstrings.
- **Docs improvements** to the package READMEs, the tutorials, and the
  per-package API docs.
- **Platform support** — Windows and macOS Intel are deferred from v0.1;
  patches that add clean support are welcome.
- **`good first issue`** tagged items are intentionally scoped for
  first-time contributors.

## What needs discussion first

Before large refactors, architectural changes, or new public-API
additions, open an issue to discuss. Saves everyone a round-trip if the
answer is "we're planning to do this differently in v0.2" or "that
touches the commercial core, which isn't in this repo."

## What belongs elsewhere

- **Changes to `knowlytix-core`**: the core is proprietary and not in
  this repo. Bugs in `knowlytix-core` behavior — [hello@knowlytix.ai](mailto:hello@knowlytix.ai).
- **Security issues**: [`SECURITY.md`](SECURITY.md), not the public
  issue tracker.
- **Licensing / commercial questions**: [hello@knowlytix.ai](mailto:hello@knowlytix.ai).

## Response times

- **Issues**: first response within 3 business days.
- **Pull requests**: first review within 5 business days. More complex
  PRs may need multiple rounds; we'll keep you in the loop.

Thanks again — looking forward to your PR.
