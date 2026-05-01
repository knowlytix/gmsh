<!--
Thanks for contributing to knowlytix / gmsh.

Reminder: this repo is a release-time mirror. Accepted PRs are
cherry-picked into the internal monorepo and surface here on the next
release — they are not merged directly. See CONTRIBUTING.md for the
full flow.
-->

## Summary

<!-- One or two sentences: what does this PR change and why? -->

## Related issue

<!-- Link the issue this addresses. "Fixes #123" / "Part of #123" / "N/A". -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing API altered or removed)
- [ ] Docs / tutorials only
- [ ] CI / tooling / build

## Which package(s)

- [ ] `knowlytix-doc`
- [ ] `knowlytix-bench`
- [ ] `knowlytix-harness`
- [ ] `tutorials/`
- [ ] Repo-level docs or CI

## Checklist

- [ ] **DCO sign-off** present on every commit (`git commit --signoff`) — see CONTRIBUTING.md
- [ ] Tests added or updated (regression test for bug fixes; coverage for new behavior)
- [ ] `pytest` passes locally
- [ ] `ruff check` passes locally
- [ ] Public-API changes updated in the relevant package README / docstrings
- [ ] No changes to `knowlytix-core` internals (those live in the private monorepo)

## Cherry-pick cadence

<!--
Maintainers cherry-pick accepted PRs into the internal monorepo. How
time-sensitive is this change?
-->

- [ ] **Immediate** — security fix, data-loss bug, or blocks another in-flight PR
- [ ] **Next release** — normal cadence, bundles with the next tagged release
- [ ] **Maintainer's call** — no strong preference

## Notes for reviewers

<!-- Anything subtle worth flagging: design trade-offs, follow-ups, migration notes. -->
