---
name: Bug report
about: Report unexpected behavior in knowlytix-doc, knowlytix-bench, or knowlytix-harness
title: "[bug] "
labels: ["bug", "needs-triage"]
---

<!--
Thanks for taking the time to file a bug. Please fill in every section
below — maintainers triage faster when the reproducer is self-contained.

Security issues: do NOT file here. See SECURITY.md for private reporting.
knowlytix-core (commercial) issues: see the "Scope" section of SECURITY.md.
-->

## What happened

<!-- One or two sentences. What did you do, what did you expect, what did you get? -->

## Reproducer

<!-- Minimum code + commands that trigger the bug. Paste the full traceback. -->

```python
# paste code here
```

```
# paste traceback / stderr here
```

## Environment

- **Python version**: <!-- `python --version` -->
- **OS**: <!-- e.g. Ubuntu 24.04, macOS 15.2 arm64 -->
- **Install method**: <!-- `pip install knowlytix-harness` / editable / from source / ... -->
- **Package versions**: <!-- paste output of `pip show knowlytix-core knowlytix-doc knowlytix-bench knowlytix-harness | grep -E '^(Name|Version)'` -->

## `knowlytix-smoke` output

<!--
Run the bundled post-install check and paste the output. If `knowlytix-smoke`
itself fails, paste what you got before it crashed.
-->

```
$ knowlytix-smoke
# paste output here
```

## Anything else

<!-- Workarounds you've tried, links to related issues, screenshots, etc. -->
