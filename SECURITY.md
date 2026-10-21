# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security concerns.** Use one of
the private channels below so we can investigate and ship a fix before
details are public.

### Preferred: GitHub Private Vulnerability Reporting

Open a private report via GitHub's PVR flow:

1. Navigate to [https://github.com/knowlytix/gmsh/security/advisories/new](https://github.com/knowlytix/gmsh/security/advisories/new)
2. Fill out the form describing the vulnerability, reproduction steps,
   and (if known) the affected versions
3. Submit — only Knowlytix maintainers see the report

### Alternative: email

If GitHub PVR isn't available to you, email
[security@knowlytix.ai](mailto:security@knowlytix.ai) with:

- A clear description of the issue
- Reproduction steps or a proof-of-concept
- The version(s) you tested against (`pip show knowlytix-harness`)
- Your disclosure timeline preference

A PGP key for confidential exchanges is available at
[knowlytix.ai/security](https://knowlytix.ai/security).

## What to expect

- **Acknowledgement**: within 2 business days.
- **Initial assessment**: within 5 business days — we'll tell you
  whether we can reproduce, what the severity looks like, and an ETA
  for a fix.
- **Coordinated disclosure**: we target a 90-day window from first
  report to public advisory. Shorter if the issue is actively exploited;
  longer with your agreement for particularly complex fixes. We'll
  credit you in the advisory unless you prefer anonymity.

## Scope

In scope:

- **`knowlytix-knowledge`**, **`knowlytix-benchmark`**, **`knowlytix-harness`** —
  the three Apache-2.0 packages whose source is in this repo.
- **`knowlytix-core`** (commercial, not in this repo): license-gate
  bypass, signature verification failures, capability-claim
  bypass. Report to the same channels — we triage across the whole stack.

Out of scope:

- **Third-party dependencies**: file those upstream with the relevant
  maintainer. Pull requests that upgrade our pins after the upstream fix
  ships are welcome.
- **Social engineering / physical attacks / DoS against knowlytix.ai**:
  report via [hello@knowlytix.ai](mailto:hello@knowlytix.ai) instead.

## Supported versions

Only the latest published minor version receives security patches.
Older minor versions can be upgraded to by pinning the patched minor;
we do not backport.

| Version | Supported |
|---|---|
| Latest minor (`2.1.0`) | ✅ Security patches |
| Older minors | ❌ Please upgrade |

## Hall of fame

Security researchers who've helped improve Knowlytix GMSH are listed at
[knowlytix.ai/security#acknowledgements](https://knowlytix.ai/security#acknowledgements).
