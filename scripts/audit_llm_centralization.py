#!/usr/bin/env python3
"""Config centralization audit (GMS-024).

Enforces that no shipped source code hardcodes a provider SDK or model
name. The raw grep pattern from the story —

    rg -n 'anthropic\\.Anthropic\\(|claude-[a-z0-9-]+|gpt-[0-9]'
       knowlytix/core knowlytix/doc finstructbench harness

— matches legitimate occurrences too (docstring examples, JSON test
fixtures, error-message hints, CLI --help strings). This script applies
the same regex but strips out those known-safe contexts so the
*actionable* hit count reaches zero.

Exit codes:
  0 — no violations beyond the explicit allowlist.
  1 — one or more new violations; they are printed with file:line and
      the matched text so they can be fixed or explicitly allowlisted.

Run locally:
  python scripts/audit_llm_centralization.py

Wired into CI in ``.github/workflows/ci.yml`` as the ``llm-audit`` job.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shipped source roots per distribution/implementation_plan.md GMS-024.
# GMS-112: gms/ moved to knowlytix/core/; legacy gms/ shim still scanned.
SHIPPED_DIRS = ("knowlytix/core", "gms", "knowlytix/knowledge", "knowlytix/benchmark", "knowlytix/harness")

PATTERN = re.compile(
    r"anthropic\.Anthropic\(|claude-[a-z0-9-]+|gpt-[0-9][a-z0-9.\-]*",
    re.IGNORECASE,
)

# Files excluded from the audit entirely. Key rule: anything that never
# ends up in a shipped wheel, or whose contents are data not code.
EXCLUDED_SUFFIXES = {
    ".md",        # docs
    ".json",      # result fixtures, test data
    ".yaml",      # configs (llm block in default.yaml is intentional doc)
    ".yml",
    ".ipynb",     # notebooks
    ".txt",
    ".csv",
    ".pdf",
}
EXCLUDED_DIR_PARTS = {
    "__pycache__",
    "tests",                     # tests legitimately reference models
    "gmsh_testing_int",          # internal book content, never ships
    "gmsh_testing_ext",          # book content → gms-book, not a wheel
    "gms_agent_builder_int",
    "gms_agent_builder_ext",
    "gms_wiki",
    "gms_llm",                   # lab tooling, not shipped
    "demo",                      # demo/ is in-repo only
    "results",                   # benchmark result JSON
}

# knowlytix/knowledge/web_agent.py is explicitly excluded from the knowlytix-knowledge
# wheel per gmsh_packaging_plan.md, so its direct anthropic.Anthropic() use
# is acceptable until that module is removed or rewritten.
FILE_ALLOWLIST = {
    "knowlytix/knowledge/web_agent.py",
}

# Exact (path, lineno, matched-text) triples that are intentionally
# retained in shipped source. Add entries here only with a reason in
# the accompanying comment. The goal is to drain this list over time.
#
# Today's retained items are either:
#   - LiteLLM-format model strings inside docstrings/error messages
#     where the AST walker below can't distinguish them from code.
#   - The foundation ``knowlytix/core/llm/config.DEFAULT_MODEL`` default, which
#     the `gms.llm.LLMSettings` API lets callers override via env var.
#     Removing it would break 30+ tests that rely on a sane default
#     before a provider-specific override is chosen. Tracked for
#     future tightening.
KNOWN_EXCEPTIONS: set[tuple[str, int, str]] = {
    # knowlytix/core/llm/config.py: the only default model in the foundation
    # package. Env var GMS_LLM_MODEL overrides it at every call site.
    # Kept intentionally until we're ready to force every caller to
    # pick a provider explicitly (post-v0.1.0 tightening).
    ("knowlytix/core/llm/config.py", 32, "claude-sonnet-4-20250514"),
    # knowlytix/core/llm/errors.py: illustrative model strings inside provider-
    # agnostic error-message hints. These are helpful examples the
    # user sees when they forget an API key — not hardcoded defaults.
    ("knowlytix/core/llm/errors.py", 73, "claude-sonnet-4-20250514"),
    ("knowlytix/core/llm/errors.py", 74, "gpt-4o"),
    ("knowlytix/core/llm/errors.py", 101, "gpt-4o"),
    # ---- CLI / in-code defaults retained until downstream migrations ----
    # These drain-list entries are the next wave of config-centralization
    # work. They're CLI flag defaults or dataclass defaults that still
    # hardcode a specific model string; the corresponding Settings classes
    # already let callers override via env var, so these remain as a
    # "friendly default" safety net. Each needs a follow-up story to
    # remove (likely part of Phase 2 packaging prep or Phase 5 CI).
    ("knowlytix/knowledge/cli.py", 22, "claude-sonnet-4-20250514"),
    ("knowlytix/knowledge/mcp_server.py", 51, "claude-sonnet-4-20250514"),
    ("knowlytix/benchmark/cli.py", 27, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/run.py", 38, "claude-opus-4-7"),
    ("knowlytix/harness/registry.py", 109, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/core.py", 55, "claude-opus-4-7"),
    ("knowlytix/harness/core.py", 64, "claude-sonnet-4-20250514"),
    # harness/core.py hosts the legacy `Harness` (ds_agent-facing) class.
    # The direct `anthropic.Anthropic()` instantiation predates GMS-021's
    # harness.doe migration and is scheduled for removal when the core
    # Harness migrates to gms.llm (separate story, not part of GMS-021's
    # harness.doe scope).
    ("knowlytix/harness/core.py", 122, "anthropic.Anthropic("),
    # knowlytix/benchmark/ingest.py: deprecated `llm_model` parameter default
    # in `ingest_markdown`. Kept as a one-release deprecation shim for
    # callers migrating from the (llm_client, llm_model) shape to the
    # new `llm_callable` seam. Both the parameter and this default will
    # be removed in the next release; entry can be deleted then.
    # Synced from upstream commit 68f6c6b (2026-04-30).
    ("knowlytix/benchmark/ingest.py", 450, "claude-sonnet-4-20250514"),
    # GMS-214c (Phase 4.10, 2026-05-01 PM): paths updated for the
    # harness.doe → harness.testing rename. Same lines/values; just the
    # path prefix changed.
    ("knowlytix/harness/testing/oracles.py", 1003, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/testing/oracles.py", 1290, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/testing/config.py", 25, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/testing/config.py", 33, "claude-sonnet-4-20250514"),
    ("knowlytix/harness/testing/cli.py", 70, "claude-sonnet-4-20250514"),
    # knowlytix/benchmark/eval/ is owned by knowlytix-bench per GMS-055 (moved from
    # top-level eval/). Hardcoded defaults here are inherited from the
    # pre-centralization legacy path; migrating them is tracked separately.
    ("knowlytix/benchmark/eval/benchmark.py", 65, "claude-sonnet-4-20250514"),
    ("knowlytix/benchmark/eval/benchmark.py", 191, "claude-sonnet-4-20250514"),
    ("knowlytix/benchmark/eval/llm_caller.py", 14, "claude-sonnet-4-20250514"),
    ("knowlytix/benchmark/eval/llm_caller.py", 40, "anthropic.Anthropic("),
    # NOTE: the 4 knowlytix/harness/graphdoe/ entries present between
    # 2026-04-21 and 2026-04-22 were drained when graphdoe was rewritten
    # to accept a provider-agnostic `knowlytix.core.llm.LLMClient` via
    # `get_llm()` (QuestionRephraser, DOEBenchmark). If graphdoe is
    # re-synced from upstream in the future, re-apply the same rewrite
    # or add the hardcodes back here with justification.
    # GMS-214b (Phase 4.10, 2026-05-01 PM): governance subpackage moved
    # from top-level governed_agent/ into knowlytix.harness.governance.
    # The plan_adversarial.assemble_attack_cohort() default model_id is
    # a friendly default for a specific adversarial-attack scenario;
    # callers can override via the function parameter. Tracked for
    # eventual migration to gms.llm.LLMSettings dispatch (separate ticket).
    # plan_adversarial: model_id defaults + direct anthropic.Anthropic()
    # for LLM-adversarial cohort generation. All 3 sites predate
    # GMS-021's gms.llm migration (the file was internal to the
    # governed_agent subpackage prior to GMS-214b's move under harness).
    # Tracked for unified migration to gms.llm.LLMSettings dispatch in
    # a separate ticket; allowlisted today as friendly-default behavior
    # that callers can override via the function parameter.
    ("knowlytix/harness/governance/plan_adversarial.py", 132, "claude-opus-4-6"),
    ("knowlytix/harness/governance/plan_adversarial.py", 182, "anthropic.Anthropic("),
    ("knowlytix/harness/governance/plan_adversarial.py", 316, "claude-opus-4-6"),
}


def _iter_py_files() -> list[Path]:
    out: list[Path] = []
    for root_name in SHIPPED_DIRS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
                continue
            if path.suffix != ".py":
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in FILE_ALLOWLIST:
                continue
            out.append(path)
    return out


def _docstring_line_spans(source: str) -> set[int]:
    """Return the set of line numbers covered by module / class / def docstrings.

    Matches in a docstring are informational examples, not code-level
    hardcoding, so they're excluded from the audit.
    """

    spans: set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        start = first.lineno
        end = getattr(first, "end_lineno", start)
        spans.update(range(start, end + 1))
    return spans


def _non_docstring_triple_string_spans(source: str) -> set[int]:
    """Return line numbers covered by triple-quoted string literals that
    are *not* docstrings.

    The simple hand-rolled comment stripper below respects single- and
    double-quoted strings on one line, but can't see that a triple-
    quoted literal spanning multiple lines is one continuous literal.
    Using the tokenizer to tag those multi-line ranges lets us skip
    them so a module-level assignment like
    ``X = \"\"\"example claude-opus-4-6\"\"\"`` is not scanned as code.

    Single-line string literals (including ``default=\"claude-…\"``)
    are still scanned — that's exactly where real hardcoded defaults
    live, and the whole point of the audit.
    """

    spans: set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return spans
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        start_line, _ = tok.start
        end_line, _ = tok.end
        if end_line == start_line:
            continue  # single-line strings stay in scope of the scanner
        spans.update(range(start_line, end_line + 1))
    return spans


def _strip_code_comment(line: str) -> str:
    """Drop ``# ...`` trailing comments while respecting single-line
    string literals on the same line.

    This is deliberately simple — it handles the common case of a
    model string inside a single-line literal (``default="claude-…"``)
    correctly. Multi-line triple-quoted literals are filtered out
    upstream by :func:`_non_docstring_triple_string_spans`.
    """

    out: list[str] = []
    in_str: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == "\\" and i + 1 < len(line):
                out.append(ch)
                out.append(line[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            out.append(ch)
        else:
            if ch in ("'", '"'):
                in_str = ch
                out.append(ch)
            elif ch == "#":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def scan_file(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    skip_lines = _docstring_line_spans(text) | _non_docstring_triple_string_spans(text)
    violations: list[tuple[int, str]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if lineno in skip_lines:
            continue
        code = _strip_code_comment(raw_line)
        if not code.strip():
            continue
        for match in PATTERN.finditer(code):
            violations.append((lineno, match.group(0)))
    return violations


def main() -> int:
    rel_root = REPO_ROOT
    errors: list[str] = []
    for path in _iter_py_files():
        rel = path.relative_to(rel_root).as_posix()
        for lineno, matched in scan_file(path):
            key = (rel, lineno, matched)
            if key in KNOWN_EXCEPTIONS:
                continue
            errors.append(f"{rel}:{lineno}: hardcoded model/SDK reference: {matched!r}")

    # Also flag KNOWN_EXCEPTIONS entries whose lines have drifted, so the
    # allowlist stays honest.
    for rel, lineno, matched in sorted(KNOWN_EXCEPTIONS):
        full = rel_root / rel
        if not full.exists():
            errors.append(
                f"{rel}:{lineno}: allowlisted file no longer exists; "
                f"remove the KNOWN_EXCEPTIONS entry."
            )
            continue
        lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
        if lineno > len(lines) or matched not in lines[lineno - 1]:
            # Help the maintainer find the new home (if any) without
            # grepping manually.
            hits = [
                new_lineno
                for new_lineno, line_text in enumerate(lines, start=1)
                if matched in line_text
            ]
            if hits:
                hint = (
                    f" still present at line {hits[0]}"
                    if len(hits) == 1
                    else f" now found at lines {hits}"
                )
            else:
                hint = " no longer appears in the file"
            errors.append(
                f"{rel}:{lineno}: allowlisted entry for {matched!r} "
                f"has drifted —{hint}. Update or remove the "
                "KNOWN_EXCEPTIONS entry."
            )

    if errors:
        print("LLM centralization audit FAILED:\n", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nFixes: replace hardcoded model strings with env-var-driven "
            "`gms.llm.LLMSettings` / `get_llm(purpose)` dispatch, or — if "
            "the occurrence is genuinely intentional — add an entry to "
            "scripts/audit_llm_centralization.py::KNOWN_EXCEPTIONS with "
            "a justifying comment.",
            file=sys.stderr,
        )
        return 1
    print(f"LLM centralization audit OK: {len(_iter_py_files())} files scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
