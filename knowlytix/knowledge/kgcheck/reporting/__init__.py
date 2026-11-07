# SPDX-License-Identifier: Apache-2.0
"""Report output. JSON is canonical; Markdown is a rendering for people."""

from knowlytix.knowledge.kgcheck.reporting.compare import (
    ReportComparison,
    compare_reports,
)
from knowlytix.knowledge.kgcheck.reporting.json_report import (
    REPORT_VERSION,
    RunInfo,
    dumps,
    report_schema,
    report_to_dict,
    stable_payload,
    write_json,
)
from knowlytix.knowledge.kgcheck.reporting.markdown_report import (
    render_markdown,
    write_markdown,
)

__all__ = [
    "ReportComparison",
    "compare_reports",
    "REPORT_VERSION",
    "RunInfo",
    "dumps",
    "report_schema",
    "report_to_dict",
    "stable_payload",
    "write_json",
    "render_markdown",
    "write_markdown",
]
