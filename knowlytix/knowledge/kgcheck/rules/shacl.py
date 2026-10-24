# SPDX-License-Identifier: Apache-2.0
"""CON007: SHACL validation of the view by an external engine (pySHACL).

The view is mapped to RDF (``kgcheck.rdf``) and validated against SHACL shapes:
a caller-supplied shapes file, the audit schema's constraints translated to
shapes, or both. Findings carry the engine's own references: the constraint
component, the source shape, the result path and the value. The engine and its
inference setting are recorded; without ``rdflib`` and ``pyshacl`` installed the
rule is unavailable. This is SHACL validation of kgcheck's RDF mapping of the
view, so it inherits that mapping's choices (see ``kgcheck.rdf``).
"""

from __future__ import annotations

from pathlib import Path

from knowlytix.knowledge.kgcheck.models import GraphView, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin

_SEVERITY = {"Violation": Severity.ERROR, "Warning": Severity.WARNING, "Info": Severity.INFO}
_WITNESS = 50


@builtin(RuleSpec(
    "CON007", "2", "constraint", "SHACL validation", Severity.ERROR, phase="C",
    views=(GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED),
    parameters={"shapes": "", "shapes_from_audit_schema": False, "inference": "none"},
    parameter_choices={"inference": ("none", "rdfs", "owlrl", "both")}, cacheable=False,
    references=("https://www.w3.org/TR/shacl/", "https://github.com/RDFLib/pySHACL"),
    basis="pySHACL validation of the view's RDF mapping (kgcheck.rdf) against the given shapes "
          "file and/or the audit schema's constraints as shapes; one finding per focus node, "
          "constraint component, source shape, path and value, at the result's SHACL severity"))
def con007(ctx: RuleContext) -> Outcome:
    try:
        import pyshacl  # noqa: F401
        from rdflib import Graph

        from knowlytix.knowledge.kgcheck import rdf
    except ImportError:
        return Outcome.unavailable("backend:pyshacl")
    shapes = _shapes(ctx, Graph, rdf)
    if shapes is None:
        return Outcome.unavailable("shacl_shapes")
    data = _data(ctx, Graph, rdf)
    results = _validate(data, shapes, str(ctx.params["inference"]))
    terms = rdf.Terms(ctx.snapshot)
    findings = [_finding(ctx, terms, key, count) for key, count in sorted(results.items())]
    untranslated = rdf.untranslated(ctx.snapshot) if ctx.params["shapes_from_audit_schema"] else []
    return Outcome(eligible=len(data), findings=tuple(findings), metrics={
        "engine": f"pyshacl {_version()}", "inference": ctx.params["inference"],
        "data_triples": len(data), "shape_triples": len(shapes),
        "validation_results": sum(results.values()), "untranslated_constraints": untranslated})


def _shapes(ctx: RuleContext, graph_type, rdf):  # type: ignore[no-untyped-def]
    path, translated = str(ctx.params["shapes"]), bool(ctx.params["shapes_from_audit_schema"])
    if not path and not translated:
        return None
    shapes = graph_type()
    if path:
        shapes.parse(Path(path).expanduser())  # a named file that cannot be read is an error
    if translated:
        shapes += rdf.shapes_from_audit_schema(ctx.snapshot)
    return shapes if len(shapes) else None


def _data(ctx: RuleContext, graph_type, rdf):  # type: ignore[no-untyped-def]
    """The graph an RDF input carried, else the RDF mapping of the view."""
    original = ctx.snapshot.parsed.get("graph.nt")
    if isinstance(original, str) and ctx.view is GraphView.STORED:
        return graph_type().parse(data=original, format="nt")
    return rdf.view_graph(ctx.snapshot, ctx.view)


Key = tuple[str, str, str, str, str, str]  # focus, component, shape, path, value, severity


def _validate(data, shapes, inference: str) -> dict[Key, int]:  # type: ignore[no-untyped-def]
    from pyshacl import validate
    from rdflib import RDF, Namespace
    sh = Namespace("http://www.w3.org/ns/shacl#")
    _, report, _ = validate(data, shacl_graph=shapes, inference=inference, abort_on_first=False,
                            allow_warnings=True, advanced=True)
    out: dict[Key, int] = {}
    for result in report.subjects(RDF.type, sh.ValidationResult):
        values = [report.value(result, p) for p in (
            sh.focusNode, sh.sourceConstraintComponent, sh.sourceShape, sh.resultPath, sh.value,
            sh.resultSeverity)]
        values[2] = _named_shape(shapes, values[2], sh)
        # data blank nodes keep their labels (canonical for RDF input); shape blank nodes do not
        key = tuple(_text(v, keep_blank=i in (0, 3, 4)) for i, v in enumerate(values))
        out[key] = out.get(key, 0) + 1  # type: ignore[index]
    return out


def _named_shape(shapes, shape, sh):  # type: ignore[no-untyped-def]
    """The shape itself, or the named node shape whose sh:property it is."""
    from rdflib import BNode
    if not isinstance(shape, BNode):
        return shape
    parent = shapes.value(predicate=sh.property, object=shape)
    return parent if parent is not None else shape


def _text(term: object, *, keep_blank: bool) -> str:
    from rdflib import BNode
    if term is None:
        return ""
    if isinstance(term, BNode):
        return f"_:{term}" if keep_blank else ""
    return str(term)


def _term(text: str):  # type: ignore[no-untyped-def]
    from rdflib import BNode, URIRef
    return BNode(text[2:]) if text.startswith("_:") else URIRef(text)


def _finding(ctx: RuleContext, terms, key: Key, count: int):  # type: ignore[no-untyped-def]
    focus, component, shape, path, value, severity = key
    node = terms.node(_term(focus)) if focus else None
    short = component.rsplit("#", 1)[-1]
    shown_path = (terms.node(_term(path)) or path) if path else ""
    return ctx.finding(
        f"SHACL {short} on {_show(node or focus)}" + (f" via {_show(shown_path)}" if path else ""),
        severity=_SEVERITY.get(severity.rsplit("#", 1)[-1], Severity.ERROR),
        affected=(node,) if node else (), discriminator="|".join(key),
        witness={"focus": focus, "component": component, "shape": shape or None,
                 "path": path or None, "value": value or None, "results": count},
        suggestion="see the shape's constraint; correct the data or the shape")


def _show(node: str) -> str:
    return repr(node.partition(":")[2]) if node.startswith(("entity:", "relation:")) else repr(node)


def _version() -> str:
    from importlib.metadata import version
    return version("pyshacl")

