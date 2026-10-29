# SPDX-License-Identifier: Apache-2.0
"""Representation-aware, read-only quality audits and reproducible report bundles.

A quality pass certifies the selected checks, never truth or semantic recall.
Native assertion syntax is not converted into unconditional domain triples.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .adapters import read_store
from .config import AuditConfig, load_profile
from .engine import audit, audit_ontology
from .incremental import code_digest
from .models import GraphView
from .registry import builtin_registry
from .reporting import RunInfo, render_markdown, report_to_dict

QUALITY_VERSION = 'kgcheck-quality/1'
HIERARCHY_RULES = frozenset({'STR003', 'STR004', 'STR005', 'STR007', 'STR008', 'DEF003'})
NATIVE_EQUIVALENTS = {'EVD001': 'ASR004', 'EVD002': 'ASR005',
                     'COV001': 'ASR006', 'NUM003': 'ASR007'}
LIMITS = [
    'A pass means selected required checks completed without blocking violations; it is not a factual accuracy or completeness certificate.',
    'Assertion citations validate document identity, offsets and quoted text, not entailment.',
    'ASR007 checks localized values/scales and recognized units/bound direction; it does not certify periods, bound strictness, aggregation or semantic scope.',
    'Discovery vocabulary is audited separately; it is not a closed vocabulary contract for source assertion phrases.',
    'Independent semantic accuracy and calibrated heuristic precision have not been measured by this audit.',
]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                   allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _read_metadata(path):
    """Read only the named JSON, without following any paths from its contents."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate metadata key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'nonfinite metadata value: {value}')
    data = path.read_bytes()
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant), hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class QualityReport:
    """Component reports and the explicit scope of their aggregate decision."""
    components: tuple
    manifest: dict
    applicability: tuple
    issues: tuple
    store_root: Path
    deployment: dict | None = None

    @property
    def outcome(self):
        outcomes = [r.evaluate_gate().outcome.value for _, r, _ in self.components]
        if 'fail' in outcomes or (self.deployment and self.deployment.get('gate') == 'fail'):
            return 'fail'
        return 'indeterminate' if self.issues or 'indeterminate' in outcomes else 'pass'

    def to_dict(self):
        return {'quality_version': QUALITY_VERSION, 'snapshot_id': self.manifest['snapshot_id'],
                'representation': self.manifest['representation'],
                'gate': {'outcome': self.outcome, 'issues': list(self.issues)},
                'semantic_accuracy_measured': False, 'limits': LIMITS,
                **({'gms_deployment': self.deployment} if self.deployment is not None else {}),
                'applicability': list(self.applicability),
                'components': {name: report_to_dict(report, run, report.evaluate_gate())
                               for name, report, run in self.components}}

    def markdown(self):
        lines = ['# OntoDisc KG quality report', '', f'Gate: **{self.outcome}**', '',
                 f"Representation: `{self.manifest['representation']}`", '',
                 f"Snapshot: `{self.manifest['snapshot_id']}`", '',
                 '## Scope and limitations', '', *['- ' + limit for limit in LIMITS], '']
        if self.deployment:
            lines += ['## GMS training and calibration', '',
                      f"Gate: **{self.deployment['gate']}**", '', self.deployment['limit'], '',
                      '```json', json.dumps({k: self.deployment[k] for k in
                          ('calibration', 'heldout', 'assertion_retrieval')}, indent=2), '```', '']
        if self.issues:
            lines += ['## Incomplete required coverage', '', *['- ' + i for i in self.issues], '']
        lines += ['## Applicability', '', '| Rule | Status | Reason |', '|---|---|---|']
        for row in self.applicability:
            reason = row['reason'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {row['rule']} | {row['status']} | {reason} |")
        for name, report, _ in self.components:
            lines += ['', '## ' + name, '', render_markdown(report, report.evaluate_gate())]
        return '\n'.join(lines) + '\n'

    def write_bundle(self, output, *, store, previous=None):
        """Write a new bundle outside the input store; never overwrite a bundle."""
        from .__main__ import check_outputs
        output = Path(output)
        if Path(store).resolve() != self.store_root:
            raise ValueError('Report destination must be checked against the audited store')
        check_outputs(Path(store), [output])
        comparison = None
        if previous is not None:
            from .reporting import compare_reports
            before, _ = _read_metadata(Path(previous))
            if before.get('quality_version') != QUALITY_VERSION:
                raise ValueError('Previous report has an unsupported quality version')
            after = self.to_dict()
            comparison = {'before': before['snapshot_id'], 'after': after['snapshot_id'], 'components': {}}
            for name in sorted(set(before['components']) | set(after['components'])):
                if name in before['components'] and name in after['components']:
                    comparison['components'][name] = compare_reports(
                        before['components'][name], after['components'][name]).to_dict()
                else:
                    comparison['components'][name] = {'comparable': False,
                        'reason': 'Component added or no longer evaluated; old findings are not resolved.'}
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Use a new quality report destination: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.kg-quality-', dir=output.parent) as tmp:
            stage = Path(tmp) / 'bundle'
            stage.mkdir()
            def save(name, value):
                p = stage / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            if comparison is not None:
                save('comparison.json', comparison)
            save('manifest.json', self.manifest)
            save('applicability.json', list(self.applicability))
            save('report.json', self.to_dict())
            (stage / 'report.md').write_text(self.markdown())
            for name, report, run in self.components:
                save(f'components/{name}.json', report_to_dict(report, run, report.evaluate_gate()))
            save('checksums.json', {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(stage.rglob('*')) if p.is_file()})
            stage.rename(output)
        return output


def audit_quality(store, *, vocabulary=None, config=None, semantic_reviewer=None):
    """Audit a saved OntoDisc KG, adding its vocabulary without requiring hierarchy.

    ``vocabulary`` is an explicit in-memory ontology dictionary (used during a
    build). Otherwise assertion stores use execution.json's final discovery
    vocabulary or seed vocabulary. Missing/malformed vocabulary prevents a
    complete quality pass. Typed sidecars remain unsupported for assertion KGs.
    Legacy stores retain the regular configured store audit and its declarations.
    """
    from knowlytix.knowledge import __version__
    store = Path(store)
    config = config or AuditConfig()
    snapshot = read_store(store, config.audit_schema)
    native = snapshot.capabilities.adapter == 'source_assertions'
    issues = []
    inputs = {a.name: a.sha256 for a in snapshot.artifacts if a.sha256}
    pointer = store / 'assertions/CURRENT'
    pointer_before = pointer.read_bytes() if native and pointer.exists() else None
    report = audit(snapshot, config)
    now = datetime.now(UTC).isoformat(timespec='seconds')
    def run_info(selected):
        return RunInfo(created_at=now, tool_version=str(__version__), configuration={
            'audit': selected.to_dict(), 'quality_version': QUALITY_VERSION,
            'code_sha256': code_digest(), 'gate': {'fail_on': 'error', 'required_rules': None}})
    from dataclasses import replace
    selected = replace(config, profile=report.profile)
    components = [('assertions' if native else 'store', report, run_info(selected))]
    metadata_path = store / 'execution.json'
    metadata_hash = None
    if native:
        if vocabulary is None:
            try:
                metadata, metadata_hash = _read_metadata(metadata_path)
                if metadata.get('generation') != snapshot.snapshot_id:
                    issues.append('execution.json does not identify the audited assertion generation')
                profile = metadata.get('profile')
                vocabulary = (json.loads(profile['ontology']) if profile is not None
                              else metadata.get('seed_ontology'))
                inputs['execution.json'] = metadata_hash
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                issues.append(f'Vocabulary metadata unavailable: {type(exc).__name__}: {exc}')
        if vocabulary is None:
            issues.append('Required vocabulary audit unavailable: supply the final vocabulary')
        else:
            vocabulary_config = AuditConfig(profile='vocabulary', view=GraphView.PROPOSAL)
            try:
                vocab = audit_ontology(vocabulary, config=vocabulary_config)
                components.append(('vocabulary', vocab, run_info(vocabulary_config)))
                inputs['vocabulary_digest'] = _digest(vocabulary)
            except (TypeError, ValueError, KeyError) as exc:
                issues.append(f'Vocabulary audit unavailable: {type(exc).__name__}: {exc}')
        if (pointer_before is None or not pointer.exists() or pointer.read_bytes() != pointer_before
                or pointer_before.decode().strip() != snapshot.snapshot_id):
            issues.append('Assertion generation changed during the quality audit')
        if metadata_hash is not None:
            try:
                changed = hashlib.sha256(metadata_path.read_bytes()).hexdigest() != metadata_hash
            except OSError:
                changed = True
            if changed:
                issues.append('Vocabulary metadata changed during the quality audit')
    if semantic_reviewer is not None:
        if not native:
            raise ValueError('AssertionSemanticReviewer requires a source-assertion KG')
        semantic_config = AuditConfig(profile='source_assertions')
        semantic = audit(snapshot, semantic_config, rules=[semantic_reviewer.rule()])
        components.append(('semantic', semantic, run_info(semantic_config)))
        if any(f.rule_id == 'ASR008' for f in semantic.findings()):
            issues.append('Semantic model review found assertions or possible omissions requiring human review')
        if not pointer.exists() or pointer.read_bytes() != pointer_before:
            issues.append('Assertion generation changed during semantic review')
    results = {r.rule_id: r for _, component, _ in components for r in component.results}
    applicability = []
    for rule_id, rule in builtin_registry().rules().items():
        result = results.get(rule_id)
        if result is not None:
            status = result.status.value
            reason = ('; '.join(result.missing_prerequisites) if status == 'unavailable'
                      else rule.spec.title)
        elif native and len(components) == 1 and rule_id in load_profile('vocabulary').rules:
            status, reason = 'unavailable', 'Required vocabulary component has no readable final vocabulary.'
        elif native and rule_id in HIERARCHY_RULES:
            status, reason = 'not_applicable', 'Source-assertion KG declares no class hierarchy.'
        elif native and rule_id in NATIVE_EQUIVALENTS:
            status, reason = 'replaced', f'Native equivalent: {NATIVE_EQUIVALENTS[rule_id]}; see its documented scope.'
        elif native and rule_id.startswith(('ART', 'NUM')):
            status, reason = 'not_applicable', 'Legacy triple-store artifact check; native snapshot contract covers this representation.'
        elif not native and rule_id.startswith('ASR'):
            status, reason = 'not_applicable', 'Requires a source-assertion representation.'
        else:
            status, reason = 'not_selected', 'Outside selected profiles; no validation claim. Requires explicit declarations or a supported representation.'
        applicability.append({'rule': rule_id, 'version': rule.spec.version, 'status': status, 'reason': reason})
    for rule_id in sorted(set(results) - set(builtin_registry().rules())):
        result = results[rule_id]
        applicability.append({'rule': rule_id, 'version': result.rule_version,
            'status': result.status.value, 'reason': 'Explicitly requested fresh semantic model review; uncalibrated.'})
    if semantic_reviewer is not None:
        # Fresh model output is part of identity; repeated calls may differ.
        from .reporting import stable_payload
        inputs['semantic_review_digest'] = _digest(stable_payload(report_to_dict(semantic)))
    deployment = None
    if native and (store / 'gms_deployment.json').exists():
        try:
            from knowlytix.knowledge.assertions.deployment import read_deployment
            deployment = read_deployment(store, generation=snapshot.snapshot_id)
            inputs['gms_deployment_digest'] = _digest(deployment)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f'GMS deployment unavailable or stale: {exc}')
    if native and (store / 'model.pt').exists() and not (store / 'gms_deployment.json').exists():
        issues.append('GMS model present without a deployment calibration/evaluation manifest')
    identity = {'components': {name: r.snapshot_id for name, r, _ in components},
                'inputs': inputs, 'configuration': selected.to_dict(), 'code_sha256': code_digest()}
    manifest = {'quality_version': QUALITY_VERSION, 'snapshot_id': _digest(identity),
                'representation': snapshot.capabilities.adapter, 'created_at': now,
                'tool_version': str(__version__), **identity,
                'profiles': {r.profile: load_profile(r.profile).version for _, r, _ in components},
                'source_snapshot': snapshot.snapshot_id, 'semantic_accuracy_measured': False}
    return QualityReport(tuple(components), manifest, tuple(applicability), tuple(issues), store.resolve(), deployment)
