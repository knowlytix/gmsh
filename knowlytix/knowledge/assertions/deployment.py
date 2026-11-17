# SPDX-License-Identifier: Apache-2.0
"""Read-only model/calibration artifact contract; no training or model imports."""
import hashlib
import json
from pathlib import Path

VERSION = 'assertion-gms/1'


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_deployment(path, *, generation=None):
    """Validate binding and all saved model/calibration artifacts before serving."""
    path = Path(path)
    report = json.loads((path / 'gms_deployment.json').read_text())
    current = (path / 'assertions/CURRENT').read_text().strip()
    if report.get('version') != VERSION or report.get('generation') != current or (generation and current != generation):
        raise ValueError('Stale or unsupported GMS deployment')
    required = {'model.pt', 'model_dims.json', 'adapter.json', 'triples.json', 'metadata.json',
                'gms_calibration.json', 'gms_evaluation.json', 'gms_cohorts.json'}
    artifacts = report['artifacts']
    if not required <= artifacts.keys():
        raise ValueError('Incomplete GMS deployment artifacts')
    for name, expected in artifacts.items():
        file = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts or file.is_symlink():
            raise ValueError('Invalid deployment artifact path')
        if _hash(file) != expected:
            raise ValueError(f'GMS deployment artifact changed: {name}')
    calibration = json.loads((path / 'gms_calibration.json').read_text())
    evaluation = json.loads((path / 'gms_evaluation.json').read_text())
    metadata = json.loads((path / 'metadata.json').read_text())
    if calibration != report['calibration'] or evaluation['heldout'] != report['heldout']:
        raise ValueError('Deployment summary disagrees with calibration/evaluation artifacts')
    if metadata.get('assertion_projection_generation') != current:
        raise ValueError('Model metadata is not bound to the assertion generation')
    if metadata.get('assertion_projection_scoring') not in {None, 'canonical_triple_v1'}:
        raise ValueError('Unknown assertion projection scoring mode')
    trust = report.get('trust', 'verified')
    if trust not in {'verified', 'unverified'} or metadata.get('assertion_projection_trust', 'verified') != trust:
        raise ValueError('Deployment trust disagrees with model metadata')
    if trust == 'unverified':
        if 'candidate_projection.json' not in artifacts or calibration.get('trust') != trust:
            raise ValueError('Incomplete unverified deployment provenance')
        from .store import compile_projections, load_snapshot
        snapshot = load_snapshot(path)
        projection = json.loads((path / 'candidate_projection.json').read_text())
        if projection != compile_projections(snapshot['index'], snapshot['graph'], candidates=True):
            raise ValueError('Candidate projection disagrees with source assertions')
    requirements = evaluation['requirements']
    retrieval = evaluation['assertion_retrieval']
    passed = (evaluation['heldout']['n_pos'] > 0 and evaluation['heldout']['n_neg'] > 0
              and evaluation['heldout']['false_accept'] <= requirements['max_false_accept']
              and evaluation['heldout']['recall'] >= requirements['min_recall']
              and (trust == 'unverified' or (retrieval['recall'] is not None
                   and retrieval['recall'] >= requirements['min_recall']))
              and retrieval['unsafe_accepts'] == 0)
    if report['gate'] != ('pass' if passed else 'fail'):
        raise ValueError('Deployment gate disagrees with evaluation results')
    return report
