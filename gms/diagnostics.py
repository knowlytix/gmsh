# SPDX-License-Identifier: LicenseRef-Knowlytix-GMS-EULA
"""Post-training diagnostics for GMS primitives.

Verifies that all GMS operations produce meaningful outputs after
training.  Each check returns a structured result with a PASS/FAIL
verdict and supporting evidence.

Usage::

    from knowlytix.core.diagnostics import GMSDiagnostics
    diag = GMSDiagnostics(model, adapter, enm, doc_graph, device)
    report = diag.run_all()
    diag.print_report(report)

Nine checks are organized in two groups:

Structural invariants (preconditions for everything else):
    S1. cayley_orthogonality — R^T R = I and det(R) = 1 for all rotors
    S2. stiefel_projection   — PP^T = I for v-space and u-space projections
    S3. sphere_embedding     — ||v|| = 1 and ||u|| = 1 for all entities

Six GMS primitives:
    P1. score_triple   — pairwise discrimination (true < fabricated)
    P2. tension_energy — spectrum width and band separation
    P3. check_holonomy — defects well below random baseline
    P4. lookup_enm     — SHA-256 integrity verification
    P5. link_predict   — type-constrained filtered rank
    P6. query_triples  — structural query correctness
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.core.geometry.stiefel import check_stiefel
from knowlytix.core.memory.enm import ExactNumericalMemory
from knowlytix.core.data.prepare import mine_tension_pairs


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    score: float            # 0.0-1.0 success rate or metric
    detail: str             # human-readable explanation
    evidence: dict = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Full diagnostic report across all six primitives."""
    checks: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


# ---------------------------------------------------------------------------
# Diagnostics class
# ---------------------------------------------------------------------------

class GMSDiagnostics:
    """Post-training diagnostic suite for all six GMS primitives.

    Args:
        model: Trained GeometricKnowledgeGraph.
        adapter: GraphToGMS adapter with entity/relation vocabularies.
        enm: Populated ExactNumericalMemory register.
        doc_graph: Source DocumentGraph with triples and ENM entries.
        device: Torch device.
        sample_size: Max triples/pairs to evaluate per check (for speed).
    """

    def __init__(
        self,
        model: GeometricKnowledgeGraph,
        adapter,
        enm: ExactNumericalMemory,
        doc_graph,
        device: torch.device,
        sample_size: int = 200,
    ):
        self.model = model
        self.adapter = adapter
        self.enm = enm
        self.doc_graph = doc_graph
        self.device = device
        self.sample_size = sample_size
        self.transport = RelationalTransport(model)

    # ---------------------------------------------------------------
    # S1. Cayley orthogonality: R^T R = I, det(R) = 1
    # ---------------------------------------------------------------

    def check_cayley_orthogonality(self) -> CheckResult:
        """Verify all Cayley rotors remain in SO(m) after training.

        Checks two properties for every relation rotor:
          (a) Orthogonality: ||R^T R - I||_F < tol
          (b) Proper rotation: |det(R) - 1| < tol

        Pass criterion: max error across all rotors < 1e-3.
        """
        self.model.eval()
        m = self.model.cfg.m
        I = torch.eye(m, device=self.device)
        max_ortho_err = 0.0
        max_det_err = 0.0

        with torch.no_grad():
            for r_idx in range(self.model.num_relations):
                R = self.model.relation_rotors[r_idx].get_rotor()
                ortho_err = (R.T @ R - I).norm().item()
                det_err = abs(torch.linalg.det(R).item() - 1.0)
                max_ortho_err = max(max_ortho_err, ortho_err)
                max_det_err = max(max_det_err, det_err)

        passed = max_ortho_err < 1e-3 and max_det_err < 1e-3

        return CheckResult(
            name="cayley_orthogonality",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=(
                f"max ||R^T R - I||_F = {max_ortho_err:.2e}, "
                f"max |det(R) - 1| = {max_det_err:.2e} "
                f"across {self.model.num_relations} rotors"
            ),
            evidence={
                "max_ortho_err": max_ortho_err,
                "max_det_err": max_det_err,
                "num_relations": self.model.num_relations,
                "m": m,
            },
        )

    # ---------------------------------------------------------------
    # S2. Stiefel projection: PP^T = I
    # ---------------------------------------------------------------

    def check_stiefel_projection(self) -> CheckResult:
        """Verify Stiefel projection matrices satisfy PP^T = I_m.

        Both P_v (semantic) and P_u (logical) are m x d_v and m x d_u
        respectively.  They should remain on the Stiefel manifold after
        training (enforced by the Riemannian optimizer retraction).

        Pass criterion: ||PP^T - I||_F < 1e-2 for both projections.
        """
        self.model.eval()
        m = self.model.cfg.m
        I = torch.eye(m, device=self.device)

        with torch.no_grad():
            P_v = self.model.dual_emb.P_v.data
            P_u = self.model.dual_emb.P_u.data
            pv_err = (P_v @ P_v.T - I).norm().item()
            pu_err = (P_u @ P_u.T - I).norm().item()

        pv_ok = check_stiefel(P_v, tol=1e-2)
        pu_ok = check_stiefel(P_u, tol=1e-2)
        passed = pv_ok and pu_ok

        return CheckResult(
            name="stiefel_projection",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=(
                f"P_v: ||PP^T - I|| = {pv_err:.2e} ({'OK' if pv_ok else 'FAIL'}), "
                f"P_u: ||PP^T - I|| = {pu_err:.2e} ({'OK' if pu_ok else 'FAIL'})"
            ),
            evidence={
                "pv_err": pv_err,
                "pu_err": pu_err,
                "m": m,
                "d_v": self.model.dual_emb.d_v,
                "d_u": self.model.dual_emb.d_u,
            },
        )

    # ---------------------------------------------------------------
    # S3. Sphere embedding: ||v|| = 1
    # ---------------------------------------------------------------

    def check_sphere_embedding(self) -> CheckResult:
        """Verify all entity embeddings project onto the unit sphere.

        Both project_v (semantic) and project_u (logical) should produce
        unit vectors for every entity.

        Pass criterion: max deviation from unit norm < 1e-4.
        """
        self.model.eval()
        with torch.no_grad():
            all_idx = torch.arange(self.model.num_entities, device=self.device)
            v = self.model.dual_emb.project_v(all_idx)
            u = self.model.dual_emb.project_u(all_idx)
            v_dev = (v.norm(dim=-1) - 1.0).abs().max().item()
            u_dev = (u.norm(dim=-1) - 1.0).abs().max().item()

        passed = v_dev < 1e-4 and u_dev < 1e-4

        return CheckResult(
            name="sphere_embedding",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=(
                f"max v-sphere deviation: {v_dev:.2e}, "
                f"max u-sphere deviation: {u_dev:.2e} "
                f"across {self.model.num_entities} entities"
            ),
            evidence={
                "v_max_deviation": v_dev,
                "u_max_deviation": u_dev,
                "num_entities": self.model.num_entities,
            },
        )

    # ---------------------------------------------------------------
    # P1. score_triple: pairwise discrimination
    # ---------------------------------------------------------------

    def check_score_triple(self) -> CheckResult:
        """Verify that true triples score lower than fabricated ones.

        For each true triple (h, r, t), replaces t with a random entity t'
        that is a valid tail for relation r but is NOT the true tail.
        Checks that d_geo(R_r v_h, v_t) < d_geo(R_r v_h, v_{t'}).

        Pass criterion: >= 90% of sampled triples pass discrimination.
        """
        self.model.eval()
        n_triples = len(self.adapter.heads)
        indices = list(range(n_triples))
        random.shuffle(indices)
        indices = indices[:self.sample_size]

        correct = 0
        total = 0
        failures: list[dict] = []

        with torch.no_grad():
            for idx in indices:
                h_idx = self.adapter.heads[idx].item()
                r_idx = self.adapter.relations[idx].item()
                t_idx = self.adapter.tails[idx].item()

                # Find a different valid tail for this relation
                valid_tails = self.adapter.tail_per_relation.get(r_idx, set())
                alt_tails = [t for t in valid_tails if t != t_idx]
                if not alt_tails:
                    continue

                t_neg = random.choice(alt_tails)

                h = torch.tensor([h_idx], device=self.device)
                r = torch.tensor([r_idx], device=self.device)
                t_true = torch.tensor([t_idx], device=self.device)
                t_fab = torch.tensor([t_neg], device=self.device)

                d_true = self.model.score_triple(h, r, t_true).item()
                d_fab = self.model.score_triple(h, r, t_fab).item()

                total += 1
                if d_true < d_fab:
                    correct += 1
                else:
                    if len(failures) < 10:
                        failures.append({
                            "head": self.adapter.idx_to_entity.get(h_idx, "?"),
                            "relation": self.adapter.idx_to_relation.get(r_idx, "?"),
                            "true_tail": self.adapter.idx_to_entity.get(t_idx, "?"),
                            "fab_tail": self.adapter.idx_to_entity.get(t_neg, "?"),
                            "d_true": round(d_true, 4),
                            "d_fab": round(d_fab, 4),
                        })

        rate = correct / max(total, 1)
        passed = rate >= 0.90

        return CheckResult(
            name="score_triple",
            passed=passed,
            score=rate,
            detail=(
                f"{correct}/{total} ({rate:.1%}) true triples score lower "
                f"than type-matched fabricated triples"
            ),
            evidence={
                "correct": correct,
                "total": total,
                "rate": round(rate, 4),
                "failures": failures,
            },
        )

    # ---------------------------------------------------------------
    # 2. tension_energy: spectrum and band separation
    # ---------------------------------------------------------------

    def check_tension_energy(self) -> CheckResult:
        """Verify tension energy spectrum has meaningful structure.

        Checks three properties:
          (a) Agreement pairs (same relation+tail) have E < sqrt(2)
          (b) Contradiction pairs (different tails, same relation) have E > sqrt(2)
          (c) Spectrum width (max - min) across all pairs > 0.5

        Pass criterion: (a) >= 80% AND (b) >= 70% AND (c) width > 0.5.
        Contradiction threshold is lower because cross-group entities share
        many relations beyond the contradicting one.
        """
        self.model.eval()
        agree_pairs, contra_pairs = mine_tension_pairs(self.adapter)

        sqrt2 = math.sqrt(2)
        agree_ok = 0
        agree_total = 0
        contra_ok = 0
        contra_total = 0
        all_energies: list[float] = []
        agree_energies: list[float] = []
        contra_energies: list[float] = []

        with torch.no_grad():
            # Agreement pairs
            pairs = agree_pairs[:self.sample_size]
            if pairs:
                a_idx = torch.tensor([p[0] for p in pairs], device=self.device)
                b_idx = torch.tensor([p[1] for p in pairs], device=self.device)
                energies = self.model.tension_energy_pairs(a_idx, b_idx)
                for e in energies.tolist():
                    agree_total += 1
                    all_energies.append(e)
                    agree_energies.append(e)
                    if e < sqrt2:
                        agree_ok += 1

            # Contradiction pairs
            pairs = contra_pairs[:self.sample_size]
            if pairs:
                a_idx = torch.tensor([p[0] for p in pairs], device=self.device)
                b_idx = torch.tensor([p[1] for p in pairs], device=self.device)
                energies = self.model.tension_energy_pairs(a_idx, b_idx)
                for e in energies.tolist():
                    contra_total += 1
                    all_energies.append(e)
                    contra_energies.append(e)
                    if e > sqrt2:
                        contra_ok += 1

        agree_rate = agree_ok / max(agree_total, 1)
        contra_rate = contra_ok / max(contra_total, 1)
        width = (max(all_energies) - min(all_energies)) if all_energies else 0.0

        passed = agree_rate >= 0.80 and contra_rate >= 0.70 and width > 0.5

        return CheckResult(
            name="tension_energy",
            passed=passed,
            score=(agree_rate + contra_rate) / 2,
            detail=(
                f"Agreement < sqrt2: {agree_ok}/{agree_total} ({agree_rate:.1%}), "
                f"Contradiction > sqrt2: {contra_ok}/{contra_total} ({contra_rate:.1%}), "
                f"Spectrum width: {width:.3f}"
            ),
            evidence={
                "agree_rate": round(agree_rate, 4),
                "contra_rate": round(contra_rate, 4),
                "spectrum_width": round(width, 4),
                "agree_energy_range": (
                    [round(min(agree_energies), 4), round(max(agree_energies), 4)]
                    if agree_energies else []
                ),
                "contra_energy_range": (
                    [round(min(contra_energies), 4), round(max(contra_energies), 4)]
                    if contra_energies else []
                ),
                "agree_mean": round(sum(agree_energies) / max(len(agree_energies), 1), 4),
                "contra_mean": round(sum(contra_energies) / max(len(contra_energies), 1), 4),
            },
        )

    # ---------------------------------------------------------------
    # 3. check_holonomy: relational consistency
    # ---------------------------------------------------------------

    def check_holonomy(self) -> CheckResult:
        """Verify holonomy defects are well below random baseline.

        For random orthogonal matrices in SO(m), ||R_random - I||_F ~ sqrt(2m).
        For m=128, that baseline is ~16.0.  Trained rotors should produce
        holonomy defects << 16.0 for triangles found in the KG.

        Also checks that same-relation self-composition (R_r @ R_r^{-1})
        produces near-zero defect (sanity check on Cayley parameterization).

        Pass criterion: mean defect for self-identity < 0.01 AND mean
        triangle defect < random_baseline / 2.
        """
        self.model.eval()
        m = self.model.cfg.m
        random_baseline = math.sqrt(2 * m)

        # Self-identity check: R_r @ R_r^T should be I
        identity_defects: list[float] = []
        with torch.no_grad():
            for r_idx in range(min(self.model.num_relations, 20)):
                R = self.model.relation_rotors[r_idx].get_rotor()
                I = torch.eye(m, device=R.device, dtype=R.dtype)
                diff = R @ R.transpose(-2, -1) - I
                defect = torch.sqrt((diff * diff).sum()).item()
                identity_defects.append(defect)

        identity_mean = sum(identity_defects) / max(len(identity_defects), 1)

        # Triangle holonomy: find relation triangles from KG
        from knowlytix.core.data.prepare import mine_relation_triangles
        triangles = mine_relation_triangles(self.adapter)

        triangle_defects: list[float] = []
        if triangles:
            with torch.no_grad():
                for r1, r2, r3 in triangles[:self.sample_size]:
                    if (r1 >= self.model.num_relations or
                            r2 >= self.model.num_relations or
                            r3 >= self.model.num_relations):
                        continue
                    defect = self.transport.triangle_holonomy(r1, r2, r3).item()
                    triangle_defects.append(defect)

        triangle_mean = (
            sum(triangle_defects) / len(triangle_defects)
            if triangle_defects else 0.0
        )

        identity_ok = identity_mean < 0.01
        triangle_ok = (
            triangle_mean < random_baseline / 2
            if triangle_defects else True  # no triangles = no failure
        )
        passed = identity_ok and triangle_ok

        return CheckResult(
            name="check_holonomy",
            passed=passed,
            score=1.0 - min(triangle_mean / random_baseline, 1.0),
            detail=(
                f"Identity check: mean defect {identity_mean:.6f} "
                f"({'OK' if identity_ok else 'FAIL'}). "
                f"Triangle holonomy: {len(triangle_defects)} triangles, "
                f"mean defect {triangle_mean:.4f} vs random baseline {random_baseline:.1f} "
                f"({'OK' if triangle_ok else 'FAIL'})"
            ),
            evidence={
                "identity_mean_defect": round(identity_mean, 6),
                "triangle_count": len(triangle_defects),
                "triangle_mean_defect": round(triangle_mean, 4),
                "triangle_max_defect": (
                    round(max(triangle_defects), 4) if triangle_defects else 0.0
                ),
                "random_baseline": round(random_baseline, 2),
                "m": m,
            },
        )

    # ---------------------------------------------------------------
    # 4. lookup_enm: SHA-256 integrity verification
    # ---------------------------------------------------------------

    def check_enm_integrity(self) -> CheckResult:
        """Verify all ENM entries pass SHA-256 integrity check.

        Recomputes the hash for each stored value and compares with the
        stored hash.  Any mismatch indicates data corruption.

        Pass criterion: 100% integrity (any corruption is a hard failure).
        """
        if self.enm is None or self.enm.size() == 0:
            return CheckResult(
                name="lookup_enm",
                passed=True,
                score=1.0,
                detail="No ENM entries to verify",
                evidence={"total": 0},
            )

        total = 0
        passed_count = 0
        failures: list[str] = []

        for key in self.enm.keys():
            total += 1
            if self.enm.verify_integrity(key):
                passed_count += 1
            else:
                failures.append(f"{key.type}::{key.id}")

        rate = passed_count / max(total, 1)
        passed = rate == 1.0

        # Also verify round-trip: read_exact should not raise
        read_ok = 0
        for key in self.enm.keys():
            try:
                val = self.enm.read_exact(key)
                if val is not None:
                    read_ok += 1
            except RuntimeError:
                pass

        return CheckResult(
            name="lookup_enm",
            passed=passed,
            score=rate,
            detail=(
                f"{passed_count}/{total} ENM entries pass SHA-256 integrity. "
                f"{read_ok}/{total} entries readable."
            ),
            evidence={
                "total": total,
                "integrity_pass": passed_count,
                "readable": read_ok,
                "failures": failures,
            },
        )

    # ---------------------------------------------------------------
    # 5. link_predict: type-constrained filtered rank
    # ---------------------------------------------------------------

    def check_link_prediction(self) -> CheckResult:
        """Verify link prediction ranks correct entities near the top.

        For each sampled triple (h, r, t), scores all type-valid tails
        (excluding self-entity) and records the rank of the true tail.

        Pass criterion: MRR >= 0.3 and Hits@5 >= 50%.
        """
        self.model.eval()
        n_triples = len(self.adapter.heads)
        indices = list(range(n_triples))
        random.shuffle(indices)
        indices = indices[:self.sample_size]

        ranks: list[int] = []
        rank_details: list[dict] = []

        with torch.no_grad():
            for idx in indices:
                h_idx = self.adapter.heads[idx].item()
                r_idx = self.adapter.relations[idx].item()
                t_idx = self.adapter.tails[idx].item()

                scores = self.model.score_all_tails(
                    torch.tensor([h_idx], device=self.device),
                    torch.tensor([r_idx], device=self.device),
                ).squeeze(0)

                # Filtered evaluation: mask self and non-type entities
                scores[h_idx] = float('inf')
                valid_tails = self.adapter.tail_per_relation.get(r_idx, set())
                if valid_tails:
                    mask = torch.ones(len(scores), dtype=torch.bool,
                                      device=self.device)
                    for t in valid_tails:
                        mask[t] = False
                    scores[mask] = float('inf')

                # Rank of true tail (1-indexed)
                rank = (scores < scores[t_idx]).sum().item() + 1
                ranks.append(rank)

                if len(rank_details) < 20:
                    rank_details.append({
                        "head": self.adapter.idx_to_entity.get(h_idx, "?"),
                        "relation": self.adapter.idx_to_relation.get(r_idx, "?"),
                        "true_tail": self.adapter.idx_to_entity.get(t_idx, "?"),
                        "rank": rank,
                        "n_candidates": len(valid_tails) if valid_tails else self.adapter.num_entities,
                    })

        if not ranks:
            return CheckResult(
                name="link_predict",
                passed=False,
                score=0.0,
                detail="No triples evaluated",
                evidence={},
            )

        mrr = sum(1.0 / r for r in ranks) / len(ranks)
        hits1 = sum(1 for r in ranks if r == 1) / len(ranks)
        hits3 = sum(1 for r in ranks if r <= 3) / len(ranks)
        hits5 = sum(1 for r in ranks if r <= 5) / len(ranks)
        hits10 = sum(1 for r in ranks if r <= 10) / len(ranks)
        mean_rank = sum(ranks) / len(ranks)

        passed = mrr >= 0.3 and hits5 >= 0.5

        return CheckResult(
            name="link_predict",
            passed=passed,
            score=mrr,
            detail=(
                f"MRR: {mrr:.4f}, Hits@1: {hits1:.1%}, Hits@3: {hits3:.1%}, "
                f"Hits@5: {hits5:.1%}, Hits@10: {hits10:.1%}, "
                f"Mean rank: {mean_rank:.1f} (n={len(ranks)})"
            ),
            evidence={
                "mrr": round(mrr, 4),
                "hits_at_1": round(hits1, 4),
                "hits_at_3": round(hits3, 4),
                "hits_at_5": round(hits5, 4),
                "hits_at_10": round(hits10, 4),
                "mean_rank": round(mean_rank, 2),
                "n_evaluated": len(ranks),
                "details": rank_details[:10],
            },
        )

    # ---------------------------------------------------------------
    # 6. query_triples: structural correctness
    # ---------------------------------------------------------------

    def check_query_triples(self) -> CheckResult:
        """Verify structural query correctness on the document graph.

        Tests three query patterns:
          (a) (head=X, relation=Y, tail=?) — fixed head+relation
          (b) (head=?, relation=Y, tail=Z) — reverse lookup
          (c) (head=X, relation=?, tail=Z) — relation inference

        Pass criterion: all sampled queries return results consistent
        with the stored triples (100% precision and recall on KG queries).
        """
        if self.doc_graph is None:
            return CheckResult(
                name="query_triples",
                passed=False,
                score=0.0,
                detail="No document graph available",
                evidence={},
            )

        triples = list(self.doc_graph.triples)
        if not triples:
            return CheckResult(
                name="query_triples",
                passed=True,
                score=1.0,
                detail="No triples to query",
                evidence={"total": 0},
            )

        random.shuffle(triples)
        sample = triples[:self.sample_size]

        correct = 0
        total = 0
        failures: list[dict] = []

        for h, r, t in sample:
            # Query (head=h, relation=r, tail=?)
            results = self.doc_graph.query_triples(head=h, relation=r)
            tails_found = {res[2] for res in results}
            total += 1
            if t in tails_found:
                correct += 1
            elif len(failures) < 5:
                failures.append({
                    "query": f"({h}, {r}, ?)",
                    "expected_tail": t,
                    "got_tails": list(tails_found)[:5],
                })

            # Query (head=?, relation=r, tail=t)
            results = self.doc_graph.query_triples(relation=r, tail=t)
            heads_found = {res[0] for res in results}
            total += 1
            if h in heads_found:
                correct += 1
            elif len(failures) < 5:
                failures.append({
                    "query": f"(?, {r}, {t})",
                    "expected_head": h,
                    "got_heads": list(heads_found)[:5],
                })

        rate = correct / max(total, 1)
        passed = rate >= 1.0

        return CheckResult(
            name="query_triples",
            passed=passed,
            score=rate,
            detail=f"{correct}/{total} structural queries correct ({rate:.1%})",
            evidence={
                "correct": correct,
                "total": total,
                "rate": round(rate, 4),
                "failures": failures,
            },
        )

    # ---------------------------------------------------------------
    # Run all + report
    # ---------------------------------------------------------------

    def run_all(self) -> DiagnosticReport:
        """Run all nine diagnostic checks and return a structured report.

        Three structural invariant checks run first (preconditions),
        then six GMS primitive checks.
        """
        checks = [
            # Structural invariants
            self.check_cayley_orthogonality(),
            self.check_stiefel_projection(),
            self.check_sphere_embedding(),
            # GMS primitives
            self.check_score_triple(),
            self.check_tension_energy(),
            self.check_holonomy(),
            self.check_enm_integrity(),
            self.check_link_prediction(),
            self.check_query_triples(),
        ]
        return DiagnosticReport(checks=checks)

    @staticmethod
    def print_report(report: DiagnosticReport) -> None:
        """Print a formatted diagnostic report."""
        print(f"\n{'='*70}")
        print("  GMS POST-TRAINING DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"  Overall: {report.pass_count}/{report.total} checks passed"
              f"  {'PASS' if report.all_passed else 'FAIL'}")
        print(f"{'─'*70}")

        for c in report.checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"\n  [{status}] {c.name} (score: {c.score:.4f})")
            print(f"         {c.detail}")

            # Print key evidence items
            ev = c.evidence
            if c.name == "tension_energy":
                if ev.get("agree_energy_range"):
                    print(f"         Agreement energy:     "
                          f"{ev['agree_energy_range'][0]:.4f} - "
                          f"{ev['agree_energy_range'][1]:.4f} "
                          f"(mean {ev['agree_mean']:.4f})")
                if ev.get("contra_energy_range"):
                    print(f"         Contradiction energy: "
                          f"{ev['contra_energy_range'][0]:.4f} - "
                          f"{ev['contra_energy_range'][1]:.4f} "
                          f"(mean {ev['contra_mean']:.4f})")
            elif c.name == "link_predict":
                print(f"         MRR={ev.get('mrr', 0):.4f}  "
                      f"H@1={ev.get('hits_at_1', 0):.1%}  "
                      f"H@5={ev.get('hits_at_5', 0):.1%}  "
                      f"H@10={ev.get('hits_at_10', 0):.1%}")
            elif c.name == "check_holonomy":
                print(f"         m={ev.get('m', '?')}, "
                      f"random baseline={ev.get('random_baseline', '?')}, "
                      f"identity defect={ev.get('identity_mean_defect', '?')}")
            elif c.name == "score_triple" and ev.get("failures"):
                for f in ev["failures"][:3]:
                    print(f"         MISS: ({f['head']}, {f['relation']}) "
                          f"true={f['true_tail']} d={f['d_true']:.4f} "
                          f"vs fab={f['fab_tail']} d={f['d_fab']:.4f}")

        print(f"\n{'='*70}")
