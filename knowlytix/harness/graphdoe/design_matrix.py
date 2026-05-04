# SPDX-License-Identifier: Apache-2.0
"""
Design Matrix Generation for DOE experiments.

Uses quasi-Monte Carlo (Sobol) or Latin Hypercube Sampling to generate
space-filling experimental designs across factor dimensions. The
"sobol+refine" method runs GPU-accelerated phi_p optimization on top
of a Sobol initialization for stronger space-filling.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .factor_loader import FactorLoader


class DesignMatrix:
    """Generate DOE design matrices for factor experiments.

    Parameters
    ----------
    factors : list[dict]
        Factor definitions (from FactorLoader).
    method : str
        Sampling method:
        - "sobol": scrambled Sobol (scipy QMC), instant
        - "sobol+refine": Sobol init + GPU phi_p refinement
          (LBFGS for n_runs <= 500, Adam otherwise)
        - "lhs": Latin Hypercube Sampling
        - "grid": full-factorial grid (subsampled to n_runs)
    n_runs : int
        Number of experimental runs (rows in design matrix).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print optimization progress for "sobol+refine".
    """

    def __init__(
        self,
        factors: list[dict],
        method: str = "sobol",
        n_runs: int = 32,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.factors = factors
        self.method = method
        self.n_runs = n_runs
        self.seed = seed
        self.verbose = verbose
        self._matrix: pd.DataFrame | None = None
        self._raw: np.ndarray | None = None
        self._refine_info: dict | None = None
        self._quality_cache: dict | None = None

    @classmethod
    def from_catalog(
        cls,
        catalog_path: str | None = None,
        group: str | None = None,
        factor_names: list[str] | None = None,
        method: str = "sobol",
        n_runs: int = 32,
        seed: int = 42,
    ) -> "DesignMatrix":
        """Create a DesignMatrix from the factor catalog.

        Parameters
        ----------
        catalog_path : str or None
            Path to factor catalog YAML. None uses bundled catalog.
        group : str or None
            Factor group name to load.
        factor_names : list[str] or None
            Explicit factor names to load (combined with group).
        method, n_runs, seed : as in __init__
        """
        loader = FactorLoader(catalog_path)
        config = {}
        if group:
            config["use_group"] = group
        if factor_names:
            config["use_factors"] = factor_names
        if not config:
            raise ValueError("Specify at least one of group= or factor_names=")
        factors = loader.load_factors(config)
        return cls(factors, method=method, n_runs=n_runs, seed=seed)

    def generate(self) -> pd.DataFrame:
        """Generate the design matrix.

        Returns
        -------
        pd.DataFrame
            One row per run, one column per factor. Categorical factors
            are mapped to their level names; continuous factors are scaled
            to [min, max].
        """
        n_factors = len(self.factors)
        if n_factors == 0:
            raise ValueError("No factors specified")

        # Generate unit hypercube samples [0, 1]^d
        raw = self._sample_unit_cube(n_factors)
        self._raw = raw

        # Map to factor levels
        columns = {}
        for i, factor in enumerate(self.factors):
            col = raw[:, i]
            if factor["type"] == "categorical":
                categories = factor["categories"]
                k = len(categories)
                indices = np.clip((col * k).astype(int), 0, k - 1)
                columns[factor["name"]] = [categories[j] for j in indices]
            elif factor["type"] == "continuous":
                lo, hi = factor["min"], factor["max"]
                scaled = col * (hi - lo) + lo
                # Round integer-like factors
                if isinstance(lo, int) and isinstance(hi, int):
                    scaled = np.round(scaled).astype(int)
                columns[factor["name"]] = scaled
            else:
                raise ValueError(f"Unknown factor type: {factor['type']}")

        self._matrix = pd.DataFrame(columns)
        return self._matrix

    @property
    def matrix(self) -> pd.DataFrame:
        """Access the generated matrix (generates on first access)."""
        if self._matrix is None:
            self.generate()
        return self._matrix

    def _sample_unit_cube(self, d: int) -> np.ndarray:
        """Sample n_runs points in [0, 1]^d."""
        if self.method == "sobol":
            return self._sobol_samples(d)

        elif self.method == "sobol+refine":
            base = self._sobol_samples(d)
            return self._refine_phi_p(base, d)

        elif self.method == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=self.seed)
            return sampler.random(n=self.n_runs)

        elif self.method == "grid":
            # Full factorial grid with evenly spaced levels
            levels_per = max(2, round(self.n_runs ** (1.0 / d)))
            axes = [np.linspace(0, 1, levels_per) for _ in range(d)]
            grid = np.array(np.meshgrid(*axes)).T.reshape(-1, d)
            # Truncate or subsample
            if len(grid) > self.n_runs:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(len(grid), self.n_runs, replace=False)
                grid = grid[idx]
            return grid

        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                "Use sobol / sobol+refine / lhs / grid."
            )

    def _sobol_samples(self, d: int) -> np.ndarray:
        """Scrambled Sobol samples — n_runs points in [0, 1]^d."""
        sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
        # Sobol generates 2^m samples; take nearest power of 2 >= n_runs
        m = max(1, math.ceil(math.log2(self.n_runs)))
        samples = sampler.random_base2(m=m)
        return samples[: self.n_runs]

    def _refine_phi_p(self, base: np.ndarray, d: int) -> np.ndarray:
        """Refine a unit-cube design via phi_p minimization.

        Uses LBFGS in float64 for n_runs <= 500, Adam in float32 otherwise.
        Categorical-only designs get cardinality-weighted dimensions so
        higher-cardinality factors receive proportionally more spread budget.
        """
        import torch  # local import — torch is heavy

        from ._optimizer import optimize

        weights = self._dim_weights()

        if self.n_runs <= 500:
            opt_kwargs = dict(
                use_lbfgs=True, dtype=torch.float64,
                p_start=5, n_stages=5,
            )
        else:
            opt_kwargs = dict(
                use_lbfgs=False, dtype=torch.float32,
                lr=0.005, p_start=50, n_stages=1, iters_per_stage=500,
            )

        refined, _, info = optimize(
            self.n_runs, d,
            p_final=50,
            seed=self.seed,
            X_init=base,
            verbose=self.verbose,
            dim_weights=weights,
            **opt_kwargs,
        )
        self._refine_info = info
        return refined

    def _dim_weights(self) -> np.ndarray | None:
        """Cardinality-based dim weights, only when all factors are categorical.

        Returns
        -------
        np.ndarray or None
            Weights of shape (n_factors,) normalized to mean 1, or None if
            any factor is continuous (uniform weighting falls back to None).
        """
        cards = []
        for f in self.factors:
            if f["type"] == "categorical":
                cards.append(len(f["categories"]))
            else:
                # Continuous: uniform weights — keep the optimizer agnostic.
                return None
        cards_arr = np.asarray(cards, dtype=np.float64)
        return cards_arr / cards_arr.mean()

    def quality(self, p: int = 50) -> dict:
        """Compute space-filling quality metrics on the unit-cube design.

        Returns a dict with phi_p, min_dist, mean_nn, level imbalance
        (mir_avg, mir_max), and coverage. Cached after first call.
        Categorical cardinalities are used for level-based metrics;
        continuous factors get a synthetic cardinality of n_runs.
        """
        if self._quality_cache is not None:
            return self._quality_cache

        if self._raw is None:
            self.generate()

        from ._quality import compute_quality

        cardinalities = []
        for f in self.factors:
            if f["type"] == "categorical":
                cardinalities.append(len(f["categories"]))
            else:
                cardinalities.append(self.n_runs)
        cards = np.asarray(cardinalities, dtype=np.float64)

        self._quality_cache = compute_quality(self._raw, cards, p=p)
        return self._quality_cache

    def coverage_report(self) -> dict:
        """Report factor level coverage in the design matrix.

        Returns
        -------
        dict
            {factor_name: {level: count}} for categorical factors,
            {factor_name: {"min": ..., "max": ..., "mean": ...}} for continuous.
        """
        df = self.matrix
        report = {}
        for factor in self.factors:
            name = factor["name"]
            if factor["type"] == "categorical":
                report[name] = df[name].value_counts().to_dict()
            else:
                report[name] = {
                    "min": float(df[name].min()),
                    "max": float(df[name].max()),
                    "mean": float(df[name].mean()),
                }
        return report

    def __len__(self):
        return self.n_runs

    def __repr__(self):
        return (
            f"DesignMatrix(factors={len(self.factors)}, "
            f"method={self.method!r}, n_runs={self.n_runs})"
        )
