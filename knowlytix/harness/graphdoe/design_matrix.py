# SPDX-License-Identifier: Apache-2.0
"""
Design Matrix Generation for DOE experiments.

Uses quasi-Monte Carlo (Sobol) or Latin Hypercube Sampling to generate
space-filling experimental designs across factor dimensions.

Adapted from DQEval (ga_verify/doe/dqeval_package/dqeval/api.py).
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
        Sampling method: "sobol", "lhs", or "grid".
    n_runs : int
        Number of experimental runs (rows in design matrix).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        factors: list[dict],
        method: str = "sobol",
        n_runs: int = 32,
        seed: int = 42,
    ):
        self.factors = factors
        self.method = method
        self.n_runs = n_runs
        self.seed = seed
        self._matrix: pd.DataFrame | None = None

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
            sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
            # Sobol generates 2^m samples; take nearest power of 2 >= n_runs
            m = max(1, math.ceil(math.log2(self.n_runs)))
            samples = sampler.random_base2(m=m)
            return samples[: self.n_runs]

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
            raise ValueError(f"Unknown method: {self.method}. Use sobol/lhs/grid.")

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
