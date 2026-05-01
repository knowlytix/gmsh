# SPDX-License-Identifier: Apache-2.0
"""
DOE Statistical Analysis — Logistic regression, analysis of deviance,
odds ratios, and factor attribution for binary outcomes.

The response variable in DOE benchmark testing is binary
(correct/incorrect).  ANOVA assumes continuous normal residuals and is
inappropriate.  This module uses:

  1. Logistic regression (GLM, binomial/logit) as the primary analysis
  2. Analysis of deviance (likelihood-ratio chi-squared) for factor
     significance — the proper analog of ANOVA for binary data
  3. Odds ratios for effect sizes — how much each factor level shifts
     the odds of a correct answer relative to the reference level
  4. McFadden's pseudo-R² as the variance-explained analog of eta²
  5. Per-level proportions with Wilson confidence intervals

Legacy ANOVA (run_anova) is retained for back-compatibility but is
no longer the recommended analysis.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .results import DOEBenchmarkResult


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


class DOEAnalyzer:
    """Statistical analysis of DOE benchmark results.

    Parameters
    ----------
    result : DOEBenchmarkResult
        Results from DOEBenchmark.run().
    """

    def __init__(self, result: DOEBenchmarkResult):
        self.result = result
        self.df = result.to_dataframe()
        self.factors = result.factors

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, factors: list[str]) -> "DOEAnalyzer":
        """Create analyzer from a pre-built DataFrame."""
        result = DOEBenchmarkResult(factors=factors)
        analyzer = cls.__new__(cls)
        analyzer.result = result
        analyzer.df = df
        analyzer.factors = factors
        return analyzer

    # ------------------------------------------------------------------
    # Primary analysis: logistic regression + analysis of deviance
    # ------------------------------------------------------------------

    def run_logistic(self, metric: str = "correct") -> pd.DataFrame:
        """Logistic regression analysis of deviance for each factor.

        For each factor, fits a logistic GLM (binomial/logit):
            logit(P(correct=1)) = β₀ + β₁·X_factor

        Tests significance via likelihood-ratio (deviance) chi-squared:
            G² = -2(LL_null - LL_model)  ~  χ²(df = n_levels - 1)

        Returns
        -------
        pd.DataFrame
            Columns: factor, deviance, p_value, pseudo_r2, significant,
            n_levels, level_rates, odds_ratios
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            warnings.warn(
                "statsmodels not installed; falling back to ANOVA. "
                "Install with: pip install statsmodels"
            )
            return self.run_anova(metric)

        rows = []
        for factor in self.factors:
            if factor not in self.df.columns:
                continue

            levels = sorted(self.df[factor].unique())
            if len(levels) < 2:
                continue

            y = self.df[metric].values.astype(float)

            # Check for zero variance (all correct or all wrong)
            if y.std() == 0:
                level_rates = {}
                for lev in levels:
                    mask = self.df[factor] == lev
                    k = int(y[mask].sum())
                    n = int(mask.sum())
                    ci_lo, ci_hi = _wilson_ci(k, n)
                    level_rates[lev] = {
                        "rate": k / n if n > 0 else 0.0,
                        "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    }
                rows.append({
                    "factor": factor,
                    "deviance": float("nan"),
                    "p_value": float("nan"),
                    "pseudo_r2": 0.0,
                    "significant": False,
                    "n_levels": len(levels),
                    "level_rates": level_rates,
                    "odds_ratios": {},
                })
                continue

            # Dummy-encode factor (reference = first level)
            dummies = pd.get_dummies(
                self.df[factor], prefix=factor, drop_first=True, dtype=float
            )

            # Null model (intercept only)
            X_null = np.ones((len(y), 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                null_model = sm.GLM(
                    y, X_null, family=sm.families.Binomial()
                ).fit(disp=0)

            # Full model (intercept + factor dummies)
            X_full = sm.add_constant(dummies.values, has_constant=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_model = sm.GLM(
                    y, X_full, family=sm.families.Binomial()
                ).fit(disp=0)

            # Likelihood-ratio (deviance) test
            G2 = null_model.deviance - full_model.deviance
            df_diff = len(levels) - 1
            p_value = 1.0 - stats.chi2.cdf(G2, df_diff)

            # McFadden's pseudo-R²
            pseudo_r2 = 1.0 - (full_model.llf / null_model.llf) if null_model.llf != 0 else 0.0

            # Per-level rates with Wilson CIs
            level_rates = {}
            for lev in levels:
                mask = self.df[factor] == lev
                k = int(y[mask].sum())
                n = int(mask.sum())
                ci_lo, ci_hi = _wilson_ci(k, n)
                level_rates[lev] = {
                    "rate": k / n if n > 0 else 0.0,
                    "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi,
                }

            # Odds ratios from logistic coefficients
            odds_ratios = {}
            ref_level = levels[0]
            odds_ratios[ref_level] = 1.0  # reference
            coef_names = [f"{factor}_{lev}" for lev in levels[1:]]
            for i, lev in enumerate(levels[1:]):
                # coefficient index: 0 is intercept, 1..d are dummies
                coef = full_model.params[i + 1]
                odds_ratios[lev] = float(np.exp(coef))

            rows.append({
                "factor": factor,
                "deviance": float(G2),
                "p_value": float(p_value),
                "pseudo_r2": float(pseudo_r2),
                "significant": p_value < 0.05,
                "n_levels": len(levels),
                "level_rates": level_rates,
                "odds_ratios": odds_ratios,
            })

        return pd.DataFrame(rows).sort_values("pseudo_r2", ascending=False)

    def run_interaction_logistic(
        self,
        factor_pairs: list[tuple[str, str]] | None = None,
        metric: str = "correct",
    ) -> pd.DataFrame:
        """Two-way interaction analysis via logistic regression.

        Compares additive model (A + B) vs interaction model (A + B + A:B)
        using a likelihood-ratio test on the deviance difference.

        Parameters
        ----------
        factor_pairs : list of (str, str) or None
            Factor pairs to test.  None tests all pairs of significant
            factors from run_logistic().
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            return self.run_interaction_anova(factor_pairs, metric)

        if factor_pairs is None:
            logistic = self.run_logistic(metric)
            sig = logistic[logistic["significant"]]["factor"].tolist()
            factor_pairs = list(combinations(sig, 2))

        rows = []
        for f1, f2 in factor_pairs:
            if f1 not in self.df.columns or f2 not in self.df.columns:
                continue

            y = self.df[metric].values.astype(float)
            if y.std() == 0:
                continue

            d1 = pd.get_dummies(self.df[f1], prefix=f1, drop_first=True, dtype=float)
            d2 = pd.get_dummies(self.df[f2], prefix=f2, drop_first=True, dtype=float)

            # Additive model: f1 + f2
            X_add = sm.add_constant(
                pd.concat([d1, d2], axis=1).values, has_constant=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                additive = sm.GLM(y, X_add, family=sm.families.Binomial()).fit(disp=0)

            # Interaction model: f1 + f2 + f1:f2
            interaction_terms = pd.DataFrame()
            for c1 in d1.columns:
                for c2 in d2.columns:
                    interaction_terms[f"{c1}:{c2}"] = d1[c1] * d2[c2]

            X_int = sm.add_constant(
                pd.concat([d1, d2, interaction_terms], axis=1).values,
                has_constant=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interact = sm.GLM(y, X_int, family=sm.families.Binomial()).fit(disp=0)

            # LR test: additive vs interaction
            G2 = additive.deviance - interact.deviance
            df_diff = len(interaction_terms.columns)
            p_value = 1.0 - stats.chi2.cdf(G2, df_diff) if df_diff > 0 else 1.0

            rows.append({
                "factor_1": f1,
                "factor_2": f2,
                "interaction_deviance": float(G2),
                "interaction_df": df_diff,
                "interaction_p_value": float(p_value),
                "significant": p_value < 0.05,
                "interaction_strength": float(G2 / len(y)) if len(y) > 0 else 0.0,
            })

        return pd.DataFrame(rows).sort_values(
            "interaction_deviance", ascending=False
        )

    # ------------------------------------------------------------------
    # Per-category and failure analysis
    # ------------------------------------------------------------------

    def by_category(self, metric: str = "correct") -> dict[str, pd.DataFrame]:
        """Run logistic analysis separately per question category.

        Returns
        -------
        dict[str, pd.DataFrame]
            {category: logistic_table}
        """
        results = {}
        for cat in sorted(self.df["category"].unique()):
            sub_df = self.df[self.df["category"] == cat]
            sub_result = DOEBenchmarkResult(factors=self.factors)
            sub_analyzer = DOEAnalyzer.__new__(DOEAnalyzer)
            sub_analyzer.result = sub_result
            sub_analyzer.df = sub_df
            sub_analyzer.factors = self.factors
            table = sub_analyzer.run_logistic(metric)
            if not table.empty:
                results[cat] = table
        return results

    def failure_analysis(self) -> pd.DataFrame:
        """Per-level failure rates with Wilson confidence intervals.

        Returns
        -------
        pd.DataFrame
            Columns: factor, level, n_total, n_failed, failure_rate,
            ci_lo, ci_hi
        """
        rows = []
        for factor in self.factors:
            if factor not in self.df.columns:
                continue
            for level in sorted(self.df[factor].unique()):
                sub = self.df[self.df[factor] == level]
                n_total = len(sub)
                n_correct = int(sub["correct"].sum())
                n_failed = n_total - n_correct
                ci_lo, ci_hi = _wilson_ci(n_failed, n_total)
                rows.append({
                    "factor": factor,
                    "level": level,
                    "n_total": n_total,
                    "n_failed": n_failed,
                    "failure_rate": n_failed / n_total if n_total > 0 else 0.0,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                })
        return (
            pd.DataFrame(rows)
            .sort_values("failure_rate", ascending=False)
        )

    # ------------------------------------------------------------------
    # Legacy ANOVA (kept for back-compatibility)
    # ------------------------------------------------------------------

    def run_anova(self, metric: str = "correct") -> pd.DataFrame:
        """One-way ANOVA for each factor (legacy — use run_logistic for
        binary outcomes).

        Treats the binary correct/incorrect response as continuous.
        This is a linear probability model approximation and is retained
        for back-compatibility.  For proper analysis of binary data, use
        run_logistic().
        """
        rows = []
        for factor in self.factors:
            if factor not in self.df.columns:
                continue

            groups = []
            levels = sorted(self.df[factor].unique())
            level_means = {}
            for level in levels:
                vals = self.df.loc[self.df[factor] == level, metric].values
                groups.append(vals)
                level_means[level] = float(np.mean(vals)) if len(vals) > 0 else 0.0

            if len(groups) < 2 or any(len(g) == 0 for g in groups):
                continue

            f_stat, p_value = stats.f_oneway(*groups)

            # Eta-squared: SS_between / SS_total
            grand_mean = self.df[metric].mean()
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
            )
            ss_total = sum((self.df[metric] - grand_mean) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            rows.append({
                "factor": factor,
                "f_stat": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_sq),
                "significant": p_value < 0.05,
                "n_levels": len(levels),
                "level_means": level_means,
            })

        return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)

    def run_interaction_anova(
        self,
        factor_pairs: list[tuple[str, str]] | None = None,
        metric: str = "correct",
    ) -> pd.DataFrame:
        """Two-way interaction analysis (legacy ANOVA-based)."""
        if factor_pairs is None:
            anova = self.run_anova(metric)
            sig = anova[anova["significant"]]["factor"].tolist()
            factor_pairs = list(combinations(sig, 2))

        rows = []
        for f1, f2 in factor_pairs:
            if f1 not in self.df.columns or f2 not in self.df.columns:
                continue

            grouped = self.df.groupby([f1, f2])[metric].mean().unstack(fill_value=0)
            f1_effects = grouped.max(axis=1) - grouped.min(axis=1)
            interaction_strength = float(f1_effects.std())

            cell_means = (
                self.df.groupby([f1, f2])[metric]
                .agg(["mean", "count"])
                .reset_index()
            )

            rows.append({
                "factor_1": f1,
                "factor_2": f2,
                "interaction_strength": interaction_strength,
                "cell_means": cell_means.to_dict("records"),
            })

        return pd.DataFrame(rows).sort_values(
            "interaction_strength", ascending=False
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self):
        """Print a comprehensive analysis report."""
        print(f"\n{'=' * 70}")
        print("DOE ANALYSIS REPORT")
        print(f"{'=' * 70}")
        print(f"  Total questions: {len(self.df)}")
        print(f"  Overall accuracy: {self.df['correct'].mean():.1%}")
        print(f"  Factors tested: {self.factors}")

        # Primary analysis: logistic regression
        logistic = self.run_logistic()
        print("\n--- Analysis of Deviance (Logistic Regression) ---")
        print(f"  {'Factor':<25} {'G²':>8} {'p-value':>10} {'R²':>8} {'Sig':>5}")
        print(f"  {'─' * 60}")
        for _, row in logistic.iterrows():
            sig = "*" if row["significant"] else ""
            g2 = f"{row['deviance']:.2f}" if not np.isnan(row['deviance']) else "NaN"
            pv = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else "NaN"
            print(
                f"  {row['factor']:<25} {g2:>8} "
                f"{pv:>10} {row['pseudo_r2']:>8.3f} {sig:>5}"
            )

        # Significant factors with odds ratios
        sig_factors = logistic[logistic["significant"]]
        if not sig_factors.empty:
            print("\n--- Significant Factors (p < 0.05) ---")
            for _, row in sig_factors.iterrows():
                print(f"\n  {row['factor']} (pseudo-R² = {row['pseudo_r2']:.3f}):")
                print(f"    {'Level':<25} {'Rate':>8} {'95% CI':>18} {'OR':>8}")
                print(f"    {'─' * 62}")
                for level, info in sorted(
                    row["level_rates"].items(),
                    key=lambda x: x[1]["rate"],
                ):
                    ci = f"[{info['ci_lo']:.3f}, {info['ci_hi']:.3f}]"
                    or_val = row["odds_ratios"].get(level, 1.0)
                    or_str = f"{or_val:.2f}" if or_val != 1.0 else "ref"
                    print(
                        f"    {level:<25} {info['rate']:>7.1%} "
                        f"{ci:>18} {or_str:>8}"
                    )

        # Top failure modes with CIs
        failures = self.failure_analysis()
        top_failures = failures[failures["failure_rate"] > 0].head(10)
        if not top_failures.empty:
            print("\n--- Top Failure Modes ---")
            print(f"  {'Factor':<20} {'Level':<20} {'Fail%':>7} {'95% CI':>18} {'N':>6}")
            print(f"  {'─' * 75}")
            for _, row in top_failures.iterrows():
                ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
                print(
                    f"  {row['factor']:<20} {row['level']:<20} "
                    f"{row['failure_rate']:>6.1%} {ci:>18} {row['n_total']:>6}"
                )

        # Per-category
        cat_results = self.by_category()
        if cat_results:
            print("\n--- Per-Category Significant Factors ---")
            for cat, cat_logistic in cat_results.items():
                sig = cat_logistic[cat_logistic["significant"]]
                if not sig.empty:
                    factors_str = ", ".join(
                        f"{r['factor']}(R²={r['pseudo_r2']:.2f})"
                        for _, r in sig.iterrows()
                    )
                    print(f"  {cat:<20} {factors_str}")

        print(f"\n{'=' * 70}")
