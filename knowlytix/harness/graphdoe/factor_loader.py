# SPDX-License-Identifier: Apache-2.0
"""
Factor Loader for FinStructBench-DOE.

Loads factors from the factor catalog and resolves factor groups,
individual factor selections, and custom factor definitions.

Adapted from DQEval (ga_verify/doe/dqeval_package/dqeval/factor_loader.py).
"""

from pathlib import Path

import yaml

# All factor category keys in the catalog
FACTOR_CATEGORIES = [
    "query_factors",
    "persona_factors",
    "reasoning_factors",
    "context_factors",
    "entity_factors",
    "conversation_factors",
    "adversarial_factors",
    "system_factors",
    "variability_factors",
    "robustness_factors",
]


class FactorLoader:
    """Load and resolve factors from catalog."""

    def __init__(self, catalog_path: str | None = None):
        if catalog_path is None:
            # Use bundled catalog
            catalog_path = str(
                Path(__file__).parent / "factor_catalog.yaml"
            )
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> dict:
        catalog_file = Path(self.catalog_path)
        if not catalog_file.exists():
            raise FileNotFoundError(f"Factor catalog not found: {self.catalog_path}")
        with open(catalog_file, "r") as f:
            return yaml.safe_load(f)

    def get_factor_by_name(self, factor_name: str) -> dict:
        """Look up a single factor definition by name."""
        for category in FACTOR_CATEGORIES:
            if category in self.catalog:
                for factor in self.catalog[category]:
                    if factor["name"] == factor_name:
                        return factor
        raise ValueError(f"Factor '{factor_name}' not found in catalog")

    def resolve_factor_group(self, group_name: str) -> list[str]:
        """Resolve a factor group to its list of factor names."""
        if "factor_groups" not in self.catalog:
            raise ValueError("No factor groups defined in catalog")
        groups = self.catalog["factor_groups"]
        if group_name not in groups:
            available = list(groups.keys())
            raise ValueError(
                f"Factor group '{group_name}' not found. Available: {available}"
            )
        return groups[group_name]["includes"]

    def load_factors(self, config: dict) -> list[dict]:
        """Load factors based on configuration.

        Supports:
        - use_group: Load a predefined factor group
        - use_factors: Load specific factors by name
        - custom_factors: Add custom factor definitions
        """
        factors = []

        if "use_group" in config:
            factor_names = self.resolve_factor_group(config["use_group"])
            for name in factor_names:
                factors.append(self.get_factor_by_name(name))

        if "use_factors" in config:
            for name in config["use_factors"]:
                factors.append(self.get_factor_by_name(name))

        if "custom_factors" in config:
            factors.extend(config["custom_factors"])

        # Deduplicate by name
        seen = set()
        unique = []
        for f in factors:
            if f["name"] not in seen:
                seen.add(f["name"])
                unique.append(f)
        return unique

    def load_by_group(self, group_name: str) -> list[dict]:
        """Convenience: load all factors in a named group."""
        return self.load_factors({"use_group": group_name})

    def load_by_names(self, names: list[str]) -> list[dict]:
        """Convenience: load factors by explicit name list."""
        return self.load_factors({"use_factors": names})

    def list_groups(self) -> dict[str, dict]:
        """Return all factor groups with descriptions."""
        return self.catalog.get("factor_groups", {})

    def list_factors(self) -> dict[str, list[str]]:
        """List all available factors by category."""
        available = {}
        for category in FACTOR_CATEGORIES:
            if category in self.catalog:
                available[category] = [f["name"] for f in self.catalog[category]]
        return available

    def validate_factors(self, factors: list[dict]) -> bool:
        """Validate factor definitions. Raises ValueError on issues."""
        for factor in factors:
            for key in ("name", "type"):
                if key not in factor:
                    raise ValueError(f"Factor missing '{key}': {factor}")
            if factor["type"] not in ("continuous", "categorical"):
                raise ValueError(
                    f"Invalid type '{factor['type']}' for '{factor['name']}'"
                )
            if factor["type"] == "continuous":
                if "min" not in factor or "max" not in factor:
                    raise ValueError(
                        f"Continuous factor '{factor['name']}' needs 'min' and 'max'"
                    )
            if factor["type"] == "categorical":
                if "cardinality" not in factor:
                    raise ValueError(
                        f"Categorical factor '{factor['name']}' needs 'cardinality'"
                    )
                if "categories" in factor:
                    if len(factor["categories"]) != factor["cardinality"]:
                        raise ValueError(
                            f"Factor '{factor['name']}': {len(factor['categories'])} "
                            f"categories but cardinality={factor['cardinality']}"
                        )
        return True
