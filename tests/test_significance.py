"""Tests for paired significance comparison helpers."""

import unittest

import pandas as pd

from evaluation.economics_comparison import build_significance_tables as build_realdata_significance_tables
from evaluation.significance import paired_wilcoxon
from evaluation.synthetic_comparison import build_significance_tables as build_synthetic_significance_tables


class SignificanceTest(unittest.TestCase):
    def test_paired_wilcoxon_groups_by_seed_and_domain_keys(self) -> None:
        rows = []
        for seed in range(5):
            rows.append({"scenario": "linear", "seed": seed, "display_name": "CMDL", "kstar_mae": 1.0 + seed * 0.01})
            rows.append({"scenario": "linear", "seed": seed, "display_name": "Plain LSTM", "kstar_mae": 2.0 + seed * 0.01})
        result = paired_wilcoxon(
            pd.DataFrame(rows),
            metric="kstar_mae",
            method_col="display_name",
            seed_col="seed",
            reference="CMDL",
            group_cols=["scenario"],
            greater_is_better=False,
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["scenario"], "linear")
        self.assertEqual(row["method"], "Plain LSTM")
        self.assertEqual(row["n_pairs"], 5)
        self.assertLess(row["mean_diff"], 0.0)
        self.assertTrue(bool(row["reference_better_mean"]))
        self.assertGreaterEqual(row["wilcoxon_p"], 0.0)

    def test_domain_builders_return_expected_tables(self) -> None:
        synthetic_rows = []
        real_rows = []
        for seed in range(5):
            synthetic_rows.extend(
                [
                    {"scenario": "linear", "seed": seed, "display_name": "CMDL", "effective_kstar_mae": 1.0, "task_loss": 0.1},
                    {"scenario": "linear", "seed": seed, "display_name": "Plain LSTM", "effective_kstar_mae": 2.0, "task_loss": 0.2},
                ]
            )
            real_rows.extend(
                [
                    {
                        "target_column": "ctfp",
                        "feature_bundle": "minimal",
                        "seed": seed,
                        "display_name": "CMDL",
                        "test_r2": 0.2,
                        "test_effective_kstar_proxy_spearman_adjusted_rho": 0.3,
                    },
                    {
                        "target_column": "ctfp",
                        "feature_bundle": "minimal",
                        "seed": seed,
                        "display_name": "Plain LSTM",
                        "test_r2": 0.1,
                        "test_effective_kstar_proxy_spearman_adjusted_rho": 0.1,
                    },
                ]
            )

        synthetic_tables = build_synthetic_significance_tables(pd.DataFrame(synthetic_rows))
        real_tables = build_realdata_significance_tables(pd.DataFrame(real_rows), domain_prefix="economics")

        self.assertIn("synthetic_significance_kstar_mae.csv", synthetic_tables)
        self.assertIn("economics_significance_test_r2.csv", real_tables)
        self.assertEqual(int(synthetic_tables["synthetic_significance_kstar_mae.csv"].iloc[0]["n_pairs"]), 5)
        self.assertEqual(int(real_tables["economics_significance_test_r2.csv"].iloc[0]["n_pairs"]), 5)


if __name__ == "__main__":
    unittest.main()
