"""Tests for proxy-shuffle negative-control helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from experiments._proxy_shuffle_control import (  # noqa: E402
    apply_proxy_shuffle_to_setup,
    build_proxy_permutation,
    ensure_safe_negative_control_output_root,
)


class ProxyShuffleNegativeControlTest(unittest.TestCase):
    """Validate proxy-only shuffling and output-root guardrails."""

    @staticmethod
    def _panel() -> SimpleNamespace:
        return SimpleNamespace(
            X_it=torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
            Y_it=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            p_i=torch.tensor(
                [
                    [0.0, 10.0],
                    [1.0, 11.0],
                    [2.0, 12.0],
                    [3.0, 13.0],
                ],
                dtype=torch.float32,
            ),
            s_i=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            entity_ids=torch.arange(4, dtype=torch.long),
            time_index=torch.arange(3, dtype=torch.long),
            entity_codes=["AAA", "BBB", "CCC", "DDD"],
            entity_names=["Alpha", "Beta", "Gamma", "Delta"],
            metadata={"domain": "toy", "years": [1, 2, 3]},
        )

    def test_proxy_shuffle_changes_only_proxy_entity_pairing(self) -> None:
        setup = SimpleNamespace(
            full_panel=self._panel(),
            train_panel=self._panel(),
            val_panel=self._panel(),
            test_panel=self._panel(),
        )
        original_proxy = setup.full_panel.p_i.clone()
        original_x = setup.full_panel.X_it.clone()
        original_y = setup.full_panel.Y_it.clone()
        original_static = setup.full_panel.s_i.clone()

        order = build_proxy_permutation(4, 10000)
        apply_proxy_shuffle_to_setup(setup, training_seed=0, proxy_perm_seed=10000)

        expected_proxy = original_proxy[torch.as_tensor(order, dtype=torch.long)]
        torch.testing.assert_close(setup.full_panel.p_i, expected_proxy)
        torch.testing.assert_close(torch.sort(setup.full_panel.p_i, dim=0).values, torch.sort(original_proxy, dim=0).values)
        torch.testing.assert_close(setup.full_panel.X_it, original_x)
        torch.testing.assert_close(setup.full_panel.Y_it, original_y)
        torch.testing.assert_close(setup.full_panel.s_i, original_static)
        self.assertEqual(setup.full_panel.entity_codes, ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(setup.full_panel.metadata["negative_control"], "proxy_shuffle")
        self.assertEqual(setup.full_panel.metadata["training_seed"], 0)
        self.assertEqual(setup.full_panel.metadata["proxy_perm_seed"], 10000)
        self.assertEqual(setup.full_panel.metadata["proxy_permutation_order"], order.tolist())

    def test_proxy_permutation_is_non_identity(self) -> None:
        order = build_proxy_permutation(2, 0)
        self.assertNotEqual(order.tolist(), [0, 1])
        self.assertEqual(sorted(order.tolist()), [0, 1])

    def test_output_root_guard_rejects_locked_and_notebook_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            safe_root = temporary_root / "outputs" / "negative_controls" / "proxy_shuffle"
            self.assertEqual(ensure_safe_negative_control_output_root(safe_root), safe_root.resolve())

            with self.assertRaises(ValueError):
                ensure_safe_negative_control_output_root(
                    temporary_root / "outputs" / "notebook_economics" / "complete_20seed_20260426"
                )
            with self.assertRaises(ValueError):
                ensure_safe_negative_control_output_root(
                    temporary_root / "outputs" / "notebook_energy" / "complete_20seed"
                )


if __name__ == "__main__":
    unittest.main()