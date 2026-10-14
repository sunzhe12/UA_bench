import random
import unittest

import numpy as np

from testsite.core import statistics as st
from testsite.core.scorer import Scorer

L3_KEYS = ["CW", "LFM", "HFM", "2FSK", "4FSK", "BPSK", "QPSK", "OFDM",
           "cargo", "cruise", "fishing", "warship", "underwater_target"]
TAXONOMY = {
    "L1": {"classes": [{"key": "active", "name": "Active"}, {"key": "passive", "name": "Passive"}]},
    "L2": {"classes": [{"key": "pulse", "name": "P"}, {"key": "communication", "name": "C"},
                       {"key": "ship_noise", "name": "S"}]},
    "L3": {"classes": [{"key": k, "name": k} for k in L3_KEYS]},
}


def _counts(pred, true):
    return st._joint_counts(st.encode(pred, L3_KEYS), st.encode(true, L3_KEYS), len(L3_KEYS))


class StatisticsTests(unittest.TestCase):
    def test_accuracy_and_macro_f1_reproduce_scorer(self):
        scorer = Scorer({"taxonomy": TAXONOMY})
        rng = random.Random(7)
        for _ in range(8):
            n = rng.choice([5, 17, 60, 200])
            true = [rng.choice(L3_KEYS) for _ in range(n)]
            pred = []
            for t in true:
                r = rng.random()
                pred.append(t if r < 0.55 else (rng.choice(L3_KEYS) if r < 0.9 else "cascade_error"))
            level = scorer.compute_level("L3", L3_KEYS, pred, true)
            accuracy, macro_f1 = st.accuracy_and_macro_f1(_counts(pred, true), len(L3_KEYS))
            self.assertAlmostEqual(accuracy, level.accuracy, places=12)
            self.assertAlmostEqual(macro_f1, level.macro_f1, places=12)

    def test_cascade_error_counts_as_wrong_not_excluded(self):
        true = ["CW", "CW"]
        pred = ["CW", "cascade_error"]
        accuracy, _ = st.accuracy_and_macro_f1(_counts(pred, true), len(L3_KEYS))
        self.assertAlmostEqual(accuracy, 0.5)

    def test_macro_f1_ignores_classes_without_support(self):
        scorer = Scorer({"taxonomy": TAXONOMY})
        true = ["CW"] * 4
        pred = ["CW"] * 4
        level = scorer.compute_level("L3", L3_KEYS, pred, true)
        _, macro_f1 = st.accuracy_and_macro_f1(_counts(pred, true), len(L3_KEYS))
        self.assertAlmostEqual(macro_f1, 1.0)
        self.assertAlmostEqual(macro_f1, level.macro_f1)

    def test_wilson_matches_known_values(self):
        lo, hi = st.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.403831, places=5)
        self.assertAlmostEqual(hi, 0.596169, places=5)

    def test_wilson_degenerate_and_bounds(self):
        self.assertEqual(st.wilson_interval(0, 20), (0.0, st.wilson_interval(0, 20)[1]))
        self.assertEqual(st.wilson_interval(0, 20)[0], 0.0)
        lo, hi = st.wilson_interval(20, 20)
        self.assertEqual(hi, 1.0)
        self.assertGreater(lo, 0.8)

    def test_channel_variants_form_one_cluster(self):
        codes, n = st.cluster_membership(["cw_000001_ch0", "cw_000001_ch1", "cw_000002_ch0"])
        self.assertEqual(n, 2)
        self.assertEqual(codes[0], codes[1])
        self.assertNotEqual(codes[0], codes[2])

    def test_bootstrap_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(3)
        true = rng.integers(0, len(L3_KEYS), 300)
        pa = np.where(rng.random(300) < 0.6, true, rng.integers(0, len(L3_KEYS), 300))
        pb = np.where(rng.random(300) < 0.4, true, rng.integers(0, len(L3_KEYS), 300))
        ids = [f"src_{i // 2}_ch{i % 2}" for i in range(300)]
        codes, n_clusters = st.cluster_membership(ids)
        first = st.cluster_bootstrap(pa, pb, true, len(L3_KEYS), codes, n_clusters,
                                     replicates=200, seed=11)
        second = st.cluster_bootstrap(pa, pb, true, len(L3_KEYS), codes, n_clusters,
                                      replicates=200, seed=11)
        self.assertEqual(first, second)

    def test_bootstrap_collapses_when_everything_is_correct(self):
        true = np.arange(100) % len(L3_KEYS)
        ids = [f"src_{i}_ch0" for i in range(100)]
        codes, n_clusters = st.cluster_membership(ids)
        out = st.cluster_bootstrap(true.copy(), true.copy(), true, len(L3_KEYS), codes, n_clusters,
                                   replicates=100, seed=5)
        self.assertAlmostEqual(out["accuracy"]["original"]["estimate"], 1.0)
        self.assertEqual(out["accuracy"]["original"]["ci95"], [1.0, 1.0])
        self.assertAlmostEqual(out["accuracy"]["delta"]["estimate"], 0.0)

    def test_delta_can_be_negative_and_ci_brackets_it(self):
        rng = np.random.default_rng(1)
        true = rng.integers(0, len(L3_KEYS), 400)
        ids = [f"src_{i // 2}_ch{i % 2}" for i in range(400)]
        codes, n_clusters = st.cluster_membership(ids)
        # silent is clearly better -> delta is negative
        pa = np.where(rng.random(400) < 0.2, true, (true + 1) % len(L3_KEYS))
        pb = np.where(rng.random(400) < 0.8, true, (true + 1) % len(L3_KEYS))
        out = st.cluster_bootstrap(pa, pb, true, len(L3_KEYS), codes, n_clusters,
                                   replicates=300, seed=9)
        delta = out["accuracy"]["delta"]
        self.assertLess(delta["estimate"], 0)
        self.assertLessEqual(delta["ci95"][1], 0)
        self.assertLess(delta["ci95"][0], delta["ci95"][1])

    def test_analytic_difference_agrees_with_bootstrap_direction(self):
        disc = {"original_only": 120, "silent_only": 30, "both_correct": 500, "neither_correct": 50}
        lo, hi = st.paired_difference_interval(disc)
        self.assertGreater(lo, 0)
        self.assertAlmostEqual((lo + hi) / 2, (120 - 30) / 700, places=12)

    def test_summarise_reports_cluster_structure_and_method(self):
        rows = []
        for i in range(120):
            sid = f"src_{i // 2}_ch{i % 2}"
            gt = L3_KEYS[i % len(L3_KEYS)]
            rows.append({"sample_id": sid, "gt": {"L3": gt},
                         "original": {"L3": gt if i % 3 else L3_KEYS[(i + 1) % len(L3_KEYS)]},
                         "silent": {"L3": gt if i % 5 else L3_KEYS[(i + 2) % len(L3_KEYS)]}})
        out = st.summarise(rows, L3_KEYS, replicates=200, seed=4)
        self.assertEqual(out["sample_structure"], {"samples": 120, "clusters": 60, "max_cluster_size": 2})
        self.assertEqual(out["method"]["resampling_unit"],
                         "source_id (channel variants of one source resampled together)")
        for group in ("accuracy", "macro_f1"):
            for key in ("original", "silent", "delta"):
                lo, hi = out[group][key]["ci95"]
                self.assertLessEqual(lo, hi)
                self.assertTrue(-1.0001 <= lo <= 1.0001 and -1.0001 <= hi <= 1.0001)
        self.assertIn("wilson_ci95", out["accuracy"]["original"])
        self.assertIn("discordant_pairs", out["accuracy"]["delta"])

    def test_bootstrap_rejects_bad_inputs(self):
        codes, n_clusters = st.cluster_membership(["a", "b"])
        with self.assertRaises(ValueError):
            st.cluster_bootstrap(np.array([0, 1]), np.array([0]), np.array([0, 1]),
                                 len(L3_KEYS), codes, n_clusters, replicates=10)
        with self.assertRaises(ValueError):
            st.cluster_bootstrap(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                                 len(L3_KEYS), codes, 0, replicates=10)


if __name__ == "__main__":
    unittest.main()
