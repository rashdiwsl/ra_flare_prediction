import unittest
import numpy as np
import pandas as pd


def make_labels(future_values, baseline, multiplier=1.15):
    """Mirror the sensitivity script's three future-observation labels."""
    future = pd.Series(future_values, dtype=float)
    observed_days = int(future.notna().sum())

    if observed_days < 1:
        raise ValueError("At least one future observation is required")

    flare = int(future.mean() > baseline * multiplier)

    return {
        "future_obs_days": observed_days,
        "flare_next_3d_min1": flare,
        "flare_next_3d_min2": flare if observed_days >= 2 else np.nan,
        "flare_next_3d_min3": flare if observed_days == 3 else np.nan,
    }


class SensitivityLabelTests(unittest.TestCase):
    def test_one_observed_future_day(self):
        result = make_labels([5.0, np.nan, np.nan], baseline=2.0)
        self.assertEqual(result["future_obs_days"], 1)
        self.assertEqual(result["flare_next_3d_min1"], 1)
        self.assertTrue(np.isnan(result["flare_next_3d_min2"]))
        self.assertTrue(np.isnan(result["flare_next_3d_min3"]))

    def test_two_observed_future_days(self):
        result = make_labels([5.0, 5.0, np.nan], baseline=2.0)
        self.assertEqual(result["future_obs_days"], 2)
        self.assertEqual(result["flare_next_3d_min1"], 1)
        self.assertEqual(result["flare_next_3d_min2"], 1)
        self.assertTrue(np.isnan(result["flare_next_3d_min3"]))

    def test_three_observed_future_days(self):
        result = make_labels([5.0, 5.0, 5.0], baseline=2.0)
        self.assertEqual(result["future_obs_days"], 3)
        self.assertEqual(result["flare_next_3d_min1"], 1)
        self.assertEqual(result["flare_next_3d_min2"], 1)
        self.assertEqual(result["flare_next_3d_min3"], 1)

    def test_boundary_is_not_positive(self):
        result = make_labels([2.3, np.nan, np.nan], baseline=2.0)
        self.assertEqual(result["flare_next_3d_min1"], 0)

    def test_no_observed_future_days_is_rejected(self):
        with self.assertRaises(ValueError):
            make_labels([np.nan, np.nan, np.nan], baseline=2.0)


if __name__ == "__main__":
    unittest.main()
