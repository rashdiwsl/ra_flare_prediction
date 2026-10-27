import unittest

import pandas as pd

from evaluate_flaredown_temporal import (
    _metric_summary,
    make_temporal_patient_split,
)


class TemporalPatientSplitTests(unittest.TestCase):
    def test_split_is_patient_disjoint_and_respects_full_label_horizon(self):
        frame = pd.DataFrame(
            {
                "user_id": [
                    "early-a", "early-a", "early-b", "early-b",
                    "crossing", "crossing", "late-a", "late-a",
                    "late-b", "late-b",
                ],
                "checkpoint_date": [
                    "2024-01-01", "2024-01-02",
                    "2024-01-01", "2024-01-02",
                    "2024-01-02", "2024-01-10",
                    "2024-01-10", "2024-01-11",
                    "2024-01-10", "2024-01-11",
                ],
            }
        )
        train, test, cutoff = make_temporal_patient_split(
            frame, test_date_fraction=0.5
        )

        train_ids = set(frame.loc[train, "user_id"])
        test_ids = set(frame.loc[test, "user_id"])
        self.assertEqual(train_ids, {"early-a", "early-b"})
        self.assertEqual(test_ids, {"late-a", "late-b"})
        self.assertFalse(train_ids.intersection(test_ids))
        self.assertEqual(cutoff, pd.Timestamp("2024-01-10"))
        self.assertLess(
            pd.to_datetime(frame.loc[train, "checkpoint_date"]).max()
            + pd.Timedelta(days=3),
            cutoff,
        )
        self.assertGreaterEqual(
            pd.to_datetime(frame.loc[test, "checkpoint_date"]).min(),
            cutoff,
        )

    def test_rejects_missing_dates_and_patient_ids(self):
        base = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "checkpoint_date": [
                    "2024-01-01", "2024-01-02",
                    "2024-01-03", "2024-01-04",
                ],
            }
        )
        missing_date = base.copy()
        missing_date.loc[0, "checkpoint_date"] = None
        with self.assertRaisesRegex(ValueError, "valid"):
            make_temporal_patient_split(missing_date)

        missing_patient = base.copy()
        missing_patient.loc[0, "user_id"] = None
        with self.assertRaisesRegex(ValueError, "Patient IDs"):
            make_temporal_patient_split(missing_patient)

    def test_baseline_threshold_preserves_strict_target_boundary(self):
        metrics = _metric_summary(
            pd.Series([0, 1, 1]).to_numpy(),
            pd.Series([0.0, 0.1, -0.1]).to_numpy(),
            0.0,
            inclusive=False,
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
