import ast
import contextlib
import io
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "prepare_flaredown_v2.py"
CHECKPOINT_CONSTANTS = {
    "PAST",
    "FUT",
    "STEP",
    "MIN_BASE_OBS",
    "FLARE_MULT",
}


def load_checkpoint_builder():
    """Load only the checkpoint helper, avoiding script-level dataset work."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_patient_checkpoints":
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign):
            assigned_names = {
                target.id
                for target in node.targets
                for target in ast.walk(target)
                if isinstance(target, ast.Name)
            }
            if CHECKPOINT_CONSTANTS & assigned_names:
                selected_nodes.append(node)

    namespace = {"np": np, "pd": pd}
    exec(
        compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SCRIPT_PATH), "exec"),
        namespace,
    )
    return namespace["build_patient_checkpoints"], namespace


def load_preparation_module():
    spec = importlib.util.spec_from_file_location("prepare_flaredown_v2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder, cls.source_constants = load_checkpoint_builder()
        cls.build_checkpoint = staticmethod(builder)

    @staticmethod
    def daily(values, counts=None, start="2026-01-01"):
        dates = pd.date_range(start, periods=len(values), freq="D")
        if counts is None:
            counts = [int(pd.notna(value)) for value in values]
        return pd.DataFrame({"dt": dates, "sev": values, "cnt": counts})

    def test_past_features_and_baseline_ignore_future_observations(self):
        past = [float(value) for value in range(1, 11)]
        low_future = self.daily(past + [0.0, 0.0, 0.0])
        high_future = self.daily(past + [20.0, 20.0, 20.0])

        low_row = self.build_checkpoint("patient", low_future)[0]
        high_row = self.build_checkpoint("patient", high_future)[0]

        self.assertEqual(low_row["checkpoint_date"], "2026-01-10")
        self.assertEqual(low_row["personal_baseline"], 5.5)
        for feature in (
            "past_mean_severity",
            "past_std_severity",
            "past_trend",
            "past_last_severity",
            "past_engagement",
            "personal_baseline",
        ):
            self.assertEqual(low_row[feature], high_row[feature])
        self.assertEqual(low_row["flare_next_3d"], 0)
        self.assertEqual(high_row["flare_next_3d"], 1)

    def test_windows_include_t_minus_six_through_t_and_t_plus_one_through_three(self):
        values = [float(value) for value in range(10)] + [6.0, 7.0, 8.0]
        counts = [100, 100, 100] + [1, 2, 3, 4, 5, 6, 7] + [8, 9, 10]

        row = self.build_checkpoint("patient", self.daily(values, counts))[0]

        self.assertEqual(row["checkpoint_date"], "2026-01-10")
        self.assertEqual(row["past_mean_severity"], 6.0)
        self.assertEqual(row["past_last_severity"], 9.0)
        self.assertEqual(row["past_engagement"], 4.0)
        self.assertEqual(row["personal_baseline"], 4.5)
        self.assertEqual(row["flare_next_3d"], 1)

    def test_missing_calendar_days_are_not_filled_with_severity(self):
        daily = self.daily([2.0] * 10 + [np.nan, 2.31, np.nan])
        daily = daily[daily["dt"] != pd.Timestamp("2026-01-11")]

        row = self.build_checkpoint("patient", daily)[0]

        self.assertEqual(row["checkpoint_date"], "2026-01-10")
        self.assertEqual(row["past_engagement"], 1.0)
        self.assertEqual(row["flare_next_3d"], 1)

    def test_minimum_baseline_past_and_future_observations_are_required(self):
        fewer_than_ten_baseline_days = self.daily(
            [2.0] * 9 + [np.nan, 2.0, np.nan, np.nan]
        )
        insufficient_past_window = self.daily(
            [2.0] * 10 + [np.nan] * 6 + [2.0, np.nan, np.nan]
        )
        no_future_observation = self.daily([2.0] * 10 + [np.nan] * 3)

        self.assertEqual(self.build_checkpoint("patient", fewer_than_ten_baseline_days), [])
        self.assertEqual(self.build_checkpoint("patient", insufficient_past_window), [])
        self.assertEqual(self.build_checkpoint("patient", no_future_observation), [])

    def test_future_target_uses_strict_greater_than_boundary(self):
        at_boundary = self.daily([2.0] * 10 + [2.3, np.nan, np.nan])
        above_boundary = self.daily([2.0] * 10 + [2.31, np.nan, np.nan])

        at_boundary_row = self.build_checkpoint("patient", at_boundary)[0]
        above_boundary_row = self.build_checkpoint("patient", above_boundary)[0]

        self.assertEqual(at_boundary_row["flare_next_3d"], 0)
        self.assertEqual(above_boundary_row["flare_next_3d"], 1)


class StreamingPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preparation = load_preparation_module()

    @staticmethod
    def source_row(user_id, date, value, trackable_type="Symptom", age="40", sex="female", name=""):
        return {
            "user_id": user_id,
            "age": age,
            "sex": sex,
            "checkin_date": date,
            "trackable_type": trackable_type,
            "trackable_name": name,
            "trackable_value": value,
        }

    def fixture_rows(self):
        rows = []
        # First source row determines the profile fields for each patient.
        rows.append(self.source_row("p-b", "", "", "Condition", "52", "unknown", "RA"))
        rows.append(self.source_row("p-a", "", "", "Condition", "41", "female", "Rheumatoid arthritis"))
        rows.append(self.source_row("p-short", "", "", "Condition", "63", "", "RA"))

        for day in range(1, 32):
            date = f"2026-01-{day:02d}"
            if day != 5:
                rows.append(self.source_row("p-a", date, "2"))
            rows.append(self.source_row("p-b", date, "5", age="99", sex="male"))
            if day <= 29:
                rows.append(self.source_row("p-short", date, "3"))

        # Duplicate symptom records on one date are averaged, as before.
        rows.append(self.source_row("p-a", "2026-01-08", "4"))
        # Invalid numeric value and invalid date must not count as observed symptom days.
        rows.append(self.source_row("p-a", "2026-01-03", "not-a-number"))
        rows.append(self.source_row("p-a", "not-a-date", "9"))
        return rows

    def test_streaming_output_matches_expected_across_chunk_boundaries(self):
        rows = self.fixture_rows()

        def chunks_factory():
            for start in range(0, len(rows), 11):
                yield pd.DataFrame(rows[start:start + 11])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            result = self.preparation.process_flaredown_chunks(
                chunks_factory,
                output_path=output_path,
                staging_directory=temporary_directory,
            )
            output = pd.read_csv(output_path, keep_default_na=False)

            self.assertEqual(
                list(output.columns),
                list(self.preparation.OUTPUT_COLUMNS),
            )
            self.assertEqual(result["rows"], 13)
            self.assertEqual(result["patients"], 2)
            self.assertEqual(result["positive_count"], 0)
            self.assertEqual(
                output[["user_id", "checkpoint_date"]].values.tolist(),
                [
                    ["p-a", date]
                    for date in (
                        "2026-01-13",
                        "2026-01-16",
                        "2026-01-19",
                        "2026-01-22",
                        "2026-01-25",
                        "2026-01-28",
                    )
                ]
                + [
                    ["p-b", date]
                    for date in (
                        "2026-01-10",
                        "2026-01-13",
                        "2026-01-16",
                        "2026-01-19",
                        "2026-01-22",
                        "2026-01-25",
                        "2026-01-28",
                    )
                ],
            )
            self.assertAlmostEqual(output.loc[0, "past_mean_severity"], 15 / 7)
            self.assertEqual(output.loc[0, "personal_baseline"], 2.0)
            self.assertEqual(output.loc[0, "age"], 41)
            self.assertEqual(output.loc[0, "sex"], "female")
            self.assertEqual(output.loc[6, "age"], 52)
            self.assertEqual(output.loc[6, "sex"], "unknown")
            self.assertFalse((output["user_id"] == "p-short").any())
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_existing_output_is_refused_without_changing_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            output_path.write_text("existing candidate\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self.preparation.process_flaredown_chunks(
                    lambda: iter(()),
                    output_path=output_path,
                    staging_directory=temporary_directory,
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "existing candidate\n",
            )

    def test_concurrent_output_creation_is_not_overwritten(self):
        rows = self.fixture_rows()

        def chunks_factory():
            yield pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            real_link = self.preparation.os.link

            def create_destination_then_link(source, destination):
                Path(destination).write_text("created concurrently\n", encoding="utf-8")
                real_link(source, destination)

            with patch.object(
                self.preparation.os,
                "link",
                side_effect=create_destination_then_link,
            ):
                with self.assertRaises(FileExistsError):
                    self.preparation.process_flaredown_chunks(
                        chunks_factory,
                        output_path=output_path,
                        staging_directory=temporary_directory,
                    )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "created concurrently\n",
            )
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_csv_writing_failure_cleans_temporary_resources(self):
        rows = self.fixture_rows()

        def chunks_factory():
            yield pd.DataFrame(rows)

        class FailingWriter:
            def writerow(self, row):
                raise OSError("fixture CSV write failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            with patch.object(
                self.preparation.csv,
                "writer",
                return_value=FailingWriter(),
            ):
                with self.assertRaisesRegex(OSError, "fixture CSV write failed"):
                    self.preparation.process_flaredown_chunks(
                        chunks_factory,
                        output_path=output_path,
                        staging_directory=temporary_directory,
                    )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_interruption_closes_connection_and_cleans_resources(self):
        real_connect = sqlite3.connect
        connections = []

        def tracking_connect(path):
            connection = real_connect(path)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            with patch.object(
                self.preparation.sqlite3,
                "connect",
                side_effect=tracking_connect,
            ):
                with patch.object(
                    self.preparation,
                    "_stage_raw_rows",
                    side_effect=KeyboardInterrupt("fixture interruption"),
                ):
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "fixture interruption"
                    ):
                        self.preparation.process_flaredown_chunks(
                            lambda: iter(()),
                            output_path=output_path,
                            staging_directory=temporary_directory,
                        )

            self.assertEqual(len(connections), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[0].execute("SELECT 1")
            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_close_failure_preserves_interruption_and_staging_file(self):
        real_connect = sqlite3.connect
        connections = []
        staging_paths = []

        class CloseFailsAfterClosing:
            def __init__(self, connection):
                self.connection = connection

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def close(self):
                self.connection.close()
                raise OSError("fixture close failed")

        def tracking_connect(path):
            staging_paths.append(Path(path))
            connection = real_connect(path)
            connections.append(connection)
            return CloseFailsAfterClosing(connection)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            stderr = io.StringIO()
            with patch.object(
                self.preparation.sqlite3,
                "connect",
                side_effect=tracking_connect,
            ):
                with patch.object(
                    self.preparation,
                    "_stage_raw_rows",
                    side_effect=KeyboardInterrupt("fixture interruption"),
                ):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaisesRegex(
                            KeyboardInterrupt, "fixture interruption"
                        ):
                            self.preparation.process_flaredown_chunks(
                                lambda: iter(()),
                                output_path=output_path,
                                staging_directory=temporary_directory,
                            )

            self.assertEqual(len(connections), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[0].execute("SELECT 1")
            self.assertEqual(len(staging_paths), 1)
            self.assertTrue(staging_paths[0].exists())
            self.assertIn("fixture close failed", stderr.getvalue())
            self.assertIn(str(staging_paths[0]), stderr.getvalue())
            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])

    def test_user_ids_must_be_strings_and_preserve_leading_zeroes(self):
        numeric_rows = pd.DataFrame(
            [self.source_row(1, "", "", "Condition", name="RA")]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "numeric.csv"
            with self.assertRaisesRegex(TypeError, "user_id values must be strings"):
                self.preparation.process_flaredown_chunks(
                    lambda: iter([numeric_rows]),
                    output_path=output_path,
                    staging_directory=temporary_directory,
                )
            self.assertFalse(output_path.exists())

        rows = []
        for user_id in ("001", "1"):
            rows.append(
                self.source_row(user_id, "", "", "Condition", name="RA")
            )
            for date in pd.date_range("2026-01-01", periods=30, freq="D"):
                rows.append(
                    self.source_row(user_id, date.strftime("%Y-%m-%d"), "2")
                )

        def chunks_factory():
            yield pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            self.preparation.process_flaredown_chunks(
                chunks_factory,
                output_path=output_path,
                staging_directory=temporary_directory,
            )
            output = pd.read_csv(output_path, dtype={"user_id": str})

            self.assertEqual(output["user_id"].drop_duplicates().tolist(), ["001", "1"])

    def test_one_shot_chunk_iterator_is_rejected(self):
        one_shot_iterator = iter(
            [
                pd.DataFrame(
                    [self.source_row("p-a", "", "", "Condition", name="RA")]
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            with self.assertRaisesRegex(ValueError, "fresh iterator"):
                self.preparation.process_flaredown_chunks(
                    lambda: one_shot_iterator,
                    output_path=output_path,
                    staging_directory=temporary_directory,
                )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_chunk_factory_must_repeat_the_same_rows(self):
        calls = 0

        def chunks_factory():
            nonlocal calls
            calls += 1
            condition_name = "RA" if calls == 1 else "not RA"
            yield pd.DataFrame(
                [
                    self.source_row(
                        "p-a", "", "", "Condition", name=condition_name
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            with self.assertRaisesRegex(ValueError, "different source rows"):
                self.preparation.process_flaredown_chunks(
                    chunks_factory,
                    output_path=output_path,
                    staging_directory=temporary_directory,
                )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )

    def test_failed_processing_does_not_leave_a_candidate_or_partial_file(self):
        calls = 0

        def failing_chunks_factory():
            nonlocal calls
            calls += 1
            if calls == 1:
                yield pd.DataFrame(
                    [
                        self.source_row(
                            "p-a", "", "", "Condition", name="RA"
                        )
                    ]
                )
                return
            raise RuntimeError("fixture read failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "candidate.csv"
            with self.assertRaisesRegex(RuntimeError, "fixture read failed"):
                self.preparation.process_flaredown_chunks(
                    failing_chunks_factory,
                    output_path=output_path,
                    staging_directory=temporary_directory,
                )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).glob(".*.partial")), [])
            self.assertEqual(
                list(Path(temporary_directory).glob("flaredown_stage_*.sqlite")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
