import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch


FUSION_PATH = Path(__file__).resolve().parents[1] / "2_fusion.py"
SPEC = importlib.util.spec_from_file_location("fusion_backend_under_test", FUSION_PATH)
fusion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fusion
SPEC.loader.exec_module(fusion)


class FusionSafetyTests(unittest.TestCase):
    def test_legacy_meta_model_is_never_loaded_or_fabricated(self):
        loaded_paths = []

        def fake_load_one(path):
            loaded_paths.append(Path(path).name)
            return object()

        with patch.object(fusion, "_load_one", side_effect=fake_load_one):
            models = fusion.load_models(models_dir="unused")

        self.assertEqual(
            loaded_paths,
            ["ra_model.pkl", "sleep_model.pkl", "hrv_model.pkl", "sl_ra_model.pkl"],
        )
        self.assertIsNone(models.meta)
        self.assertFalse(models.fusion_available)
        self.assertNotIn("meta", models.load_errors)

    def test_meta_artifact_in_model_directory_is_ignored_without_deserialization(self):
        with tempfile.TemporaryDirectory() as models_dir:
            artifact = Path(models_dir) / "meta_model.pkl"
            artifact.write_bytes(b"not a serialized model")
            real_load_one = fusion._load_one

            with patch.object(
                fusion, "_load_one", wraps=real_load_one
            ) as load_one, patch.object(fusion.joblib, "load") as deserialize:
                models = fusion.load_models(models_dir=models_dir)

            requested = [Path(call.args[0]).name for call in load_one.call_args_list]
            self.assertEqual(
                requested,
                ["ra_model.pkl", "sleep_model.pkl", "hrv_model.pkl", "sl_ra_model.pkl"],
            )
            deserialize.assert_not_called()
            self.assertIsNone(models.meta)
            self.assertFalse(models.fusion_available)

    def test_base_model_availability_does_not_imply_fusion_availability(self):
        with patch.object(fusion, "_load_one", return_value=object()):
            models = fusion.load_models(models_dir="unused")

        self.assertTrue(models.any_real)
        self.assertTrue(models.models_ok)
        self.assertFalse(models.fusion_available)
        self.assertIsNone(models.meta)

    def test_metrics_labels_identify_available_values_as_base_not_fusion_metrics(self):
        status, scope = fusion.model_metrics_labels(metrics_available=True)

        self.assertIn("Base-model", status)
        self.assertIn("not fusion validation", status)
        self.assertIn("generation time unverified", status)
        self.assertIn("base-model metrics", scope)
        self.assertIn("not flare-risk or fusion metrics", scope)

    def test_metrics_reference_label_does_not_claim_fusion_performance(self):
        status, scope = fusion.model_metrics_labels(metrics_available=False)

        self.assertIn("Metrics unavailable", status)
        self.assertIn("not flare-risk or fusion metrics", scope)

    def test_get_model_metrics_accepts_structurally_valid_base_metrics(self):
        valid_metrics = {
            "Example dataset": {
                "Example model": {
                    "acc": 0.8,
                    "f1": 0.75,
                    "auc": 0.82,
                    "pr_auc": 0.79,
                    "precision": 0.7,
                    "recall": 0.8,
                    "test_rows": 12,
                    "target": "dataset-specific target",
                    "split": "held-out test split",
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            metrics_path.write_text(json.dumps(valid_metrics), encoding="utf-8")

            metrics, available = fusion.get_model_metrics(str(metrics_path))

        self.assertTrue(available)
        self.assertEqual(metrics, valid_metrics)

    def test_get_model_metrics_rejects_malformed_json_and_invalid_structure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            for contents in ('{"dataset":', '{"dataset": {"model": {"acc": 0.9}}}'):
                with self.subTest(contents=contents):
                    metrics_path.write_text(contents, encoding="utf-8")

                    metrics, available = fusion.get_model_metrics(str(metrics_path))

                    self.assertEqual(metrics, {})
                    self.assertFalse(available)

    def test_get_model_metrics_reports_missing_file_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "missing.json"

            metrics, available = fusion.get_model_metrics(str(metrics_path))

        self.assertEqual(metrics, {})
        self.assertFalse(available)

    def test_checked_in_metrics_file_matches_supported_structure(self):
        metrics, available = fusion.get_model_metrics()

        self.assertTrue(available)
        self.assertTrue(metrics)

    def test_api_system_endpoint_reports_fusion_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "api-test.sqlite"
            with patch.dict(os.environ, {"DB_PATH": str(db_path)}), patch(
                "joblib.load", side_effect=AssertionError("model deserialization blocked")
            ) as deserialize:
                api_module = importlib.import_module("api")
                try:
                    from fastapi.testclient import TestClient

                    response = TestClient(api_module.app).get("/api/system")
                finally:
                    sys.modules.pop("api", None)
                    sys.modules.pop("fusion_backend", None)

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIs(payload["fusion_model_available"], False)
            self.assertIn("unavailable", payload["fusion_model_status"].lower())
            self.assertIn("independent", payload["fusion_model_status"].lower())
            self.assertEqual(payload["risk_method"], "experimental heuristic")
            self.assertIs(
                payload["trained_fusion_predictions_used_by_risk_score"], False
            )
            self.assertFalse(
                any(
                    "meta_model.pkl" in str(call.args[0])
                    for call in deserialize.call_args_list
                )
            )

    def test_production_cookie_is_secure_by_default(self):
        with patch.dict(
            os.environ,
            {"APP_ENV": "production", "DB_PATH": ":memory:"},
        ):
            os.environ.pop("COOKIE_SECURE", None)
            with patch(
                "joblib.load",
                side_effect=AssertionError("model loading blocked"),
            ):
                api_module = importlib.import_module("api")
                try:
                    self.assertTrue(api_module.COOKIE_SECURE)
                    response = api_module.Response()
                    api_module.set_session_cookie(response, "test-session")
                    self.assertIn(
                        "secure", response.headers["set-cookie"].lower()
                    )
                finally:
                    sys.modules.pop("api", None)
                    sys.modules.pop("fusion_backend", None)

    def test_production_rejects_insecure_cookie_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "api-test.sqlite"
            with patch.dict(
                os.environ,
                {
                    "APP_ENV": "production",
                    "COOKIE_SECURE": "false",
                    "DB_PATH": str(db_path),
                },
            ), patch(
                "joblib.load",
                side_effect=AssertionError("model loading blocked"),
            ):
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "COOKIE_SECURE cannot be disabled"
                    ):
                        importlib.import_module("api")
                finally:
                    sys.modules.pop("api", None)

    def test_assessment_endpoints_return_context_and_no_fake_forecast(self):
        with patch.dict(os.environ, {"DB_PATH": ":memory:"}), patch(
            "joblib.load", side_effect=AssertionError("model deserialization blocked")
        ):
            api_module = importlib.import_module("api")
            try:
                from fastapi.testclient import TestClient

                class WeatherClientStub:
                    def current(self, _city):
                        return SimpleNamespace(
                            temp=29.0,
                            humidity=78.0,
                            pressure=1011.0,
                            desc="Partly cloudy",
                            is_live=False,
                        )

                    def forecast(self, *_args, **_kwargs):
                        raise AssertionError("No projected forecast should be generated")

                request_data = {
                    "age": 42,
                    "symptom_change": "worse",
                    "city": "Colombo",
                    "fn_vals": {letter: 1 for letter in "ABCDEFGHIJ"},
                    "pain": 7,
                    "global_estimate": 6,
                    "fatigue": 8,
                    "flares_30d": 2,
                    "duration_years": 4,
                    "sleep_diff": 2,
                    "anxiety": 1,
                    "activity_min": 20,
                    "heart_rate": 0,
                }
                with patch.object(api_module.fusion, "WeatherClient", WeatherClientStub), patch.object(
                    api_module.fusion,
                    "compute_risk",
                    return_value=(0.37, {"Pain VAS": 0.12}),
                ):
                    with TestClient(api_module.app) as client:
                        guest_response = client.post(
                            "/api/assessments/guest",
                            json=request_data,
                        )
                        with patch.object(
                            api_module, "get_session_username", return_value="test-user"
                        ), patch.object(
                            api_module.fusion, "save_assessment", return_value=True
                        ):
                            authenticated_response = client.post(
                                "/api/assessments",
                                json=request_data,
                            )
            finally:
                sys.modules.pop("api", None)
                sys.modules.pop("fusion_backend", None)

        self.assertEqual(guest_response.status_code, 200, guest_response.text)
        self.assertEqual(authenticated_response.status_code, 200, authenticated_response.text)
        for response in (guest_response, authenticated_response):
            with self.subTest(endpoint=response.request.url.path):
                payload = response.json()
                self.assertEqual(payload["assessment"]["experimental_score"], 0.37)
                self.assertEqual(payload["assessment"]["peak_experimental_score"], 0.37)
                self.assertEqual(
                    payload["assessment"]["reported_inputs"],
                    {
                        "symptom_change": "worse",
                        "fatigue": 8.0,
                        "sleep_difficulty": 2,
                        "anxiety": 1,
                        "flares_30d": 2,
                        "activity_min": 20,
                        "heart_rate_bpm": None,
                    },
                )
                for measure in (
                    "morning_stiffness",
                    "tender_joint_count",
                    "swollen_joint_count",
                    "esr",
                    "crp",
                ):
                    self.assertNotIn(measure, payload["assessment"]["reported_inputs"])
                self.assertEqual(payload["forecast"], [])
                self.assertTrue(any(
                    "No clinically validated three-day RA flare forecast" in item
                    for item in payload["limitations"]
                ))
                self.assertEqual(payload["weather"]["is_live"], False)

    def test_rule_based_risk_does_not_consume_meta_model(self):
        class InvalidMetaModel:
            def predict_proba(self, _inputs):
                raise AssertionError("legacy meta-model must never be called")

        inputs = fusion.RiskInputs(
            pain=5,
            fn_score=4,
            global_estimate=5,
            fatigue=5,
            flares_30d=2,
            duration_years=4,
            sleep_diff=1,
            anxiety=1,
            activity_min=30,
            heart_rate=72,
            rapid3=10,
            temp=28,
            humidity=70,
            pressure=1010,
        )

        score, contributions = fusion.compute_risk(
            inputs,
            fusion.ModelBundle(meta=InvalidMetaModel()),
        )

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertTrue(contributions)


if __name__ == "__main__":
    unittest.main()
