"""
2_fusion.py — RA FlareGuard shared backend
============================================================================
Pure-Python backend module (no Streamlit calls) used by 3_patient_console.py.
Keeping this framework-agnostic means it can also be unit-tested or reused
headlessly (e.g. from a notebook or a CLI) without pulling in Streamlit.

Responsibilities:
  1. Weather context retrieval with a fallback for legacy interface use.
  2. Legacy base-model loading helpers for the old Streamlit console. The
     unavailable fusion model is never loaded or replaced with a fake model.
  3. Legacy score calculators retained for backwards compatibility. The React
     API uses project-specific questionnaire summary functions instead.
  4. Legacy heuristic risk functions retained for the old Streamlit console;
     they are not a trained or validated multimodal flare predictor and are
     not called by the React/FastAPI patient assessment endpoint.
  5. Historical dataset metrics and account/history helper functions.

Reference standards implemented here:
  RAPID3      Pincus et al., J Rheumatol 2008;35:2136
  MDHAQ       Pincus & Swearingen, Arthritis Rheum 1999;42:2220
  HAQ-DI      Fries et al., Arthritis Rheum 1980;23:137
  Pain VAS    Huskisson, Lancet 1974;2:1127
  Fatigue VAS Wolfe et al., J Rheumatol 1996;23:1407
  RADAI       Stucki et al., Arthritis Rheum 1995;38:795
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import warnings
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests
from database import connect as connect_database, execute as execute_sql
from database import INTEGRITY_ERRORS

LOG = logging.getLogger(__name__)

# Load optional local environment variables from a .env file beside this file.
# The app still works without python-dotenv or a .env file; in that case,
# OWM_API_KEY can be supplied through the operating system environment.
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

warnings.filterwarnings("ignore")

try:
    import joblib
except Exception:  # pragma: no cover - joblib should normally be installed
    joblib = None

# ─────────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if load_dotenv is not None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "flareguard.db"))
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEEDBACK_PATH = os.path.join(BASE_DIR, "patient_feedback.csv")

API_KEY = os.environ.get("OWM_API_KEY", "").strip()
WEATHER_TIMEOUT = 5  # seconds — keep the UI responsive even when the API hangs


# ═════════════════════════════════════════════════════════════════════════
# 1. WEATHER — OpenWeatherMap with a strict try/except + Sri Lankan fallback
# ═════════════════════════════════════════════════════════════════════════

# Approximate seasonal averages (temp °C, humidity %, pressure hPa) used
# whenever the live API is unreachable, unconfigured, or returns an error —
# so a demo or viva session never breaks because of network conditions.
SL_CITY_DEFAULTS = {
    "Colombo": (30, 78, 1010), "Kandy": (26, 72, 1012), "Galle": (29, 80, 1009),
    "Jaffna": (33, 65, 1008), "Negombo": (30, 77, 1010), "Trincomalee": (32, 68, 1007),
    "Anuradhapura": (31, 67, 1008), "Matara": (28, 81, 1009), "Kurunegala": (29, 73, 1010),
    "Badulla": (22, 70, 1015), "Ratnapura": (28, 82, 1009),
}
GENERIC_FALLBACK = (29, 75, 1010)
CITIES = SL_CITY_DEFAULTS  # convenience alias used by the UI's city picker


def _default_for(city: str):
    return SL_CITY_DEFAULTS.get(city, GENERIC_FALLBACK)


@dataclass
class WeatherReading:
    temp: float
    humidity: float
    pressure: float
    desc: str
    is_live: bool
    date: Optional[str] = None


class WeatherClient:
    """OpenWeatherMap client. Every network call is wrapped in a strict
    try/except; on ANY failure (timeout, bad key, unknown city, malformed
    payload) we seamlessly return a realistic Sri Lankan fallback reading
    instead of raising, so the wizard never gets stuck."""

    def __init__(self, api_key: str = API_KEY, timeout: int = WEATHER_TIMEOUT):
        self.api_key = api_key
        self.timeout = timeout

    def current(self, city: str) -> WeatherReading:
        city = (city or "").strip()
        if not city:
            t, h, p = GENERIC_FALLBACK
            return WeatherReading(t, h, p, "no location entered", False)
        if not self.api_key:
            t, h, p = _default_for(city)
            return WeatherReading(t, h, p, "estimated (no live weather API key configured)", False)
        try:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": self.api_key, "units": "metric"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            if str(data.get("cod")) != "200":
                raise ValueError(data.get("message", "unknown API error"))
            return WeatherReading(
                float(data["main"]["temp"]), float(data["main"]["humidity"]),
                float(data["main"]["pressure"]), data["weather"][0]["description"], True,
            )
        except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
            t, h, p = _default_for(city)
            return WeatherReading(
                t, h, p,
                f"live weather unavailable ({exc.__class__.__name__}) — showing Sri Lankan estimate for '{city}'",
                False,
            )

    def forecast(self, city: str, days: int = 3) -> list[WeatherReading]:
        """Returns `days` daily readings starting tomorrow. Falls back to a
        synthetic-but-plausible short trend anchored on `current()` if the
        live 5-day/3-hour forecast endpoint fails or is unavailable."""
        city = (city or "").strip()
        base = self.current(city)

        if self.api_key and city:
            try:
                r = requests.get(
                    "https://api.openweathermap.org/data/2.5/forecast",
                    params={"q": city, "appid": self.api_key, "units": "metric", "cnt": 24},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                if str(data.get("cod")) != "200":
                    raise ValueError(data.get("message", "unknown API error"))
                daily: dict = {}
                for item in data["list"]:
                    d = item["dt_txt"].split(" ")[0]
                    bucket = daily.setdefault(d, {"t": [], "h": [], "p": [], "desc": []})
                    bucket["t"].append(item["main"]["temp"])
                    bucket["h"].append(item["main"]["humidity"])
                    bucket["p"].append(item["main"]["pressure"])
                    bucket["desc"].append(item["weather"][0]["description"])
                today = datetime.now().strftime("%Y-%m-%d")
                out = []
                for d in sorted(daily):
                    if d == today or len(out) >= days:
                        continue
                    v = daily[d]
                    out.append(WeatherReading(
                        round(sum(v["t"]) / len(v["t"]), 1),
                        round(sum(v["h"]) / len(v["h"])),
                        round(sum(v["p"]) / len(v["p"])),
                        max(set(v["desc"]), key=v["desc"].count), True, d,
                    ))
                if len(out) >= days:
                    return out[:days]
                raise ValueError("insufficient forecast data returned")
            except (requests.RequestException, ValueError, KeyError, TypeError, OSError):
                pass  # fall through to the synthetic trend below

        rng = np.random.default_rng(zlib.crc32(city.encode("utf-8")))
        out = []
        for i in range(days):
            out.append(WeatherReading(
                round(base.temp + rng.uniform(-1.5, 1.5), 1),
                min(100, base.humidity + int(rng.integers(-5, 8))),
                base.pressure + int(rng.integers(-3, 3)),
                "estimated (Sri Lankan seasonal fallback)", False,
                (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            ))
        return out


# ═════════════════════════════════════════════════════════════════════════
# 2. MODELS — loading with per-model isolation + calibrated fallback
# ═════════════════════════════════════════════════════════════════════════

class CalibratedFallbackModel:
    """Legacy compatibility stand-in for the old Streamlit model panel.

    It generates a formula-based pseudo-score, not a trained prediction and
    not a calibrated probability. The React/FastAPI patient assessment does
    not use this fallback. Do not report its outputs as model performance.
    """

    def __init__(self, name: str, n_features: int, weights: Optional[list] = None):
        self.name = name
        self.n_features_in_ = n_features
        self.weights = weights or [1.0 / n_features] * n_features
        self.is_fallback = True

    def predict_proba(self, X):
        X = np.atleast_2d(np.array(X, dtype=float))
        w = np.array(self.weights[: X.shape[1]], dtype=float)
        if len(w) < X.shape[1]:
            pad_val = w.mean() if len(w) else 0.1
            w = np.pad(w, (0, X.shape[1] - len(w)), constant_values=pad_val)
        scale = np.abs(X).max(axis=0)
        scale[scale == 0] = 1.0
        norm = np.clip(X / scale, 0, 3)
        score = np.clip((norm * w).sum(axis=1) / (w.sum() + 1e-6), 0, 1)
        return np.column_stack([1 - score, score])


@dataclass
class ModelBundle:
    ra: object = None
    sleep: object = None
    hrv: object = None
    sl_ra: object = None
    meta: object = None  # Reserved compatibility field; supervised fusion is unavailable.
    fusion_available: bool = False
    load_errors: dict = field(default_factory=dict)   # {slot: "ExcType: message"}
    models_ok: bool = False   # True only if every real model loaded cleanly
    any_real: bool = False    # True if at least one real model loaded


# (attribute name, filename, expected feature count for its fallback)
_MODEL_SPECS = [
    ("ra", "ra_model.pkl", 10),
    ("sleep", "sleep_model.pkl", 6),
    ("hrv", "hrv_model.pkl", 14),
    ("sl_ra", "sl_ra_model.pkl", 8),
]


def _load_one(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not installed in this environment")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{os.path.basename(path)} not found in models/")
    obj = joblib.load(path)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{os.path.basename(path)} does not contain a classifier with predict_proba()")
    return model


def load_models(models_dir: str = MODELS_DIR) -> ModelBundle:
    """Load legacy base models without loading or fabricating a fusion model.

    Base-model failures are isolated and backfilled for legacy compatibility.
    Their outputs are not combined by ``compute_risk`` because the models use
    different targets and feature schemas.
    """
    bundle = ModelBundle()
    slots: dict = {}

    for attr, filename, n_feat in _MODEL_SPECS:
        try:
            slots[attr] = _load_one(os.path.join(models_dir, filename))
            bundle.any_real = True
        except Exception as exc:
            bundle.load_errors[attr] = f"{type(exc).__name__}: {exc}"
            slots[attr] = CalibratedFallbackModel(attr, n_feat)

    bundle.ra, bundle.sleep, bundle.hrv, bundle.sl_ra = (
        slots["ra"], slots["sleep"], slots["hrv"], slots["sl_ra"]
    )
    bundle.models_ok = len(bundle.load_errors) == 0
    return bundle


def safe_predict_proba(model, inputs) -> float:
    """Strict probability helper. Feature padding/truncation is prohibited:
    silently changing a model's feature contract can produce meaningless scores."""
    try:
        arr = np.asarray(inputs, dtype=float).reshape(1, -1)
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and arr.shape[1] != int(expected):
            raise ValueError(f"Feature mismatch: received {arr.shape[1]}, expected {expected}")
        proba = model.predict_proba(arr)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return float(np.clip(proba[positive_index], 0, 1))
    except Exception:
        return 0.5


# ═════════════════════════════════════════════════════════════════════════
# 3. CLINICAL CALCULATORS — RAPID3 / HAQ-DI / MDHAQ / VAS
# ═════════════════════════════════════════════════════════════════════════

RAPID3_QUESTIONS = [
    ("A", "Dress yourself, including tying shoelaces and doing buttons?", "HAQ item 1"),
    ("B", "Get in and out of bed?", "HAQ item 2"),
    ("C", "Lift a full cup or glass to your mouth?", "HAQ item 5"),
    ("D", "Walk outdoors on flat ground?", "HAQ item 9"),
    ("E", "Wash and dry your entire body?", "HAQ item 12"),
    ("F", "Bend down to pick up clothing from the floor?", "HAQ item 14"),
    ("G", "Turn regular faucets (taps) on and off?", "HAQ item 16"),
    ("H", "Get in and out of a car, bus, train, or vehicle?", "HAQ item 18"),
    ("I", "Walk two miles or about three kilometres?", "HAQ item 10"),
    ("J", "Participate in recreational activities as you wish?", "HAQ item 20"),
]
DIFFICULTY_LABELS = {0: "No difficulty", 1: "Some difficulty", 2: "Much difficulty", 3: "Unable to do"}
DIFFICULTY_SHORT = {0: "0 · None", 1: "1 · Some", 2: "2 · Much", 3: "3 · Unable"}
DIFFICULTY_COLORS = {0: "#4ade80", 1: "#facc15", 2: "#fb923c", 3: "#f87171"}


def rapid3_function_score(fn_vals: dict) -> float:
    """Return the ten-item RAPID3 function component on a 0-10 scale.

    This is not the official HAQ-DI score. The ten selected function items
    are summed (0-30) and divided by 3 to obtain the RAPID3 function component.
    """
    return round(sum(fn_vals.values()) / 3, 2)


def prototype_function_score(fn_vals: dict) -> float:
    """Score the prototype's ten selected function items on a 0–10 scale.

    This is a project-specific calculation and is not an official HAQ-DI or
    RAPID3 score unless the exact licensed instrument and protocol are verified.
    """
    return round(sum(float(v) for v in fn_vals.values()) / 3, 2)


def prototype_questionnaire_score(fn_vals: dict, pain: float, global_estimate: float) -> float:
    """Project-specific function + pain + global summary on a nominal 0–30 scale."""
    return round(prototype_function_score(fn_vals) + float(pain) + float(global_estimate), 2)


def haq_di_score(fn_vals: dict) -> float:
    """Backward-compatible legacy name; do not label this as official HAQ-DI."""
    return rapid3_function_score(fn_vals)


def rapid3_total(fn_vals: dict, pain: float, global_estimate: float) -> float:
    """RAPID3 total: function (0-10) + pain VAS (0-10) + patient global (0-10)."""
    return rapid3_function_score(fn_vals) + float(pain) + float(global_estimate)


def rapid3_category(score: float):
    if score <= 3:
        return "Near Remission", "#4ade80"
    if score <= 6:
        return "Low Severity", "#60a5fa"
    if score <= 12:
        return "Moderate Severity", "#fbbf24"
    return "High Severity", "#f87171"


# ═════════════════════════════════════════════════════════════════════════
# 4. FUSED FLARE-RISK ENGINE + EXPLAINABILITY (XAI)
# ═════════════════════════════════════════════════════════════════════════

def _clip01(v: float) -> float:
    return float(np.clip(v, 0, 1))


def _norm(v: float, lo: float, hi: float) -> float:
    return _clip01((v - lo) / (hi - lo))


@dataclass
class RiskInputs:
    pain: float
    fn_score: float
    global_estimate: float
    fatigue: float
    flares_30d: int
    duration_years: int
    sleep_diff: int
    anxiety: int
    activity_min: int
    heart_rate: int
    rapid3: float
    temp: float
    humidity: float
    pressure: float


def _submodel_inputs(x: RiskInputs):
    sp = x.fn_score * 2.8
    stp = (x.global_estimate / 10) * 12
    sleep_hours_proxy = max(0, 8 - x.sleep_diff * 1.8)
    sleep_quality_proxy = max(1, 10 - x.sleep_diff * 3)
    stress = 1 + x.anxiety * 3
    energy = max(1, 10 - x.global_estimate)

    ra_in = [x.pain, sp, stp, x.fatigue, x.flares_30d, x.duration_years,
             x.humidity, x.temp, x.pressure, 1 if x.humidity > 75 else 0]
    slra_in = [round(x.rapid3 / 5.5, 2), x.pain * 7.6, sp, x.pain, x.fatigue,
               x.duration_years * 12, stp, 10 - energy]
    sleep_in = [sleep_hours_proxy, sleep_quality_proxy, stress, x.activity_min, x.heart_rate, energy]

    rmssd = max(10, 60 - stress * 4 - x.heart_rate * 0.2 + sleep_quality_proxy * 2)
    sdnn = rmssd * 1.3
    lf_hf = 1 + stress * 0.3 - sleep_quality_proxy * 0.1
    hrv_in = [rmssd, rmssd, sdnn, rmssd, rmssd * 0.8, rmssd / max(sdnn, 1), x.heart_rate,
              max(0, 50 - stress * 3), max(0, 30 - stress * 2), rmssd * 0.7, sdnn * 1.2,
              stress * 0.5, (stress - 5) * 0.2, lf_hf]
    return ra_in, slra_in, sleep_in, hrv_in


def compute_risk(x: RiskInputs, models: ModelBundle):
    """Return an experimental heuristic score and additive display components.

    IMPORTANT: This is NOT a calibrated flare probability. The available
    dataset models have incompatible targets and feature schemas; therefore
    no trained model prediction is mixed into this score. The clinical and
    environmental weights are normalised to sum to one (0.65 + 0.15 + 0.05
    = 0.85 before normalisation). A future fusion model must be independently
    trained, calibrated, and evaluated before it is connected here.
    """
    clinical_components = {
        "Pain VAS": _norm(x.pain, 0, 10) * 0.25,
        "RAPID3 Function Component": _norm(x.fn_score, 0, 10) * 0.25,
        "Patient Global Estimate": _norm(x.global_estimate, 0, 10) * 0.20,
        "Flare History (30d)": _norm(x.flares_30d, 0, 30) * 0.15,
        "Sleep Difficulty": (1 - _norm(max(0, 8 - x.sleep_diff * 1.8), 0, 12)) * 0.05,
        "Fatigue VAS": _norm(x.fatigue, 0, 10) * 0.10,
    }
    clinical_score = float(sum(clinical_components.values()))
    humidity_raw = _norm(x.humidity, 40, 100) * 0.15
    temp_raw = _norm(max(0, 32 - x.temp), 0, 20) * 0.05

    # Normalise the stated 65% clinical + 15% humidity + 5% temperature
    # heuristic so the score remains in [0,1] without pretending missing
    # ML weight is real evidence.
    denominator = 0.85
    score = (clinical_score * 0.65 + humidity_raw + temp_raw) / denominator

    if x.humidity > 85:
        score = min(1, score + 0.04)
    elif x.humidity > 80:
        score = min(1, score + 0.02)
    elif x.humidity > 75:
        score = min(1, score + 0.01)
    if x.rapid3 > 12:
        score = min(1, score + 0.05)
    elif x.rapid3 > 6:
        score = min(1, score + 0.025)
    if x.pain >= 7 and x.sleep_diff >= 2:
        score = min(1, score + 0.03)
    if x.rapid3 <= 3 and x.flares_30d <= 2:
        score = min(score, 0.25)

    contributions = {k: v * 0.65 / denominator for k, v in clinical_components.items()}
    contributions["Humidity (heuristic)" ] = humidity_raw / denominator
    contributions["Temperature (heuristic)"] = temp_raw / denominator
    return _clip01(score), contributions


def project_forward(pain, fatigue, sleep_diff, anxiety, flares_30d, day_offset):
    """Simple deterministic projection of patient-reported fields (NOT a trained
    forecasting model). Poor sleep and frequent recent flares push the projected
    symptoms slightly up; anxiety and sleep are carried forward unchanged, so a
    higher anxiety answer can never lower the projected risk."""
    sleep_penalty = max(0, sleep_diff - 1) * 0.04 * day_offset
    flare_pressure = flares_30d * 0.005 * day_offset
    projected_pain = float(np.clip(pain + flare_pressure + sleep_penalty * 0.5, 0, 10))
    projected_fatigue = float(np.clip(fatigue + sleep_penalty, 0, 10))
    projected_anxiety = float(np.clip(anxiety, 0, 3))
    projected_sleep_diff = float(np.clip(sleep_diff, 0, 3))
    return projected_pain, projected_fatigue, projected_anxiety, projected_sleep_diff


def risk_level(r: float) -> str:
    return "high" if r > 0.60 else "medium" if r > 0.30 else "low"


def risk_color(r: float) -> str:
    return {"high": "#f87171", "medium": "#fbbf24", "low": "#4ade80"}[risk_level(r)]


def risk_label(r: float) -> str:
    return {"high": "Higher heuristic score", "medium": "Intermediate heuristic score", "low": "Lower heuristic score"}[risk_level(r)]


def risk_emoji(r: float) -> str:
    return {"high": "🔴", "medium": "🟡", "low": "🟢"}[risk_level(r)]


# ═════════════════════════════════════════════════════════════════════════
# 5. MODEL PERFORMANCE METRICS (viva "Model Performance" tab)
# ═════════════════════════════════════════════════════════════════════════

def _valid_metrics_payload(data: object) -> bool:
    """Check the per-dataset, per-model metric structure written by training."""
    if not isinstance(data, dict) or not data:
        return False

    required = {
        "acc", "f1", "auc", "pr_auc", "precision", "recall",
        "test_rows", "target", "split",
    }
    bounded_metrics = ("acc", "f1", "precision", "recall")
    optional_bounded_metrics = ("auc", "pr_auc")

    for dataset, models in data.items():
        if not isinstance(dataset, str) or not dataset.strip():
            return False
        if not isinstance(models, dict) or not models:
            return False
        for model_name, metrics in models.items():
            if not isinstance(model_name, str) or not model_name.strip():
                return False
            if not isinstance(metrics, dict) or not required.issubset(metrics):
                return False
            for key in bounded_metrics:
                value = metrics[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1
                ):
                    return False
            for key in optional_bounded_metrics:
                value = metrics[key]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1
                ):
                    return False
            if (
                isinstance(metrics["test_rows"], bool)
                or not isinstance(metrics["test_rows"], int)
                or metrics["test_rows"] <= 0
                or not isinstance(metrics["target"], str)
                or not metrics["target"].strip()
                or not isinstance(metrics["split"], str)
                or not metrics["split"].strip()
            ):
                return False
    return True


def get_model_metrics(path: str = METRICS_PATH):
    """Return structurally valid base-model metrics, or ({}, False).

    The availability flag validates the file's structure; it does not verify
    when the metrics were generated or independently reproduce their results.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}, False
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        LOG.warning("Could not read model metrics from %s: %s", path, exc)
        return {}, False

    if _valid_metrics_payload(data):
        return data, True
    LOG.warning("Model metrics at %s have an invalid structure", path)
    return {}, False


def model_metrics_labels(metrics_available: bool) -> tuple[str, str]:
    """Return explicit scope labels for the separate base-model metric display."""
    status = (
        "Base-model test metrics only — not fusion validation; generation time unverified"
        if metrics_available
        else "Metrics unavailable — models/metrics.json is missing, unreadable, or invalid"
    )
    scope = (
        "These are base-model metrics for their own dataset-specific targets, "
        "not flare-risk or fusion metrics."
    )
    return status, scope


def save_metrics(summary: dict, path: str = METRICS_PATH) -> bool:
    """Optional helper a training script can call to persist real
    cross-validation results, e.g.:
        save_metrics({"RA Clinical": {"Logistic Regression": {...}, ...}})
    so the Step-7 dashboard shows live figures instead of the demo table."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return True
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════
# 6. AUTHENTICATION & ASSESSMENT HISTORY (SQLite)
# ═════════════════════════════════════════════════════════════════════════

def _connect(db_path: Optional[str] = None):
    # DATABASE_URL selects persistent PostgreSQL in hosting; otherwise SQLite.
    return connect_database(db_path or DB_PATH)


def init_db(db_path: str = DB_PATH) -> None:
    with closing(_connect(db_path)) as conn, conn:
        execute_sql(conn, """
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                salt         TEXT NOT NULL,
                pw_hash      TEXT NOT NULL,
                display_name TEXT,
                created_at   TEXT NOT NULL
            )
        """)
        execute_sql(conn, """
            CREATE TABLE IF NOT EXISTS assessments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                city            TEXT,
                rapid3          REAL,
                fn_score        REAL,
                pain            REAL,
                global_estimate REAL,
                peak_risk       REAL,
                risk_level      TEXT
            )
        """)


def _hash_password(password: str, salt: Optional[str] = None):
    salt = salt or secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000).hex()
    return salt, pw_hash


def create_user(username: str, password: str, display_name: str = "") -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    salt, pw_hash = _hash_password(password)
    try:
        with closing(_connect()) as conn, conn:
            execute_sql(conn,
                "INSERT INTO users (username, salt, pw_hash, display_name, created_at) VALUES (?,?,?,?,?)",
                (username, salt, pw_hash, (display_name or username).strip(), datetime.now().isoformat()),
            )
        return True, "Account created — you can now log in."
    except Exception as exc:
        from database import INTEGRITY_ERRORS
        if isinstance(exc, INTEGRITY_ERRORS):
            return False, "That username is already taken."
        return False, f"Could not create account ({exc.__class__.__name__})."


def verify_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    try:
        with closing(_connect()) as conn:
            row = execute_sql(conn,
                "SELECT salt, pw_hash, display_name FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row:
            return False, "No account with that username."
        salt, stored_hash, display_name = row
        _, computed_hash = _hash_password(password, salt)
        if hmac.compare_digest(computed_hash, stored_hash):
            return True, display_name or username
        return False, "Incorrect password."
    except Exception as exc:
        return False, f"Login failed ({exc.__class__.__name__})."


def save_assessment(username: str, city: str, rapid3: float, fn_score: float, pain: float,
                     global_estimate: float, peak_risk: Optional[float], level: Optional[str]) -> bool:
    try:
        with closing(_connect()) as conn, conn:
            execute_sql(conn,
                """INSERT INTO assessments
                   (username, created_at, city, rapid3, fn_score, pain, global_estimate, peak_risk, risk_level)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                ((username or "guest").strip().lower(), datetime.now().isoformat(), city,
                 rapid3, fn_score, pain, global_estimate, peak_risk, level),
            )
        return True
    except Exception:
        return False



def get_history(username: str, limit: int | None = None):
    """Return recent-first account history; optionally limit the number of records."""
    try:
        with closing(_connect()) as conn:
            query = """
                SELECT created_at, city, rapid3, peak_risk, risk_level
                FROM assessments
                WHERE username = ?
                ORDER BY created_at DESC
            """
            params = [(username or "guest").strip().lower()]

            if limit is not None:
                limit = max(1, int(limit))
                query += " LIMIT ?"
                params.append(limit)

            rows = execute_sql(conn, query, tuple(params)).fetchall()
        return rows
    except Exception:
        LOG.exception("Failed to retrieve assessment history")
        return []


# ═════════════════════════════════════════════════════════════════════════
# 7. DEMO / QUICK-VIVA SAMPLE DATA
# ═════════════════════════════════════════════════════════════════════════

DEMO_PATIENT = dict(
    name="Demo Patient (Evaluator)",
    age=52,
    city="Colombo",
    fn_vals={"A": 1, "B": 1, "C": 0, "D": 2, "E": 1, "F": 2, "G": 0, "H": 1, "I": 2, "J": 1},
    sleep_diff=2,
    anxiety_val=1,
    fatigue=6.0,
    pain=6.5,
    global_s=6.0,
    flares30=4,
    duration=7,
    activity=15,
    heart_rate=88,
)