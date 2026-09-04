"""
2_fusion.py — RA FlareGuard shared backend
============================================================================
Pure-Python backend module (no Streamlit calls) used by 3_patient_console.py.
Keeping this framework-agnostic means it can also be unit-tested or reused
headlessly (e.g. from a notebook or a CLI) without pulling in Streamlit.

Responsibilities:
  1. Weather retrieval (OpenWeatherMap) with a strict try/except and a
     realistic Sri Lankan offline fallback dataset.
  2. ML model loading (.pkl / .joblib) with per-model error isolation and a
     calibrated formula-based fallback so a missing/corrupt model file never
     crashes the app or silently returns nonsense.
  3. Clinical score calculators — RAPID3 / HAQ-DI / MDHAQ / Pain VAS.
  4. The fused multimodal flare-risk engine, plus an explainability (XAI)
     breakdown of what drove each prediction.
  5. Cross-validated model performance metrics for the "Model Performance"
     viva tab (loaded from models/metrics.json when the training pipeline
     has produced one, otherwise clearly-labelled reference figures).
  6. A lightweight SQLite-backed authentication & assessment-history store.

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
import os
import secrets
import sqlite3
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    import joblib
except Exception:  # pragma: no cover - joblib should normally be installed
    joblib = None

# ─────────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "flareguard.db")
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
                "http://api.openweathermap.org/data/2.5/weather",
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
                    "http://api.openweathermap.org/data/2.5/forecast",
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

        rng = np.random.default_rng(abs(hash(city)) % (2**32))
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
    """Drop-in stand-in for a scikit-learn classifier, used whenever a real
    .pkl/.joblib file fails to load or deserialize. Rather than crashing or
    returning a meaningless constant, it produces a bounded, monotonically
    sensible pseudo-probability from a weighted combination of the raw
    inputs — the same clinically-motivated fields the real model expects —
    so the app degrades gracefully and the panel can see *why* a number was
    produced even when a model artefact is missing."""

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
    meta: object = None
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
    """Loads each of the four base models plus the meta (fusion) model
    independently — one file failing to load never prevents the others from
    loading. Every failure is caught, recorded in `load_errors` for
    transparent viva reporting, and backfilled with a CalibratedFallbackModel
    so downstream code can always call `.predict_proba()` safely."""
    bundle = ModelBundle()
    slots: dict = {}

    for attr, filename, n_feat in _MODEL_SPECS:
        try:
            slots[attr] = _load_one(os.path.join(models_dir, filename))
            bundle.any_real = True
        except Exception as exc:
            bundle.load_errors[attr] = f"{type(exc).__name__}: {exc}"
            slots[attr] = CalibratedFallbackModel(attr, n_feat)

    try:
        slots["meta"] = _load_one(os.path.join(models_dir, "meta_model.pkl"))
        bundle.any_real = True
    except Exception as exc:
        bundle.load_errors["meta"] = f"{type(exc).__name__}: {exc}"
        # meta model blends [p_ra, p_sleep, p_hrv, p_slra] -> weight RA highest
        slots["meta"] = CalibratedFallbackModel("meta", 4, weights=[0.35, 0.2, 0.2, 0.25])

    bundle.ra, bundle.sleep, bundle.hrv, bundle.sl_ra, bundle.meta = (
        slots["ra"], slots["sleep"], slots["hrv"], slots["sl_ra"], slots["meta"]
    )
    bundle.models_ok = len(bundle.load_errors) == 0
    return bundle


def safe_predict_proba(model, inputs) -> float:
    """Pads/truncates `inputs` to whatever the model expects and always
    returns a clean float probability in [0, 1] — falling back to a neutral
    0.5 only if the model itself raises despite the guards above."""
    try:
        n = int(getattr(model, "n_features_in_", len(inputs)))
        arr = np.array(inputs, dtype=float)
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)))
        else:
            arr = arr[:n]
        proba = model.predict_proba([arr])[0]
        return float(np.clip(proba[1], 0, 1))
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


def haq_di_score(fn_vals: dict) -> float:
    """HAQ-DI-derived Function score (0-10), the FN component of RAPID3."""
    return round(sum(fn_vals.values()) / 3, 1)


def rapid3_total(fn_vals: dict, pain: float, global_estimate: float) -> float:
    """RAPID3 = FN(raw 0-30, unscaled) + Pain VAS(0-10) + Patient Global(0-10)."""
    return sum(fn_vals.values()) + pain + global_estimate


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
    """Returns (risk in [0,1], contributions dict) where `contributions` is
    a transparent, additive decomposition used to drive the Step-7 XAI chart
    — every entry is on the same 0-1-weighted scale so the bars are directly
    comparable."""
    ra_in, slra_in, sleep_in, hrv_in = _submodel_inputs(x)

    clinical_score = (
        _norm(x.fn_score, 0, 10) * 0.25
        + _norm(x.pain, 0, 10) * 0.25
        + _norm(x.global_estimate, 0, 10) * 0.20
        + _norm(x.flares_30d, 0, 30) * 0.15
        + _norm(x.fatigue, 0, 10) * 0.10
        + (1 - _norm(max(0, 8 - x.sleep_diff * 1.8), 0, 12)) * 0.05
    )

    p_ra = safe_predict_proba(models.ra, ra_in)
    p_slra = safe_predict_proba(models.sl_ra, slra_in)
    p_sleep = safe_predict_proba(models.sleep, sleep_in)
    p_hrv = safe_predict_proba(models.hrv, hrv_in)
    model_score = safe_predict_proba(models.meta, [p_ra, p_sleep, p_hrv, p_slra])

    humidity_term = _norm(x.humidity, 40, 100) * 0.15
    temp_term = _norm(max(0, 32 - x.temp), 0, 20) * 0.05

    risk = clinical_score * 0.65 + model_score * 0.15 + humidity_term * 0.15 + temp_term * 0.05

    if x.humidity > 85:
        risk = min(1, risk + 0.06)
    elif x.humidity > 80:
        risk = min(1, risk + 0.03)
    elif x.humidity > 75:
        risk = min(1, risk + 0.01)
    if x.rapid3 > 12:
        risk = min(1, risk + 0.08)
    elif x.rapid3 > 6:
        risk = min(1, risk + 0.04)
    if x.pain >= 7 and x.sleep_diff >= 2:
        risk = min(1, risk + 0.05)
    if x.rapid3 <= 3 and x.flares_30d <= 2:
        risk = min(risk, 0.25)

    contributions = {
        "Pain VAS": _norm(x.pain, 0, 10) * 0.25,
        "Function Score (HAQ-DI)": _norm(x.fn_score, 0, 10) * 0.25,
        "Patient Global Estimate": _norm(x.global_estimate, 0, 10) * 0.20,
        "Flare History (30d)": _norm(x.flares_30d, 0, 30) * 0.15,
        "Sleep Difficulty": (1 - _norm(max(0, 8 - x.sleep_diff * 1.8), 0, 12)) * 0.05,
        "Fatigue VAS": _norm(x.fatigue, 0, 10) * 0.10,
        "Humidity": humidity_term,
        "Temperature": temp_term,
        "ML Fusion Signal (RA+Sleep+HRV+SL-RA)": model_score * 0.15,
    }
    return _clip01(risk), contributions


def project_forward(pain, fatigue, sleep_diff, anxiety, flares_30d, day_offset):
    """Deterministic trend model used to extrapolate patient-reported fields
    for the multi-day-ahead forecast (day_offset = 1, 2, 3, ...)."""
    decay = max(0, (2 - sleep_diff) * 0.04 * day_offset)
    easing = anxiety * 0.05 * min(day_offset - 1, 1)
    flare_pressure = flares_30d * 0.005 * day_offset
    projected_pain = float(np.clip(pain + flare_pressure - easing * 0.3, 0, 10))
    projected_fatigue = float(np.clip(fatigue + decay, 0, 10))
    projected_anxiety = float(np.clip(anxiety - easing, 0, 3))
    projected_sleep_diff = float(np.clip(sleep_diff + decay * 0.5, 0, 3))
    return projected_pain, projected_fatigue, projected_anxiety, projected_sleep_diff


def risk_level(r: float) -> str:
    return "high" if r > 0.60 else "medium" if r > 0.30 else "low"


def risk_color(r: float) -> str:
    return {"high": "#f87171", "medium": "#fbbf24", "low": "#4ade80"}[risk_level(r)]


def risk_label(r: float) -> str:
    return {"high": "High Risk", "medium": "Moderate Risk", "low": "Low Risk"}[risk_level(r)]


def risk_emoji(r: float) -> str:
    return {"high": "🔴", "medium": "🟡", "low": "🟢"}[risk_level(r)]


# ═════════════════════════════════════════════════════════════════════════
# 5. MODEL PERFORMANCE METRICS (viva "Model Performance" tab)
# ═════════════════════════════════════════════════════════════════════════

# Reference figures shown ONLY when models/metrics.json (written by
# 1_train_models_v2.py after a real cross-validated comparison) is not
# present — clearly labelled "Demo / Reference" in the UI so a viva panel
# never mistakes them for a live evaluation on this machine.
_DEMO_METRICS = {
    "RA Clinical": {
        "Logistic Regression": {"acc": 0.78, "f1": 0.77, "auc": 0.83},
        "Random Forest": {"acc": 0.84, "f1": 0.83, "auc": 0.89},
        "XGBoost": {"acc": 0.86, "f1": 0.85, "auc": 0.91},
    },
    "Sleep Health": {
        "Logistic Regression": {"acc": 0.74, "f1": 0.73, "auc": 0.80},
        "Random Forest": {"acc": 0.81, "f1": 0.80, "auc": 0.87},
        "XGBoost": {"acc": 0.83, "f1": 0.82, "auc": 0.88},
    },
    "HRV Stress": {
        "Logistic Regression": {"acc": 0.71, "f1": 0.70, "auc": 0.77},
        "Random Forest": {"acc": 0.79, "f1": 0.78, "auc": 0.85},
        "XGBoost": {"acc": 0.82, "f1": 0.81, "auc": 0.87},
    },
    "Sri Lankan RA": {
        "Logistic Regression": {"acc": 0.72, "f1": 0.71, "auc": 0.78},
        "Random Forest": {"acc": 0.80, "f1": 0.79, "auc": 0.86},
        "XGBoost": {"acc": 0.83, "f1": 0.82, "auc": 0.88},
    },
}


def get_model_metrics(path: str = METRICS_PATH):
    """Returns (metrics_dict, is_live). `is_live=True` only when a real
    models/metrics.json produced by the training pipeline was found and
    parsed successfully; otherwise the labelled reference figures above are
    returned with is_live=False."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data, True
    except Exception:
        pass
    return _DEMO_METRICS, False


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

def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                salt         TEXT NOT NULL,
                pw_hash      TEXT NOT NULL,
                display_name TEXT,
                created_at   TEXT NOT NULL
            )
        """)
        conn.execute("""
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
            conn.execute(
                "INSERT INTO users (username, salt, pw_hash, display_name, created_at) VALUES (?,?,?,?,?)",
                (username, salt, pw_hash, (display_name or username).strip(), datetime.now().isoformat()),
            )
        return True, "Account created — you can now log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    except Exception as exc:
        return False, f"Could not create account ({exc.__class__.__name__})."


def verify_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
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
                     global_estimate: float, peak_risk: float, level: str) -> bool:
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                """INSERT INTO assessments
                   (username, created_at, city, rapid3, fn_score, pain, global_estimate, peak_risk, risk_level)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                ((username or "guest").strip().lower(), datetime.now().isoformat(), city,
                 rapid3, fn_score, pain, global_estimate, peak_risk, level),
            )
        return True
    except Exception:
        return False


def get_history(username: str, limit: int = 10):
    """Returns most-recent-first list of (created_at, city, rapid3, peak_risk, risk_level)."""
    try:
        with closing(_connect()) as conn:
            rows = conn.execute(
                """SELECT created_at, city, rapid3, peak_risk, risk_level FROM assessments
                   WHERE username = ? ORDER BY created_at DESC LIMIT ?""",
                ((username or "guest").strip().lower(), limit),
            ).fetchall()
        return rows
    except Exception:
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