"""
RA FlareGuard — Patient Console (Research Prototype)
=============================================================================
Clinical standards:
  RAPID3      Pincus et al., J Rheumatol 2008;35:2136
  MDHAQ       Pincus & Swearingen, Arthritis Rheum 1999;42:2220
  HAQ-DI      Fries et al., Arthritis Rheum 1980;23:137
  VAS         Huskisson, Lancet 1974;2:1127
  Fatigue VAS Wolfe et al., J Rheumatol 1996;23:1407
  RADAI       Stucki et al., Arthritis Rheum 1995;38:795
  DAS28       Prevoo et al., Arthritis Rheum 1995;38:44 (DAS28-ESR/CRP)

This file is the Streamlit presentation layer only. Base-model loading,
weather retrieval, RAPID3/HAQ-DI scoring, experimental heuristic risk scoring,
and persistence
of the *existing* fields live in the framework-agnostic backend module
`2_fusion.py`, loaded below via importlib (its filename starts with a
digit, so it can't be `import`-ed directly as a regular Python identifier).

Backend call signatures, argument order, and score-calculation usage from the
original console are preserved in this rewrite — only the Streamlit
presentation/production layer around them was touched.

PRODUCTION NOTES (read before a live clinical / viva deployment)
-------------------------------------------------------------------------
1. De-identification: no field in this console ever collects a real
   patient name. Every record is keyed on a self-chosen, de-identified
   "Patient ID / Subject Code" (e.g. PATIENT_001).
2. Zero pre-fills: every clinical input starts at its scale minimum (0)
   or empty. The one exception is Heart Rate: the on-screen default is
   0 ("not measured"), but a physiologically implausible 0 bpm is never
   sent to the risk model — see `_safe_heart_rate()`, which substitutes
   a neutral resting-HR baseline *only* for the model call, never for
   what's displayed or stored.
3. DAS28 (swollen/tender joint counts + optional ESR/CRP) is collected
   and displayed for clinical completeness and audit, but by design is
   NOT wired into `fusion.compute_risk()` — the heuristic's existing input
   contract is left unchanged.
4. Auto-save: every form widget below is bound directly to
   `st.session_state` via its `key`, so a value is captured the instant
   it changes — not only when "Continue" is pressed. A browser refresh
   or crash mid-form loses nothing already answered.
5. Local persistence (zero-downtime): every completed assessment is
   written straight to a local SQLite database, `flareguard.db`, in the
   `patient_assessments` table — no network round-trip, no cloud
   provider, no outage window. `init_db()` auto-creates the database
   file and table on first run if they don't already exist. This
   replaces the previous Supabase-backed cloud audit log so the console
   keeps working end-to-end even with no internet connection at all.
   Local history via `fusion.save_assessment()` / `fusion.get_history()`
   remains the separate store the in-app "Previous Scores" panel reads
   from, unaffected by this change.
"""

import csv
import importlib.util
import logging
import math
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import streamlit as st

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("raflareguard")

APP_VERSION = "v2.7-experimental-score-clarified"
SITE_LABEL = "Patient Self-Assessment · Research Preview"

# ─────────────────────────────────────────────────────────────────────────
# Load the backend module (2_fusion.py) from disk
# ─────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION_PATH = os.path.join(_HERE, "2_fusion.py")
_LOCAL_DB_PATH = os.path.join(_HERE, "flareguard.db")

try:
    _spec = importlib.util.spec_from_file_location("fusion_backend", _FUSION_PATH)
    fusion = importlib.util.module_from_spec(_spec)
    sys.modules["fusion_backend"] = fusion   # required so dataclasses in 2_fusion.py resolve
    _spec.loader.exec_module(fusion)
    _BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit if 2_fusion.py itself is broken/missing
    fusion = None
    _BACKEND_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

st.set_page_config(
    page_title="RA FlareGuard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if fusion is None:
    st.error(
        "⚠️ **RA FlareGuard could not start** — the backend module `2_fusion.py` "
        f"failed to load.\n\n**Details:** {_BACKEND_IMPORT_ERROR}\n\n"
        "Make sure `2_fusion.py` is in the same folder as this file and try again."
    )
    st.stop()

try:
    import plotly.graph_objects as pgo
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

# ─────────────────────────────────────────────────────────────────────────
# LOCAL SQLITE PERSISTENCE LAYER — zero-downtime replacement for the
# previous cloud (Supabase) audit log. No network dependency: every
# assessment is committed straight to a local database file that lives
# next to this script. Every function here is defensive (never raises
# into the UI) so a locked/corrupt DB file degrades to a readable error
# banner instead of crashing the console mid-assessment.
# ─────────────────────────────────────────────────────────────────────────
_PATIENT_ASSESSMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    patient_id TEXT,
    pain_score INTEGER,
    swollen_joint_count INTEGER,
    tender_joint_count INTEGER,
    rapid3_score REAL,
    haq_di_score REAL,
    predicted_flare_risk REAL,
    risk_category TEXT
)
"""

_PATIENT_PROFILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_profiles (
    username TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    age INTEGER NOT NULL,
    city TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def get_patient_profile(username: str) -> dict | None:
    """Load the saved profile belonging to one registered account."""
    if not username or username.startswith("guest_") or username == "viva_evaluator":
        return None

    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT patient_id, age, city
            FROM patient_profiles
            WHERE username = ?
            """,
            (username.strip().lower(),),
        ).fetchone()

        if row is None:
            return None

        return {
            "patient_id": row[0],
            "age": int(row[1]),
            "city": row[2],
        }
    finally:
        conn.close()


def save_patient_profile(username: str, patient_id: str, age: int, city: str) -> bool:
    """Create or update a registered user's patient profile."""
    if (
        not username
        or username.startswith("guest_")
        or username == "viva_evaluator"
    ):
        return False

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO patient_profiles
                (username, patient_id, age, city, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                patient_id = excluded.patient_id,
                age = excluded.age,
                city = excluded.city,
                updated_at = excluded.updated_at
            """,
            (
                username.strip().lower(),
                patient_id.strip(),
                int(age),
                city.strip(),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return True
    except Exception as exc:
        LOG.error("Patient profile save failed: %s", exc)
        return False
    finally:
        conn.close()


def get_db_connection():
    """One short-lived connection per call — SQLite handles this cheaply
    and it avoids holding a connection open across Streamlit reruns."""
    conn = sqlite3.connect(_LOCAL_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  # safer for concurrent Streamlit reruns
    return conn



def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.execute(_PATIENT_ASSESSMENTS_SCHEMA)
        conn.execute(_PATIENT_PROFILES_SCHEMA)
        conn.commit()
    finally:
        conn.close()



def save_patient_record(data_dict: dict) -> bool:
    """Commits one row to the local patient_assessments table.
    `data_dict` keys map 1:1 onto the table's columns (any extra keys
    are ignored; missing keys are stored as NULL). Returns True on a
    confirmed commit, False on any failure — never raises."""
    columns = [
        "patient_id", "pain_score", "swollen_joint_count", "tender_joint_count",
        "rapid3_score", "haq_di_score", "predicted_flare_risk", "risk_category",
    ]
    values = [data_dict.get(c) for c in columns]
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO patient_assessments ({', '.join(columns)}) VALUES ({placeholders})"
    try:
        conn = get_db_connection()
        try:
            conn.execute(sql, values)
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:
        LOG.error("Local SQLite insert into patient_assessments failed: %s", exc)
        return False


def get_device_viewport() -> str:
    """Best-effort device/browser profile string for the audit trail.
    Uses st.context (Streamlit >=1.37) when available; never raises."""
    try:
        ua = st.context.headers.get("User-Agent", "")
        if ua:
            return ua[:180]
    except Exception:
        pass
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Local, theme-owned color mapping.
# The backend's fusion.risk_color()/fusion.rapid3_category() colors were
# tuned for a dark UI. We keep using fusion.risk_level()/fusion.rapid3_category()
# for their *labels* (unchanged scoring/category logic) but resolve display
# colors ourselves so the whole dashboard follows the light clinical palette.
# ─────────────────────────────────────────────────────────────────────────
LEVEL_STYLES = {
    "high":   dict(text="#991B1B", bg="#FEE2E2", border="#EF4444"),
    "medium": dict(text="#92400E", bg="#FEF3C7", border="#F59E0B"),
    "low":    dict(text="#065F46", bg="#ECFDF5", border="#10B981"),
}


def level_style(level, key=None):
    s = LEVEL_STYLES.get(level, LEVEL_STYLES["low"])
    return s.get(key) if key else s


# (cap, label, text, bg, border) — mirrors the original ≤3 / 4-6 / 7-12 / >12 tiers
R3_TIERS = [
    (3,   "Near Remission", "#065F46", "#ECFDF5", "#10B981"),
    (6,   "Low",            "#3730A3", "#EEF2FF", "#6366F1"),
    (12,  "Moderate",       "#92400E", "#FEF3C7", "#F59E0B"),
    (999, "High",           "#991B1B", "#FEE2E2", "#EF4444"),
]

# DAS28 (Prevoo et al. 1995) standard clinical cut-points
DAS28_TIERS = [
    (2.6, "Remission", "#065F46", "#ECFDF5", "#10B981"),
    (3.2, "Low",       "#3730A3", "#EEF2FF", "#6366F1"),
    (5.1, "Moderate",  "#92400E", "#FEF3C7", "#F59E0B"),
    (999, "High",      "#991B1B", "#FEE2E2", "#EF4444"),
]


def r3_style(r3_value):
    for cap, label, text, bg, border in R3_TIERS:
        if r3_value <= cap:
            return dict(label=label, text=text, bg=bg, border=border)
    last = R3_TIERS[-1]
    return dict(label=last[1], text=last[2], bg=last[3], border=last[4])


def das28_style(value):
    for cap, label, text, bg, border in DAS28_TIERS:
        if value <= cap:
            return dict(label=label, text=text, bg=bg, border=border)
    last = DAS28_TIERS[-1]
    return dict(label=last[1], text=last[2], bg=last[3], border=last[4])


def compute_das28(tjc, sjc, global_health_0_10, lab_type, lab_value):
    """Returns dict(score, style) or None if no lab value was provided.
    Standard DAS28-ESR / DAS28-CRP (Prevoo et al., Arthritis Rheum 1995).
    Joint counts and global health are clamped to 0–28 / 0–10 before use."""
    if lab_type not in ("ESR", "CRP") or lab_value is None or lab_value <= 0:
        return None
    tjc = max(0, min(28, tjc))
    sjc = max(0, min(28, sjc))
    gh100 = max(0.0, min(10.0, global_health_0_10)) * 10.0  # 0–10 → 0–100mm VAS
    try:
        if lab_type == "ESR":
            score = 0.56 * math.sqrt(tjc) + 0.28 * math.sqrt(sjc) + 0.70 * math.log(max(lab_value, 1.0)) + 0.014 * gh100
        else:  # CRP
            score = 0.56 * math.sqrt(tjc) + 0.28 * math.sqrt(sjc) + 0.36 * math.log(lab_value + 1.0) + 0.014 * gh100 + 0.96
    except (ValueError, ZeroDivisionError):
        return None
    return dict(score=round(score, 2), style=das28_style(score))


def _safe_heart_rate(raw_hr):
    """The on-screen default/minimum for Heart Rate is 0 ("not measured"),
    per the zero-pre-fill requirement — but 0 bpm is not a valid model
    input. Substitute a neutral resting-HR baseline for the model call
    only; the raw entered value (including 0) is still what's displayed
    and stored in the audit record."""
    return raw_hr if raw_hr and raw_hr > 0 else 72


def valid_patient_id(pid: str) -> bool:
    pid = (pid or "").strip()
    if len(pid) < 3 or len(pid) > 40:
        return False
    return all(c.isalnum() or c in "_-" for c in pid)


def clamp(value, lo, hi):
    """Defensive numeric clamp used before any value is handed to the
    scoring functions or the ML models, so a stray out-of-range value
    can never reach fusion.compute_risk() / fusion.rapid3_total()."""
    try:
        v = type(lo)(value)
    except Exception:
        v = value
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────
# Styling — light clinical dashboard theme
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════════════════
   DESIGN TOKENS — single source of truth for the whole console.
   Radius scale: 8px (controls/buttons) · 12px (cards/panels) · 999px (pills)
   ═══════════════════════════════════════════════════════════════════════ */
:root{
  --bg:#F8FAFC; --surface:#F1F5F9; --card:#FFFFFF;
  --border:#E2E8F0; --border2:#CBD5E1;
  --teal:#0EA5E9; --teal2:#0284C7;
  --blue:#0EA5E9; --violet:#6366F1;
  --amber:#F59E0B; --amber-bg:#FEF3C7; --amber-text:#92400E;
  --red:#EF4444;   --red-bg:#FEE2E2;   --red-text:#991B1B;
  --green:#10B981; --green-bg:#ECFDF5; --green-text:#065F46;
  --text:#0F172A;  --text2:#475569; --text3:#64748B;
  --focus:#0EA5E9;

  --radius-control:8px;
  --radius-card:12px;
  --radius-pill:999px;

  --space-page: clamp(14px, 4vw, 24px);
  --space-card: clamp(14px, 3vw, 20px);

  --touch-min:44px;

  --shadow-sm:0 1px 2px rgba(15,23,42,0.04);
  --shadow-card:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.04);
  --shadow-raised:0 8px 16px -4px rgba(15,23,42,0.10);
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,[class*="css"]{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,sans-serif!important;
  background:var(--bg)!important;
  color:var(--text)!important;
  -webkit-font-smoothing:antialiased;
  font-size:16px;
  line-height:1.5;
}

#MainMenu,footer,header{visibility:hidden}
[data-testid="stAppViewContainer"], [data-testid="stHeader"]{background:var(--bg)!important}

.block-container{
  padding:0!important;
  max-width:720px!important;
  margin:0 auto!important;
  width:100%;
}

a:focus-visible, button:focus-visible, input:focus-visible,
textarea:focus-visible, select:focus-visible,
[role="radio"]:focus-visible, [role="slider"]:focus-visible,
[tabindex]:focus-visible{
  outline:2px solid var(--focus)!important;
  outline-offset:2px!important;
  border-radius:var(--radius-control);
}
::selection{background:rgba(14,165,233,0.22)}

.env-banner{
  background:#0F172A;color:#E2E8F0;font-size:0.68rem;font-weight:600;
  letter-spacing:0.04em;text-align:center;padding:6px var(--space-page);
  display:flex;justify-content:center;gap:10px;flex-wrap:wrap;
}
.env-banner b{color:#7DD3FC}
.env-banner .sep{color:#475569}

.live-ticker{
  position:sticky;top:0;z-index:101;
  background:var(--card);border-bottom:1px solid var(--border);
  padding:10px var(--space-page);display:flex;gap:14px;flex-wrap:wrap;
  font-size:0.72rem;color:var(--text2);
  box-shadow:var(--shadow-sm);
}
.live-ticker b{color:var(--teal2);font-weight:700}

.prog-wrap{
  position:sticky;top:0;z-index:100;
  background:var(--bg);border-bottom:1px solid var(--border);
  padding:12px 0 10px;
}
.prog-inner{max-width:720px;margin:0 auto;padding:0 var(--space-page)}
.prog-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:10px;flex-wrap:wrap}
.prog-label{font-size:0.72rem;font-weight:700;color:var(--teal2);text-transform:uppercase;letter-spacing:0.1em}
.prog-step{font-size:0.72rem;color:var(--text3);font-weight:600;white-space:nowrap}
.prog-track{height:4px;background:var(--border);border-radius:var(--radius-pill);overflow:hidden}
.prog-fill{height:100%;border-radius:var(--radius-pill);background:linear-gradient(90deg,var(--teal2),var(--teal));transition:width 0.4s ease}
.prog-dots{display:flex;gap:5px;margin-top:8px}
.prog-dot{flex:1;height:3px;border-radius:var(--radius-pill);background:var(--border)}
.prog-dot.done{background:var(--teal)}
.prog-dot.active{background:var(--teal2)}
.autosave-tag{font-size:0.62rem;color:var(--text3);display:flex;align-items:center;gap:4px;white-space:nowrap}
.autosave-dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block}

.step-wrap{padding:var(--space-page)}
.step-eyebrow{
  font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
  color:var(--teal2);margin-bottom:8px;
}
.step-title{
  font-family:'Inter',system-ui,sans-serif;font-weight:800;
  font-size:clamp(1.35rem, 1.05rem + 1.3vw, 1.75rem);
  color:var(--text);line-height:1.2;margin-bottom:8px;letter-spacing:-0.01em;
}
.step-title em{color:var(--teal2);font-style:normal}
.step-desc{font-size:0.86rem;color:var(--text2);line-height:1.65;margin-bottom:12px;max-width:60ch}
.step-cite{
  display:inline-flex;align-items:center;gap:5px;background:#EFF6FF;border:1px solid #DBEAFE;
  border-radius:var(--radius-control);padding:5px 10px;font-size:0.66rem;color:var(--teal2);
  margin-bottom:12px;font-weight:500;
}

.qcard{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius-card);
  padding:var(--space-card);margin-bottom:12px;box-shadow:var(--shadow-card);
}
.qcard-hdr{
  font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
  color:var(--text2);margin-bottom:10px;display:flex;align-items:center;gap:7px;
}
.qcard-hdr::before{content:'';width:3px;height:12px;background:var(--teal);border-radius:2px;display:inline-block}

.field-help{font-size:0.72rem;color:var(--text3);line-height:1.5;margin-top:4px}

.q-item{padding:10px 0;border-bottom:1px solid var(--border)}
.q-item:last-child{border-bottom:none}
.q-top{display:flex;gap:9px;margin-bottom:7px}
.q-badge{
  width:24px;height:24px;background:var(--surface);border:1px solid var(--border2);
  border-radius:var(--radius-control);display:flex;align-items:center;justify-content:center;
  font-size:0.65rem;font-weight:800;color:var(--teal2);flex-shrink:0;margin-top:1px;
}
.q-text{font-size:0.86rem;color:var(--text);line-height:1.5;font-weight:500}
.q-ref{font-size:0.65rem;color:var(--text3);margin-top:2px}

.vas-grad{height:6px;border-radius:var(--radius-pill);background:linear-gradient(to right,var(--green),var(--amber),var(--red));margin:6px 0 4px}
.vas-ends{display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text3);margin-bottom:8px;font-weight:500}

.scale-chips{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 10px}
.scale-chip{display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:var(--radius-pill);font-size:0.7rem;font-weight:600}

.nav-bar{background:var(--bg);border-top:1px solid var(--border);padding:14px 0;margin-top:6px;
  position:sticky;bottom:0;z-index:90;}
.nav-inner{max-width:720px;margin:0 auto;padding:0 var(--space-page);display:flex;gap:10px}

.stButton>button,
div[data-testid="stFormSubmitButton"] button,
.stDownloadButton>button{
  font-family:'Inter',system-ui,sans-serif!important;
  font-size:0.88rem!important;font-weight:600!important;
  border-radius:var(--radius-control)!important;
  padding:0.7rem 1.4rem!important;
  min-height:var(--touch-min)!important;
  transition:all 0.15s ease!important;border:none!important;
  width:100%;
}
.stButton>button[kind="primary"],
.stButton>button:not([kind]),
div[data-testid="stFormSubmitButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
div[data-testid="stFormSubmitButton"] button:not([kind]){
  background:linear-gradient(135deg,var(--teal2),var(--teal))!important;
  color:#fff!important;
  box-shadow:0 3px 12px rgba(2,132,199,0.25)!important;
}
.stButton>button[kind="primary"]:hover,
.stButton>button:not([kind]):hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stFormSubmitButton"] button:not([kind]):hover{
  transform:translateY(-1px)!important;
  box-shadow:0 5px 16px rgba(2,132,199,0.35)!important;
}
.stButton>button[kind="primary"]:active,
.stButton>button:not([kind]):active,
div[data-testid="stFormSubmitButton"] button:active{transform:translateY(0)!important}
.stButton>button[kind="primary"]:disabled,
.stButton>button:not([kind]):disabled,
div[data-testid="stFormSubmitButton"] button:disabled{
  opacity:0.5!important;box-shadow:none!important;cursor:not-allowed!important;transform:none!important;
}
.stButton>button[kind="secondary"],
div[data-testid="stFormSubmitButton"] button[kind="secondary"],
div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"]{
  background:var(--card)!important;color:var(--text2)!important;
  border:1px solid var(--border2)!important;
  box-shadow:none!important;
}
.stButton>button[kind="secondary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="secondary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"]:hover{
  border-color:var(--teal)!important;color:var(--teal2)!important;background:#F0F9FF!important;
}
.stButton>button:focus-visible,
div[data-testid="stFormSubmitButton"] button:focus-visible{
  outline:2px solid var(--focus)!important;outline-offset:2px!important;
}

div[data-testid="stSlider"] div[role="slider"],
div[data-testid="stSelectSlider"] div[role="slider"]{
  background:var(--teal)!important;border:2px solid #fff!important;
  box-shadow:0 0 0 3px rgba(14,165,233,0.18)!important;
  min-width:20px!important;min-height:20px!important;
}
div[data-testid="stSlider"] div[role="slider"]:focus-visible,
div[data-testid="stSelectSlider"] div[role="slider"]:focus-visible{
  box-shadow:0 0 0 4px rgba(14,165,233,0.35)!important;
}

.stTextInput input,.stNumberInput input,.stTextInput>div>div>input,.stTextArea textarea{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:var(--radius-control)!important;
  font-family:'Inter',system-ui,sans-serif!important;font-size:0.9rem!important;
  padding:11px 14px!important;
  min-height:var(--touch-min)!important;
}
.stTextArea textarea{min-height:80px!important}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{
  border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(14,165,233,0.12)!important;outline:none!important;
}
.stTextInput input:disabled,.stNumberInput input:disabled{
  background:var(--surface)!important;color:var(--text3)!important;cursor:not-allowed!important;
}
.stNumberInput button{
  background:var(--card)!important;border-color:var(--border2)!important;color:var(--text2)!important;
  min-width:var(--touch-min)!important;
}

.stSelectbox>div>div{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:var(--radius-control)!important;
  font-family:'Inter',system-ui,sans-serif!important;font-size:0.9rem!important;
  min-height:var(--touch-min)!important;
}
.stSelectbox>div>div:focus-within{
  border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(14,165,233,0.12)!important;
}

label{color:var(--text2)!important;font-size:0.8rem!important;font-family:'Inter',system-ui,sans-serif!important;font-weight:600!important}
.stMarkdown p{color:var(--text2)!important;font-size:0.84rem!important}

[data-testid="stCaptionContainer"], .stCaption{
  color:var(--text3)!important;font-size:0.74rem!important;line-height:1.5!important;
}

div[data-testid="stAlert"]{
  border-radius:var(--radius-card)!important;
  border:1px solid transparent!important;
  padding:14px 16px!important;
  font-size:0.84rem!important;
}
div[data-testid="stAlertContentSuccess"], div[data-baseweb="notification"][kind="success"]{color:var(--green-text)!important}
div[data-testid="stAlertContentError"]{color:var(--red-text)!important}
.stSuccess{background:var(--green-bg)!important;border-color:#6EE7B7!important}
.stError{background:var(--red-bg)!important;border-color:#FCA5A5!important}
.stWarning{background:var(--amber-bg)!important;border-color:#FCD34D!important}
.stInfo{background:#EFF6FF!important;border-color:#BFDBFE!important}

.result-hero{text-align:center;padding:clamp(20px,5vw,30px) 0 22px}
.result-name{font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--text3);margin-bottom:8px}
.result-score{font-family:'Inter',system-ui,sans-serif;font-weight:800;font-size:clamp(3rem,8vw,4.6rem);line-height:1;margin-bottom:8px;letter-spacing:-0.02em}
.result-pill{display:inline-block;font-size:0.76rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:6px 16px;border-radius:var(--radius-pill);margin-bottom:12px}
.result-pill.high  {background:var(--red-bg);color:var(--red-text);border:1px solid #FCA5A5}
.result-pill.medium{background:var(--amber-bg);color:var(--amber-text);border:1px solid #FCD34D}
.result-pill.low   {background:var(--green-bg);color:var(--green-text);border:1px solid #6EE7B7}
.result-meta{font-size:0.78rem;color:var(--text3)}

.fc-row{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:22px 0}
.fc-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius-card);padding:14px 8px;text-align:center;position:relative;overflow:hidden;box-shadow:var(--shadow-card)}
.fc-box.now{border-color:var(--teal)}
.fc-box.now::before{content:'NOW';position:absolute;top:0;left:0;right:0;background:var(--teal2);color:#fff;font-size:0.52rem;font-weight:800;letter-spacing:0.2em;padding:3px 0}
.fc-day{font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text3);margin-top:5px}
.fc-pct{font-family:'Inter',system-ui,sans-serif;font-weight:800;font-size:clamp(1.4rem,4.5vw,1.9rem);line-height:1.1;margin:4px 0}
.fc-badge{font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;padding:2px 8px;border-radius:var(--radius-pill);display:inline-block;margin-bottom:5px}
.fc-badge.high  {background:var(--red-bg);color:var(--red-text)}
.fc-badge.medium{background:var(--amber-bg);color:var(--amber-text)}
.fc-badge.low   {background:var(--green-bg);color:var(--green-text)}
.fc-wx{font-size:0.62rem;color:var(--text3);line-height:1.4}
.fc-live{font-size:0.54rem;color:var(--teal2);margin-top:4px;font-weight:600}

.r3-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius-card);padding:var(--space-card);margin:14px 0;box-shadow:var(--shadow-card)}
.r3-row{margin-bottom:14px}
.r3-top{display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:5px;gap:8px;flex-wrap:wrap}
.r3-lbl{color:var(--text2)}.r3-val{color:var(--text);font-weight:600}
.r3-track{height:6px;background:var(--surface);border-radius:var(--radius-pill);overflow:hidden}
.r3-fill{height:100%;border-radius:var(--radius-pill);transition:width 0.6s ease}
.r3-total{border-top:1px solid var(--border);padding-top:14px;margin-top:6px}
.r3-cats{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:14px}
.r3-cat{border-radius:var(--radius-control);padding:8px 4px;text-align:center}
.r3-catlbl{font-size:0.54rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.06em;font-weight:600}
.r3-catval{font-size:0.72rem;font-weight:700;margin-top:3px}

.adv{border-radius:var(--radius-card);padding:var(--space-card);margin:14px 0;border-left:4px solid transparent;box-shadow:var(--shadow-card)}
.adv.high  {background:var(--red-bg);border-left-color:var(--red);color:var(--red-text)}
.adv.medium{background:var(--amber-bg);border-left-color:var(--amber);color:var(--amber-text)}
.adv.low   {background:var(--green-bg);border-left-color:var(--green);color:var(--green-text)}
.adv-hdr{font-size:0.92rem;font-weight:700;margin-bottom:12px;color:var(--text)}
.adv ul{list-style:none;padding:0}
.adv li{font-size:0.82rem;color:var(--text2);padding:6px 0;border-bottom:1px solid rgba(15,23,42,0.06);display:flex;gap:9px;line-height:1.55}
.adv li:last-child{border-bottom:none}
.adv-dot{flex-shrink:0;margin-top:5px;width:5px;height:5px;border-radius:50%}
.adv-dot.high  {background:var(--red)}
.adv-dot.medium{background:var(--amber)}
.adv-dot.low   {background:var(--green)}

.chips{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.chip{background:var(--card);border:1px solid var(--border);border-radius:var(--radius-control);padding:6px 12px;font-size:0.72rem;color:var(--text2);display:flex;align-items:center;gap:5px;box-shadow:var(--shadow-sm);min-height:32px}
.chip b{color:var(--text);font-weight:600}
.chip.ok{border-color:#6EE7B7;background:var(--green-bg);color:var(--green-text)}
.chip.warn{border-color:#FCD34D;background:var(--amber-bg);color:var(--amber-text)}

.disc{margin-top:16px;padding:14px 16px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-control);font-size:0.7rem;color:var(--text2);line-height:1.65}
.disc b{color:var(--text)}

.demo-banner{background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid var(--teal2);border-radius:var(--radius-control);padding:12px 16px;
  font-size:0.78rem;color:#0C4A6E;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.demo-banner b{color:#0C4A6E}

.privacy-banner{background:#F0FDFA;border:1px solid #99F6E4;border-left:4px solid #0D9488;border-radius:var(--radius-control);padding:12px 16px;
  font-size:0.76rem;color:#134E4A;margin-bottom:16px;line-height:1.55}
.privacy-banner b{color:#0F766E}

.model-status{display:inline-flex;align-items:center;gap:5px;font-size:0.68rem;font-weight:600;padding:5px 11px;border-radius:var(--radius-pill)}
.model-status.ok{background:var(--green-bg);color:var(--green-text);border:1px solid #6EE7B7}
.model-status.fallback{background:var(--amber-bg);color:var(--amber-text);border:1px solid #FCD34D}

.auth-wrap{padding:var(--space-page)}
.auth-hero{text-align:center;padding:14px 0 20px}

/* ── Single-row 0–10 / 0–3 rating scales — never wraps on any viewport ── */
div[data-testid="stRadio"] > div[role="radiogroup"]{
  display:flex; flex-wrap:nowrap; gap:6px; margin:8px 0 4px;
  overflow-x:auto; -webkit-overflow-scrolling:touch; scrollbar-width:thin;
  padding-bottom:2px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label{
  position:relative; flex:1 1 0; min-width:30px; min-height:var(--touch-min);
  display:flex; align-items:center; justify-content:center;
  background:var(--card); border:1.5px solid var(--border2); border-radius:var(--radius-control);
  padding:8px 2px; cursor:pointer; transition:all .15s ease; margin:0!important;
  white-space:nowrap;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover{border-color:var(--teal)}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(135deg,var(--teal2),var(--teal));
  border-color:var(--teal2); box-shadow:0 3px 10px rgba(2,132,199,0.25);
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p{color:#fff!important;font-weight:700!important}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:focus-visible){
  outline:2px solid var(--focus); outline-offset:2px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child{
  position:absolute; opacity:0; pointer-events:none; width:1px; height:1px; overflow:hidden;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label p{
  color:var(--text2)!important; font-size:0.72rem!important; font-weight:600!important;
  text-align:center; margin:0!important; white-space:nowrap;
}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"]{flex-wrap:nowrap}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label{flex:1 1 0; min-width:26px; min-height:40px; padding:8px 1px}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label p{font-size:0.68rem!important}
div[data-testid="stVerticalBlock"]{ gap:0.4rem!important; }
div[data-testid="element-container"]{ margin-bottom:0!important; }

.stTabs [data-baseweb="tab-list"]{gap:4px}
.stTabs [data-baseweb="tab"]{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  border-radius:var(--radius-control) var(--radius-control) 0 0!important;color:var(--text2)!important;
  min-height:var(--touch-min)!important;display:flex!important;align-items:center!important;
}
.stTabs [aria-selected="true"]{color:var(--teal2)!important;border-color:var(--teal)!important}

.streamlit-expanderHeader{
  background:var(--card)!important;border-radius:var(--radius-control)!important;color:var(--text)!important;
  min-height:var(--touch-min)!important;
}

@media (min-width:960px){
  .block-container{max-width:760px!important}
  .step-wrap{padding:32px 8px}
  .qcard{padding:24px 28px}
}

@media (max-width:640px){
  .fc-row{grid-template-columns:repeat(2,1fr);gap:8px}
  .r3-cats{grid-template-columns:repeat(2,1fr)}
  .nav-inner{flex-direction:column}
  .nav-inner .stButton>button,
  .nav-inner div[data-testid="stFormSubmitButton"] button{width:100%}
  .result-score{font-size:3rem}
  .step-title{font-size:1.4rem}
  .live-ticker{gap:10px;font-size:0.68rem}
  .chips{gap:6px}
  div[data-testid="stRadio"] > div[role="radiogroup"] > label{min-width:26px}
}

@media (max-width:400px){
  .fc-row{grid-template-columns:repeat(2,1fr)}
  .r3-cats{grid-template-columns:repeat(2,1fr)}
  :root{--space-page:12px;--space-card:14px}
  div[data-testid="stRadio"] > div[role="radiogroup"] > label{min-width:22px;padding:8px 0}
  div[data-testid="stRadio"] > div[role="radiogroup"] > label p{font-size:0.62rem!important}
}

@media (prefers-reduced-motion: reduce){
  *{transition:none!important;animation:none!important}
}

/* ═══════════════════════════════════════════════════════════════════════
   FIX (login-page look): Streamlit wraps every text box in its own BaseWeb
   container. The rules above styled only the inner <input>, so the outer
   container kept the browser/OS theme (dark border + dark eye-icon box).
   Style the wrapper too, and draw ONE clean border instead of two.
   ═══════════════════════════════════════════════════════════════════════ */
div[data-baseweb="input"],
div[data-baseweb="textarea"]{
  background:var(--card)!important;
  border:1px solid var(--border2)!important;
  border-radius:var(--radius-control)!important;
  box-shadow:none!important;overflow:hidden;
}
div[data-baseweb="base-input"]{
  background:transparent!important;border:none!important;box-shadow:none!important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
  background:transparent!important;border:none!important;box-shadow:none!important;
}
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus{
  border:none!important;box-shadow:none!important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within{
  border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(14,165,233,0.12)!important;
}
/* password show/hide (eye) button */
.stTextInput div[data-baseweb="input"] button{
  background:transparent!important;border:none!important;box-shadow:none!important;
  color:var(--text2)!important;
}
.stTextInput div[data-baseweb="input"] button:hover{background:var(--surface)!important;color:var(--teal2)!important}

/* Log In / Sign Up / Guest tabs: equal width, teal accent (was red default) */
.stTabs [data-baseweb="tab-list"]{gap:6px!important;width:100%}
.stTabs [data-baseweb="tab"]{flex:1 1 0!important;justify-content:center!important;padding:0 10px!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--teal)!important}
.stTabs [data-baseweb="tab-border"]{background:var(--border)!important}
/* hide the empty card outlines left by wrapper <div>s opened in one st.markdown call and closed in another */
.qcard:empty{display:none!important}

/* ═══════════════════════════════════════════════════════════════════════
   READABILITY PASS — larger type, answer buttons that fill the row and line
   up with the question text (they were flush-left while text had 24px inset)
   ═══════════════════════════════════════════════════════════════════════ */
div[data-testid="stRadio"]{width:100%!important;box-sizing:border-box;padding:0 24px!important}
div[data-testid="stRadio"] > div[role="radiogroup"]{width:100%!important}
div[data-testid="stRadio"] > div[role="radiogroup"] > label{min-height:46px}
div[data-testid="stRadio"] > div[role="radiogroup"] > label p{font-size:0.9rem!important}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label{min-height:46px}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label p{font-size:0.9rem!important}
.q-text{font-size:1rem}
.q-ref{font-size:0.74rem}
.q-badge{font-size:0.74rem}
.step-desc{font-size:0.95rem}
.field-help{font-size:0.8rem}
.scale-chip{font-size:0.78rem}
.vas-ends{font-size:0.78rem}
.prog-label,.prog-step{font-size:0.8rem}
.step-eyebrow{font-size:0.74rem}
@media (max-width:400px){
  div[data-testid="stRadio"]{padding:0 12px!important}
  div[data-testid="stRadio"] > div[role="radiogroup"] > label p{font-size:0.74rem!important}
  .nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label p{font-size:0.78rem!important}
}

/* patient-facing simplification: hide clinical meta lines (citations stay in the results-page disclaimer) */
.q-ref,.step-cite,.scale-chips{display:none!important}
/* answer rows: Streamlit sizes st.radio to its content by default — make the row full width */
[data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[data-testid="element-container"]:has([data-testid="stRadio"]){width:100%!important}
div[data-testid="stRadio"] > div[role="radiogroup"]{overflow-x:visible!important}

/* ── FlareGuard application-wide dashboard refresh ── */
:root {
  --bg: #F5F8FB;
  --surface: #EDF3F7;
  --card: #FFFFFF;
  --border: #DCE6ED;
  --border2: #C5D5DF;
  --teal: #0F766E;
  --teal2: #115E59;
  --blue: #173B4D;
  --text: #142C3A;
  --text2: #526777;
  --text3: #718493;
  --focus: #0F766E;
  --radius-card: 16px;
  --shadow-card: 0 4px 18px rgba(23, 59, 77, 0.06);
}
.stApp {
  background: #F5F8FB;
}
[data-testid="stHeader"] {
  background: rgba(245,248,251,0.92);
}
[data-testid="stSidebar"] {
  background: #173B4D;
  border-right: 1px solid #254C5F;
}
[data-testid="stSidebar"] * {
  color: #F0F7FA;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #D6E5EC;
}
[data-testid="stSidebar"] div.stButton > button {
  width: 100%;
  text-align: left;
  justify-content: flex-start;
  min-height: 44px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  background: rgba(255,255,255,0.055);
  color: #F0F7FA;
  font-weight: 600;
}
[data-testid="stSidebar"] div.stButton > button:hover {
  border-color: #73D6C6;
  background: rgba(115,214,198,0.14);
  color: #FFFFFF;
}
[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.18);
}
[data-testid="stMainBlockContainer"] {
  max-width: 1500px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}
div[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #DCE6ED;
  padding: 18px 20px;
  border-radius: 15px;
  box-shadow: 0 4px 18px rgba(23,59,77,0.04);
}
div[data-testid="stMetricLabel"] {
  color: #526777;
  font-weight: 600;
}
div[data-testid="stMetricValue"] {
  color: #173B4D;
}
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
  border-radius: 10px;
  min-height: 42px;
  font-weight: 600;
  transition: all 120ms ease;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stTextArea"] textarea {
  border-radius: 10px;
}
div[data-testid="stTabs"] button {
  font-weight: 600;
}
.fg-page-heading {
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: #173B4D;
  line-height: 1.2;
}
.fg-page-subtitle {
  color: #647887;
  margin-top: 6px;
  margin-bottom: 22px;
  font-size: 0.98rem;
}
.fg-panel {
  background: #FFFFFF;
  border: 1px solid #DCE6ED;
  border-radius: 16px;
  padding: 20px;
  margin: 8px 0 16px;
  box-shadow: 0 4px 18px rgba(23,59,77,0.04);
}
.fg-panel-title {
  font-size: 1.05rem;
  font-weight: 750;
  color: #173B4D;
  margin-bottom: 6px;
}
.fg-muted {
  color: #647887;
  font-size: 0.9rem;
}
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] {
    padding: 1rem 0.8rem 2rem;
  }
  .fg-page-heading {
    font-size: 1.55rem;
  }
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Constants sourced from the backend
# ─────────────────────────────────────────────────────────────────────────
CITIES = fusion.CITIES
RAPID3_Qs = fusion.RAPID3_QUESTIONS
DIFF_SHORT = fusion.DIFFICULTY_SHORT
TOTAL_STEPS = 6  # 0 Welcome · 1 Demographics · 2 RAPID3 · 3 DAS28 · 4 Lifestyle&Sensor · 5 Results
STEP_NAMES = ["Welcome", "Demographics", "Symptom Survey", "DAS28 Markers", "Lifestyle & Sensor", "Results"]

# ─────────────────────────────────────────────────────────────────────────
# Init backend resources (local DB + models) — cached per Streamlit
# session, guarded so a broken DB/model file shows a friendly message
# instead of a raw traceback during a live demo.
# ─────────────────────────────────────────────────────────────────────────
try:
    init_db()
    _DB_INIT_ERROR = None
except Exception as exc:
    _DB_INIT_ERROR = f"{type(exc).__name__}: {exc}"
    LOG.error("Local flareguard.db init failed: %s", _DB_INIT_ERROR)

if _DB_INIT_ERROR:
    st.error(
        "⚠️ **Local database could not be initialised.** Assessments can still be "
        "completed, but results won't be saved to `flareguard.db` this session. "
        f"(Technical detail logged: {_DB_INIT_ERROR})"
    )

# fusion's own local history mechanism (existing patient-account feature —
# unrelated to and unaffected by the flareguard.db audit table above).
try:
    fusion.init_db()
except Exception as exc:
    LOG.error("fusion.init_db() failed: %s", exc)


@st.cache_resource(show_spinner=False)
def get_models():
    return fusion.load_models()


try:
    MODELS = get_models()
    MODELS_OK = MODELS.models_ok
    _MODEL_INIT_ERROR = None
except Exception as exc:
    MODELS = None
    MODELS_OK = False
    _MODEL_INIT_ERROR = f"{type(exc).__name__}: {exc}"
    LOG.error("Model load failed: %s", _MODEL_INIT_ERROR)

weather_client = fusion.WeatherClient()

if _MODEL_INIT_ERROR:
    st.error(
        "⚠️ **Prediction models could not be loaded.** The console will still let you "
        "fill in the assessment, but the flare-risk forecast may be unavailable. "
        f"(Technical detail logged: {_MODEL_INIT_ERROR})"
    )

# ─────────────────────────────────────────────────────────────────────────
# Tap-to-select pill widgets — every widget is bound directly to a
# st.session_state key that matches the field it represents, so a value
# is captured in session_state the instant the user taps it (auto-save),
# not only when a "Continue" button is later pressed.
# ─────────────────────────────────────────────────────────────────────────
def diff_pills(field_key):
    """0-3 difficulty scale as tap-to-select pill buttons, auto-saving
    straight into st.session_state[field_key]."""
    opts = [0, 1, 2, 3]
    widget_key = f"w_{field_key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = clamp(st.session_state.get(field_key, 0), 0, 3)
    val = st.radio(field_key, options=opts, format_func=lambda x: DIFF_SHORT[x],
                    horizontal=True, key=widget_key, label_visibility="collapsed")
    st.session_state[field_key] = clamp(val, 0, 3)
    return st.session_state[field_key]


def nrs_pills(field_key):
    """0-10 numeric rating scale (validated NRS equivalent of a VAS) as
    tap-to-select pills, auto-saving straight into st.session_state[field_key]."""
    opts = list(range(11))
    widget_key = f"w_{field_key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = int(round(clamp(st.session_state.get(field_key, 0.0), 0.0, 10.0)))
    st.markdown('<div class="nrs-wrap">', unsafe_allow_html=True)
    val = st.radio(field_key, options=opts, format_func=lambda x: str(x),
                    horizontal=True, key=widget_key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state[field_key] = float(clamp(val, 0, 10))
    return st.session_state[field_key]

# ─────────────────────────────────────────────────────────────────────────
# Session state — every clinical field starts at a clean baseline (0 / empty)
# so a fresh session never biases a real patient toward a "demo-looking"
# answer. Quick Viva Demo Mode is the only path that pre-fills anything,
# and it's an explicit, clearly-labelled opt-in.
# ─────────────────────────────────────────────────────────────────────────
SENSITIVE_KEYS = [
    "patient_id", "age", "city", "fn_vals", "sleep_diff", "anxiety_val",
    "fatigue", "pain", "global_s", "flares30", "duration", "activity",
    "heart_rate", "das_tjc", "das_sjc", "das_lab_type", "das_lab_value",
    "results", "confirm_submit_cb",
]


def init_session():
    defaults = dict(
        authenticated=False, username=None, display_name=None, demo_mode=False,
        step=0, form_started_at=None,
        patient_id="", city="", age=0,
        fn_vals={l: 0 for l, _, _ in RAPID3_Qs},
        sleep_diff=0, anxiety_val=0,
        fatigue=0.0, pain=0.0, global_s=0.0,
        flares30=0, duration=0, activity=0, heart_rate=0,
        das_tjc=0, das_sjc=0, das_lab_type="Not available", das_lab_value=0.0,
        confirm_reset=False,
        results=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


def clear_sensitive_session():
    """Clinical-safety reset: wipes every patient-identifying / clinical
    field from RAM (not just the UI) so the next person at this device
    never sees a trace of the previous patient's data, without forcing a
    fresh login."""
    for k in list(st.session_state.keys()):
        if k in SENSITIVE_KEYS or k.startswith("w_"):
            del st.session_state[k]
    st.session_state.step = 0
    st.session_state.confirm_reset = False
    init_session()


def scoreboard():
    """Live-updating HAQ-DI / Pain / RAPID3 ticker shown at the top of every
    question step, reading the auto-saved values already in session_state —
    so it always reflects the very latest tap, with no separate save step."""
    fv = st.session_state.fn_vals
    p = st.session_state.pain
    g = st.session_state.global_s
    try:
        fn_score = fusion.haq_di_score(fv)
        r3 = fusion.rapid3_total(fv, p, g)
    except Exception as exc:
        LOG.error("Scoreboard calculation failed: %s", exc)
        st.markdown(
            '<div class="live-ticker">⚠️ Live score preview unavailable right now — '
            'your answers are still being saved.</div>', unsafe_allow_html=True)
        return
    r3style = r3_style(r3)
    st.markdown(f"""
    <div class="live-ticker">
      <span>🧮 Function score (prototype)&nbsp;<b>{fn_score:.1f}/10</b></span>
      <span>🎚 Pain VAS&nbsp;<b>{p:.1f}/10</b></span>
      <span>📊 Prototype score&nbsp;<b style="color:{r3style['text']}">{r3:.1f}/30 · {r3style['label']}</b></span>
      <span class="autosave-tag"><span class="autosave-dot"></span>Auto-saved</span>
    </div>""", unsafe_allow_html=True)


def run_assessment():
    """Reads the fully-collected wizard inputs from session_state, fetches
    weather (with automatic fallback), fuses the multimodal risk models,
    projects a 3-day-ahead trend, computes the optional DAS28 score, stores
    everything the Step-5 dashboard needs into st.session_state.results,
    persists to fusion's local history, and commits the audit record to
    the local `flareguard.db` SQLite database. Every external call is
    wrapped so a single failure degrades gracefully instead of crashing
    the live demo."""
    t_start = time.monotonic()

    # Defensive re-clamp — belt-and-braces before anything reaches the models.
    fn_vals = {k: clamp(v, 0, 3) for k, v in st.session_state.fn_vals.items()}
    st.session_state.pain = clamp(st.session_state.pain, 0.0, 10.0)
    st.session_state.global_s = clamp(st.session_state.global_s, 0.0, 10.0)
    st.session_state.fatigue = clamp(st.session_state.fatigue, 0.0, 10.0)
    st.session_state.das_sjc = int(clamp(st.session_state.das_sjc, 0, 28))
    st.session_state.das_tjc = int(clamp(st.session_state.das_tjc, 0, 28))

    try:
        fn_score = fusion.haq_di_score(fn_vals)
        r3 = fusion.rapid3_total(fn_vals, st.session_state.pain, st.session_state.global_s)
    except Exception as exc:
        st.error("We couldn't calculate your RAPID3/HAQ-DI scores right now. Please try again.")
        LOG.error("RAPID3/HAQ-DI calculation failed: %s", exc)
        return

    das = compute_das28(
        st.session_state.das_tjc, st.session_state.das_sjc, st.session_state.global_s,
        st.session_state.das_lab_type.split()[0] if st.session_state.das_lab_type != "Not available" else "None",
        st.session_state.das_lab_value,
    )

    try:
        with st.spinner("Fetching weather information and calculating the experimental score…"):
            wx_today = weather_client.current(st.session_state.city)
            wx_forecast = weather_client.forecast(st.session_state.city, days=3)

            hr_for_model = _safe_heart_rate(st.session_state.heart_rate)

            def build_inputs(pain, fatigue, sleep_diff, anxiety, rapid3_val, temp, hum, pres):
                return fusion.RiskInputs(
                    pain=pain, fn_score=fn_score, global_estimate=st.session_state.global_s,
                    fatigue=fatigue, flares_30d=st.session_state.flares30,
                    duration_years=st.session_state.duration, sleep_diff=sleep_diff,
                    anxiety=anxiety, activity_min=st.session_state.activity,
                    heart_rate=hr_for_model, rapid3=rapid3_val,
                    temp=temp, humidity=hum, pressure=pres,
                )

            today_inputs = build_inputs(
                st.session_state.pain, st.session_state.fatigue,
                st.session_state.sleep_diff, st.session_state.anxiety_val,
                r3, wx_today.temp, wx_today.humidity, wx_today.pressure,
            )
            today_risk, today_contrib = fusion.compute_risk(today_inputs, MODELS)

            risks = [today_risk]
            weathers = [wx_today]
            contribs = [today_contrib]
            for i, f in enumerate(wx_forecast):
                pp, ff, aa, sd = fusion.project_forward(
                    st.session_state.pain, st.session_state.fatigue,
                    st.session_state.sleep_diff, st.session_state.anxiety_val,
                    st.session_state.flares30, i + 1,
                )
                fr3 = sum(fn_vals.values()) + pp + st.session_state.global_s
                f_inputs = build_inputs(pp, ff, sd, aa, fr3, f.temp, f.humidity, f.pressure)
                fr, fcontrib = fusion.compute_risk(f_inputs, MODELS)
                risks.append(fr)
                weathers.append(f)
                contribs.append(fcontrib)
    except Exception as exc:
        st.error(
            "⚠️ We couldn't calculate the experimental score right now — this may be a temporary "
            "weather-service or connectivity issue. Please try again in a moment."
        )
        LOG.error("Forecast/risk computation failed: %s", exc)
        return

    peak_idx = int(np.argmax(risks))
    peak_risk = risks[peak_idx]
    level = fusion.risk_level(peak_risk)
    submission_latency_ms = int((time.monotonic() - t_start) * 1000)

    st.session_state.results = dict(
        fn_score=fn_score, r3=r3, das28=das,
        pain=st.session_state.pain, gs=st.session_state.global_s, fatigue=st.session_state.fatigue,
        risks=risks, weathers=weathers, contributions=contribs[peak_idx], peak_idx=peak_idx,
        saved_to_db=False, submission_latency_ms=submission_latency_ms,
    )

    # Primary persistence — fusion's own local history (existing
    # patient-account "Previous Scores" feature — a separate store from
    # the flareguard.db audit table below).
    try:
        fusion.save_assessment(
            st.session_state.username, st.session_state.city, r3, fn_score,
            st.session_state.pain, st.session_state.global_s, peak_risk, level,
        )
    except Exception as exc:
        LOG.error("Local save_assessment failed: %s", exc)
        st.warning("Your assessment is ready, but we couldn't save it to your local history right now.")

    # Local SQLite audit record — zero-downtime, no cloud provider needed.
    try:
        record = dict(
            patient_id=st.session_state.patient_id,
            pain_score=int(round(st.session_state.pain)),
            swollen_joint_count=int(st.session_state.das_sjc),
            tender_joint_count=int(st.session_state.das_tjc),
            rapid3_score=float(r3),
            haq_di_score=float(fn_score),
            predicted_flare_risk=float(peak_risk),
            risk_category=level,
        )
        st.session_state.results["saved_to_db"] = save_patient_record(record)
    except Exception as exc:
        LOG.error("flareguard.db payload build failed: %s", exc)
        st.session_state.results["saved_to_db"] = False

    if not st.session_state.results["saved_to_db"]:
        st.warning(
            "Your assessment is ready, but we couldn't commit this assessment to the local "
            "database (`flareguard.db`) right now. The rest of your results below are unaffected."
        )


def enter_demo_mode():
    """Explicit, clearly-labelled opt-in path for viva evaluators only.
    This is the one place pre-filled data is allowed — the real intake
    form (Steps 1–4) always starts from a clean baseline."""
    d = fusion.DEMO_PATIENT
    st.session_state.authenticated = True
    st.session_state.demo_mode = True
    st.session_state.username = "viva_evaluator"
    st.session_state.display_name = "Evaluator (Demo Mode)"
    st.session_state.patient_id = "PATIENT_DEMO01"
    st.session_state.age = d.get("age", 45)
    st.session_state.city = d.get("city", "Colombo")
    st.session_state.fn_vals = dict(d["fn_vals"])
    st.session_state.sleep_diff = d["sleep_diff"]
    st.session_state.anxiety_val = d["anxiety_val"]
    st.session_state.fatigue = d["fatigue"]
    st.session_state.pain = d["pain"]
    st.session_state.global_s = d["global_s"]
    st.session_state.flares30 = d["flares30"]
    st.session_state.duration = d["duration"]
    st.session_state.activity = d["activity"]
    st.session_state.heart_rate = d["heart_rate"]
    st.session_state.das_tjc = 4
    st.session_state.das_sjc = 3
    st.session_state.das_lab_type = "Not available"
    # Clear any pill-widget cache so the demo values actually render.
    for k in list(st.session_state.keys()):
        if k.startswith("w_"):
            del st.session_state[k]
    run_assessment()
    st.session_state.step = 5


def secrets_token():
    import secrets as _s
    return _s.token_hex(3)


# ─────────────────────────────────────────────────────────────────────────
# Environment / workspace context banner — shown on every screen,
# including the auth gate, so it's always clear which mode is active.
# ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="env-banner">
  <span>{SITE_LABEL}</span>
  <span class="sep">·</span>
  <span>Experimental tool · Not a medical diagnosis</span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# AUTH GATE — login / sign up / guest / quick viva demo
# ─────────────────────────────────────────────────────────────────────────
def render_auth_gate():
    st.markdown("""
    <div class="auth-wrap">
      <div class="auth-hero">
        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.16em;color:var(--teal2);margin-bottom:10px">
          RA FlareGuard · Sri Lanka
        </div>
        <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:1.9rem;color:var(--text);
             line-height:1.2;letter-spacing:-0.01em">
          Understand your symptoms<br><span style="color:var(--teal2)">Track changes over time</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="auth-wrap" style="padding-top:0">',
        unsafe_allow_html=True,
    )

    if st.button(
        "⚡ Evaluator Demo Mode",
        type="primary",
        use_container_width=True,
    ):
        enter_demo_mode()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="auth-wrap" style="padding-top:0">',
        unsafe_allow_html=True,
    )

    tab_login, tab_signup, tab_guest = st.tabs(
        ["Log In", "Sign Up", "Continue as Guest"]
    )

    # ── Login ───────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")

            submitted = st.form_submit_button(
                "Log In",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                try:
                    ok, msg = fusion.verify_user(u, p)
                except Exception as exc:
                    ok = False
                    msg = (
                        "Login is temporarily unavailable — "
                        "please try again shortly."
                    )
                    LOG.error("verify_user failed: %s", exc)

                if ok:
                    username = u.strip().lower()

                    st.session_state.authenticated = True
                    st.session_state.demo_mode = False
                    st.session_state.username = username
                    st.session_state.display_name = msg

                    # Clear stale patient details and widget values.
                    for key in list(st.session_state.keys()):
                        if (
                            key in {
                                "patient_id",
                                "age",
                                "city",
                                "results",
                            }
                            or key.startswith("w_")
                        ):
                            del st.session_state[key]

                    # Restore the profile saved for this account.
                    try:
                        profile = get_patient_profile(username)

                        if profile:
                            st.session_state.patient_id = profile["patient_id"]
                            st.session_state.age = profile["age"]
                            st.session_state.city = profile["city"]

                    except Exception as exc:
                        LOG.error(
                            "Could not restore patient profile: %s",
                            exc,
                        )

                    st.success(f"Welcome back, {msg}!")
                    st.rerun()

                else:
                    st.error(msg)

        st.caption(
            "Log in to access your saved assessment history and patient profile."
        )

    # ── Sign up ─────────────────────────────────────────────────────────
    with tab_signup:
        with st.form("signup_form"):
            su = st.text_input("Choose a username")
            sn = st.text_input("Display name (optional)")
            sp = st.text_input("Choose a password", type="password")
            sp2 = st.text_input("Confirm password", type="password")

            submitted = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                if sp != sp2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        ok, msg = fusion.create_user(su, sp, sn)
                    except Exception as exc:
                        ok = False
                        msg = (
                            "Account creation is temporarily unavailable — "
                            "please try again shortly."
                        )
                        LOG.error("create_user failed: %s", exc)

                    if ok:
                        st.success(
                            msg + " Please log in from the 'Log In' tab."
                        )
                    else:
                        st.error(msg)

    # ── Guest access ────────────────────────────────────────────────────
    with tab_guest:
        st.markdown(
            '<p style="font-size:0.8rem;color:var(--text2);margin-bottom:10px">'
            "Try the assessment without creating an account. Guest results are not "
            "linked to your registered account history.</p>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Continue as Guest →",
            use_container_width=True,
        ):
            # Clear values left by another account or previous session.
            for key in list(st.session_state.keys()):
                if (
                    key in {
                        "patient_id",
                        "age",
                        "city",
                        "results",
                    }
                    or key.startswith("w_")
                ):
                    del st.session_state[key]

            st.session_state.authenticated = True
            st.session_state.demo_mode = False
            st.session_state.username = f"guest_{secrets_token()}"
            st.session_state.display_name = "Guest"

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.authenticated:
    render_auth_gate()
    st.stop()
# ─────────────────────────────────────────────────────────────────────────
# Demo-mode banner (persists across all steps once active)
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.demo_mode:
    st.markdown("""
    <div style="padding:10px 24px 0">
      <div class="demo-banner">
        <span>⚡ <b>Demo Mode</b> — showing a pre-populated sample assessment.</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session-safety controls — visible once authenticated, on every step,
# so a clinician can end a consultation and wipe patient state at any time.
# ─────────────────────────────────────────────────────────────────────────
_sig_col1, _sig_col2, _sig_col3 = st.columns([4, 1.35, 1])
with _sig_col1:
    st.markdown(
        f'<div style="padding:8px 8px 0;font-size:0.85rem;color:var(--text3)">'
        f'Account: <b style="color:var(--text2)">{st.session_state.display_name or st.session_state.username}</b></div>',
        unsafe_allow_html=True)
with _sig_col2:
    if st.button("🧹 Clear Session", key="clear_session_top", help="Clear the current answers without logging out", use_container_width=True):
        st.session_state["_confirm_clear_top"] = True
with _sig_col3:
    if st.button("↪ Log out", key="logout_top", help="Log out and return to the welcome screen", use_container_width=True):
        # Clear clinical and identity state so another person using this
        # browser session cannot inherit the previous participant's answers.
        for _key in list(st.session_state.keys()):
            if _key not in {"_streamlit_run_id"}:
                del st.session_state[_key]
        st.session_state.authenticated = False
        st.session_state.demo_mode = False
        st.session_state.username = None
        st.session_state.display_name = None
        init_session()
        st.rerun()

if st.session_state.get("_confirm_clear_top"):
    st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
    st.warning(
        "This clears the current patient's answers and results from this device's memory "
        "(records already committed to flareguard.db are unaffected). Continue?"
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Cancel", key="cancel_clear_top"):
            st.session_state["_confirm_clear_top"] = False
            st.rerun()
    with cc2:
        if st.button("Yes, clear patient data", type="primary", key="confirm_clear_top"):
            clear_sensitive_session()
            st.session_state["_confirm_clear_top"] = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# FLAREGUARD APP SHELL — dashboard, previous sessions, trends and profile
# ─────────────────────────────────────────────────────────────────────────
def fg_load_history():
    """Read the existing account history without changing its storage."""
    username = st.session_state.get("username")
    if not username:
        return []

    try:
        rows = fusion.get_history(username) or []
    except Exception as exc:
        LOG.error("Dashboard history retrieval failed: %s", exc)
        return []

    parsed = []
    for row in rows:
        try:
            created_at, city, rapid3, risk, level = row[:5]
            parsed.append({
                "created_at": str(created_at or ""),
                "city": str(city or ""),
                "rapid3": float(rapid3 or 0),
                "risk": float(risk or 0),
                "level": str(level or "Unknown"),
            })
        except (TypeError, ValueError, IndexError):
            continue

    # Timestamps are stored as ISO-like strings by the existing history feature.
    parsed.sort(key=lambda item: item["created_at"], reverse=True)
    return parsed


def fg_heading(title, subtitle):
    st.markdown(
        f'<div class="fg-page-heading">{title}</div>'
        f'<div class="fg-page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def fg_render_dashboard():
    username = st.session_state.get("username") or ""
    display_name = st.session_state.get("display_name") or username or "Patient"
    patient_id = st.session_state.get("patient_id") or "Not assigned yet"
    city = st.session_state.get("city") or "Not set"
    history = fg_load_history()

    fg_heading(
        f"Welcome back, {display_name}",
        "Your personal RA monitoring space. Review your history or start a new assessment.",
    )

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            '<div class="fg-panel">'
            '<div class="fg-panel-title">Your health, at a glance</div>'
            '<div class="fg-muted">Track your reported symptoms and experimental flare-risk scores over time.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="fg-panel">'
            '<div class="fg-panel-title">Patient record</div>'
            f'<div><b>{patient_id}</b></div>'
            f'<div class="fg-muted">{city}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    latest = history[0] if history else None
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Saved assessments", len(history))
    with m2:
        st.metric(
            "Latest prototype score",
            f'{latest["rapid3"]:.1f}/30' if latest else "—",
        )
    with m3:
        st.metric(
            "Latest recorded risk",
            f'{latest["risk"]:.0%}' if latest else "—",
        )
    with m4:
        st.metric("Profile city", city)

    st.markdown("### Quick actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("＋  Start new assessment", type="primary",
                     use_container_width=True, key="dash_new_assessment"):
            st.session_state.app_view = "New Assessment"
            st.session_state.step = 0
            st.session_state.results = None
            st.rerun()
    with a2:
        if st.button("▤  Previous sessions", use_container_width=True,
                     key="dash_previous_sessions"):
            st.session_state.app_view = "Previous Sessions"
            st.rerun()
    with a3:
        if st.button("↗  View risk trends", use_container_width=True,
                     key="dash_risk_trends"):
            st.session_state.app_view = "Risk Trends"
            st.rerun()

    st.markdown("### Recent assessments")
    if history:
        recent = history[:5]
        table = []
        for item in recent:
            try:
                date_label = datetime.fromisoformat(
                    item["created_at"]
                ).strftime("%d %b %Y, %H:%M")
            except Exception:
                date_label = item["created_at"] or "Date unavailable"

            table.append({
                "Date": date_label,
                "City": item["city"],
                "RAPID3": f'{item["rapid3"]:.1f}/30',
                "Experimental risk": f'{item["risk"]:.0%}',
                "Category": item["level"].title(),
            })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Risk values are experimental research scores, not clinically validated "
            "probabilities. Historical scores do not replace medical advice."
        )
    else:
        st.info(
            "No previous assessments are available in the account history yet. "
            "Complete an assessment while signed in to start building your history."
        )


def fg_render_history():
    if st.button("← Back to Dashboard", key="fg_back_history", use_container_width=False):
        st.session_state.app_view = "Dashboard"
        st.rerun()
    st.write("")
    history = fg_load_history()
    fg_heading(
        "Previous Sessions",
        "Review assessments saved to your existing patient-account history.",
    )

    if not history:
        st.info(
            "No saved sessions were found for this account. If you expected to see "
            "older records, they may be stored in a different local history database."
        )
        return

    st.markdown(
        f'<div class="fg-panel"><div class="fg-panel-title">'
        f'{len(history)} saved assessment(s)</div>'
        '<div class="fg-muted">Showing the history currently returned by your account history store.</div></div>',
        unsafe_allow_html=True,
    )

    table = []
    for item in history:
        try:
            date_label = datetime.fromisoformat(
                item["created_at"]
            ).strftime("%d %b %Y, %H:%M")
        except Exception:
            date_label = item["created_at"] or "Date unavailable"

        table.append({
            "Date": date_label,
            "City": item["city"],
            "Prototype score": f'{item["rapid3"]:.1f}/30',
            "Experimental score": f'{item["risk"]:.0%}',
            "Risk category": item["level"].title(),
        })

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.warning(
        "Research prototype: historical risk scores are experimental and are not "
        "clinically validated probabilities or a substitute for rheumatology care."
    )


def fg_render_trends():
    if st.button("← Back to Dashboard", key="fg_back_trends", use_container_width=False):
        st.session_state.app_view = "Dashboard"
        st.rerun()
    st.write("")
    history = fg_load_history()
    fg_heading(
        "Risk Trends",
        "Explore changes in recorded Prototype scores and experimental risk estimates.",
    )

    if not history:
        st.info("Complete and save an assessment to begin viewing trends.")
        return

    chronological = list(reversed(history))
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Assessments available", len(chronological))
    with m2:
        st.metric("Most recent risk", f'{history[0]["risk"]:.0%}')

    st.markdown("### Experimental risk over saved sessions")
    st.line_chart(
        {"Risk (%)": [item["risk"] * 100 for item in chronological]},
        y_label="Experimental score (%)",
        x_label="Saved sessions (oldest to newest)",
    )

    st.markdown("### Prototype score history")
    st.line_chart(
        {"RAPID3": [item["rapid3"] for item in chronological]},
        y_label="Prototype score",
        x_label="Saved sessions (oldest to newest)",
    )

    st.caption(
        "Each point represents a saved assessment, not a daily continuous measurement. "
        "These charts describe recorded history; they do not establish a clinical trend "
        "or validate the risk model."
    )


def fg_render_profile():
    if st.button("← Back to Dashboard", key="fg_back_profile", use_container_width=False):
        st.session_state.app_view = "Dashboard"
        st.rerun()
    st.write("")
    username = st.session_state.get("username") or ""
    display_name = st.session_state.get("display_name") or username or "Patient"
    profile = None

    if username and not username.startswith("guest_") and username != "viva_evaluator":
        try:
            profile = get_patient_profile(username)
        except Exception as exc:
            LOG.error("Profile display failed: %s", exc)

    fg_heading(
        "My Profile",
        "View the account and patient-record information currently available in this session.",
    )

    patient_id = (
        profile.get("patient_id") if profile
        else st.session_state.get("patient_id") or "Not assigned yet"
    )
    age = profile.get("age") if profile else st.session_state.get("age") or "Not provided"
    city = profile.get("city") if profile else st.session_state.get("city") or "Not provided"

    st.markdown('<div class="fg-panel">', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Account display name**")
        st.write(display_name)
        st.markdown("**Username**")
        st.write(username or "Guest session")
    with p2:
        st.markdown("**Patient ID**")
        st.write(patient_id)
        st.markdown("**Age**")
        st.write(age)
        st.markdown("**City / town**")
        st.write(city)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("demo_mode"):
        st.info("Demo mode is active. The displayed details may be sample data.")

    st.caption(
        "Profile details are displayed from the existing session/profile store. "
        "This page does not edit or overwrite your saved profile."
    )


def fg_render_app_navigation():
    st.session_state.setdefault("app_view", "Dashboard")
    allowed_views = {
        "Dashboard", "New Assessment", "Previous Sessions", "Risk Trends", "My Profile"
    }
    if st.session_state.app_view not in allowed_views:
        st.session_state.app_view = "Dashboard"

    with st.sidebar:
        st.markdown("## 🫀 FlareGuard")
        st.caption("RA monitoring · Research prototype")
        st.divider()

        st.markdown("**WORKSPACE**")
        items = [
            ("Dashboard", "⌂  Dashboard"),
            ("New Assessment", "＋  New Assessment"),
            ("Previous Sessions", "▤  Previous Sessions"),
            ("Risk Trends", "↗  Risk Trends"),
            ("My Profile", "◉  My Profile"),
        ]

        for view_name, label in items:
            if st.button(
                label,
                key=f"fg_nav_{view_name.lower().replace(' ', '_')}",
                use_container_width=True,
                type="primary" if st.session_state.app_view == view_name else "secondary",
            ):
                if view_name == "New Assessment":
                    st.session_state.step = 0
                    st.session_state.results = None
                st.session_state.app_view = view_name
                st.rerun()

        st.divider()
        st.caption(
            f"Signed in: {st.session_state.get('display_name') or st.session_state.get('username') or 'Patient'}"
        )
        st.caption("Experimental score · Not for medical decisions")

    view = st.session_state.app_view

    if view == "Dashboard":
        fg_render_dashboard()
        st.stop()
    elif view == "Previous Sessions":
        fg_render_history()
        st.stop()
    elif view == "Risk Trends":
        fg_render_trends()
        st.stop()
    elif view == "My Profile":
        fg_render_profile()
        st.stop()

    # Only the New Assessment view reaches the original six-step workflow.
    if view == "New Assessment":
        return

    st.session_state.app_view = "Dashboard"
    fg_render_dashboard()
    st.stop()



# Readability improvements for the complete application
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px !important;
}
[data-testid="stAppViewContainer"] {
    background: #F5F8FC !important;
    color: #172B3A !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #263B4A !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
}
[data-testid="stMarkdownContainer"] h1 {
    color: #173B4D !important;
    font-size: 2rem !important;
    font-weight: 750 !important;
}
[data-testid="stMarkdownContainer"] h2 {
    color: #173B4D !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}
[data-testid="stMarkdownContainer"] h3 {
    color: #173B4D !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #526777 !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #243B4A !important;
}
[data-testid="stButton"] button {
    min-height: 42px !important;
    font-size: 0.96rem !important;
    font-weight: 650 !important;
    border-radius: 10px !important;
}
[data-testid="stMetricLabel"] {
    color: #425969 !important;
    font-size: 0.92rem !important;
}
[data-testid="stMetricValue"] {
    color: #173B4D !important;
    font-size: 1.8rem !important;
    font-weight: 750 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #D5E0E8 !important;
    border-radius: 10px !important;
}
input, textarea, [data-baseweb="select"] {
    font-size: 1rem !important;
}
button:focus-visible, input:focus-visible {
    outline: 3px solid #087E8B !important;
    outline-offset: 2px !important;
}
</style>
""", unsafe_allow_html=True)

fg_render_app_navigation()

# Progress bar
# ─────────────────────────────────────────────────────────────────────────
step = st.session_state.step
if step < TOTAL_STEPS:
    pct = int(step / (TOTAL_STEPS - 1) * 100) if TOTAL_STEPS > 1 else 0
    current_name = STEP_NAMES[step] if step < len(STEP_NAMES) else ""
    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-inner">
        <div class="prog-row">
          <span class="prog-label">{current_name}</span>
          <span class="prog-step">Step {step + 1} of {TOTAL_STEPS}</span>
        </div>
        <div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>
      </div>
    </div>""", unsafe_allow_html=True)
    if step >= 1 and not st.session_state.demo_mode:
        st.session_state.form_started_at = st.session_state.form_started_at or datetime.now(timezone.utc).isoformat()

# ─────────────────────────────────────────────────────────────────────────
# Nav helpers
# ─────────────────────────────────────────────────────────────────────────
def go(n):
    st.session_state.step = n
    st.rerun()


def nav(back=True, next_label="Continue →", next_fn=None, back_step=None, next_disabled=False):
    st.markdown('<div class="nav-bar"><div class="nav-inner">', unsafe_allow_html=True)
    if back:
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("← Back", key=f"back_{step}", type="secondary"):
                go(step - 1 if back_step is None else back_step)
        with c2:
            if st.button(next_label, key=f"next_{step}", type="primary", disabled=next_disabled):
                if next_fn:
                    next_fn()
                else:
                    go(step + 1)
    else:
        if st.button(next_label, key=f"next_{step}", type="primary", disabled=next_disabled):
            if next_fn:
                next_fn()
            else:
                go(step + 1)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — WELCOME
# ══════════════════════════════════════════════════════════════════════════
if step == 0:
    model_badge = '<span class="model-status fallback">🧪 Research prototype · ML fusion not validated</span>'
    st.markdown("""
    <div class="step-wrap" style="padding-top:10px">
      <div style="text-align:center;padding:10px 0 14px">
        <div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.16em;color:var(--teal2);margin-bottom:8px">RA FlareGuard · Sri Lanka</div>
        <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:1.8rem;color:var(--text);
             line-height:1.2;margin-bottom:10px;letter-spacing:-0.01em">
          Understand your symptoms<br><span style="color:var(--teal2)">one step at a time</span>
        </div>
        <div style="font-size:1rem;color:var(--text2);line-height:1.6;max-width:440px;margin:0 auto">
          Tell us how you have been feeling and how symptoms affect your daily life.
          Your answers create a research assessment to help you review your symptoms.
        </div>
      </div>
      <div class="qcard" style="padding:14px 18px">
        <div class="qcard-hdr" style="margin-bottom:10px">What you will fill in</div>
        <div style="font-size:1rem;color:var(--text);line-height:1.5">
          <div style="padding:7px 0;border-bottom:1px solid var(--border)"><b>1.</b> A code (not your name), your age and your town</div>
          <div style="padding:7px 0;border-bottom:1px solid var(--border)"><b>2.</b> 10 daily-activity questions &mdash; tap one answer each</div>
          <div style="padding:7px 0;border-bottom:1px solid var(--border)"><b>3.</b> Your pain and overall feeling, from 0 to 10</div>
          <div style="padding:7px 0;border-bottom:1px solid var(--border)"><b>4.</b> Doctor's numbers <span style="color:var(--text3)">(optional &mdash; skip if you don't have them)</span></div>
          <div style="padding:7px 0"><b>5.</b> Sleep, tiredness, recent flares and pulse</div>
        </div>
      </div>
      <div class="field-help" style="margin-top:10px;font-size:0.85rem;text-align:center">
        &#128274; We never ask for your name. Your answers save automatically on this device.
      </div>
    </div>
    """, unsafe_allow_html=True)
    nav(back=False, next_label="Start Assessment →")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — SECTION 01 · PATIENT DEMOGRAPHICS & ANONYMIZED BASELINE
# ══════════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">STEP 1 OF 5 · ABOUT YOU</div>
      <div class="step-title">Let’s start <em>with you</em></div>
      <div class="step-desc">Three quick details so we can personalise your forecast and check the weather in your area.</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        username = st.session_state.get("username") or ""
        is_registered = (
            st.session_state.get("authenticated", False)
            and not st.session_state.get("demo_mode", False)
            and not username.startswith("guest_")
            and username != "viva_evaluator"
        )

        # Show the existing ID for a registered account, or derive a
        # stable ID if this account does not have a saved profile yet.
        if is_registered:
            profile = get_patient_profile(username)
            if profile:
                st.session_state.patient_id = profile["patient_id"]
            elif not st.session_state.get("patient_id"):
                import hashlib
                suffix = hashlib.sha256(
                    username.encode("utf-8")
                ).hexdigest()[:8].upper()
                st.session_state.patient_id = f"FG-{suffix}"

        patient_id_display = st.session_state.get("patient_id") or "Assigned when you continue"
        st.markdown(
            f'<div class="privacy-banner">🔒 <b>Patient ID:</b> '
            f'<code>{patient_id_display}</code><br>'
            'This ID is assigned automatically to help identify your assessment record. You do not need to enter it manually.</div>',
            unsafe_allow_html=True,
        )
        # Widget keys use a "w_" prefix and are mirrored into plain session keys each run.
        # (Assigning to a key that a live widget owns raises StreamlitAPIException.)
        if "w_patient_id" not in st.session_state:
            st.session_state.w_patient_id = st.session_state.get("patient_id", "")
        st.session_state.patient_id = st.session_state.w_patient_id or ""
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Your age *", min_value=0, max_value=120,
                             value=int(st.session_state.age or 0), key="w_age",
                             help="Leave at 0 until you enter a real value — required to continue.")
            st.session_state.age = st.session_state.w_age
        with c2:
            city_options = [""] + list(CITIES.keys())
            if st.session_state.city and st.session_state.city not in city_options:
                city_options = city_options + [st.session_state.city]
            city_val = st.selectbox(
                "Your city / town *",
                options=city_options,
                index=city_options.index(st.session_state.city) if st.session_state.city in city_options else 0,
                accept_new_options=True,
                placeholder="Type or select your city",
                key="w_city",
            )
            st.session_state.city = (city_val or "").strip()
        st.markdown(
            '<div class="field-help">Can\'t find your town? Just type it in.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        def save1():
            username = st.session_state.get("username") or ""
            is_registered = (
                st.session_state.get("authenticated", False)
                and not st.session_state.get("demo_mode", False)
                and not username.startswith("guest_")
                and username != "viva_evaluator"
            )

            age = st.session_state.age
            city = (st.session_state.city or "").strip()

            if not age or age < 18 or age > 90:
                st.error("Please enter a valid age between 18 and 90.")
                return

            if not city:
                st.error("Please enter your city or town.")
                return

            if is_registered:
                # Restore the existing ID if this account already has a profile.
                profile = get_patient_profile(username)
                if profile:
                    pid = profile["patient_id"]
                else:
                    # Stable ID for this registered account.
                    import hashlib
                    suffix = hashlib.sha256(username.encode("utf-8")).hexdigest()[:8].upper()
                    pid = f"FG-{suffix}"

                st.session_state.patient_id = pid

                if not save_patient_profile(username, pid, int(age), city):
                    st.error("Could not save your patient profile. Please try again.")
                    return
            else:
                # Separate, non-account ID for guest/demo sessions.
                if not st.session_state.get("patient_id"):
                    st.session_state.patient_id = f"FG-GUEST-{secrets_token().upper()}"

            st.session_state.age = int(age)
            st.session_state.city = city
            go(2)

        nav(back=False, next_label="Continue →", next_fn=save1)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — YOUR DAILY ACTIVITIES AND SYMPTOMS
# ══════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Part 2 · Daily activities</div>
      <div class="step-title">What can you <em>do — and how do you feel?</em></div>
      <div class="step-desc">Tap <b style="color:var(--text)">one answer</b> for each question.</div>
      <div class="step-cite">📖 Pincus et al., J Rheumatol 2008;35:2136 · HAQ-DI: Fries et al., Arthritis Rheum 1980 · Pain VAS: Huskisson, Lancet 1974</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 24px">
    <div class="scale-chips">
      <span class="scale-chip" style="background:#ECFDF5;color:#065F46">0 — No difficulty</span>
      <span class="scale-chip" style="background:#FEF3C7;color:#92400E">1 — Some difficulty</span>
      <span class="scale-chip" style="background:#FFEDD5;color:#9A3412">2 — Much difficulty</span>
      <span class="scale-chip" style="background:#FEE2E2;color:#991B1B">3 — Unable to do</span>
    </div>
    <p style="font-size:0.74rem;color:var(--text3);margin-bottom:18px;font-style:italic">
      Think about the <b style="color:var(--text2)">past 7 days</b> when answering each question.
    </p>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        fn_vals = dict(st.session_state.fn_vals)
        for letter, question, ref in RAPID3_Qs:
            st.markdown(f"""
            <div class="q-item">
              <div class="q-top">
                <div class="q-badge">{letter}</div>
                <div>
                  <div class="q-text">{question}</div>
                  <div class="q-ref">{ref} · MDHAQ 1{letter.lower()}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            fn_field_key = f"fn_{letter}"
            if fn_field_key not in st.session_state:
                st.session_state[fn_field_key] = fn_vals.get(letter, 0)
            v = diff_pills(fn_field_key)
            fn_vals[letter] = v
        st.session_state.fn_vals = fn_vals
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0 24px">
        <div class="qcard" style="margin-top:6px">
          <div class="qcard-hdr">Pain this past week</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:11px">
            "How much pain have you had because
            of your condition <b style="color:var(--text)">over the past week?</b>"
          </div>
        </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="padding:0 24px"><div class="vas-grad"></div></div>', unsafe_allow_html=True)
        pain = nrs_pills("pain")
        st.markdown('<div style="padding:0 24px"><div class="vas-ends"><span>0 — No pain at all</span><span>10 — Worst possible pain</span></div></div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0 24px">
        <div class="qcard" style="margin-top:18px">
          <div class="qcard-hdr">Overall, how are you doing?</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:11px">
            "Considering all the ways your illness
            affects you at this time — <b style="color:var(--text)">how are you doing overall?</b>"
          </div>
        </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="padding:0 24px"><div class="vas-grad" style="background:linear-gradient(to right,var(--green),var(--red))"></div></div>',
                    unsafe_allow_html=True)
        gs = nrs_pills("global_s")
        st.markdown('<div style="padding:0 24px"><div class="vas-ends"><span>0 — Very well</span><span>10 — Very poorly</span></div></div>',
                    unsafe_allow_html=True)

    scoreboard()
    nav(next_fn=lambda: go(3))

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — OPTIONAL CLINICAL INFORMATION
# ══════════════════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Part 3 · Optional</div>
      <div class="step-title">Numbers from <em>your doctor</em></div>
      <div class="step-desc">
        <b style="color:var(--text)">You can skip this page.</b> Only open the box below if your
        doctor has given you joint counts or a blood test result (ESR or CRP).
        Otherwise just press Continue.
      </div>
      <div class="step-cite">📖 DAS28: Prevoo et al., Arthritis Rheum 1995;38:44</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Add my doctor's numbers (optional)", expanded=False):
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown("""
        <div class="qcard">
          <div class="qcard-hdr">28-Joint Count</div>
          <div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
            Count of joints your clinician (or you, if self-assessing) currently finds
            swollen or tender, out of the standard 28 joints assessed.
          </div>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Swollen Joint Count (0–28)", min_value=0, max_value=28, key="das_sjc")
        with c2:
            st.number_input("Tender Joint Count (0–28)", min_value=0, max_value=28, key="das_tjc")

        st.markdown("""
        <div class="qcard" style="margin-top:14px">
          <div class="qcard-hdr">Optional lab value — ESR or CRP</div>
          <div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
            Only fill this in if you have a recent result on hand. Leave as
            "Not available" to skip — your joint counts above are still recorded either way.
          </div>
        </div>""", unsafe_allow_html=True)
        lab_options = ["Not available", "ESR (mm/hr)", "CRP (mg/L)"]
        st.selectbox(
            "Lab value type",
            options=lab_options,
            index=lab_options.index(st.session_state.das_lab_type) if st.session_state.das_lab_type in lab_options else 0,
            key="das_lab_type",
        )
        if st.session_state.das_lab_type == "ESR (mm/hr)":
            st.number_input("ESR value (mm/hr)", min_value=0.0, max_value=150.0, step=1.0, key="das_lab_value")
        elif st.session_state.das_lab_type == "CRP (mg/L)":
            st.number_input("CRP value (mg/L)", min_value=0.0, max_value=300.0, step=0.5, key="das_lab_value")

        preview = compute_das28(
            st.session_state.das_tjc, st.session_state.das_sjc, st.session_state.global_s,
            st.session_state.das_lab_type.split()[0] if st.session_state.das_lab_type != "Not available" else "None",
            st.session_state.das_lab_value,
        )
        if preview:
            pv_style = preview["style"]
            st.markdown(f"""
            <div class="chips" style="margin-top:6px">
              <span class="chip" style="background:{pv_style['bg']};color:{pv_style['text']};border-color:{pv_style['border']}">
                DAS28-{st.session_state.das_lab_type.split()[0]} <b>{preview['score']:.2f}</b> · {pv_style['label']}
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="field-help" style="margin-top:6px">'
                'Joint counts will be recorded for the clinical audit trail. '
                'Add an ESR or CRP value above to also calculate a DAS28 score.</div>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    nav(next_fn=lambda: go(4))

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — SLEEP, ENERGY AND RECENT SYMPTOMS
# ══════════════════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Part 4 · Last questions</div>
      <div class="step-title">Sleep, tiredness <em>&amp; recent flares</em></div>
      <div class="step-desc">Six short questions. The pulse question is <b style="color:var(--text)">optional</b>.</div>
      <div class="step-cite">📖 MDHAQ: Pincus 2009 · Fatigue VAS: Wolfe et al. 1996 · Flare count: RADAI — Stucki et al. 1995</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown('<div class="qcard-hdr" style="margin:6px 0 4px">Sleep and mood &mdash; past 7 days</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="q-item">
          <div class="q-top">
            <div class="q-badge">K</div>
            <div>
              <div class="q-text">Get a good night's sleep?</div>
              <div class="q-ref">MDHAQ item 1k · Sleep quality</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        diff_pills("sleep_diff")

        st.markdown("""
        <div class="q-item" style="margin-top:9px">
          <div class="q-top">
            <div class="q-badge">L</div>
            <div>
              <div class="q-text">Deal with feelings of anxiety or being nervous?</div>
              <div class="q-ref">MDHAQ item 1l · Anxiety</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        diff_pills("anxiety_val")

        st.markdown("""
        <div class="qcard" style="margin-top:18px">
          <div class="qcard-hdr">Tiredness this past week</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:9px">
            How much <b style="color:var(--text)">fatigue or tiredness</b> have you had because of your RA <b style="color:var(--text)">over the past week?</b>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="vas-grad"></div>', unsafe_allow_html=True)
        nrs_pills("fatigue")
        st.markdown('<div class="vas-ends"><span>0 — No fatigue</span><span>10 — Worst possible</span></div>',
                    unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:14px"><div class="qcard-hdr">Flares in the last 30 days</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many times did your joint pain <b style="color:var(--text)">suddenly get much worse</b>
        in the <b style="color:var(--text)">past 30 days?</b></div>""", unsafe_allow_html=True)
        st.number_input("Flares", 0, 30, key="flares30", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">Time since diagnosis</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many years ago were you first diagnosed with RA?</div>""", unsafe_allow_html=True)
        st.number_input("Years", 0, 50, key="duration", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">Activity today</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many minutes of <b style="color:var(--text)">walking or light movement</b> did you do today?</div>""",
                    unsafe_allow_html=True)
        st.number_input("Minutes", 0, 180, key="activity", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">Pulse (optional)</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        What is your <b style="color:var(--text)">pulse right now?</b> Check a phone or watch.
        Not sure? Leave it at 0.</div>""", unsafe_allow_html=True)
        st.number_input("bpm", 0, 220, key="heart_rate", label_visibility="collapsed")
        st.markdown("""<div style="font-size:0.8rem;color:var(--text3);margin:5px 0 12px">
        Normal resting pulse: 60 to 100.</div></div>""", unsafe_allow_html=True)

        confirm = st.checkbox(
            "I confirm the information I've provided in this assessment is accurate to the "
            "best of my knowledge.", value=st.session_state.get("confirm_submit_cb", False), key="confirm_submit_cb",
        )
        if not confirm:
            st.markdown(
                '<div class="field-help">Please confirm the checkbox above to calculate your forecast.</div>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def save4():
        run_assessment()
        go(5)

    nav(next_label="View My Assessment →", next_fn=save4, next_disabled=not confirm)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — SECTION 05 · PREDICTIVE MODEL ASSESSMENT & CLINICAL RISK SUMMARY
# ══════════════════════════════════════════════════════════════════════════
elif step == 5:
    res = st.session_state.results
    if not res:
        go(0)
    else:
        risks = res["risks"]
        weathers = res["weathers"]
        peak_idx = res["peak_idx"]
        peak = risks[peak_idx]
        plvl = fusion.risk_level(peak)
        pstyle = level_style(plvl)
        pcol = pstyle["text"]
        r3 = res["r3"]
        r3sty = r3_style(r3)
        das = res.get("das28")
        contributions = res["contributions"]

        if res.get("saved_to_db"):
            st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
            st.success(
                f"✅ Assessment Successfully Saved to Local Database · "
                f"Patient ID: {st.session_state.patient_id}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="step-wrap">
          <div class="step-eyebrow">Your results · {st.session_state.city}</div>
          <div class="result-hero">
            <div class="result-name">Your symptom assessment · {datetime.now().strftime('%d %b %Y')}</div>
            <div style="font-size:0.82rem;font-weight:700;color:var(--text2);margin:12px 0 2px">YOUR PROTOTYPE SCORE</div>
            <div class="result-score" style="color:{r3sty['text']};font-size:clamp(2.8rem,7vw,4rem)">{r3:.1f}<span style="font-size:1.25rem;color:var(--text2)"> / 30</span></div>
            <div class="result-pill">Experimental questionnaire summary · {r3sty['label']}</div>
            <div class="result-meta">Your selected answers about function, pain and overall health contribute to this prototype questionnaire summary.</div>
            <div style="margin:18px auto 0;max-width:520px;padding:12px 16px;border:1px solid var(--border);border-radius:12px;background:var(--card);text-align:left">
              <div style="font-weight:700;color:var(--text);margin-bottom:4px">About the experimental heuristic score ({peak * 100:.0f}/100 points)</div>
              <div style="font-size:0.88rem;line-height:1.55;color:var(--text2)">This is a separate experimental calculation using selected assessment answers and weather inputs. It is not a recognised clinical score and does not represent a probability or percentage chance of an RA flare.</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # The four trained datasets have different targets and feature schemas.
        # They are deliberately NOT blended into this console score until a
        # validated, schema-matched fusion model has been trained and evaluated.
        db_chip = (
            '<span class="chip ok">💾 Saved to local database</span>' if res.get("saved_to_db")
            else '<span class="chip warn">⚠️ Not saved to local database</span>'
        )
        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="chips">
          <span class="chip">📍 <b>{st.session_state.city}</b></span>
          <span class="chip">🌡 <b>{weathers[0].temp:.1f}°C</b> · {weathers[0].humidity:.0f}% humidity</span>
          <span class="chip">📊 Prototype questionnaire <b style="color:{r3sty['text']}">{r3:.1f}/30</b> · {r3sty['label']}</span>
          <span class="chip warn">🧪 <b>Research prototype score</b></span>
          {db_chip}
          <span class="chip">⏱ <b>{res.get('submission_latency_ms', 0)} ms</b> to compute</span>
        </div>
        </div>""", unsafe_allow_html=True)
        # Patient-centred interpretation: describe reported answers, not an unvalidated forecast.
        symptom_flags = []
        pain = float(res.get("pain", 0) or 0)
        if pain >= 7:
            symptom_flags.append("You reported a high pain level.")
        elif pain >= 4:
            symptom_flags.append("You reported a moderate pain level.")
        else:
            symptom_flags.append("You reported a lower pain level on the scale today.")
        if res.get("fatigue", 0) >= 7:
            symptom_flags.append("You reported a high fatigue level.")
        if res.get("sleep_diff", 0) >= 2:
            symptom_flags.append("You reported sleep difficulty.")
        if not symptom_flags:
            symptom_flags.append("Your assessment responses have been recorded.")
        st.markdown("### What your answers tell you")
        st.write(" ".join(symptom_flags))
        st.caption("This summary describes the answers you entered today; it does not diagnose a flare or predict what will happen next.")
        st.info(
            "**About the experimental percentage:** this is generated by FlareGuard's current rule-based calculation using selected assessment answers and weather inputs. "
            "It is not a recognised clinical score and does not mean you have that percentage chance of an RA flare."
        )
        st.markdown("### What can you do with this result?")
        st.markdown(
            "- Review the symptom summary and your separate RAPID3 result.\n"
            "- Compare this assessment with your previous sessions to notice changes, if you have previous records.\n"
            "- Share the assessment with your healthcare professional if you have concerns.\n"
            "- Do not change medication based on this prototype. Seek medical advice for new, severe, or worsening symptoms regardless of the score."
        )

        # ── Results are split into tabs so the page no longer needs a long scroll ──
        tab_sum, tab_why, tab_model, tab_fb = st.tabs(
            ["📊 What this means", "🔍 Why this score", "🧪 Research details", "💬 Feedback"]
        )

        with tab_sum:
            # ── 1. Experimental calculation details (not a flare probability) ─────────────────────────────────
            st.markdown('<div style="padding:0 24px"><div class="qcard">', unsafe_allow_html=True)
            st.markdown('<div class="qcard-hdr">Experimental calculation details (not a flare probability)</div>', unsafe_allow_html=True)
            if _PLOTLY_OK:
                try:
                    gauge = pgo.Figure(pgo.Indicator(
                        mode="gauge+number",
                        value=peak * 100,
                        number={"suffix": " pts", "font": {"size": 34, "color": pcol}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#94A3B8", "tickwidth": 1},
                            "bar": {"color": pcol, "thickness": 0.28},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                            ],
                            "threshold": {"line": {"color": pcol, "width": 3}, "thickness": 0.9, "value": peak * 100},
                        },
                    ))
                    gauge.update_layout(
                        height=180, margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0F172A", "family": "Inter"},
                    )
                    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
                except Exception as exc:
                    LOG.error("Gauge render failed: %s", exc)
                    st.info(f"Experimental heuristic score: **{peak * 100:.0f}/100 points** ({fusion.risk_label(peak)}) — chart temporarily unavailable.")
            else:
                st.info(f"Experimental heuristic score: **{peak * 100:.0f}/100 points** ({fusion.risk_label(peak)}).")

            st.markdown("### Your assessment today")
            st.write(
                f"**Pain:** {res['pain']:.1f}/10  ·  "
                f"**Physical function:** {res['fn_score']:.1f}/10  ·  "
                f"**Overall health:** {res['gs']:.1f}/10  ·  "
                f"**Fatigue:** {res['fatigue']:.1f}/10"
            )
            st.caption(
                "These are the responses you reported for this assessment. The prototype does not currently "
                "provide a validated prediction for tomorrow or the following three days, so future-day score cards are not shown."
            )
            st.markdown('</div></div>', unsafe_allow_html=True)

            # Patient-facing next steps: do not infer treatment from the experimental score.
            st.markdown("""
            <div style="padding:0 24px"><div class="adv">
              <div class="adv-hdr">What should I do next?</div>
              <ul>
                <li>Review your reported symptoms and your RAPID3 result with your healthcare professional if you have concerns.</li>
                <li>Do not start, stop, or change medication because of this experimental score.</li>
                <li>Contact your healthcare professional about new, severe, or worsening symptoms, regardless of the score.</li>
              </ul>
            </div>
            <div class="disc">Research prototype only. The experimental score is not a validated clinical risk estimate, diagnosis, or treatment recommendation.</div>
            </div>""", unsafe_allow_html=True)

        with tab_why:
            # ── 2. RAPID3 Score Breakdown ───────────────────────────────────────
            fn_sc = res["fn_score"]; pain = res["pain"]; gs = res["gs"]; fat = res["fatigue"]
            fnp = fn_sc / 10 * 100; pp = pain / 10 * 100; gp = gs / 10 * 100; fp_ = fat / 10 * 100; r3p = r3 / 30 * 100

            cats_html = "".join(f"""<div class="r3-cat" style="background:{bg};
                border:{'2px' if r3sty['label'] == label else '1px'} solid {border}">
                <div class="r3-catlbl">{cap_label}</div>
                <div class="r3-catval" style="color:{text}">{label}</div>
                </div>""" for (cap_label, label, text, bg, border) in [
                    ("≤3", R3_TIERS[0][1], R3_TIERS[0][2], R3_TIERS[0][3], R3_TIERS[0][4]),
                    ("4–6", R3_TIERS[1][1], R3_TIERS[1][2], R3_TIERS[1][3], R3_TIERS[1][4]),
                    ("7–12", R3_TIERS[2][1], R3_TIERS[2][2], R3_TIERS[2][3], R3_TIERS[2][4]),
                    (">12", R3_TIERS[3][1], R3_TIERS[3][2], R3_TIERS[3][3], R3_TIERS[3][4]),
                ])

            st.markdown(f"""
            <div style="padding:0 24px">
            <div class="r3-card">
              <div class="qcard-hdr">Prototype Questionnaire Score Breakdown</div>
              <div class="r3-row">
                <div class="r3-top"><span class="r3-lbl">Function component (10 selected activities)</span><span class="r3-val">{fn_sc:.1f}/10</span></div>
                <div class="r3-track"><div class="r3-fill" style="width:{fnp:.0f}%;background:var(--teal2)"></div></div>
              </div>
              <div class="r3-row">
                <div class="r3-top"><span class="r3-lbl">Pain VAS (Huskisson 1974)</span><span class="r3-val">{pain:.1f}/10</span></div>
                <div class="r3-track"><div class="r3-fill" style="width:{pp:.0f}%;background:var(--amber)"></div></div>
              </div>
              <div class="r3-row">
                <div class="r3-top"><span class="r3-lbl">Patient Global Estimate</span><span class="r3-val">{gs:.1f}/10</span></div>
                <div class="r3-track"><div class="r3-fill" style="width:{gp:.0f}%;background:var(--violet)"></div></div>
              </div>
              <div class="r3-row" style="opacity:0.75">
                <div class="r3-top"><span class="r3-lbl">Fatigue VAS (Wolfe 1996) — supplementary</span><span class="r3-val">{fat:.1f}/10</span></div>
                <div class="r3-track"><div class="r3-fill" style="width:{fp_:.0f}%;background:var(--teal)"></div></div>
              </div>
              <div class="r3-total">
                <div class="r3-top">
                  <span style="font-weight:700;color:var(--text);font-size:0.78rem">Prototype total score</span>
                  <span style="color:{r3sty['text']};font-size:0.9rem;font-weight:700">{r3:.1f}/30 · {r3sty['label']}</span>
                </div>
                <div class="r3-track" style="height:8px"><div class="r3-fill" style="width:{r3p:.0f}%;background:{r3sty['border']}"></div></div>
              </div>
              <div class="r3-cats">{cats_html}</div>
            </div>
            </div>""", unsafe_allow_html=True)

            # ── 2b. DAS28 Clinical Markers ──────────────────────────────────────
            if das:
                dsty = das["style"]
                das28_body = f"""
                <span class="chip" style="background:{dsty['bg']};color:{dsty['text']};border-color:{dsty['border']}">
                  DAS28 <b>{das['score']:.2f}</b> · {dsty['label']}
                </span>"""
            else:
                das28_body = '<span class="chip">No lab value provided — joint counts recorded for audit only</span>'
            st.markdown(f"""
            <div style="padding:0 24px">
            <div class="qcard">
              <div class="qcard-hdr">DAS28 Clinical Markers · Prevoo et al. 1995</div>
              <div class="chips" style="margin:0 0 8px">
                <span class="chip">Swollen joints <b>{st.session_state.das_sjc}/28</b></span>
                <span class="chip">Tender joints <b>{st.session_state.das_tjc}/28</b></span>
                {das28_body}
              </div>
              <div class="field-help">
                DAS28 not used as an ML model input in this version — recorded for clinical
                completeness and audit only.
              </div>
            </div>
            </div>""", unsafe_allow_html=True)

            # ── 3. Feature Importance / XAI Chart ───────────────────────────────
            st.markdown('<div style="padding:0 24px"><div class="qcard">', unsafe_allow_html=True)
            st.markdown('<div class="qcard-hdr">🔍 Explainability — Inputs used by the experimental calculation</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">'
                'Additive decomposition of the peak-day experimental score — each bar shows how much that '
                'factor contributed to the final heuristic score.</div>', unsafe_allow_html=True)
            if _PLOTLY_OK:
                try:
                    items = sorted(contributions.items(), key=lambda kv: kv[1])
                    labels = [k for k, _ in items]
                    values = [v for _, v in items]
                    bar_colors = ["#EF4444" if v == max(values) else "#0EA5E9" for v in values]
                    xai_fig = pgo.Figure(pgo.Bar(
                        x=values, y=labels, orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:.2f}" for v in values], textposition="outside",
                    ))
                    xai_fig.update_layout(
                        height=340, margin=dict(l=10, r=30, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#0F172A", "family": "Inter", "size": 11},
                        xaxis={"title": "Contribution to risk score", "gridcolor": "#E2E8F0", "zerolinecolor": "#E2E8F0"},
                        yaxis={"gridcolor": "#E2E8F0"},
                    )
                    st.plotly_chart(xai_fig, use_container_width=True, config={"displayModeBar": False})
                except Exception as exc:
                    LOG.error("XAI chart render failed: %s", exc)
                    for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]):
                        st.write(f"**{k}**: {v:.2f}")
            else:
                for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]):
                    st.write(f"**{k}**: {v:.2f}")
            st.markdown('</div></div>', unsafe_allow_html=True)

        with tab_model:
            st.info(
                "**Research status:** the console currently uses a rule-based clinical-symptom and "
                "weather heuristic only. The separately trained RA, sleep, HRV, and Sri Lankan RA models "
                "predict different targets and expect different feature schemas, so they are intentionally "
                "not fused into this score. A validated, schema-matched meta-model and independent fusion "
                "evaluation are still required. Base-model metrics below are not fusion-model metrics."
            )
            # ── 4. Model Performance Metrics (expandable) ───────────────────────
            st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
            try:
                metrics, metrics_available = fusion.get_model_metrics()
            except Exception as exc:
                metrics, metrics_available = {}, False
                LOG.error("get_model_metrics failed: %s", exc)
            with st.expander("📈 Model Performance Metrics (single held-out test split)"):
                metrics_status, metrics_scope = fusion.model_metrics_labels(metrics_available)
                st.markdown(
                    f'<span class="model-status fallback">{metrics_status}</span>',
                    unsafe_allow_html=True,
                )
                if not metrics:
                    st.info(
                        "No valid base-model metrics are available. The file may be missing, unreadable, "
                        "or have an unsupported structure."
                    )
                else:
                    st.caption(
                        "Accuracy, weighted F1, and ROC-AUC on ONE held-out test split, compared across "
                        "the three candidate base learners for each sub-model."
                    )
                    st.caption(
                        f"{metrics_scope} RA Clinical and Sri Lankan RA have small cohorts, so estimates may be noisy. "
                        "The HRV dataset lacks reliable subject/session IDs; feature-space grouping is only a proxy "
                        "and cannot establish patient-level generalisation. Re-run training before quoting metrics."
                    )
                    for sub_model, rows in metrics.items():
                        st.markdown(f"**{sub_model}**")
                        table_rows = []
                        for algo, m in rows.items():
                            table_rows.append({
                                "Algorithm": algo,
                                "Accuracy": f"{m['acc']:.3f}",
                                "F1-Score": f"{m['f1']:.3f}",
                                "ROC-AUC": f"{m['auc']:.3f}",
                            })
                        st.table(table_rows)
                if MODELS and MODELS.load_errors:
                    st.markdown("**This session's model load status**")
                    for slot, err in MODELS.load_errors.items():
                        st.markdown(f"- `{slot}` → calibrated fallback used ({err})")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Assessment History ─────────────────────────────────────────────
            try:
                history = fusion.get_history(st.session_state.username)
            except Exception as exc:
                history = []
                LOG.error("get_history failed: %s", exc)
            if history:
                st.markdown('<div style="padding:0 24px;margin-top:18px">', unsafe_allow_html=True)
                with st.expander(f"🕓 Your Previous RAPID3 Scores ({len(history)})"):
                    for created_at, city_h, rapid3_h, peak_h, level_h in history:
                        try:
                            ts = datetime.fromisoformat(created_at).strftime("%d %b %Y, %H:%M")
                        except Exception:
                            ts = created_at
                        st.markdown(
                            f"- **{ts}** · {city_h} · RAPID3 **{rapid3_h:.1f}/30** · "
                            f"Experimental score **{peak_h:.0%}** ({level_h.title()})"
                        )
                st.markdown('</div>', unsafe_allow_html=True)

        with tab_fb:
            # ── Patient Feedback ─────────────────────────────────────────────────
            st.markdown("""
            <div style="padding:0 24px;margin-top:22px">
            <div class="qcard">
              <div class="qcard-hdr">📝 Patient Feedback — Help improve this system</div>
              <div style="font-size:0.78rem;color:var(--text2);margin-bottom:14px">
                Your feedback helps us improve the system. This takes less than 1 minute.
              </div>
            </div>
            </div>""", unsafe_allow_html=True)

            with st.form("feedback_form"):
                st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

                ease = st.select_slider(
                    "How easy was the system to use?",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: {1: "Very difficult", 2: "Difficult", 3: "Okay", 4: "Easy", 5: "Very easy"}[x],
                    value=4,
                )
                clarity = st.select_slider(
                    "How clear were the questions?",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: {1: "Very confusing", 2: "Confusing", 3: "Okay", 4: "Clear", 5: "Very clear"}[x],
                    value=4,
                )
                accuracy = st.selectbox(
                    "Did the risk prediction match how you were feeling?",
                    ["Yes — it matched well", "Somewhat — partially correct", "No — it did not match"],
                )
                would_use = st.selectbox(
                    "Would you use this system regularly?",
                    ["Yes, definitely", "Maybe", "No"],
                )
                comments = st.text_area(
                    "Any comments or suggestions?",
                    placeholder="Optional — write anything here...",
                    height=80,
                )

                fb_submitted = st.form_submit_button("Submit Feedback")

                if fb_submitted:
                    file_exists = os.path.isfile(fusion.FEEDBACK_PATH)
                    try:
                        with open(fusion.FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["timestamp", "patient_id", "city", "age", "risk",
                                                  "ease", "clarity", "accuracy", "would_use", "comments"])
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M"),
                                st.session_state.patient_id, st.session_state.city, st.session_state.age,
                                f"{peak:.0%} {plvl.upper()}",
                                ease, clarity, accuracy, would_use, comments,
                            ])
                        st.success("✅ Thank you! Your feedback has been recorded.")
                    except Exception as exc:
                        LOG.error("Feedback CSV write failed: %s", exc)
                        st.error("Could not save feedback right now. Please try again.")

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        if not st.session_state.confirm_reset:
            if st.button("← Start New Assessment", type="secondary"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("This will clear the current assessment. Are you sure?")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Cancel", type="secondary", key="cancel_reset"):
                    st.session_state.confirm_reset = False
                    st.rerun()
            with rc2:
                if st.button("Yes, start new assessment", type="primary", key="confirm_reset_btn"):
                    keep_auth = dict(
                        authenticated=st.session_state.authenticated,
                        username=st.session_state.username,
                        display_name=st.session_state.display_name,
                    )
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    for k, v in keep_auth.items():
                        st.session_state[k] = v
                    st.session_state.demo_mode = False
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
