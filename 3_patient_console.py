"""
RA FlareGuard — Patient Console (Production Build)
=============================================================================
Clinical standards:
  RAPID3      Pincus et al., J Rheumatol 2008;35:2136
  MDHAQ       Pincus & Swearingen, Arthritis Rheum 1999;42:2220
  HAQ-DI      Fries et al., Arthritis Rheum 1980;23:137
  VAS         Huskisson, Lancet 1974;2:1127
  Fatigue VAS Wolfe et al., J Rheumatol 1996;23:1407
  RADAI       Stucki et al., Arthritis Rheum 1995;38:795
  DAS28       Prevoo et al., Arthritis Rheum 1995;38:44 (DAS28-ESR/CRP)

This file is the Streamlit presentation layer only. All model loading,
weather retrieval, RAPID3/HAQ-DI scoring, risk fusion, and persistence
of the *existing* fields live in the framework-agnostic backend module
`2_fusion.py`, loaded below via importlib (its filename starts with a
digit, so it can't be `import`-ed directly as a regular Python identifier).

Every fusion.* call signature, argument order, and score-calculation
usage from the original console is preserved unchanged in this rewrite —
only the Streamlit presentation/production layer around it was touched.

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
   NOT wired into `fusion.compute_risk()` — the trained model's input
   contract is left untouched.
4. Auto-save: every form widget below is bound directly to
   `st.session_state` via its `key`, so a value is captured the instant
   it changes — not only when "Continue" is pressed. A browser refresh
   or crash mid-form loses nothing already answered.
5. Offline resilience: Supabase writes are wrapped in bounded retry
   logic. If every retry fails (e.g. clinic Wi-Fi drop), the payload is
   appended to a local `pending_sync.json` queue instead of being
   dropped, and is retried automatically on the next successful run of
   this console. Local persistence via `fusion.save_assessment()` /
   `fusion.get_history()` remains the primary store the in-app
   "Previous Scores" panel reads from, independent of cloud status.
"""

import csv
import importlib.util
import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import streamlit as st

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("raflareguard")

APP_VERSION = "v2.4-production"
SITE_LABEL = "Clinical Environment · Site 01 · Patient Self-Assessment Mode"

# ─────────────────────────────────────────────────────────────────────────
# Load the backend module (2_fusion.py) from disk
# ─────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION_PATH = os.path.join(_HERE, "2_fusion.py")
_PENDING_SYNC_PATH = os.path.join(_HERE, "pending_sync.json")

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
    layout="centered",
    initial_sidebar_state="collapsed",
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
# Optional Supabase client — cloud audit log with retry + offline queue.
# Never blocks the app: any failure here degrades to "local only", and
# the payload is queued for a later automatic retry instead of being lost.
# ─────────────────────────────────────────────────────────────────────────
_SUPABASE_CLIENT = None
_SUPABASE_INIT_TRIED = False


def get_supabase_client():
    """Lazily build a Supabase client from st.secrets. Returns None (and
    logs, never raises) if the `supabase` package isn't installed or the
    SUPABASE_URL / SUPABASE_KEY secrets aren't configured."""
    global _SUPABASE_CLIENT, _SUPABASE_INIT_TRIED
    if _SUPABASE_INIT_TRIED:
        return _SUPABASE_CLIENT
    _SUPABASE_INIT_TRIED = True
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        _SUPABASE_CLIENT = create_client(url, key)
        LOG.info("Supabase client initialised.")
    except Exception as exc:
        LOG.warning("Supabase unavailable, falling back to local-only persistence: %s", exc)
        _SUPABASE_CLIENT = None
    return _SUPABASE_CLIENT


def _queue_pending_sync(payload: dict) -> None:
    """Append a failed payload to the local pending_sync.json queue file
    so nothing is lost on a Wi-Fi drop. Best-effort — if even the local
    queue write fails, we log it and move on rather than crash the app."""
    try:
        queue = []
        if os.path.isfile(_PENDING_SYNC_PATH):
            try:
                with open(_PENDING_SYNC_PATH, "r", encoding="utf-8") as f:
                    queue = json.load(f)
                if not isinstance(queue, list):
                    queue = []
            except Exception:
                queue = []
        queue.append(payload)
        with open(_PENDING_SYNC_PATH, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2, default=str)
    except Exception as exc:
        LOG.error("Failed to write pending_sync.json queue: %s", exc)


def _flush_pending_sync(client) -> int:
    """Try to resend anything queued from a previous offline session.
    Returns the number of records successfully flushed. Never raises."""
    if client is None or not os.path.isfile(_PENDING_SYNC_PATH):
        return 0
    try:
        with open(_PENDING_SYNC_PATH, "r", encoding="utf-8") as f:
            queue = json.load(f)
        if not isinstance(queue, list) or not queue:
            return 0
    except Exception as exc:
        LOG.error("Failed to read pending_sync.json queue: %s", exc)
        return 0

    still_pending = []
    flushed = 0
    for payload in queue:
        try:
            client.table("patient_assessments").insert(payload).execute()
            flushed += 1
        except Exception as exc:
            LOG.warning("Retry flush failed for a queued record: %s", exc)
            still_pending.append(payload)

    try:
        if still_pending:
            with open(_PENDING_SYNC_PATH, "w", encoding="utf-8") as f:
                json.dump(still_pending, f, indent=2, default=str)
        elif os.path.isfile(_PENDING_SYNC_PATH):
            os.remove(_PENDING_SYNC_PATH)
    except Exception as exc:
        LOG.error("Failed to update pending_sync.json after flush: %s", exc)

    return flushed


def save_to_supabase(payload: dict, max_retries: int = 3, backoff_seconds: float = 0.6) -> bool:
    """Insert one row into `patient_assessments` with bounded retry/backoff.
    On total failure, queues the payload locally instead of dropping it.
    Returns True only on a confirmed successful cloud insert."""
    client = get_supabase_client()
    if client is None:
        _queue_pending_sync(payload)
        return False

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            client.table("patient_assessments").insert(payload).execute()
            return True
        except Exception as exc:
            last_exc = exc
            LOG.warning("Supabase insert attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)

    LOG.error("Supabase insert failed after %d attempts: %s", max_retries, last_exc)
    _queue_pending_sync(payload)
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Constants sourced from the backend
# ─────────────────────────────────────────────────────────────────────────
CITIES = fusion.CITIES
RAPID3_Qs = fusion.RAPID3_QUESTIONS
DIFF_SHORT = fusion.DIFFICULTY_SHORT
TOTAL_STEPS = 6  # 0 Welcome · 1 Demographics · 2 RAPID3 · 3 DAS28 · 4 Lifestyle&Sensor · 5 Results
STEP_NAMES = ["Welcome", "Demographics", "RAPID3 Survey", "DAS28 Markers", "Lifestyle & Sensor", "Results"]

# ─────────────────────────────────────────────────────────────────────────
# Init backend resources (DB + models) — cached per Streamlit session,
# guarded so a broken DB/model file shows a friendly message instead of
# a raw traceback during a live demo.
# ─────────────────────────────────────────────────────────────────────────
try:
    fusion.init_db()
    _DB_INIT_ERROR = None
except Exception as exc:
    _DB_INIT_ERROR = f"{type(exc).__name__}: {exc}"
    LOG.error("Local DB init failed: %s", _DB_INIT_ERROR)


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

# Try to flush any records left over from a previous offline session,
# once per Streamlit session (not on every rerun).
if "sync_flush_attempted" not in st.session_state:
    st.session_state.sync_flush_attempted = True
    _flushed_ct = _flush_pending_sync(get_supabase_client())
    if _flushed_ct:
        st.session_state["_flush_notice"] = _flushed_ct

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
      <span>🧮 HAQ-DI Function&nbsp;<b>{fn_score:.1f}/10</b></span>
      <span>🎚 Pain VAS&nbsp;<b>{p:.1f}/10</b></span>
      <span>📊 RAPID3&nbsp;<b style="color:{r3style['text']}">{r3:.1f}/30 · {r3style['label']}</b></span>
      <span class="autosave-tag"><span class="autosave-dot"></span>Auto-saved</span>
    </div>""", unsafe_allow_html=True)


def run_assessment():
    """Reads the fully-collected wizard inputs from session_state, fetches
    weather (with automatic fallback), fuses the multimodal risk models,
    projects a 3-day-ahead trend, computes the optional DAS28 score, stores
    everything the Step-5 dashboard needs into st.session_state.results,
    persists to local history, and mirrors the record to Supabase with
    retry + offline-queue fallback. Every external call is wrapped so a
    single failure degrades gracefully instead of crashing the live demo."""
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
        with st.spinner("Fetching live weather & computing your forecast…"):
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
            "⚠️ We couldn't complete your forecast right now — this is usually a temporary "
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
        synced_to_cloud=False, submission_latency_ms=submission_latency_ms,
    )

    # Primary persistence — local history (existing patient-account feature).
    try:
        fusion.save_assessment(
            st.session_state.username, st.session_state.city, r3, fn_score,
            st.session_state.pain, st.session_state.global_s, peak_risk, level,
        )
    except Exception as exc:
        LOG.error("Local save_assessment failed: %s", exc)
        st.warning("Your forecast is ready, but we couldn't save it to your local history right now.")

    # Secondary, cloud audit log — Supabase, with retry + offline queue and
    # a full clinical-audit metadata envelope.
    try:
        payload = dict(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            patient_id=st.session_state.patient_id,
            pain_score=int(round(st.session_state.pain)),
            swollen_joint_count=int(st.session_state.das_sjc),
            tender_joint_count=int(st.session_state.das_tjc),
            rapid3_score=float(r3),
            haq_di_score=float(fn_score),
            predicted_flare_risk=float(peak_risk),
            risk_category=level,
            app_version=APP_VERSION,
            client_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            submission_latency_ms=submission_latency_ms,
            device_viewport=get_device_viewport(),
        )
        st.session_state.results["synced_to_cloud"] = save_to_supabase(payload)
    except Exception as exc:
        LOG.error("Supabase payload build failed: %s", exc)
        st.session_state.results["synced_to_cloud"] = False


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
  <span>Build <b>{APP_VERSION}</b></span>
</div>""", unsafe_allow_html=True)

if st.session_state.get("_flush_notice"):
    st.success(
        f"☁️ Reconnected — {st.session_state['_flush_notice']} previously offline "
        "record(s) have now synced to the cloud audit log."
    )
    del st.session_state["_flush_notice"]

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
          Know your flare risk<br><span style="color:var(--teal2)">3 days ahead</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap" style="padding-top:0">', unsafe_allow_html=True)
    st.markdown("""
    <div class="demo-banner">
      <span>🎓 <b>Evaluator / Viva panel?</b> Skip login and jump straight to a fully
      populated results dashboard with sample patient data.</span>
    </div>""", unsafe_allow_html=True)
    if st.button("⚡ Quick Viva / Evaluator Demo Mode", type="primary", use_container_width=True):
        enter_demo_mode()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap" style="padding-top:0">', unsafe_allow_html=True)
    tab_login, tab_signup, tab_guest = st.tabs(["Log In", "Sign Up", "Continue as Guest"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
            if submitted:
                try:
                    ok, msg = fusion.verify_user(u, p)
                except Exception as exc:
                    ok, msg = False, "Login is temporarily unavailable — please try again shortly."
                    LOG.error("verify_user failed: %s", exc)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.demo_mode = False
                    st.session_state.username = u.strip().lower()
                    st.session_state.display_name = msg
                    st.success(f"Welcome back, {msg}!")
                    st.rerun()
                else:
                    st.error(msg)
        st.caption("Logging in lets us remember your previous RAPID3 scores between visits.")

    with tab_signup:
        with st.form("signup_form"):
            su = st.text_input("Choose a username")
            sn = st.text_input("Display name (optional)")
            sp = st.text_input("Choose a password", type="password")
            sp2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            if submitted:
                if sp != sp2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        ok, msg = fusion.create_user(su, sp, sn)
                    except Exception as exc:
                        ok, msg = False, "Account creation is temporarily unavailable — please try again shortly."
                        LOG.error("create_user failed: %s", exc)
                    if ok:
                        st.success(msg + " Please log in from the 'Log In' tab.")
                    else:
                        st.error(msg)

    with tab_guest:
        st.markdown(
            '<p style="font-size:0.8rem;color:var(--text2);margin-bottom:10px">'
            "Try the full assessment without creating an account. Your results won't be "
            "saved to a personal history, but you'll still see the complete dashboard.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Continue as Guest →", use_container_width=True):
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
        <span>⚡ <b>Quick Viva Demo Mode</b> — showing a pre-populated sample assessment.</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session-safety controls — visible once authenticated, on every step,
# so a clinician can end a consultation and wipe patient state at any time.
# ─────────────────────────────────────────────────────────────────────────
_sig_col1, _sig_col2 = st.columns([4, 1])
with _sig_col1:
    st.markdown(
        f'<div style="padding:8px 24px 0;font-size:0.7rem;color:var(--text3)">'
        f'Signed in as <b style="color:var(--text2)">{st.session_state.display_name or st.session_state.username}</b></div>',
        unsafe_allow_html=True)
with _sig_col2:
    st.markdown('<div style="padding:8px 24px 0">', unsafe_allow_html=True)
    if st.button("🧹 Clear Session", key="clear_session_top", help="End this consultation and wipe patient data from memory"):
        st.session_state["_confirm_clear_top"] = True
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("_confirm_clear_top"):
    st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
    st.warning(
        "This clears the current patient's answers and results from this device's memory "
        "(local history and any synced cloud record are unaffected). Continue?"
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
    model_badge = (
        '<span class="model-status ok">✓ All ML models loaded</span>' if MODELS_OK else
        f'<span class="model-status fallback">⚠ {len(MODELS.load_errors) if MODELS else "?"} model(s) using calibrated fallback</span>'
    )
    st.markdown(f"""
    <div class="step-wrap" style="padding-top:10px">
      <div style="text-align:center;padding:10px 0 16px">
        <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.16em;color:var(--teal2);margin-bottom:8px">
          RA FlareGuard · Sri Lanka
        </div>
        <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:1.7rem;color:var(--text);
             line-height:1.2;margin-bottom:10px;letter-spacing:-0.01em">
          Know your flare risk<br><span style="color:var(--teal2)">3 days ahead</span>
        </div>
        <div style="font-size:0.8rem;color:var(--text2);line-height:1.6;
             max-width:420px;margin:0 auto 14px">
          Answer a short clinically validated questionnaire and get a
          personalised 3-day flare risk forecast — including live Sri Lankan
          weather conditions.
        </div>
        <div style="display:flex;gap:7px;justify-content:center;flex-wrap:wrap;margin-bottom:12px">
          <span class="chip">⚕️ RAPID3 / MDHAQ</span>
          <span class="chip">📊 HAQ-DI validated</span>
          <span class="chip">🦴 DAS28 markers</span>
          <span class="chip">🌤 Live weather</span>
          <span class="chip">🤖 ML models</span>
        </div>
        <div style="margin-bottom:16px">{model_badge}</div>
      </div>
      <div class="qcard" style="padding:12px 16px">
        <div class="qcard-hdr" style="margin-bottom:8px">What to expect</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:10px 12px;font-size:0.72rem;color:var(--text2)">
            <b style="color:var(--text)">⏱ 3–5 minutes</b><br>to complete
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:10px 12px;font-size:0.72rem;color:var(--text2)">
            <b style="color:var(--text)">📋 5 sections</b><br>step by step
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:10px 12px;font-size:0.72rem;color:var(--text2)">
            <b style="color:var(--text)">🗓 3-day forecast</b><br>with live weather
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:10px 12px;font-size:0.72rem;color:var(--text2)">
            <b style="color:var(--text)">🔬 Clinical standard</b><br>RAPID3 / HAQ-DI / DAS28
          </div>
        </div>
      </div>
      <div class="privacy-banner">
        🔒 <b>De-identified by design.</b> This tool never asks for your real name — only a
        self-chosen Patient ID / Subject Code that you control.
      </div>
      <div class="field-help" style="margin-top:6px">
        💾 Every answer auto-saves as you go — an accidental refresh or app crash mid-form
        will not lose what you've already entered.
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
      <div class="step-eyebrow">Section 01 · Patient Demographics &amp; Anonymized Baseline</div>
      <div class="step-title">About <em>this record</em></div>
      <div class="step-desc">Just a de-identified baseline so we can personalise your forecast and fetch live weather for your area.</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        st.markdown("""
        <div class="privacy-banner">
          🔒 <b>Do not enter your real name or national ID.</b> Use a de-identified code
          only you (or your clinician) can map back to your identity — e.g.
          <code>PATIENT_001</code>. Letters, numbers, <code>_</code> and <code>-</code> only, 3–40 characters.
        </div>""", unsafe_allow_html=True)
        st.text_input(
            "Patient ID / Subject Code *",
            key="patient_id",
            placeholder="e.g. PATIENT_001",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Your age *", min_value=0, max_value=120, key="age",
                             help="Leave at 0 until you enter a real value — required to continue.")
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
            '<div class="field-help">Not in the list? Just type your own city/town — any location works. '
            'None of this is pre-filled — every field starts blank for a genuine, unbiased assessment.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def save1():
        pid = (st.session_state.patient_id or "").strip()
        if not valid_patient_id(pid):
            st.error("Please enter a valid Patient ID (3–40 characters: letters, numbers, _ or - only)."); return
        age = st.session_state.age
        if not age or age < 18 or age > 90:
            st.error("Please enter a valid age between 18 and 90."); return
        if not st.session_state.city:
            st.error("Please enter your city or town."); return
        st.session_state.patient_id = pid
        go(2)

    nav(next_label="Continue →", next_fn=save1)

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — SECTION 02 · RAPID3 SURVEY (FUNCTION + PAIN VAS + GLOBAL ESTIMATE)
# ══════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 02 · RAPID3 Survey (Functional Disability, Pain, Global Estimate)</div>
      <div class="step-title">What can you <em>do — and how do you feel?</em></div>
      <div class="step-desc">
        The complete RAPID3 questionnaire — the same tool rheumatologists worldwide use to
        measure functional disability, pain, and overall disease impact in RA patients.
      </div>
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
          <div class="qcard-hdr">Pain VAS</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:11px">
            <b style="color:var(--text)">Official wording:</b> "How much pain have you had because
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
          <div class="qcard-hdr">Patient Global Estimate</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:11px">
            <b style="color:var(--text)">Official wording:</b> "Considering all the ways your illness
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
# STEP 3 — SECTION 03 · DAS28 CLINICAL MARKERS
# ══════════════════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 03 · DAS28 Clinical Markers</div>
      <div class="step-title">Joint counts <em>&amp; lab markers</em></div>
      <div class="step-desc">
        Swollen and tender joint counts (28-joint assessment) are recorded for your clinical
        audit trail. If you have a recent ESR or CRP lab result, we can also calculate your
        full DAS28 disease-activity score — <b style="color:var(--text)">this is entirely optional.</b>
      </div>
      <div class="step-cite">📖 DAS28: Prevoo et al., Arthritis Rheum 1995;38:44</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
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

    scoreboard()
    nav(next_fn=lambda: go(4))

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — SECTION 04 · LIFESTYLE & SENSOR METRICS
# ══════════════════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 04 · Lifestyle &amp; Sensor Metrics</div>
      <div class="step-title">Sleep, stress <em>&amp; recent history</em></div>
      <div class="step-desc">
        MDHAQ wellbeing items, fatigue, activity, flare/RA history, and a simple heart-rate
        reading used to estimate your HRV stress signal.
      </div>
      <div class="step-cite">📖 MDHAQ: Pincus 2009 · Fatigue VAS: Wolfe et al. 1996 · Flare count: RADAI — Stucki et al. 1995</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown("""
        <div class="qcard">
          <div class="qcard-hdr">MDHAQ Supplement — same 0–3 scale</div>
        </div>""", unsafe_allow_html=True)

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
          <div class="qcard-hdr">Fatigue VAS · Wolfe et al. 1996</div>
          <div style="font-size:0.8rem;color:var(--text2);margin-bottom:9px">
            How much <b style="color:var(--text)">fatigue or tiredness</b> have you had because of your RA <b style="color:var(--text)">over the past week?</b>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="vas-grad"></div>', unsafe_allow_html=True)
        nrs_pills("fatigue")
        st.markdown('<div class="vas-ends"><span>0 — No fatigue</span><span>10 — Worst possible</span></div>',
                    unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:14px"><div class="qcard-hdr">Flare history — RADAI adapted</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many times did your joint pain <b style="color:var(--text)">suddenly get much worse</b>
        in the <b style="color:var(--text)">past 30 days?</b></div>""", unsafe_allow_html=True)
        st.number_input("Flares", 0, 30, key="flares30", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">RA duration</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many years ago were you first diagnosed with RA?</div>""", unsafe_allow_html=True)
        st.number_input("Years", 0, 50, key="duration", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">Activity today — MDHAQ exercise item</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        How many minutes of <b style="color:var(--text)">walking or light movement</b> did you do today?</div>""",
                    unsafe_allow_html=True)
        st.number_input("Minutes", 0, 180, key="activity", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:11px"><div class="qcard-hdr">Heart rate — HRV estimation (optional)</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:var(--text2);margin-bottom:9px">
        What is your <b style="color:var(--text)">current pulse rate?</b>
        Check your phone health app, smartwatch, or count for 15 seconds × 4.
        Leave at 0 if you don't know it right now — we'll use a neutral baseline for the
        forecast and simply note it as "not measured" in your record.</div>""", unsafe_allow_html=True)
        st.number_input("bpm", 0, 220, key="heart_rate", label_visibility="collapsed")
        st.markdown("""<div style="font-size:0.68rem;color:var(--text3);margin-top:5px">
        Normal resting pulse = 60 to 100 beats per minute</div></div>""", unsafe_allow_html=True)

        confirm = st.checkbox(
            "I confirm the information I've provided in this assessment is accurate to the "
            "best of my knowledge.", value=st.session_state.get("confirm_submit_cb", False), key="confirm_submit_cb",
        )
        if not confirm:
            st.markdown(
                '<div class="field-help">Please confirm the checkbox above to calculate your forecast.</div>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    scoreboard()

    def save4():
        run_assessment()
        go(5)

    nav(next_label="Calculate My Forecast →", next_fn=save4, next_disabled=not confirm)

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

        st.markdown(f"""
        <div class="step-wrap">
          <div class="step-eyebrow">Section 05 · Predictive Model Assessment · {st.session_state.city}</div>
          <div class="result-hero">
            <div class="result-name">Patient {st.session_state.patient_id} · Age {st.session_state.age}</div>
            <div class="result-score" style="color:{pcol}">{peak:.0%}</div>
            <div class="result-pill {plvl}">{fusion.risk_emoji(peak)} {fusion.risk_label(peak).replace('Risk','').strip()} Risk</div>
            <div class="result-meta">
              {datetime.now().strftime('%d %b %Y')} &nbsp;·&nbsp;
              Peak flare risk over the next 3 days
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        real_ct = 5 - len(MODELS.load_errors) if MODELS else 0
        cloud_chip = (
            '<span class="chip ok">☁️ Synced to cloud audit log</span>' if res.get("synced_to_cloud")
            else '<span class="chip warn">💾 Saved locally &amp; queued for cloud sync (offline)</span>'
        )
        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="chips">
          <span class="chip">📍 <b>{st.session_state.city}</b></span>
          <span class="chip">🌡 <b>{weathers[0].temp:.1f}°C</b> · {weathers[0].humidity:.0f}% humidity</span>
          <span class="chip">📊 RAPID3 <b style="color:{r3sty['text']}">{r3:.1f}/30</b> · {r3sty['label']}</span>
          <span class="chip">🤖 <b>{real_ct}/5 ML models live</b>{'' if MODELS_OK else ' · fallback active'}</span>
          {cloud_chip}
          <span class="chip">⏱ <b>{res.get('submission_latency_ms', 0)} ms</b> to compute</span>
        </div>
        </div>""", unsafe_allow_html=True)

        # ── 1. Peak 3-Day Flare Risk Gauge ─────────────────────────────────
        st.markdown('<div style="padding:0 24px"><div class="qcard">', unsafe_allow_html=True)
        st.markdown('<div class="qcard-hdr">Peak 3-Day Flare Risk Gauge</div>', unsafe_allow_html=True)
        if _PLOTLY_OK:
            try:
                gauge = pgo.Figure(pgo.Indicator(
                    mode="gauge+number",
                    value=peak * 100,
                    number={"suffix": "%", "font": {"size": 40, "color": pcol}},
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
                    height=220, margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0F172A", "family": "Inter"},
                )
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            except Exception as exc:
                LOG.error("Gauge render failed: %s", exc)
                st.info(f"Peak risk: **{peak:.0%}** ({fusion.risk_label(peak)}) — chart temporarily unavailable.")
        else:
            st.info(f"Peak risk: **{peak:.0%}** ({fusion.risk_label(peak)}).")

        day_labels = ["Today", "Tomorrow", "Day +2", "Day +3"]
        fc_html = ""
        for i, (label, risk, w) in enumerate(zip(day_labels, risks, weathers)):
            lvl = fusion.risk_level(risk)
            col = level_style(lvl, "text")
            now = "now" if i == 0 else ""
            emoji = fusion.risk_emoji(risk); lb = fusion.risk_label(risk).replace("Risk", "").strip()
            date_str = w.date if w.date else datetime.now().strftime("%d %b")
            live_tag = '<div class="fc-live">● live</div>' if w.is_live else ""
            fc_html += f"""
            <div class="fc-box {now}">
              <div class="fc-day">{label}</div>
              <div style="font-size:0.6rem;color:var(--text3);margin-bottom:5px">{date_str}</div>
              <div class="fc-pct" style="color:{col}">{risk:.0%}</div>
              <div class="fc-badge {lvl}">{emoji} {lb}</div>
              <div class="fc-wx">{w.temp:.1f}°C · {w.humidity:.0f}%<br>{w.desc[:22]}</div>
              {live_tag}
            </div>"""
        st.markdown(f'<div class="fc-row">{fc_html}</div>', unsafe_allow_html=True)
        st.caption("4-day weather trend shown above (today + 3-day forecast) drives the peak risk gauge.")
        st.markdown('</div></div>', unsafe_allow_html=True)

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
          <div class="qcard-hdr">RAPID3 Score Breakdown · Pincus et al. 2008</div>
          <div class="r3-row">
            <div class="r3-top"><span class="r3-lbl">Function Score A–J (HAQ-DI)</span><span class="r3-val">{fn_sc:.1f}/10</span></div>
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
              <span style="font-weight:700;color:var(--text);font-size:0.78rem">Total RAPID3</span>
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
        st.markdown('<div class="qcard-hdr">🔍 Explainability — Top Risk Contributors</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">'
            'Additive decomposition of the peak-day risk score — each bar shows how much that '
            'factor contributed to the final prediction.</div>', unsafe_allow_html=True)
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

        # ── 4. Model Performance Metrics (expandable) ───────────────────────
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        try:
            metrics, metrics_live = fusion.get_model_metrics()
        except Exception as exc:
            metrics, metrics_live = {}, False
            LOG.error("get_model_metrics failed: %s", exc)
        with st.expander("📈 Model Performance Metrics (cross-validation)"):
            if not metrics:
                st.info("Model performance metrics are temporarily unavailable.")
            else:
                st.markdown(
                    ('<span class="model-status ok">✓ Live metrics from models/metrics.json</span>' if metrics_live
                     else '<span class="model-status fallback">⚠ Demo / reference figures — no models/metrics.json found</span>'),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Accuracy, weighted F1, and ROC-AUC from a held-out cross-validation split, "
                    "compared across the three candidate base learners for each sub-model."
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

        # ── Advisory ─────────────────────────────────────────────────────────
        if plvl == "high":
            title = "🔴  High Flare Risk — Action Recommended"
            items = [
                "Contact your rheumatologist or doctor today — do not wait",
                "Rest as much as possible — avoid heavy lifting",
                "Take your prescribed medicines on time",
                "Apply warm or cold packs to painful joints",
                "Avoid going outside during peak midday heat (11am–3pm)",
            ]
            if weathers[0].humidity > 75 or any(w.humidity > 75 for w in weathers[1:]):
                items.append(f"High humidity in {st.session_state.city} — stay indoors, use a fan or AC")
        elif plvl == "medium":
            title = "🟡  Moderate Risk — Take Precautions"
            items = [
                "Aim for at least 7–8 hours of sleep tonight",
                "Avoid stressful or physically demanding activities",
                "Drink plenty of water — stay well hydrated",
                "Gentle stretching only — no strenuous exercise",
                "Call your doctor if pain suddenly worsens",
            ]
            if any(w.humidity > 75 for w in weathers[1:]):
                items.append(f"Elevated humidity expected in {st.session_state.city} — take cool breaks")
        else:
            title = "🟢  Low Risk — You Are Doing Well"
            items = [
                "Continue your current routine and medication schedule",
                "Stay hydrated — drink water regularly",
                "Light walking is good for your joints",
                "Attend any planned appointments",
            ]

        adv_html = "".join(f'<li><div class="adv-dot {plvl}"></div>{it}</li>' for it in items)
        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="adv {plvl}">
          <div class="adv-hdr">{title}</div>
          <ul>{adv_html}</ul>
        </div>
        <div class="disc">
          ⚕️ &nbsp;<b>Clinical Decision Support Only.</b>
          Based on RAPID3/MDHAQ (Pincus 2008), HAQ-DI (Fries 1980), VAS Pain (Huskisson 1974),
          Fatigue VAS (Wolfe 1996), RADAI (Stucki 1995), DAS28 (Prevoo 1995).
          This tool does <b>not replace</b> evaluation by a qualified rheumatologist.
        </div>
        </div>""", unsafe_allow_html=True)

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
                        f"Peak risk **{peak_h:.0%}** ({level_h.title()})"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

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