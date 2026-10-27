
"""
api.py — RA FlareGuard FastAPI backend
Provides authentication, assessments, experimental risk scoring,
and per-user assessment history. No validated future flare forecast is
currently available.
"""

import hashlib
import importlib.util
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from database import connect as connect_database
from database import execute as execute_sql
from database import is_postgres


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FUSION_PATH = BASE_DIR / "2_fusion.py"
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "flareguard.db")))

SESSION_DURATION_HOURS = 12
SESSION_COOKIE_NAME = "flareguard_session"

# Local development defaults. Configure these environment variables
# when deploying the frontend and backend.
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

# Secure cookies are the safe default. Local HTTP development may explicitly
# disable this only when APP_ENV is set to development.
APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
COOKIE_SECURE_VALUE = os.getenv("COOKIE_SECURE", "true").strip().lower()
if COOKIE_SECURE_VALUE not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
    raise RuntimeError("COOKIE_SECURE must be set to a boolean value.")
COOKIE_SECURE = COOKIE_SECURE_VALUE in {"1", "true", "yes", "on"}
if APP_ENV in {"prod", "production"} and not COOKIE_SECURE:
    raise RuntimeError("COOKIE_SECURE cannot be disabled when APP_ENV is production.")
if not COOKIE_SECURE and APP_ENV != "development":
    raise RuntimeError("COOKIE_SECURE may be disabled only when APP_ENV is development.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RA FlareGuard API",
    description="Research-preview API for the RA FlareGuard application.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# LOAD EXISTING FUSION BACKEND
# ============================================================

def load_fusion_backend():
    """Load the shared Python backend without importing Streamlit."""
    if not FUSION_PATH.is_file():
        raise RuntimeError(f"Backend file not found: {FUSION_PATH}")

    spec = importlib.util.spec_from_file_location(
        "fusion_backend",
        FUSION_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the fusion backend.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["fusion_backend"] = module
    spec.loader.exec_module(module)

    return module


fusion = load_fusion_backend()
fusion.init_db(str(DB_PATH))

# Load the available models once when the API starts.
# The current compute_risk() function does not combine model predictions.
models = fusion.load_models()


# ============================================================
# DATABASE AND SESSION HELPERS
# ============================================================

def connect_db():
    return connect_database(str(DB_PATH))


def init_session_table():
    with connect_db() as conn:
        execute_sql(conn,
            """
            CREATE TABLE IF NOT EXISTS api_sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        execute_sql(conn,
            "CREATE INDEX IF NOT EXISTS idx_sessions_username "
            "ON api_sessions(username)"
        )


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(username: str):
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_DURATION_HOURS)

    with connect_db() as conn:
        # Remove expired sessions and older sessions for this account.
        execute_sql(conn,
            "DELETE FROM api_sessions WHERE expires_at <= ?",
            (now.isoformat(),),
        )
        execute_sql(conn,
            "DELETE FROM api_sessions WHERE username = ?",
            (username,),
        )
        execute_sql(conn,
            """
            INSERT INTO api_sessions
                (token_hash, username, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                hash_session_token(token),
                username,
                expires_at.isoformat(),
                now.isoformat(),
            ),
        )

    return token, expires_at


def get_session_username(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE_NAME)

    if not token:
        raise HTTPException(status_code=401, detail="Please log in.")

    token_hash = hash_session_token(token)
    now = datetime.now(timezone.utc)

    with connect_db() as conn:
        row = execute_sql(conn,
            """
            SELECT username, expires_at
            FROM api_sessions
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

        if not row:
            raise HTTPException(
                status_code=401,
                detail="Session is invalid. Please log in again.",
            )

        try:
            expires_at = datetime.fromisoformat(row[1])
        except (ValueError, TypeError):
            execute_sql(conn,
                "DELETE FROM api_sessions WHERE token_hash = ?",
                (token_hash,),
            )
            raise HTTPException(
                status_code=401,
                detail="Session is invalid. Please log in again.",
            )

        if expires_at <= now:
            execute_sql(conn,
                "DELETE FROM api_sessions WHERE token_hash = ?",
                (token_hash,),
            )
            raise HTTPException(
                status_code=401,
                detail="Session expired. Please log in again.",
            )

        user_exists = execute_sql(conn,
            "SELECT 1 FROM users WHERE username = ?",
            (row[0],),
        ).fetchone()

        if not user_exists:
            execute_sql(conn,
                "DELETE FROM api_sessions WHERE token_hash = ?",
                (token_hash,),
            )
            raise HTTPException(
                status_code=401,
                detail="Account is unavailable.",
            )

    return row[0]


def set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_DURATION_HOURS * 60 * 60,
        path="/",
    )


def delete_session_cookie(response: Response):
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


init_session_table()


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class AssessmentRequest(BaseModel):
    """
    Inputs supported by the existing experimental risk engine.

    fn_vals must contain all ten keys A-J, each with a value from 0 to 3.
    Pain and global_estimate are on 0-10 scales.
    sleep_diff and anxiety use 0-3 scales.
    """

    age: int = Field(ge=1, le=120)
    symptom_change: str = Field(default="same", pattern="^(better|same|worse)$")
    city: str = Field(default="Colombo", min_length=1, max_length=100)

    fn_vals: dict[str, int]

    pain: float = Field(ge=0, le=10)
    global_estimate: float = Field(ge=0, le=10)
    fatigue: float = Field(ge=0, le=10)

    flares_30d: int = Field(ge=0, le=30)
    duration_years: int = Field(ge=0, le=100)

    sleep_diff: int = Field(ge=0, le=3)
    anxiety: int = Field(ge=0, le=3)

    activity_min: int = Field(ge=0, le=1440)
    heart_rate: int = Field(default=0, ge=0, le=250)


def _reported_input_summary(assessment: AssessmentRequest) -> dict:
    return {
        "symptom_change": assessment.symptom_change,
        "fatigue": assessment.fatigue,
        "sleep_difficulty": assessment.sleep_diff,
        "anxiety": assessment.anxiety,
        "flares_30d": assessment.flares_30d,
        "activity_min": assessment.activity_min,
        "heart_rate_bpm": assessment.heart_rate or None,
    }


# ============================================================
# GENERAL ENDPOINTS
# ============================================================

@app.get("/api/root")
def root():
    return {
        "message": "RA FlareGuard API is running",
        "status": "ok",
        "version": app.version,
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "RA FlareGuard",
        "mode": "research_preview",
    }


@app.get("/api/system")
def system_info():
    return {
        "application": "RA FlareGuard",
        "backend_loaded": True,
        "database_type": "postgresql" if is_postgres() else "sqlite",
        "database_file_exists": None if is_postgres() else DB_PATH.is_file(),
        "database_name": "postgresql" if is_postgres() else DB_PATH.name,
        "risk_method": "experimental heuristic",
        "clinically_validated": False,
        "fusion_model_available": models.fusion_available,
        "fusion_model_status": "unavailable: no independent patient-aligned fusion labels",
        "trained_fusion_predictions_used_by_risk_score": False,
        "models_loaded_without_errors": models.models_ok,
        "real_model_available": models.any_real,
        "model_load_errors": models.load_errors,
    }


# ============================================================
# REGISTRATION
# ============================================================

@app.post("/api/auth/register")
def register(request: RegisterRequest):
    username = request.username.strip().lower()
    display_name = request.display_name.strip()

    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="Username must contain at least 3 characters.",
        )

    success, message = fusion.create_user(
        username=username,
        password=request.password,
        display_name=display_name or username,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": "Account created successfully.",
        "username": username,
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(request: LoginRequest, response: Response):
    username = request.username.strip().lower()

    # IMPORTANT: verify_user() returns a tuple (is_valid, message).
    # Checking the tuple itself would incorrectly accept invalid passwords.
    is_valid, result = fusion.verify_user(
        username,
        request.password,
    )

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    with connect_db() as conn:
        row = execute_sql(conn,
            "SELECT display_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    token, expires_at = create_session(username)
    set_session_cookie(response, token)

    return {
        "success": True,
        "message": "Login successful.",
        "username": username,
        "display_name": row[0] if row else result,
        "expires_at": expires_at.isoformat(),
    }


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/api/auth/me")
def current_user(request: Request):
    username = get_session_username(request)

    with connect_db() as conn:
        row = execute_sql(conn,
            "SELECT display_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    return {
        "authenticated": True,
        "username": username,
        "display_name": row[0] if row else username,
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)

    if token:
        with connect_db() as conn:
            execute_sql(conn,
                "DELETE FROM api_sessions WHERE token_hash = ?",
                (hash_session_token(token),),
            )

    delete_session_cookie(response)

    return {
        "success": True,
        "message": "Logged out successfully.",
    }


# ============================================================
# SUBMIT ASSESSMENT
# ============================================================

@app.post("/api/assessments")
def create_assessment(
    assessment: AssessmentRequest,
    request: Request,
):
    """
    Calculate an experimental score and save its summary.

    Only authenticated users may submit persistent assessments.
    This endpoint does not claim clinical validation or provide a
    medically calibrated flare probability.
    """
    username = get_session_username(request)

    required_keys = set("ABCDEFGHIJ")
    supplied_keys = set(assessment.fn_vals.keys())

    if supplied_keys != required_keys:
        raise HTTPException(
            status_code=422,
            detail=(
                "fn_vals must contain exactly these keys: "
                "A, B, C, D, E, F, G, H, I, J."
            ),
        )

    for key, value in assessment.fn_vals.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(
                status_code=422,
                detail=f"fn_vals[{key}] must be an integer from 0 to 3.",
            )
        if value < 0 or value > 3:
            raise HTTPException(
                status_code=422,
                detail=f"fn_vals[{key}] must be from 0 to 3.",
            )

    city = assessment.city.strip()
    if not city:
        raise HTTPException(
            status_code=422,
            detail="City cannot be empty.",
        )

    try:
        # Existing clinical-score functions.
        fn_score = fusion.haq_di_score(assessment.fn_vals)
        rapid3 = fusion.rapid3_total(
            assessment.fn_vals,
            assessment.pain,
            assessment.global_estimate,
        )

        # Use the existing weather client. If live weather is unavailable,
        # the backend returns an estimated fallback reading.
        weather = fusion.WeatherClient().current(city)

        # Preserve the existing app's convention: zero means unavailable.
        # The model calculation uses 72 as a proxy in that case.
        heart_rate_for_model = assessment.heart_rate or 72

        risk_inputs = fusion.RiskInputs(
            pain=assessment.pain,
            fn_score=fn_score,
            global_estimate=assessment.global_estimate,
            fatigue=assessment.fatigue,
            flares_30d=assessment.flares_30d,
            duration_years=assessment.duration_years,
            sleep_diff=assessment.sleep_diff,
            anxiety=assessment.anxiety,
            activity_min=assessment.activity_min,
            heart_rate=heart_rate_for_model,
            rapid3=rapid3,
            temp=weather.temp,
            humidity=weather.humidity,
            pressure=weather.pressure,
        )

        risk_score, contributions = fusion.compute_risk(
            risk_inputs,
            models,
        )

        peak_risk = risk_score
        level = fusion.risk_level(peak_risk)

        saved = fusion.save_assessment(
            username=username,
            city=city,
            rapid3=rapid3,
            fn_score=fn_score,
            pain=assessment.pain,
            global_estimate=assessment.global_estimate,
            peak_risk=peak_risk,
            level=level,
        )

        if not saved:
            raise HTTPException(
                status_code=500,
                detail="The assessment was calculated but could not be saved.",
            )

        category, category_color = fusion.rapid3_category(rapid3)

        return {
            "success": True,
            "message": "Assessment calculated and saved.",
            "assessment": {
                "username": username,
                "city": city,
                "created_at": datetime.now().isoformat(),
                "rapid3": round(rapid3, 2),
                "rapid3_category": category,
                "function_score": round(fn_score, 2),
                "pain": assessment.pain,
                "global_estimate": assessment.global_estimate,
                "reported_inputs": _reported_input_summary(assessment),
                "experimental_score": round(risk_score, 4),
                "risk_level": fusion.risk_level(risk_score),
                "risk_label": fusion.risk_label(risk_score),
                "risk_color": fusion.risk_color(risk_score),
                "peak_experimental_score": round(peak_risk, 4),
                "peak_risk_level": level,
                "contributions": {
                    key: round(float(value), 4)
                    for key, value in contributions.items()
                },
            },
            "weather": {
                "city": city,
                "temperature": weather.temp,
                "humidity": weather.humidity,
                "pressure": weather.pressure,
                "description": weather.desc,
                "is_live": weather.is_live,
            },
            "forecast": [],
            "limitations": [
                "The score is an experimental heuristic, not a calibrated flare probability.",
                "The risk calculation does not combine trained model predictions.",
                "No clinically validated three-day RA flare forecast is available.",
                "Weather may use an estimated fallback instead of live measurements.",
                "This research prototype is not a diagnosis or a substitute for clinical care.",
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        # Avoid returning internal paths or stack traces to the client.
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed ({type(exc).__name__}).",
        ) from exc



# ============================================================
# GUEST ASSESSMENT (NO LOGIN, NOT SAVED TO ACCOUNT HISTORY)
# ============================================================

@app.post("/api/assessments/guest")
def create_guest_assessment(assessment: AssessmentRequest):
    """
    Calculate an assessment for a guest without requiring authentication
    and without saving it to a user account or persistent assessment history.

    The response uses the same experimental scoring and projection functions
    as the authenticated assessment endpoint. It is not a clinically
    validated flare probability or a trained multimodal prediction.
    """
    required_keys = set("ABCDEFGHIJ")
    supplied_keys = set(assessment.fn_vals.keys())

    if supplied_keys != required_keys:
        raise HTTPException(
            status_code=422,
            detail=(
                "fn_vals must contain exactly these keys: "
                "A, B, C, D, E, F, G, H, I, J."
            ),
        )

    for key, value in assessment.fn_vals.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(
                status_code=422,
                detail=f"fn_vals[{key}] must be an integer from 0 to 3.",
            )
        if value < 0 or value > 3:
            raise HTTPException(
                status_code=422,
                detail=f"fn_vals[{key}] must be from 0 to 3.",
            )

    city = assessment.city.strip()
    if not city:
        raise HTTPException(status_code=422, detail="City cannot be empty.")

    try:
        fn_score = fusion.haq_di_score(assessment.fn_vals)
        rapid3 = fusion.rapid3_total(
            assessment.fn_vals,
            assessment.pain,
            assessment.global_estimate,
        )

        weather = fusion.WeatherClient().current(city)
        heart_rate_for_model = assessment.heart_rate or 72

        risk_inputs = fusion.RiskInputs(
            pain=assessment.pain,
            fn_score=fn_score,
            global_estimate=assessment.global_estimate,
            fatigue=assessment.fatigue,
            flares_30d=assessment.flares_30d,
            duration_years=assessment.duration_years,
            sleep_diff=assessment.sleep_diff,
            anxiety=assessment.anxiety,
            activity_min=assessment.activity_min,
            heart_rate=heart_rate_for_model,
            rapid3=rapid3,
            temp=weather.temp,
            humidity=weather.humidity,
            pressure=weather.pressure,
        )

        risk_score, contributions = fusion.compute_risk(risk_inputs, models)

        peak_risk = risk_score
        level = fusion.risk_level(peak_risk)
        category, _category_color = fusion.rapid3_category(rapid3)

        return {
            "success": True,
            "message": "Guest assessment calculated. It was not saved to account history.",
            "assessment": {
                "username": "Guest",
                "age": assessment.age,
                "symptom_change": assessment.symptom_change,
                "city": city,
                "created_at": datetime.now().isoformat(),
                "rapid3": round(rapid3, 2),
                "rapid3_category": category,
                "function_score": round(fn_score, 2),
                "pain": assessment.pain,
                "global_estimate": assessment.global_estimate,
                "reported_inputs": _reported_input_summary(assessment),
                "experimental_score": round(risk_score, 4),
                "risk_level": fusion.risk_level(risk_score),
                "risk_label": fusion.risk_label(risk_score),
                "risk_color": fusion.risk_color(risk_score),
                "peak_experimental_score": round(peak_risk, 4),
                "peak_risk_level": level,
                "contributions": {
                    key: round(float(value), 4)
                    for key, value in contributions.items()
                },
            },
            "weather": {
                "city": city,
                "temperature": weather.temp,
                "humidity": weather.humidity,
                "pressure": weather.pressure,
                "description": weather.desc,
                "is_live": weather.is_live,
            },
            "forecast": [],
            "limitations": [
                "The score is an experimental heuristic, not a calibrated flare probability.",
                "The risk calculation does not combine trained model predictions.",
                "No clinically validated three-day RA flare forecast is available.",
                "Weather may use an estimated fallback instead of live measurements.",
                "This research prototype is not a diagnosis or a substitute for clinical care.",
                "Guest assessments are not saved to account history.",
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose internal paths or stack traces to clients.
        raise HTTPException(
            status_code=500,
            detail=f"Guest assessment failed ({type(exc).__name__}).",
        ) from exc


# ============================================================
# ASSESSMENT HISTORY
# ============================================================

@app.get("/api/assessments")
def assessment_history(
    request: Request,
    limit: int | None = None,
):
    """Return the authenticated user's assessment history, newest first."""
    username = get_session_username(request)

    if limit is not None and (limit < 1 or limit > 100):
        raise HTTPException(
            status_code=422,
            detail="When provided, limit must be between 1 and 100.",
        )

    try:
        rows = fusion.get_history(username, limit=limit)

        return {
            "success": True,
            "username": username,
            "count": len(rows),
            "assessments": [
                {
                    "created_at": row[0],
                    "city": row[1],
                    "rapid3": row[2],
                    "peak_experimental_score": row[3],
                    "risk_level": row[4],
                }
                for row in rows
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve history ({type(exc).__name__}).",
        ) from exc


# ============================================================
# RUN WITH:
# python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
# ============================================================


# --- Serve the built React app (single-service deployment) ---
from fastapi.staticfiles import StaticFiles
_dist = BASE_DIR / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
