# RA FlareGuard Research Prototype

FlareGuard is a research prototype for exploring rheumatoid arthritis (RA)
symptom assessment alongside clinical, physiological, and environmental data.
Its displayed risk score is experimental and must not be interpreted as a
calibrated flare probability, diagnosis, or treatment recommendation.

## Project status and methodology

The project contains separate model pipelines for Flaredown symptom data,
Sleep Health, SWELL HRV, and a Sri Lankan RA clinical dataset. These datasets
have different populations, feature schemas, observation units, and targets.
Their base models are evaluated separately for their respective
dataset-specific targets; their predictions are not combined into a trained
RA flare prediction.

The legacy learned fusion model is unavailable. It is not loaded or used by
the application. The current displayed risk score uses an experimental
rule-based heuristic based on assessment inputs and available weather
context. A valid, independently evaluated fusion model is not currently
available.

## Model performance

The checked-in `models/metrics.json` contains separate candidate base-model
results from held-out evaluations. The following are the recorded accuracy
values, in Logistic Regression / Random Forest / XGBoost order:

| Dataset | Recorded accuracy values | Evaluation target |
|---------|--------------------------|-------------------|
| RA Clinical | 0.673 / 0.762 / 0.683 | Dataset-defined future 3-day symptom-worsening proxy |
| Sleep Health | 0.893 / 0.907 / 0.933 | Sleep disorder present vs. no disorder |
| HRV Stress | 0.686 / 1.000 / 0.960 | Dataset stress condition; row-level split |
| Sri Lankan RA | 0.500 / 0.688 / 0.500 | cDAI above cohort median; not confirmed flare |

These values describe base models on their own dataset-specific targets. They
are not fusion-model metrics, are not evidence of clinical validity, and have
not been independently reproduced as part of this documentation. In
particular, the HRV split lacks verified subject/session identifiers, and the
RA Clinical and Sri Lankan RA evaluations have small held-out cohorts. See
`models/metrics.json` for the other recorded metrics and split descriptions.

## Implemented features

- Patient-friendly symptom and function assessments.
- Weather context for supported Sri Lankan locations.
- An experimental rule-based score with explanatory components.
- Historical assessment storage and review.
- Separate dataset-specific model training and evaluation pipelines.

The score is not a validated prediction for tomorrow or the following three
days. A valid independently evaluated fusion model would require aligned
observations and an independent RA flare target shared across modalities,
followed by leakage-aware training and held-out evaluation.

## System workflow

1. The user submits symptom and function assessment responses.
2. The application obtains weather context where available.
3. The application calculates and displays its experimental rule-based score.
4. Assessment history is stored for later review.
5. Separately, base models may be trained and evaluated on their own
   dataset-specific targets; those outputs are not inputs to the displayed
   risk score.

## Demo and presentation

- Demo video:
  https://drive.google.com/file/d/1b8L1t0CE5Pz0CVhBLDsdnRcjX8JzGxYT/view?usp=drive_link
- Presentation slides:
  https://docs.google.com/presentation/d/1QjLsukGy7AuADmjqd1waOIliCktjqNa-/edit?usp=sharing&ouid=118082880975097147141&rtpof=true&sd=true

## Run

Run the application using the project's configured environment and entry
point. The training script is separate from application use; running training
is not required to calculate the experimental rule-based score.

## Deployment configuration

The provided [Dockerfile](./Dockerfile) builds the React frontend and runs
FastAPI, which serves that build from the same origin. This is the supported
deployment arrangement for the current cookie-based authentication. Start the
container with the platform-provided `PORT`; the Docker command runs:

```text
uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}
```

Configure these values in the hosting platform's environment settings rather
than committing credentials:

- `APP_ENV=production` identifies production. Production is the default when
  this variable is unset.
- `COOKIE_SECURE=true` is the default and is required in production. The app
  refuses to start in production if it is explicitly disabled. For local
  plain-HTTP development only, set both `APP_ENV=development` and
  `COOKIE_SECURE=false`.
- `DATABASE_URL` should point to a persistent PostgreSQL service for hosted
  deployments. Alternatively, set `DB_PATH` to a path on storage that the
  hosting provider explicitly guarantees to persist. The default SQLite
  database is inside the application directory and may be lost when an
  ephemeral container is replaced; the platform does not automatically make
  it persistent.
- `FRONTEND_ORIGINS` is a comma-separated list of exact browser origins when
  cross-origin requests are needed. The default is for local Vite development.
  If frontend and API are separate, keep them same-site and configure the
  exact frontend origin; the current `SameSite=Lax` session cookie does not
  support a cross-site frontend/API arrangement. Prefer the same-origin
  frontend served by FastAPI.
- `OWM_API_KEY` is optional and must remain a server-side environment secret.
  Without it, weather information uses the existing estimated fallback.
- `PORT` is normally supplied by the hosting platform; locally, the Docker
  command defaults to port `10000`.

Do not put database credentials, weather keys, passwords, or other secrets in
`VITE_*` variables or frontend source: frontend build-time variables are
embedded in public browser assets. `.env` files and local database files are
excluded from Git and the Docker build context; provide production settings
through the host's secret/environment configuration.
The disabled legacy `models/meta_model.pkl` is also excluded from the Docker
build context and is not loaded or used by the application.

After deployment, check `GET /api/health` for service health and `GET
/api/system` for runtime configuration status. The system response is
operational metadata, not evidence of clinical validation; base-model
availability does not mean learned fusion is available.
