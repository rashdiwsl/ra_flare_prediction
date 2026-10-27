import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import {
  getCurrentUser,
  getAssessmentHistory,
  login,
  logout,
  register,
  submitAssessment as sendAssessment,
  submitGuestAssessment,
  type AssessmentData,
  type AssessmentResult,
  type AuthUser,
} from './services/api'
import './App.css'

type Page = 'Overview' | 'My Symptoms' | 'Assessment' | 'Results' | 'Forecast History'
type AuthMode = 'login' | 'register'

const navigation: { label: Page; icon: string }[] = [
  { label: 'Overview', icon: '⌂' },
  { label: 'My Symptoms', icon: '◷' },
  { label: 'Assessment', icon: '✓' },
  { label: 'Results', icon: '◉' },
  { label: 'Forecast History', icon: '▤' },
]

const contributionLabels: Record<string, string> = {
  'Pain VAS': 'Pain response',
  'RAPID3 Function Component': 'Daily-function responses',
  'Patient Global Estimate': 'Overall-health response',
  'Flare History (30d)': 'Reported symptom worsening in the past 30 days',
  'Sleep Difficulty': 'Sleep response',
  'Fatigue VAS': 'Fatigue response',
  'Humidity (heuristic)': 'Humidity input',
  'Temperature (heuristic)': 'Temperature input',
}

function scorePoints(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'Not available'
  }
  return `${(value * 100).toFixed(1)} / 100 score points`
}


function AuthScreen({
  onAuthenticated,
  onGuest,
}: {
  onAuthenticated: (user: AuthUser) => void
  onGuest: () => void
}) {
  const [mode, setMode] = useState<AuthMode>('login')
  const [username, setUsername] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')

    if (mode === 'register' && password !== confirmPassword) {
      setError('The passwords do not match.')
      return
    }

    setBusy(true)

    try {
      if (mode === 'register') {
        await register({
          username: username.trim(),
          password,
          display_name: displayName.trim(),
        })
      }

      await login({
        username: username.trim(),
        password,
      })

      const authenticatedUser = await getCurrentUser()
      onAuthenticated(authenticatedUser)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to authenticate. Please try again.',
      )
    } finally {
      setBusy(false)
    }
  }

  function switchMode() {
    setMode((current) =>
      current === 'login' ? 'register' : 'login',
    )
    setError('')
    setPassword('')
    setConfirmPassword('')
  }

  return (
    <main className="auth-page">
      <section className="auth-visual">
        <div className="auth-visual-content">
          <a
            className="auth-visual-brand"
            href="#"
            onClick={(event) => event.preventDefault()}
          >
            <span className="brand-mark">F</span>
            <span>FlareGuard</span>
          </a>

          <div className="auth-visual-copy">
            <p className="auth-visual-eyebrow">
              YOUR HEALTH, IN FOCUS
            </p>

            <h2>
              Your health journey,
              <br />
              <span>understood.</span>
            </h2>

            <p className="auth-visual-description">
              A clearer view of your symptoms, patterns, and
              wellbeing. Take a moment to understand how you feel.
            </p>
          </div>

          <div className="auth-visual-panel">
            <div className="auth-panel-icon">♡</div>
            <div>
              <strong>Your wellbeing, at a glance</strong>
              <p>
                Explore symptom tracking and discover how daily
                check-ins can help you reflect on changes over time.
              </p>
            </div>
          </div>

          <div className="auth-visual-bottom">
            <span className="auth-status-dot" />
            <span>Research prototype · Demonstration only</span>
          </div>
        </div>
      </section>

      <section className="auth-form-side">
        <div className="auth-card">
          <a
            className="brand auth-brand"
            href="#"
            onClick={(event) => event.preventDefault()}
          >
            <span className="brand-mark">F</span>
            <span>
              <strong>FlareGuard</strong>
              <small>YOUR HEALTH, IN FOCUS</small>
            </span>
          </a>

          <div className="auth-heading">
            <p className="eyebrow">PATIENT SPACE</p>

            <h1>
              {mode === 'login'
                ? 'Welcome back'
                : 'Create your account'}
            </h1>

            <p className="auth-description">
              {mode === 'login'
                ? 'Sign in to continue to your personal health space.'
                : 'Create an account to explore the FlareGuard research prototype.'}
            </p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === 'register' && (
              <label>
                Display name
                <input
                  type="text"
                  autoComplete="name"
                  value={displayName}
                  onChange={(event) =>
                    setDisplayName(event.target.value)
                  }
                  maxLength={100}
                  required
                  placeholder="How should we address you?"
                />
              </label>
            )}

            <label>
              Username
              <input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(event) =>
                  setUsername(event.target.value)
                }
                minLength={mode === 'register' ? 3 : 1}
                maxLength={50}
                required
                placeholder="Enter your username"
              />
            </label>

            <label>
              Password
              <input
                type="password"
                autoComplete={
                  mode === 'login'
                    ? 'current-password'
                    : 'new-password'
                }
                value={password}
                onChange={(event) =>
                  setPassword(event.target.value)
                }
                minLength={mode === 'register' ? 10 : 1}
                maxLength={128}
                required
                placeholder={
                  mode === 'register'
                    ? 'At least 10 characters'
                    : 'Enter your password'
                }
              />
            </label>

            {mode === 'register' && (
              <label>
                Confirm password
                <input
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) =>
                    setConfirmPassword(event.target.value)
                  }
                  minLength={10}
                  maxLength={128}
                  required
                  placeholder="Enter your password again"
                />
              </label>
            )}

            {error && (
              <p className="auth-error" role="alert">
                {error}
              </p>
            )}

            <button
              className="primary-button auth-submit"
              type="submit"
              disabled={busy}
            >
              {busy
                ? 'Please wait…'
                : mode === 'login'
                  ? 'Sign in to FlareGuard'
                  : 'Create account'}
              {!busy && <span aria-hidden="true">→</span>}
            </button>
          </form>

          <div className="guest-divider">
            <span>OR</span>
          </div>

          <button
            type="button"
            className="guest-button"
            onClick={onGuest}
          >
            <span aria-hidden="true">♙</span>
            Continue as Guest
          </button>

          <p className="guest-caption">
            Explore the interface without creating an account.
          </p>

          <p className="auth-switch">
            {mode === 'login'
              ? 'New to FlareGuard? '
              : 'Already have an account? '}
            <button type="button" onClick={switchMode}>
              {mode === 'login'
                ? 'Create an account'
                : 'Sign in'}
            </button>
          </p>

          <p className="auth-disclaimer">
            Research prototype · Demo data is illustrative ·
            Not medical advice
          </p>
        </div>
      </section>
    </main>
  )
}

function App() {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isGuest, setIsGuest] = useState(false)
  const [checkingSession, setCheckingSession] = useState(true)
  const [authError, setAuthError] = useState('')
  const [activePage, setActivePage] = useState<Page>('Overview')
  const [loggingOut, setLoggingOut] = useState(false)
  const [latestAssessment, setLatestAssessment] = useState<AssessmentResult | null>(null)
  const [historyItems, setHistoryItems] = useState<import('./services/api').HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const assessmentRequestVersion = useRef(0)
  const historyRequestVersion = useRef(0)

  function clearSessionAssessmentState() {
    assessmentRequestVersion.current += 1
    historyRequestVersion.current += 1
    setLatestAssessment(null)
    setHistoryItems([])
    setHistoryLoading(false)
    setHistoryError('')
  }

  function navigateTo(page: Page) {
    assessmentRequestVersion.current += 1
    setActivePage(page)
  }

  function startAssessment() {
    assessmentRequestVersion.current += 1
    setLatestAssessment(null)
    setActivePage('Assessment')
  }

  const loadAssessmentHistory = useCallback(async () => {
    const requestVersion = ++historyRequestVersion.current

    if (isGuest || !user) {
      setHistoryItems([])
      setHistoryError('')
      setHistoryLoading(false)
      return
    }

    setHistoryLoading(true)
    setHistoryError('')

    try {
      const response = await getAssessmentHistory()
      if (requestVersion === historyRequestVersion.current) {
        setHistoryItems(response.assessments)
      }
    } catch (err) {
      if (requestVersion === historyRequestVersion.current) {
        setHistoryError(
          err instanceof Error ? err.message : 'Unable to load assessment history.',
        )
      }
    } finally {
      if (requestVersion === historyRequestVersion.current) {
        setHistoryLoading(false)
      }
    }
  }, [isGuest, user])

  useEffect(() => {
    let cancelled = false

    getCurrentUser()
      .then((currentUser) => {
        if (!cancelled) {
          setUser(currentUser)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setUser(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCheckingSession(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (checkingSession || !user || isGuest) {
      return
    }

    let cancelled = false
    const requestVersion = ++historyRequestVersion.current

    getAssessmentHistory()
      .then((response) => {
        if (!cancelled && requestVersion === historyRequestVersion.current) {
          setHistoryItems(response.assessments)
          setHistoryError('')
        }
      })
      .catch((err: unknown) => {
        if (!cancelled && requestVersion === historyRequestVersion.current) {
          setHistoryError(
            err instanceof Error
              ? err.message
              : 'Unable to load assessment history.',
          )
        }
      })
      .finally(() => {
        if (!cancelled && requestVersion === historyRequestVersion.current) {
          setHistoryLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [user, isGuest, checkingSession])

  async function handleLogout() {
    setLoggingOut(true)
    setAuthError('')

    try {
      await logout()
      clearSessionAssessmentState()
      setUser(null)
      setIsGuest(false)
      setActivePage('Overview')
    } catch (err) {
      setAuthError(
        err instanceof Error
          ? `Logout failed: ${err.message}`
          : 'Logout failed. Please try again.',
      )
    } finally {
      setLoggingOut(false)
    }
  }

  function handleGuestExit() {
    clearSessionAssessmentState()
    setIsGuest(false)
    setActivePage('Overview')
    setAuthError('')
  }

  if (checkingSession) {
    return (
      <main className="auth-page">
        <div className="auth-loading">
          <span className="brand-mark">F</span>
          <p>Preparing your FlareGuard space…</p>
        </div>
      </main>
    )
  }

  if (!user && !isGuest) {
    return (
      <AuthScreen
        onAuthenticated={(authenticatedUser) => {
          clearSessionAssessmentState()
          setUser(authenticatedUser)
          setIsGuest(false)
          setActivePage('Overview')
        }}
        onGuest={() => {
          clearSessionAssessmentState()
          setIsGuest(true)
          setActivePage('Overview')
        }}
      />
    )
  }

  const displayName = isGuest
    ? 'Guest'
    : user?.display_name || user?.username || 'User'

  const avatarLetter = displayName.charAt(0).toUpperCase()
  const resultAssessment = latestAssessment?.assessment
  const resultScore = resultAssessment?.experimental_score
  const resultContributions = Object.entries(resultAssessment?.contributions ?? {})
    .filter(([, value]) => Number.isFinite(value))
    .sort((left, right) => right[1] - left[1])
  const reportedInputs = resultAssessment?.reported_inputs
  const sleepLabels = ['Very well', 'Okay', 'Poorly', 'Very poorly']
  const anxietyLabels = ['Not at all', 'A little', 'Quite a bit', 'Extremely']
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault()
            navigateTo('Overview')
          }}
        >
          <span className="brand-mark">F</span>
          <span>
            <strong>FlareGuard</strong>
            <small>YOUR HEALTH, IN FOCUS</small>
          </span>
        </a>

        <div className="nav-heading">PATIENT SPACE</div>

        <nav className="navigation" aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.label}
              type="button"
              className={`nav-item ${
                activePage === item.label ? 'active' : ''
              }`}
              aria-current={
                activePage === item.label ? 'page' : undefined
              }
              onClick={() => {
                if (item.label === 'Assessment') {
                  startAssessment()
                } else {
                  navigateTo(item.label)
                }
              }}
            >
              <span className="nav-icon" aria-hidden="true">
                {item.icon}
              </span>
              {item.label}
              {item.label === 'Assessment' && (
                <span className="nav-dot" />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <span className="support-icon">♡</span>
            <strong>Small steps matter.</strong>
            <p>
              Tracking how you feel can help you understand your
              patterns over time.
            </p>
          </div>

          <div className="profile">
            <div className="avatar">{avatarLetter}</div>

            <div className="profile-text">
              <strong>{displayName}</strong>
              <small>
                {isGuest ? 'Guest session' : 'Signed-in account'}
              </small>
            </div>
          </div>

          {isGuest ? (
            <button
              type="button"
              className="nav-item logout-button"
              onClick={handleGuestExit}
            >
              ↪ Exit Guest Mode
            </button>
          ) : (
            <button
              type="button"
              className="nav-item logout-button"
              onClick={handleLogout}
              disabled={loggingOut}
            >
              {loggingOut ? 'Signing out…' : '↪ Sign out'}
            </button>
          )}

          {authError && (
            <p className="auth-error" role="alert">
              {authError}
            </p>
          )}
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Patient Space <span>/</span> {activePage}
          </div>

          <div className="topbar-right">
            <span className="demo-badge">
              <span /> {isGuest ? 'GUEST SESSION' : 'RESEARCH PROTOTYPE'}
            </span>
            <div className="top-avatar">{avatarLetter}</div>
          </div>
        </header>

        <div className="page-content">
          {isGuest && (
            <div className="guest-banner" role="status">
              <div>
                <strong>You are exploring as a guest.</strong>
                <p>
                  Your submitted assessment is calculated by the research prototype. Guest results stay in this browser session and are not saved to an account.
                </p>
              </div>

              <button type="button" onClick={handleGuestExit}>
                Exit Guest Mode
              </button>
            </div>
          )}

          <div className="welcome-row">
            <div>
              <p className="eyebrow">YOUR PERSONAL HEALTH SPACE</p>

              <h1>
                {activePage === 'Overview'
                  ? `Good day, ${displayName}`
                  : activePage}
              </h1>

              <p className="subtitle">
                {activePage === 'Overview'
                  ? 'A little check-in today can help you notice changes over time.'
                  : `Explore your ${activePage.toLowerCase()} information in this demonstration.`}
              </p>
            </div>

            <button
              type="button"
              className="primary-button"
              onClick={startAssessment}
            >
              <span>＋</span> Record symptoms
            </button>
          </div>

          <div className="notice">
            <span className="notice-icon">ⓘ</span>
            <p>
              <strong>Research prototype.</strong> Only values from assessments you submit are shown here. Scores are experimental and are not clinically validated flare probabilities.
            </p>
          </div>

          {activePage === 'Overview' && (
            <section className="overview-dashboard" aria-label="Health overview">
              <div className="overview-metrics">
                <article className="overview-metric-card">
                  <span className="overview-metric-icon blue" aria-hidden="true">◉</span>
                  <span className="overview-metric-label">LATEST PROJECT SUMMARY</span>
                  <strong>
                    {latestAssessment
                      ? latestAssessment.assessment.rapid3.toFixed(1)
                      : historyItems[0]?.rapid3 == null
                        ? '—'
                        : historyItems[0].rapid3.toFixed(1)}
                  </strong>
                  <small>
                    {latestAssessment
                      ? 'Project-specific summary; not a disease-activity measure'
                      : historyItems[0]
                        ? 'From your latest saved assessment'
                        : 'Complete a check-in to see this value'}
                  </small>
                </article>

                <article className="overview-metric-card">
                  <span className="overview-metric-icon coral" aria-hidden="true">♡</span>
                  <span className="overview-metric-label">PAIN REPORTED</span>
                  {/* FIX: <small> is now real JSX, not text inside a string */}
                  <strong>
                    {latestAssessment ? (
                      <>
                        {latestAssessment.assessment.pain}
                        <small>/10</small>
                      </>
                    ) : (
                      '—'
                    )}
                  </strong>
                  <small>{latestAssessment ? 'In your latest check-in' : 'Record a check-in to see this value'}</small>
                </article>

                <article className="overview-metric-card">
                  <span className="overview-metric-icon violet" aria-hidden="true">↗</span>
                  <span className="overview-metric-label">FUNCTION SCORE</span>
                  <strong>{latestAssessment ? latestAssessment.assessment.function_score.toFixed(1) : '—'}</strong>
                  <small>{latestAssessment ? 'Based on your activity answers' : 'From your latest check-in'}</small>
                </article>

                <article className="overview-metric-card">
                  <span className="overview-metric-icon mint" aria-hidden="true">⌁</span>
                  <span className="overview-metric-label">EXPERIMENTAL SCORE · POINTS</span>
                  <strong>
                    {latestAssessment
                      ? scorePoints(latestAssessment.assessment.experimental_score)
                      : '—'}
                  </strong>
                  <small>Rule-based score, not a flare probability</small>
                </article>
              </div>

              <div className="overview-content-grid">
                <section className="overview-panel overview-trend-panel">
                  <div className="overview-panel-heading">
                    <div>
                      <p className="eyebrow">YOUR PROGRESS</p>
                      <h2>Recent check-ins</h2>
                      <p>
                        {isGuest
                          ? 'Guest assessments are not saved to your check-in history'
                          : 'Project-specific patient-reported summaries from saved check-ins'}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="overview-link-button"
                      onClick={() => navigateTo('My Symptoms')}
                    >
                      View history <span aria-hidden="true">→</span>
                    </button>
                  </div>

                  {historyItems.filter((item) => item.rapid3 != null).length > 1 ? (
                    <div className="overview-chart-empty" role="status">
                      <strong>Historical check-ins are available</strong>
                      <p>
                        A question-set version is not stored with older summaries,
                        so this prototype does not plot a score trend. Changes in
                        reported symptoms alone do not confirm a clinical flare.
                      </p>
                    </div>
                  ) : (
                    <div className="overview-chart-empty">
                      <span aria-hidden="true">⌁</span>
                      <strong>Your progress will appear here</strong>
                      <p>
                        {isGuest
                          ? 'Guest check-ins are temporary. Sign in to track your progress over time.'
                          : 'Complete at least two saved check-ins to see your score history.'}
                      </p>
                      <button type="button" className="overview-link-button" onClick={startAssessment}>
                        Start a check-in <span aria-hidden="true">→</span>
                      </button>
                    </div>
                  )}
                  <p className="overview-chart-note">These project-specific summaries are not validated clinical measures or clinician-assessed disease activity.</p>
                </section>

                <section className="overview-panel overview-checkin-panel">
                  <div className="overview-checkin-art" aria-hidden="true">
                    <span className="overview-art-orbit orbit-one" />
                    <span className="overview-art-orbit orbit-two" />
                    <span className="overview-art-heart">♡</span>
                    <span className="overview-art-spark spark-one">✦</span>
                    <span className="overview-art-spark spark-two">✦</span>
                  </div>
                  <p className="eyebrow">A MOMENT FOR YOU</p>
                  <h2>How are you feeling today?</h2>
                  <p className="overview-checkin-copy">A quick check-in helps you reflect on symptoms and daily activities over time.</p>
                  <button type="button" className="primary-button" onClick={startAssessment}>
                    Start check-in <span aria-hidden="true">→</span>
                  </button>
                  {latestAssessment && (
                    <button type="button" className="overview-results-link" onClick={() => navigateTo('Results')}>
                      Review your latest results
                    </button>
                  )}
                </section>
              </div>

              <section className="overview-panel overview-recent-panel">
                <div className="overview-panel-heading">
                  <div>
                    <p className="eyebrow">YOUR ACTIVITY</p>
                    <h2>Recent assessments</h2>
                  </div>
                  <button type="button" className="overview-link-button" onClick={() => navigateTo('My Symptoms')}>
                    See all <span aria-hidden="true">→</span>
                  </button>
                </div>
                {isGuest ? (
                  latestAssessment ? (
                    <article className="overview-activity-row">
                      <span className="overview-activity-icon" aria-hidden="true">✓</span>
                      <div><strong>Check-in completed</strong><small>{new Date(latestAssessment.assessment.created_at).toLocaleString()}</small></div>
                      <span className="overview-activity-value">Project summary {latestAssessment.assessment.rapid3.toFixed(1)}/30</span>
                    </article>
                  ) : (
                    <p className="overview-activity-empty">Your check-ins will appear here after you complete an assessment.</p>
                  )
                ) : historyLoading ? (
                  <p className="overview-activity-empty">Loading your saved assessments…</p>
                ) : historyError ? (
                  <p className="overview-activity-empty" role="alert">Could not load your saved assessments: {historyError}</p>
                ) : historyItems.length > 0 ? (
                  historyItems.slice(0, 3).map((item, index) => (
                    <article className="overview-activity-row" key={`${item.created_at}-${index}`}>
                      <span className="overview-activity-icon" aria-hidden="true">✓</span>
                      <div><strong>Check-in completed</strong><small>{new Date(item.created_at).toLocaleString()}{item.city ? ` · ${item.city}` : ''}</small></div>
                      <span className="overview-activity-value">{item.rapid3 == null ? 'Summary unavailable' : `Project summary ${item.rapid3.toFixed(1)}/30`}</span>
                    </article>
                  ))
                ) : (
                  <p className="overview-activity-empty">No saved assessments yet. Complete your first check-in to get started.</p>
                )}
              </section>
            </section>
          )}

          {activePage === 'Results' && (
            <section className="panel secondary-page results-page" aria-labelledby="assessment-results-title">
              {!latestAssessment ? (
                <div className="results-empty">
                  <span className="results-empty-icon" aria-hidden="true">✓</span>
                  <h2 id="assessment-results-title">Your assessment summary</h2>
                  <p>Complete an assessment to see a summary based on your answers.</p>
                  <button type="button" className="primary-button" onClick={startAssessment}>
                    Start assessment <span aria-hidden="true">→</span>
                  </button>
                </div>
              ) : (
                <>
                  <div className="results-heading">
                    <div>
                      <p className="eyebrow">YOUR LATEST CHECK-IN</p>
                      <h2 id="assessment-results-title">Assessment summary</h2>
                      <p className="results-date">{new Date(latestAssessment.assessment.created_at).toLocaleString()}</p>
                    </div>
                    <span className="results-session-label">{isGuest ? 'Guest result' : 'Saved assessment'}</span>
                  </div>

                  <article className="risk-result-card">
                    <div className="risk-result-topline">
                      <span className="risk-result-label">EXPERIMENTAL RULE-BASED SCORE</span>
                    </div>
                    <h3 className="risk-result-value">
                      {scorePoints(resultScore)}
                    </h3>
                    <p className="risk-result-explainer">
                      These are score points from the current rule-based calculation, not a percentage chance or prediction of an RA flare. It combines selected symptom answers with temperature and humidity inputs; trained model predictions are not used.
                    </p>
                  </article>

                  <div className="result-metrics-grid">
                    <article className="result-metric-card">
                      <span className="result-metric-icon" aria-hidden="true">◉</span>
                      <span className="result-metric-label">Project symptom &amp; function summary</span>
                      <strong>{Number.isFinite(latestAssessment.assessment.rapid3) ? latestAssessment.assessment.rapid3.toFixed(1) : 'Not available'}</strong>
                      <span className="result-metric-note">0–30 project scale; not a validated RAPID3 score</span>
                    </article>
                    <article className="result-metric-card">
                      <span className="result-metric-icon" aria-hidden="true">↗</span>
                      <span className="result-metric-label">Daily-function responses</span>
                      <strong>{Number.isFinite(latestAssessment.assessment.function_score) ? latestAssessment.assessment.function_score.toFixed(1) : 'Not available'}</strong>
                      <span className="result-metric-note">Project scale; not a validated function index</span>
                    </article>
                    <article className="result-metric-card">
                      <span className="result-metric-icon" aria-hidden="true">♡</span>
                      <span className="result-metric-label">Pain reported</span>
                      <strong>{Number.isFinite(latestAssessment.assessment.pain) ? `${latestAssessment.assessment.pain}/10` : 'Not available'}</strong>
                      <span className="result-metric-note">Your answer</span>
                    </article>
                  </div>

                  <section className="patient-meaning-section" aria-labelledby="assessment-details-title">
                    <div className="patient-section-heading">
                      <span className="patient-section-icon" aria-hidden="true">♡</span>
                      <div>
                        <p className="eyebrow">IN PLAIN LANGUAGE</p>
                        <h3 id="assessment-details-title">What your answers tell you</h3>
                      </div>
                    </div>
                    <p>
                      This summary reflects only the information returned for this check-in. It does not establish whether your RA is active or in remission.
                    </p>
                    <div className="patient-observation-grid">
                      <div>
                        <strong>{Number.isFinite(latestAssessment.assessment.pain) ? `${latestAssessment.assessment.pain}/10` : 'Not available'}</strong>
                        <span>Pain you reported</span>
                      </div>
                      <div>
                        <strong>{Number.isFinite(latestAssessment.assessment.global_estimate) ? `${latestAssessment.assessment.global_estimate}/10` : 'Not available'}</strong>
                        <span>Overall health rating you reported</span>
                      </div>
                    </div>
                  </section>

                  <section className="result-details-section" aria-labelledby="result-context-title">
                    <div className="patient-section-heading">
                      <span className="patient-section-icon" aria-hidden="true">▤</span>
                      <div>
                        <p className="eyebrow">ASSESSMENT CONTEXT</p>
                        <h3 id="result-context-title">Information available for this check-in</h3>
                      </div>
                    </div>
                    <div className="result-details-grid">
                      <article>
                        <h4>Symptoms and function</h4>
                        <dl>
                          <div><dt>Change from your usual symptoms</dt><dd>{reportedInputs ? ({ better: 'Better', same: 'About the same', worse: 'Worse' }[reportedInputs.symptom_change] ?? 'Not returned') : 'Not returned'}</dd></div>
                          <div><dt>Fatigue</dt><dd>{reportedInputs ? `${reportedInputs.fatigue}/10` : 'Not returned'}</dd></div>
                          <div><dt>Recent worsening episodes reported</dt><dd>{reportedInputs?.flares_30d ?? 'Not returned'}</dd></div>
                          <div><dt>Physical activity today</dt><dd>{reportedInputs ? `${reportedInputs.activity_min} minutes` : 'Not returned'}</dd></div>
                          <div><dt>Project symptom and function summary</dt><dd>{Number.isFinite(latestAssessment.assessment.rapid3) ? `${latestAssessment.assessment.rapid3.toFixed(1)} / 30` : 'Not available'}</dd></div>
                        </dl>
                        <p>Change from usual symptoms and physical activity are recorded for context, but are not used by the current heuristic.</p>
                      </article>
                      <article>
                        <h4>Sleep and stress</h4>
                        <dl>
                          <div><dt>Sleep last night</dt><dd>{reportedInputs && sleepLabels[reportedInputs.sleep_difficulty] ? sleepLabels[reportedInputs.sleep_difficulty] : 'Not returned'}</dd></div>
                          <div><dt>Stress or anxiety reported</dt><dd>{reportedInputs && anxietyLabels[reportedInputs.anxiety] ? anxietyLabels[reportedInputs.anxiety] : 'Not returned'}</dd></div>
                        </dl>
                        <p>Sleep and fatigue are used by the heuristic; stress or anxiety is recorded for context but is not used in the score.</p>
                      </article>
                      <article>
                        <h4>HRV information</h4>
                        <p>No HRV measurement was collected for this check-in. HRV model predictions are not used in the score.</p>
                        <p>Optional pulse: {reportedInputs?.heart_rate_bpm == null ? 'Not provided' : `${reportedInputs.heart_rate_bpm} bpm`}. Pulse is not an HRV measurement and is not used by the current heuristic.</p>
                      </article>
                      <article>
                        <h4>Environmental context</h4>
                        {latestAssessment.weather ? (
                          <>
                            <p>{latestAssessment.weather.description || 'Weather description unavailable'} for {latestAssessment.weather.city || latestAssessment.assessment.city || 'your selected city'}.</p>
                            <p>Temperature: {Number.isFinite(latestAssessment.weather.temperature) ? `${latestAssessment.weather.temperature} °C` : 'Not available'}; humidity: {Number.isFinite(latestAssessment.weather.humidity) ? `${latestAssessment.weather.humidity}%` : 'Not available'}; pressure: {Number.isFinite(latestAssessment.weather.pressure) ? `${latestAssessment.weather.pressure} hPa` : 'Not available'}.</p>
                            <p>{latestAssessment.weather.is_live ? 'Live weather data' : 'Estimated weather fallback; live data was unavailable'}.</p>
                            <p>Temperature and humidity are used by the heuristic; pressure is shown for context only.</p>
                          </>
                        ) : (
                          <p>Environmental information was not returned for this check-in.</p>
                        )}
                      </article>
                    </div>
                  </section>

                  <section className="result-details-section" aria-labelledby="result-factors-title">
                    <div className="patient-section-heading">
                      <span className="patient-section-icon" aria-hidden="true">i</span>
                      <div>
                        <p className="eyebrow">HOW THE SCORE IS BUILT</p>
                        <h3 id="result-factors-title">Factors in the current calculation</h3>
                      </div>
                    </div>
                    <p>These are the additive terms returned by the heuristic, shown as score points before its final rule adjustments. They are not probabilities, model outputs, or clinical risk estimates.</p>
                    {resultContributions.length > 0 ? (
                      <ul className="result-factor-list">
                        {resultContributions.map(([name, value]) => (
                          <li key={name}>
                            <span>{contributionLabels[name] ?? name}</span>
                            <strong>{(value * 100).toFixed(1)} score points</strong>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p role="status">Factor details are not available for this check-in.</p>
                    )}
                  </section>

                  <section className="patient-next-steps" aria-labelledby="monitoring-plan-title">
                    <div className="patient-section-heading">
                      <span className="patient-section-icon" aria-hidden="true">✓</span>
                      <div>
                        <p className="eyebrow">NOT A FORECAST</p>
                        <h3 id="monitoring-plan-title">Next 3 Days: Symptom Monitoring Plan</h3>
                      </div>
                    </div>
                    <ul>
                      <li><strong>Day 1:</strong> Note your pain, fatigue, sleep, and daily function in your own words.</li>
                      <li><strong>Day 2:</strong> Record the same symptoms and note any changes compared with yesterday.</li>
                      <li><strong>Day 3:</strong> Record symptoms again and share persistent or concerning changes with your healthcare professional.</li>
                    </ul>
                    <p>Continue your existing treatment plan. Contact your rheumatology team or healthcare professional if symptoms worsen, persist, or concern you. Do not change medication based on this page.</p>
                    <p>This is a reminder to monitor and record symptoms. It does not predict whether a flare will occur.</p>
                  </section>

                  <aside className="result-safety-notice" role="note">
                    <strong>Research prototype — not for medical decisions.</strong>
                    <p>This is not a diagnosis and does not provide a clinically validated prediction. Do not start, stop, or change medication based on this score. Contact your healthcare professional if symptoms worsen or cause concern.</p>
                  </aside>

                  <section className="result-details-section" aria-labelledby="clinical-measures-title">
                    <div className="patient-section-heading">
                      <span className="patient-section-icon" aria-hidden="true">+</span>
                      <div>
                        <p className="eyebrow">CLINICAL CONTEXT</p>
                        <h3 id="clinical-measures-title">Patient-reported symptoms are not a clinical disease-activity assessment</h3>
                      </div>
                    </div>
                    <p>
                      This check-in records pain, fatigue, patient global health, and selected daily-function and sleep responses. It does not collect morning stiffness, clinician-counted tender or swollen joints, or ESR/CRP inflammatory markers. These missing clinical measures are not inferred.
                    </p>
                    <p>
                      The 0–30 summary above uses this prototype’s selected questions and arithmetic; it is not a validated RAPID3 instrument. This page does not calculate DAS28, CDAI, or SDAI. Symptom reports and this heuristic cannot diagnose a flare, remission, or active RA.
                    </p>
                    <p>
                      Clinical guidelines describe disease-activity monitoring and treatment targets as part of care with a healthcare professional. The references below provide clinical context; they do not validate this prototype’s score.
                    </p>
                    <ul className="clinical-reference-list">
                      <li>
                        <a href="https://www.nice.org.uk/guidance/ng100/chapter/recommendations" target="_blank" rel="noreferrer">
                          NICE NG100: Rheumatoid arthritis in adults—management
                        </a>
                        {' '}— includes disease-activity monitoring, flare management, and rapid access to specialist care if the condition suddenly worsens.
                      </li>
                      <li>
                        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9273041/" target="_blank" rel="noreferrer">
                          ACR 2021 Guideline for the Treatment of Rheumatoid Arthritis
                        </a>
                        {' '}— Arthritis Care &amp; Research 73(7):924–939; defines treat-to-target using frequent monitoring and validated instruments toward low disease activity or remission.
                      </li>
                      <li>
                        <a href="https://ard.bmj.com/content/80/10/1278" target="_blank" rel="noreferrer">
                          2021 EULAR recommendations for self-management in inflammatory arthritis
                        </a>
                        {' '}— Annals of the Rheumatic Diseases 80(10):1278–1285; supports patient education and self-management resources as part of routine care.
                      </li>
                    </ul>
                  </section>

                  <details className="interpretation-details">
                    <summary>Technical details and important limitations</summary>
                    <ul>
                      <li>The displayed score rescales the current rule-based heuristic to 0–100 score points. It is not a percentage chance.</li>
                      <li>The calculation uses patient-reported pain, function, overall health, recent worsening episodes, sleep difficulty, fatigue, humidity, and temperature. Threshold adjustments can change the final value from the displayed factor sum. Stress, activity, symptom-change response, optional pulse, and weather pressure are not used.</li>
                      <li>Trained modality model predictions are not combined, and a trained fusion model is unavailable.</li>
                      <li>No validated three-day RA flare forecast is available. The monitoring plan above is not a forecast.</li>
                      <li>Weather may be an estimate when live data is unavailable; optional pulse is not used by the current heuristic.</li>
                    </ul>
                  </details>

                  {isGuest && (
                    <p className="guest-result-note">Guest result only: this assessment was not saved to an account history.</p>
                  )}

                  <div className="results-actions">
                    <button type="button" className="primary-button" onClick={startAssessment}>
                      Update assessment <span aria-hidden="true">→</span>
                    </button>
                    <button type="button" className="secondary-button" onClick={() => navigateTo('My Symptoms')}>
                      View symptom history
                    </button>
                  </div>
                </>
              )}
            </section>
          )}

          {activePage === 'My Symptoms' && (
            <section className="panel secondary-page">
              <h2>Symptom history</h2>
              {isGuest ? (
                <>
                  <p>Guest assessments are temporary and are not saved as account history.</p>
                  {latestAssessment ? (
                    <div className="history-row">
                      <strong>{new Date(latestAssessment.assessment.created_at).toLocaleString()}</strong>
                      <span>Pain: {latestAssessment.assessment.pain}/10</span>
                      <span>Project-specific symptom and function summary: {latestAssessment.assessment.rapid3.toFixed(1)} / 30 (not a validated instrument)</span>
                    </div>
                  ) : (
                    <div className="empty-state">Complete an assessment to see its result here.</div>
                  )}
                </>
              ) : (
                <>
                  <p>Your saved assessments, newest first. Earlier question-set versions are not recorded, so this history does not compare changes in disease activity.</p>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void loadAssessmentHistory()}
                    disabled={historyLoading}
                  >
                    {historyLoading ? 'Refreshing history…' : 'Refresh history'}
                  </button>

                  {historyError && (
                    <div className="empty-state" role="alert">
                      Could not load saved history: {historyError}
                    </div>
                  )}

                  {historyLoading ? (
                    <p>Loading your saved assessments…</p>
                  ) : historyItems.length > 0 ? (
                    <div className="history-list">
                      {historyItems.map((item, index) => (
                        <article
                          className="history-row"
                          key={`${item.created_at}-${index}`}
                        >
                          <strong>{new Date(item.created_at).toLocaleString()}</strong>
                          <span>City: {item.city || 'Not recorded'}</span>
                          <span>
                            Project-specific symptom and function summary: {item.rapid3 == null ? 'Not available' : `${item.rapid3.toFixed(1)} / 30`} (not a validated instrument)
                          </span>
                          <span>
                            Experimental score: Not available in historical records (they store a legacy peak value).
                          </span>
                        </article>
                      ))}
                    </div>
                  ) : !historyError ? (
                    <div className="empty-state">
                      No saved assessments found for this account. Complete an assessment to begin your history.
                    </div>
                  ) : null}
                  <p className="chart-note">
                    Experimental scores are prototype outputs, not clinically validated probabilities or medical advice.
                  </p>
                </>
              )}
            </section>
          )}

          {activePage === 'Assessment' && (
            <AssessmentPage
              isGuest={isGuest}
              onSubmissionStart={() => {
                setLatestAssessment(null)
                return ++assessmentRequestVersion.current
              }}
              isSubmissionCurrent={(requestVersion) =>
                requestVersion === assessmentRequestVersion.current
              }
              onResult={(requestVersion, result) => {
                if (requestVersion !== assessmentRequestVersion.current) {
                  return
                }
                setLatestAssessment(result)
                if (!isGuest) {
                  void loadAssessmentHistory()
                }
                setActivePage('Results')
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }}
            />
          )}

          {activePage === 'Forecast History' && (
            <section className="panel secondary-page">
              <span className="stat-icon lavender">▤</span>
              <h2>Forecast availability</h2>
              <p>
                FlareGuard does not currently provide a clinically validated
                72-hour RA flare forecast. Use the symptom monitoring plan on
                your assessment summary to record changes; it is not a forecast.
              </p>
              <div className="empty-state">
                No validated forecast results are available.
              </div>
            </section>
          )}

          <footer className="footer">
            <span>
              FlareGuard <span className="footer-dot">•</span> Research
              prototype
            </span>
            <span>
              For research and demonstration only · Not medical advice
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}


function AssessmentPage({
  isGuest,
  onSubmissionStart,
  isSubmissionCurrent,
  onResult,
}: {
  isGuest: boolean
  onSubmissionStart: () => number
  isSubmissionCurrent: (requestVersion: number) => boolean
  onResult: (requestVersion: number, result: AssessmentResult) => void
}) {
  const [step, setStep] = useState(0)
  const [age, setAge] = useState('')
  const [city, setCity] = useState('Colombo')
  const [pain, setPain] = useState('0')
  const [globalEstimate, setGlobalEstimate] = useState('0')
  const [fatigue, setFatigue] = useState('0')
  const [flares30d, setFlares30d] = useState('0')
  const [symptomChange, setSymptomChange] = useState('same')
  const [durationYears, setDurationYears] = useState('1')
  const [sleepDiff, setSleepDiff] = useState('1')
  const [anxiety, setAnxiety] = useState('1')
  const [activityMin, setActivityMin] = useState('30')
  const [heartRate, setHeartRate] = useState('')
  const functionQuestions = [
    ['A', 'Getting dressed and ready'],
    ['B', 'Getting up from a chair or bed'],
    ['C', 'Eating or using everyday utensils'],
    ['D', 'Walking on level ground'],
    ['E', 'Washing and personal care'],
    ['F', 'Reaching for things above your head'],
    ['G', 'Opening jars or gripping objects'],
    ['H', 'Doing usual household activities'],
    ['I', 'Climbing steps or stairs'],
    ['J', 'Shopping or other daily errands'],
  ]
  const [fnVals, setFnVals] = useState<Record<string, number>>({ A: 0, B: 0, C: 0, D: 0, E: 0, F: 0, G: 0, H: 0, I: 0, J: 0 })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const stepTitles = ['About you', 'Symptoms today', 'Daily activities', 'Recent patterns', 'Review & result']
  const progress = ((step + 1) / stepTitles.length) * 100

  function validateStep() {
    if (step === 0) {
      const ageNumber = Number(age)
      if (!age.trim() || !Number.isInteger(ageNumber) || ageNumber < 1 || ageNumber > 120) {
        setError('Please enter your age in years (1–120).')
        return false
      }
      if (!city.trim()) {
        setError('Please enter your city.')
        return false
      }
    }
    setError('')
    return true
  }

  function nextStep() {
    if (!validateStep()) return
    setStep(current => Math.min(current + 1, stepTitles.length - 1))
    setError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function previousStep() {
    setError('')
    setStep(current => Math.max(current - 1, 0))
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!validateStep()) return
    setBusy(true)
    setError('')
    const requestVersion = onSubmissionStart()

    const data: AssessmentData = {
      age: Number(age),
      symptom_change: symptomChange,
      city: city.trim(),
      fn_vals: fnVals,
      pain: Number(pain),
      global_estimate: Number(globalEstimate),
      fatigue: Number(fatigue),
      flares_30d: Number(flares30d),
      duration_years: Number(durationYears),
      sleep_diff: Number(sleepDiff),
      anxiety: Number(anxiety),
      activity_min: Number(activityMin),
      heart_rate: Number(heartRate) || 0,
    }

    try {
      const response = isGuest
        ? await submitGuestAssessment(data)
        : await sendAssessment(data)
      if (!isSubmissionCurrent(requestVersion)) {
        return
      }
      if (response.success !== true || !response.assessment) {
        throw new Error('The API did not return a completed assessment. Please try again.')
      }
      onResult(requestVersion, response)
    } catch (err) {
      if (isSubmissionCurrent(requestVersion)) {
        setError(err instanceof Error ? err.message : 'Unable to calculate this assessment. Please try again.')
      }
    } finally {
      setBusy(false)
    }
  }

  const scale = (label: string, value: string, setter: (value: string) => void, low: string, high: string) => (
    <label className="guided-question">{label}<strong>{value}/10</strong>
      <input type="range" min="0" max="10" step="1" value={value} onChange={e => setter(e.target.value)} />
      <span className="range-hints"><span>{low}</span><span>{high}</span></span>
    </label>
  )

  return (
    <section className="panel assessment-panel guided-assessment">
      <div className="assessment-heading">
        <span className="eyebrow">{isGuest ? 'GUEST ASSESSMENT' : 'PATIENT ASSESSMENT'}</span>
        <h2>A quick check-in about your health</h2>
        <p>Answer at your own pace. There are five short steps, and you can go back to change your answers.</p>
      </div>
      {isGuest && <div className="guest-banner" role="status"><div><strong>Guest mode — no account required</strong><p>Your result is temporary and is not saved to an account.</p></div></div>}

      <div className="guided-progress" aria-label={`Step ${step + 1} of ${stepTitles.length}`}>
        <div className="guided-progress-top"><span>Step {step + 1} of {stepTitles.length}</span><strong>{stepTitles[step]}</strong></div>
        <div className="guided-progress-track"><div className="guided-progress-fill" style={{ width: `${progress}%` }} /></div>
        <div className="guided-step-dots">{stepTitles.map((title, index) => <span key={title} className={index <= step ? 'complete' : ''} aria-label={title}>{index + 1}</span>)}</div>
      </div>

      <form onSubmit={handleSubmit} aria-busy={busy}>
        {step === 0 && <div className="guided-step-content">
          <div className="guided-intro"><span className="guided-emoji">👋</span><div><h3>Let's start with a little about you</h3><p>This information helps describe the people taking part in the research.</p></div></div>
          <label className="guided-question">How old are you? <span className="required-note">Required</span>
            <input type="number" min="1" max="120" step="1" required value={age} onChange={e => setAge(e.target.value)} placeholder="Enter age in years" />
            <small>Please enter your age, not your date of birth.</small>
          </label>
          <label className="guided-question">Which city are you in? <span className="required-note">Required</span>
            <input type="text" required maxLength={100} value={city} onChange={e => setCity(e.target.value)} placeholder="e.g. Colombo" />
            <small>Used to retrieve local environmental information when available.</small>
          </label>
          <label className="guided-question">How long have you had RA or related symptoms?
            <select value={durationYears} onChange={e => setDurationYears(e.target.value)}><option value="0">Less than 1 year / not sure</option><option value="1">1–2 years</option><option value="3">3–5 years</option><option value="6">6–10 years</option><option value="11">More than 10 years</option></select>
          </label>
        </div>}

        {step === 1 && <div className="guided-step-content">
          <div className="guided-intro"><span className="guided-emoji">💬</span><div><h3>How are you feeling today?</h3><p>Choose the number that best matches how you feel right now.</p></div></div>
          {scale('How much joint pain do you have?', pain, setPain, 'No pain', 'Worst pain')}
          {scale('How would you rate your overall health?', globalEstimate, setGlobalEstimate, 'Very good', 'Very poor')}
          {scale('How tired or exhausted do you feel?', fatigue, setFatigue, 'Not tired', 'Extremely tired')}
          <label className="guided-question">Compared with your usual symptoms, how do you feel today?
            <div className="choice-grid">{[['better','🙂','Better'],['same','😐','About the same'],['worse','🙁','Worse']].map(([value, emoji, label]) => <button key={value} type="button" className={`choice-card ${symptomChange === value ? 'selected' : ''}`} onClick={() => setSymptomChange(value)}><span>{emoji}</span>{label}</button>)}</div>
          </label>
          <label className="guided-question">How many times have your symptoms flared or worsened in the past 30 days?
            <input type="number" min="0" max="30" step="1" value={flares30d} onChange={e => setFlares30d(e.target.value)} required />
            <small>The choice above and this count are separate details; enter a count if you remember one.</small>
          </label>
        </div>}

        {step === 2 && <div className="guided-step-content">
          <div className="guided-intro"><span className="guided-emoji">🧩</span><div><h3>How are everyday activities going?</h3><p>Think about the past week. Choose the option that fits you best for each activity.</p></div></div>
          <p className="research-note">These are brief research questions about daily function. They are not presented as a validated HAQ-DI questionnaire.</p>
          <div className="function-rating-fields guided-function-fields">
            {functionQuestions.map(([key, question]) => <label className="guided-question" key={key}>{question}
              <select value={fnVals[key]} onChange={e => setFnVals(current => ({ ...current, [key]: Number(e.target.value) }))}>
                <option value={0}>No difficulty</option><option value={1}>Some difficulty</option><option value={2}>A lot of difficulty</option><option value={3}>Unable to do</option>
              </select>
            </label>)}
          </div>
        </div>}

        {step === 3 && <div className="guided-step-content">
          <div className="guided-intro"><span className="guided-emoji">🌙</span><div><h3>A few things about your recent routine</h3><p>These extra questions explore possible patterns. It's okay if you don't know your heart rate.</p></div></div>
          <label className="guided-question">How well did you sleep last night?
            <select value={sleepDiff} onChange={e => setSleepDiff(e.target.value)}><option value="0">Very well</option><option value="1">Okay</option><option value="2">Poorly</option><option value="3">Very poorly</option></select>
          </label>
          <label className="guided-question">How stressed or anxious have you felt recently?
            <select value={anxiety} onChange={e => setAnxiety(e.target.value)}><option value="0">Not at all</option><option value="1">A little</option><option value="2">Quite a bit</option><option value="3">Extremely</option></select>
          </label>
          <label className="guided-question">About how many minutes of physical activity did you do today?
            <input type="number" min="0" max="1440" step="1" required value={activityMin} onChange={e => setActivityMin(e.target.value)} />
            <small>Include walking, exercise, or other movement. Enter 0 if none.</small>
          </label>
          <label className="guided-question">Heart rate (beats per minute) <span className="optional-note">Optional</span>
            <input type="number" min="30" max="220" step="1" value={heartRate} onChange={e => setHeartRate(e.target.value)} placeholder="Leave blank if unknown" />
            <small>Only enter a measured value if you know it. FlareGuard does not measure your heart rate.</small>
          </label>
        </div>}

        {step === 4 && <div className="guided-step-content">
          <div className="guided-intro"><span className="guided-emoji">✅</span><div><h3>Review your answers</h3><p>Check your answers before calculating your result. You can go back to edit any step.</p></div></div>
          <div className="review-list">
            <div><span>Age</span><strong>{age} years</strong><button type="button" onClick={() => setStep(0)}>Edit</button></div>
            <div><span>City</span><strong>{city}</strong><button type="button" onClick={() => setStep(0)}>Edit</button></div>
            <div><span>Joint pain</span><strong>{pain}/10</strong><button type="button" onClick={() => setStep(1)}>Edit</button></div>
            <div><span>Overall health</span><strong>{globalEstimate}/10</strong><button type="button" onClick={() => setStep(1)}>Edit</button></div>
            <div><span>Fatigue</span><strong>{fatigue}/10</strong><button type="button" onClick={() => setStep(1)}>Edit</button></div>
            <div><span>Daily activity responses</span><strong>{Object.keys(fnVals).length} answered</strong><button type="button" onClick={() => setStep(2)}>Edit</button></div>
            <div><span>Sleep and stress</span><strong>{['Very well','Okay','Poorly','Very poorly'][Number(sleepDiff)]}; {['Not at all','A little','Quite a bit','Extremely'][Number(anxiety)]} stress</strong><button type="button" onClick={() => setStep(3)}>Edit</button></div>
            <div><span>Physical activity</span><strong>{activityMin} minutes</strong><button type="button" onClick={() => setStep(3)}>Edit</button></div>
          </div>
          <p className="research-note">Your age is collected for research context. It is not currently used by the risk calculation. The score is experimental and is not a diagnosis or a clinically validated flare probability.</p>
        </div>}

        {error && <div className="auth-error" role="alert"><strong>Please check:</strong> {error}</div>}
        {busy && <p role="status" aria-live="polite">Calculating your assessment. Please wait.</p>}
        <div className="guided-actions">
          {step > 0 && <button className="secondary-button" type="button" onClick={previousStep} disabled={busy}>← Back</button>}
          {step < stepTitles.length - 1
            ? <button className="primary-button" type="button" onClick={nextStep}>Continue →</button>
            : <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Calculating your result…' : 'Calculate my assessment'}</button>}
        </div>
      </form>

    </section>
  )
}

export default App