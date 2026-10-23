
import { useCallback, useEffect, useState, type FormEvent } from 'react'
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

type Page = 'Overview' | 'My Symptoms' | 'Assessment' | 'Forecast History'
type AuthMode = 'login' | 'register'

const navigation: { label: Page; icon: string }[] = [
  { label: 'Overview', icon: '⌂' },
  { label: 'My Symptoms', icon: '◷' },
  { label: 'Assessment', icon: '✓' },
  { label: 'Forecast History', icon: '▤' },
]


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

  const loadAssessmentHistory = useCallback(async () => {
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
      setHistoryItems(response.assessments)
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : 'Unable to load assessment history.',
      )
    } finally {
      setHistoryLoading(false)
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

    getAssessmentHistory()
      .then((response) => {
        if (!cancelled) {
          setHistoryItems(response.assessments)
          setHistoryError('')
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setHistoryError(
            err instanceof Error
              ? err.message
              : 'Unable to load assessment history.',
          )
        }
      })
      .finally(() => {
        if (!cancelled) {
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
          setUser(authenticatedUser)
          setIsGuest(false)
        }}
        onGuest={() => {
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault()
            setActivePage('Overview')
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
              onClick={() => setActivePage(item.label)}
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
              onClick={() => setActivePage('Assessment')}
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
            <section className="panel secondary-page">
              <h2>Your assessment overview</h2>
              {!latestAssessment ? (
                <div className="empty-state">
                  <h3>No assessment yet</h3>
                  <p>Complete an assessment to see the result calculated from your answers. No sample health values are displayed.</p>
                  <button type="button" className="primary-button" onClick={() => setActivePage('Assessment')}>
                    Start assessment →
                  </button>
                </div>
              ) : (
                <>
                  <p>Latest assessment · {new Date(latestAssessment.assessment.created_at).toLocaleString()}</p>
                  <div className="stats-grid">
                    <article className="stat-card">
                      <div className="stat-label">RAPID3 SCORE</div>
                      <div className="stat-value">{latestAssessment.assessment.rapid3.toFixed(1)}</div>
                      <p>{latestAssessment.assessment.rapid3_category}</p>
                    </article>
                    <article className="stat-card">
                      <div className="stat-label">EXPERIMENTAL SCORE</div>
                      <div className="stat-value">{(latestAssessment.assessment.experimental_score * 100).toFixed(1)}<span>%</span></div>
                      <p>{latestAssessment.assessment.risk_label}</p>
                    </article>
                    <article className="stat-card">
                      <div className="stat-label">PAIN LEVEL</div>
                      <div className="stat-value">{latestAssessment.assessment.pain}<span>/10</span></div>
                      <p>Reported by you</p>
                    </article>
                    <article className="stat-card">
                      <div className="stat-label">FUNCTION SCORE</div>
                      <div className="stat-value">{latestAssessment.assessment.function_score.toFixed(1)}</div>
                      <p>Calculated from function responses</p>
                    </article>
                  </div>
                  <p className="chart-note">Experimental score is a heuristic output, not a validated probability of an RA flare. This prototype does not provide medical advice.</p>
                  <button type="button" className="primary-button" onClick={() => setActivePage('Assessment')}>Complete another assessment →</button>
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
                      <span>RAPID3: {latestAssessment.assessment.rapid3.toFixed(1)}</span>
                    </div>
                  ) : (
                    <div className="empty-state">Complete an assessment to see its result here.</div>
                  )}
                </>
              ) : (
                <>
                  <p>Your saved assessments, newest first.</p>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void loadAssessmentHistory()}
                    disabled={historyLoading}
                  >
                    {historyLoading ? 'Refreshing?' : 'Refresh history'}
                  </button>

                  {historyError && (
                    <div className="empty-state" role="alert">
                      Could not load saved history: {historyError}
                    </div>
                  )}

                  {historyLoading ? (
                    <p>Loading your saved assessments?</p>
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
                            RAPID3: {item.rapid3 == null ? 'Not available' : item.rapid3.toFixed(1)}
                          </span>
                          <span>
                            Experimental score:{' '}
                            {item.peak_experimental_score == null
                              ? 'Not available'
                              : `${(item.peak_experimental_score * 100).toFixed(1)}%`}
                          </span>
                          <span>Risk level: {item.risk_level || 'Not available'}</span>
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
  <AssessmentPage isGuest={isGuest} onResult={setLatestAssessment} />
)


}

          {activePage === 'Forecast History' && (
            <section className="panel secondary-page">
              <span className="stat-icon lavender">▤</span>
              <h2>Forecast history</h2>
              <p>
                No real forecasts are connected yet. This page will
                display timestamped model outputs and later recorded
                outcomes when a properly evaluated prediction pipeline
                is integrated.
              </p>
              <div className="empty-state">
                No forecasts available in this demo.
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
  onResult,
}: {
  isGuest: boolean
  onResult: (result: AssessmentResult) => void
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
  const [result, setResult] = useState<AssessmentResult | null>(null)
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
    setResult(null)

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
      setResult(response)
      onResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to calculate this assessment. Please try again.')
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

      <form onSubmit={handleSubmit}>
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
        <div className="guided-actions">
          {step > 0 && <button className="secondary-button" type="button" onClick={previousStep} disabled={busy}>← Back</button>}
          {step < stepTitles.length - 1
            ? <button className="primary-button" type="button" onClick={nextStep}>Continue →</button>
            : <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Calculating your result…' : 'Calculate my assessment'}</button>}
        </div>
      </form>

      {result && <section className="form-success assessment-result" aria-live="polite">
        <h3>Your assessment result</h3>
        <p><strong>RAPID3-style score:</strong> {result.assessment.rapid3.toFixed(1)} — {result.assessment.rapid3_category}</p>
        <p><strong>Function score:</strong> {result.assessment.function_score.toFixed(1)}</p>
        <p><strong>Experimental score:</strong> {(result.assessment.experimental_score * 100).toFixed(1)}%</p>
        <p><strong>Risk category:</strong> {result.assessment.risk_label}</p>
        <p><strong>Location:</strong> {result.weather.city}. Weather source: {result.weather.is_live ? 'live data' : 'estimated/fallback data'}.</p>
        <h4>Important limitations</h4><ul>{result.limitations.map(item => <li key={item}>{item}</li>)}</ul>
        <p><strong>Please remember:</strong> this experimental result does not diagnose RA or replace advice from your healthcare professional.</p>
        {isGuest && <p>Guest result only: this assessment was not saved to an account history.</p>}
      </section>}
    </section>
  )
}

export default App
