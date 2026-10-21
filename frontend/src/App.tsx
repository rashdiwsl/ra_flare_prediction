
import { useEffect, useState, type FormEvent } from 'react'
import {
  getCurrentUser,
  login,
  logout,
  register,
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

const symptomData = [
  { day: 'Mon', pain: 4, fatigue: 5 },
  { day: 'Tue', pain: 3, fatigue: 4 },
  { day: 'Wed', pain: 5, fatigue: 6 },
  { day: 'Thu', pain: 4, fatigue: 5 },
  { day: 'Fri', pain: 6, fatigue: 7 },
  { day: 'Sat', pain: 3, fatigue: 4 },
  { day: 'Sun', pain: 4, fatigue: 3 },
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
              <span /> DEMO DATA
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
                  The information shown is illustrative demo data.
                  Assessment entries are not saved permanently.
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
              <strong>Demonstration interface.</strong> All patient
              values below are illustrative, not real patient records
              or validated clinical predictions.
            </p>
          </div>

          {activePage === 'Overview' && (
            <>
              <section className="section-heading">
                <div>
                  <h2>Your week at a glance</h2>
                  <p>
                    A simple overview of your recent symptom check-ins.
                  </p>
                </div>
                <span className="period-label">
                  ILLUSTRATIVE · 7 DAYS
                </span>
              </section>

              <div className="stats-grid">
                <article className="stat-card">
                  <div className="stat-top">
                    <span className="stat-icon mint">◷</span>
                    <span className="stat-label">PAIN LEVEL</span>
                  </div>
                  <div className="stat-value">
                    4.1 <span>/ 10</span>
                  </div>
                  <p className="stat-foot">
                    <span className="neutral-pill">Demo value</span>
                    Average this week
                  </p>
                  <div className="mini-bars pain-bars">
                    {[35, 25, 50, 40, 65, 30, 40].map(
                      (height, index) => (
                        <i
                          key={index}
                          style={{ height: `${height}%` }}
                        />
                      ),
                    )}
                  </div>
                </article>

                <article className="stat-card">
                  <div className="stat-top">
                    <span className="stat-icon lavender">☼</span>
                    <span className="stat-label">FATIGUE</span>
                  </div>
                  <div className="stat-value">
                    4.9 <span>/ 10</span>
                  </div>
                  <p className="stat-foot">
                    <span className="neutral-pill">Demo value</span>
                    Average this week
                  </p>
                  <div className="mini-bars fatigue-bars">
                    {[45, 35, 60, 48, 75, 38, 28].map(
                      (height, index) => (
                        <i
                          key={index}
                          style={{ height: `${height}%` }}
                        />
                      ),
                    )}
                  </div>
                </article>

                <article className="stat-card">
                  <div className="stat-top">
                    <span className="stat-icon peach">☾</span>
                    <span className="stat-label">SLEEP QUALITY</span>
                  </div>
                  <div className="stat-value">
                    6.5 <span>/ 10</span>
                  </div>
                  <p className="stat-foot">
                    <span className="neutral-pill">Demo value</span>
                    Average this week
                  </p>
                  <div className="sleep-progress">
                    <span />
                  </div>
                  <p className="progress-caption">
                    Illustrative sleep rating
                  </p>
                </article>

                <article className="stat-card checkin-card">
                  <div className="stat-top">
                    <span className="stat-icon blue">✓</span>
                    <span className="stat-label">CHECK-INS</span>
                  </div>
                  <div className="stat-value">
                    7 <span>days</span>
                  </div>
                  <p className="stat-foot">
                    Example tracking streak
                  </p>
                  <div className="checkin-days">
                    {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map(
                      (day, index) => (
                        <span key={index}>{day}</span>
                      ),
                    )}
                  </div>
                </article>
              </div>

              <div className="content-grid">
                <section className="panel trend-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Symptom patterns</h2>
                      <p>
                        Illustrative daily ratings on a 0–10 scale
                      </p>
                    </div>
                    <span className="chart-tag">Last 7 days</span>
                  </div>

                  <div className="chart-legend">
                    <span>
                      <i className="legend-pain" /> Pain
                    </span>
                    <span>
                      <i className="legend-fatigue" /> Fatigue
                    </span>
                  </div>

                  <div className="chart">
                    <div className="y-labels">
                      <span>10</span>
                      <span>8</span>
                      <span>6</span>
                      <span>4</span>
                      <span>2</span>
                      <span>0</span>
                    </div>

                    <div className="chart-area">
                      {[0, 1, 2, 3, 4, 5].map((line) => (
                        <div className="grid-line" key={line} />
                      ))}

                      <svg
                        viewBox="0 0 700 200"
                        preserveAspectRatio="none"
                        role="img"
                        aria-label="Illustrative pain and fatigue trends over seven days"
                      >
                        <polyline
                          points={symptomData
                            .map(
                              (day, index) =>
                                `${index * 116 + 8},${200 - day.pain * 18}`,
                            )
                            .join(' ')}
                          className="pain-line"
                        />

                        <polyline
                          points={symptomData
                            .map(
                              (day, index) =>
                                `${index * 116 + 8},${200 - day.fatigue * 18}`,
                            )
                            .join(' ')}
                          className="fatigue-line"
                        />

                        {symptomData.map((day, index) => (
                          <circle
                            key={`pain-${day.day}`}
                            cx={index * 116 + 8}
                            cy={200 - day.pain * 18}
                            r="4"
                            className="pain-point"
                          />
                        ))}

                        {symptomData.map((day, index) => (
                          <circle
                            key={`fatigue-${day.day}`}
                            cx={index * 116 + 8}
                            cy={200 - day.fatigue * 18}
                            r="4"
                            className="fatigue-point"
                          />
                        ))}
                      </svg>

                      <div className="x-labels">
                        {symptomData.map((day) => (
                          <span key={day.day}>{day.day}</span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <p className="chart-note">
                    Example data only. This chart does not represent a
                    real person's health.
                  </p>
                </section>

                <section className="panel checkin-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Daily check-in</h2>
                      <p>Take a moment for yourself</p>
                    </div>
                    <span className="checkin-spark">✧</span>
                  </div>

                  <div className="checkin-illustration">
                    <span>♡</span>
                    <div>
                      <i />
                      <i />
                      <i />
                    </div>
                  </div>

                  <h3>How are you feeling today?</h3>
                  <p className="checkin-description">
                    Your experience matters. Record how you feel to
                    build a personal symptom history.
                  </p>

                  <button
                    type="button"
                    className="primary-button full-button"
                    onClick={() => setActivePage('Assessment')}
                  >
                    Start a check-in <span>→</span>
                  </button>

                  <p className="time-hint">
                    Usually takes 2–3 minutes
                  </p>
                </section>
              </div>

              <section className="panel recent-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Recent activity</h2>
                    <p>Example entries for the demonstration</p>
                  </div>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setActivePage('My Symptoms')}
                  >
                    View symptom history →
                  </button>
                </div>

                <div className="activity-row">
                  <span className="activity-icon mint">✓</span>
                  <div className="activity-main">
                    <strong>Symptom check-in recorded</strong>
                    <small>Illustrative entry · Monday</small>
                  </div>
                  <span className="activity-value">Pain: 4/10</span>
                </div>

                <div className="activity-row">
                  <span className="activity-icon lavender">☾</span>
                  <div className="activity-main">
                    <strong>Sleep rating recorded</strong>
                    <small>Illustrative entry · Sunday</small>
                  </div>
                  <span className="activity-value">Sleep: 6/10</span>
                </div>
              </section>
            </>
          )}

          {activePage === 'My Symptoms' && (
            <section className="panel secondary-page">
              <h2>Symptom history</h2>
              <p>
                Example symptom ratings. Real history will appear after
                a connected, persistent data source is implemented.
              </p>

              {symptomData.map((day) => (
                <div className="history-row" key={day.day}>
                  <strong>{day.day}</strong>
                  <span>Pain: {day.pain}/10</span>
                  <span>Fatigue: {day.fatigue}/10</span>
                </div>
              ))}
            </section>
          )}

          {activePage === 'Assessment' && (
  <AssessmentPage isGuest={isGuest} />
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


function AssessmentPage({ isGuest }: { isGuest: boolean }) {
  const [pain, setPain] = useState('0')
  const [fatigue, setFatigue] = useState('0')
  const [sleep, setSleep] = useState('5')
  const [notes, setNotes] = useState('')
  const [saved, setSaved] = useState(false)

  function submitAssessment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    // Demonstration only: no API request or database write occurs.
    // Guest and patient submissions are not persisted by this function.
    setSaved(true)
  }

  return (
    <section className="panel assessment-panel">
      <div className="assessment-heading">
        <span className="eyebrow">
          {isGuest ? 'GUEST DEMONSTRATION' : 'PATIENT CHECK-IN'}
        </span>

        <h2>How are you feeling today?</h2>

        <p>
          Record your symptoms and review your check-in.
          {isGuest
            ? ' Your responses are temporary and will not be saved to a patient account.'
            : ' This prototype currently does not save assessment submissions to your account.'}
        </p>
      </div>

      {isGuest && (
        <div className="guest-banner" role="status">
          <div>
            <strong>Guest mode — temporary assessment</strong>
            <p>
              You can complete the assessment without registering.
              Your responses are not stored in patient history.
            </p>
          </div>
        </div>
      )}

      <form onSubmit={submitAssessment}>
        <label>
          Pain level <strong>{pain}/10</strong>
          <input
            type="range"
            min="0"
            max="10"
            value={pain}
            onChange={(event) => {
              setPain(event.target.value)
              setSaved(false)
            }}
          />
        </label>

        <label>
          Fatigue level <strong>{fatigue}/10</strong>
          <input
            type="range"
            min="0"
            max="10"
            value={fatigue}
            onChange={(event) => {
              setFatigue(event.target.value)
              setSaved(false)
            }}
          />
        </label>

        <label>
          Sleep quality <strong>{sleep}/10</strong>
          <input
            type="range"
            min="0"
            max="10"
            value={sleep}
            onChange={(event) => {
              setSleep(event.target.value)
              setSaved(false)
            }}
          />
        </label>

        <label>
          Optional notes
          <textarea
            rows={4}
            maxLength={1000}
            value={notes}
            onChange={(event) => {
              setNotes(event.target.value)
              setSaved(false)
            }}
            placeholder="How have you been feeling?"
          />
        </label>

        <button className="primary-button" type="submit">
          Review my check-in
        </button>

        {saved && (
          <div className="form-success" role="status">
            <h3>Check-in completed</h3>

            <p>
              Pain: {pain}/10 · Fatigue: {fatigue}/10 · Sleep quality: {sleep}/10
            </p>

            {notes.trim() && (
              <p>
                <strong>Your notes:</strong> {notes}
              </p>
            )}

            <p>
              {isGuest
                ? 'This is a temporary demonstration result. It has not been saved to a patient account.'
                : 'This is a demonstration result. It has not been saved to your patient history.'}
            </p>
          </div>
        )}
      </form>
    </section>
  )
}


export default App
