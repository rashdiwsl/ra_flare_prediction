
import { useState } from 'react'
import './App.css'

type Page = 'Overview' | 'My Symptoms' | 'Assessment' | 'Forecast History'

const navigation: { label: Page; icon: string }[] = [
  { label: 'Overview', icon: '⌂' },
  { label: 'My Symptoms', icon: '◷' },
  { label: 'Assessment', icon: '✚' },
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

function App() {
  const [activePage, setActivePage] = useState<Page>('Overview')

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={() => setActivePage('Overview')}>
          <span className="brand-mark">F</span>
          <span>
            <strong>FlareGuard</strong>
            <small>YOUR HEALTH, IN FOCUS</small>
          </span>
        </a>

        <div className="nav-heading">PATIENT SPACE</div>
        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${activePage === item.label ? 'active' : ''}`}
              onClick={() => setActivePage(item.label)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
              {item.label === 'Assessment' && <span className="nav-dot" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <span className="support-icon">♡</span>
            <strong>Small steps matter.</strong>
            <p>Tracking how you feel can help you understand your patterns over time.</p>
          </div>
          <div className="profile">
            <div className="avatar">D</div>
            <div className="profile-text">
              <strong>Demo Patient</strong>
              <small>Patient account</small>
            </div>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Patient Space <span>/</span> {activePage}</div>
          <div className="topbar-right">
            <span className="demo-badge"><span /> DEMO DATA</span>
            <div className="top-avatar">D</div>
          </div>
        </header>

        <div className="page-content">
          <div className="welcome-row">
            <div>
              <p className="eyebrow">YOUR PERSONAL HEALTH SPACE</p>
              <h1>{activePage === 'Overview' ? 'Good day, Demo Patient' : activePage}</h1>
              <p className="subtitle">
                {activePage === 'Overview'
                  ? 'A little check-in today can help you notice changes over time.'
                  : `Explore your ${activePage.toLowerCase()} information in this demonstration.`}
              </p>
            </div>
            <button className="primary-button" onClick={() => setActivePage('Assessment')}>
              <span>＋</span> Record symptoms
            </button>
          </div>

          <div className="notice">
            <span className="notice-icon">ⓘ</span>
            <p><strong>Demonstration interface.</strong> All patient values below are illustrative, not real patient records or validated clinical predictions.</p>
          </div>

          {activePage === 'Overview' && (
            <>
              <section className="section-heading">
                <div>
                  <h2>Your week at a glance</h2>
                  <p>A simple overview of your recent symptom check-ins.</p>
                </div>
                <span className="period-label">ILLUSTRATIVE · 7 DAYS</span>
              </section>

              <div className="stats-grid">
                <article className="stat-card">
                  <div className="stat-top"><span className="stat-icon mint">◷</span><span className="stat-label">PAIN LEVEL</span></div>
                  <div className="stat-value">4.1 <span>/ 10</span></div>
                  <p className="stat-foot"><span className="neutral-pill">Demo value</span> Average this week</p>
                  <div className="mini-bars pain-bars">{[35, 25, 50, 40, 65, 30, 40].map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</div>
                </article>

                <article className="stat-card">
                  <div className="stat-top"><span className="stat-icon lavender">☼</span><span className="stat-label">FATIGUE</span></div>
                  <div className="stat-value">4.9 <span>/ 10</span></div>
                  <p className="stat-foot"><span className="neutral-pill">Demo value</span> Average this week</p>
                  <div className="mini-bars fatigue-bars">{[45, 35, 60, 48, 75, 38, 28].map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</div>
                </article>

                <article className="stat-card">
                  <div className="stat-top"><span className="stat-icon peach">☾</span><span className="stat-label">SLEEP QUALITY</span></div>
                  <div className="stat-value">6.5 <span>/ 10</span></div>
                  <p className="stat-foot"><span className="neutral-pill">Demo value</span> Average this week</p>
                  <div className="sleep-progress"><span /></div>
                  <p className="progress-caption">Illustrative sleep rating</p>
                </article>

                <article className="stat-card checkin-card">
                  <div className="stat-top"><span className="stat-icon blue">✓</span><span className="stat-label">CHECK-INS</span></div>
                  <div className="stat-value">7 <span>days</span></div>
                  <p className="stat-foot">Example tracking streak</p>
                  <div className="checkin-days">{['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((d, i) => <span key={i}>{d}</span>)}</div>
                </article>
              </div>

              <div className="content-grid">
                <section className="panel trend-panel">
                  <div className="panel-heading">
                    <div><h2>Symptom patterns</h2><p>Illustrative daily ratings on a 0–10 scale</p></div>
                    <span className="chart-tag">Last 7 days</span>
                  </div>

                  <div className="chart-legend">
                    <span><i className="legend-pain" /> Pain</span>
                    <span><i className="legend-fatigue" /> Fatigue</span>
                  </div>

                  <div className="chart">
                    <div className="y-labels"><span>10</span><span>8</span><span>6</span><span>4</span><span>2</span><span>0</span></div>
                    <div className="chart-area">
                      {[0, 1, 2, 3, 4, 5].map((line) => <div className="grid-line" key={line} />)}
                      <svg viewBox="0 0 700 200" preserveAspectRatio="none" role="img" aria-label="Illustrative pain and fatigue trend lines for seven days">
                        <polyline points={symptomData.map((d, i) => `${i * 116 + 8},${200 - d.pain * 18}`).join(' ')} className="pain-line" />
                        <polyline points={symptomData.map((d, i) => `${i * 116 + 8},${200 - d.fatigue * 18}`).join(' ')} className="fatigue-line" />
                        {symptomData.map((d, i) => <circle key={`p${i}`} cx={i * 116 + 8} cy={200 - d.pain * 18} r="4" className="pain-point" />)}
                        {symptomData.map((d, i) => <circle key={`f${i}`} cx={i * 116 + 8} cy={200 - d.fatigue * 18} r="4" className="fatigue-point" />)}
                      </svg>
                      <div className="x-labels">{symptomData.map((d) => <span key={d.day}>{d.day}</span>)}</div>
                    </div>
                  </div>
                  <p className="chart-note">Example data only. This chart does not represent a real person's health.</p>
                </section>

                <section className="panel checkin-panel">
                  <div className="panel-heading"><div><h2>Daily check-in</h2><p>Take a moment for yourself</p></div><span className="checkin-spark">✳</span></div>
                  <div className="checkin-illustration"><span>♡</span><div><i /><i /><i /></div></div>
                  <h3>How are you feeling today?</h3>
                  <p className="checkin-description">Your experience matters. Record how you feel to build a personal symptom history.</p>
                  <button className="primary-button full-button" onClick={() => setActivePage('Assessment')}>Start a check-in <span>→</span></button>
                  <p className="time-hint">Usually takes 2–3 minutes</p>
                </section>
              </div>

              <section className="panel recent-panel">
                <div className="panel-heading"><div><h2>Recent activity</h2><p>Example entries for the demonstration</p></div><button className="text-button" onClick={() => setActivePage('My Symptoms')}>View symptom history →</button></div>
                <div className="activity-row"><span className="activity-icon mint">✓</span><div className="activity-main"><strong>Symptom check-in recorded</strong><small>Illustrative entry · Monday</small></div><span className="activity-value">Pain: 4/10</span></div>
                <div className="activity-row"><span className="activity-icon lavender">☾</span><div className="activity-main"><strong>Sleep rating recorded</strong><small>Illustrative entry · Sunday</small></div><span className="activity-value">Sleep: 6/10</span></div>
              </section>
            </>
          )}

          {activePage === 'My Symptoms' && (
            <section className="panel secondary-page">
              <h2>Symptom history</h2>
              <p>Example symptom ratings. Real history will appear after a connected, persistent data source is implemented.</p>
              {symptomData.map((d) => <div className="history-row" key={d.day}><strong>{d.day}</strong><span>Pain: {d.pain}/10</span><span>Fatigue: {d.fatigue}/10</span></div>)}
            </section>
          )}

          {activePage === 'Assessment' && <AssessmentPage />}

          {activePage === 'Forecast History' && (
            <section className="panel secondary-page">
              <span className="stat-icon lavender">▤</span>
              <h2>Forecast history</h2>
              <p>No real forecasts are connected yet. This page will display timestamped model outputs and later recorded outcomes when a properly evaluated prediction pipeline is integrated.</p>
              <div className="empty-state">No forecasts available in this demo.</div>
            </section>
          )}

          <footer className="footer"><span>FlareGuard <span className="footer-dot">•</span> Research prototype</span><span>For research and demonstration only · Not medical advice</span></footer>
        </div>
      </main>
    </div>
  )
}

function AssessmentPage() {
  const [pain, setPain] = useState('0')
  const [fatigue, setFatigue] = useState('0')
  const [sleep, setSleep] = useState('5')
  const [notes, setNotes] = useState('')
  const [saved, setSaved] = useState(false)

  function submitAssessment(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSaved(true)
  }

  return (
    <section className="panel assessment-panel">
      <h2>How are you feeling today?</h2>
      <p>This interactive form is a frontend demonstration. Submissions are not sent to a server or saved permanently.</p>
      <form onSubmit={submitAssessment}>
        <label>Pain level <strong>{pain}/10</strong><input type="range" min="0" max="10" value={pain} onChange={(e) => { setPain(e.target.value); setSaved(false) }} /></label>
        <label>Fatigue level <strong>{fatigue}/10</strong><input type="range" min="0" max="10" value={fatigue} onChange={(e) => { setFatigue(e.target.value); setSaved(false) }} /></label>
        <label>Sleep quality <strong>{sleep}/10</strong><input type="range" min="0" max="10" value={sleep} onChange={(e) => { setSleep(e.target.value); setSaved(false) }} /></label>
        <label>Optional notes<textarea rows={4} value={notes} onChange={(e) => { setNotes(e.target.value); setSaved(false) }} placeholder="How have you been feeling?" /></label>
        <button className="primary-button" type="submit">Review check-in</button>
        {saved && <p className="form-success">Demo check-in reviewed: pain {pain}/10, fatigue {fatigue}/10, sleep {sleep}/10. This has not been saved permanently.</p>}
      </form>
    </section>
  )
}

export default App
