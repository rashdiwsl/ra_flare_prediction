
import { useEffect, useState } from 'react'
import {
  getDailyCheckins,
  type DailyCheckin,
} from './services/api'

function formatDate(value: string) {
  const date = new Date(`${value}T00:00:00`)
  if (Number.isNaN(date.getTime())) return value

  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function formatSymptomChange(value: string) {
  if (value === 'better') return 'Feeling better'
  if (value === 'worse') return 'Feeling worse'
  return 'About the same'
}

export default function DailySymptomHistory() {
  const [items, setItems] = useState<DailyCheckin[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function loadHistory() {
    setLoading(true)
    setError('')

    try {
      const result = await getDailyCheckins(30)
      setItems(result.checkins)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load symptom history.',
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadHistory()
  }, [])

  if (loading) {
    return (
      <section className="daily-history">
        <h2>Daily Symptom History</h2>
        <p>Loading your saved check-ins...</p>
      </section>
    )
  }

  if (error) {
    return (
      <section className="daily-history">
        <h2>Daily Symptom History</h2>
        <p className="daily-history-error">{error}</p>
        <button type="button" onClick={() => void loadHistory()}>
          Try again
        </button>
      </section>
    )
  }

  return (
    <section className="daily-history">
      <div className="daily-history-heading">
        <div>
          <h2>Daily Symptom History</h2>
          <p>Review symptoms and assessment scores you have saved.</p>
        </div>

        <button
          type="button"
          onClick={() => void loadHistory()}
          className="daily-history-refresh"
        >
          Refresh
        </button>
      </div>

      <div className="daily-history-note">
        These records show your reported symptoms and experimental
        assessment results. They are not a confirmed flare forecast.
      </div>

      {items.length === 0 ? (
        <div className="daily-history-empty">
          <h3>No check-ins yet</h3>
          <p>
            Complete and save an assessment while logged in. Your saved
            check-ins will appear here.
          </p>
        </div>
      ) : (
        <>
          <p className="daily-history-count">
            {items.length} saved {items.length === 1 ? 'record' : 'records'}
          </p>

          <div className="daily-history-list">
            {items.map((item) => (
              <article
                className="daily-history-card"
                key={item.created_at}
              >
                <div className="daily-history-card-heading">
                  <div>
                    <h3>{formatDate(item.date)}</h3>
                    <p>
                      {formatSymptomChange(item.symptoms.symptom_change)}
                    </p>
                  </div>
                  <span className="daily-history-score">
                    RAPID3: {Number(item.rapid3).toFixed(1)}
                  </span>
                </div>

                <div className="daily-history-metrics">
                  <div>
                    <span>Pain</span>
                    <strong>{item.symptoms.pain}/10</strong>
                  </div>
                  <div>
                    <span>Fatigue</span>
                    <strong>{item.symptoms.fatigue}/10</strong>
                  </div>
                  <div>
                    <span>Overall health</span>
                    <strong>{item.symptoms.global_estimate}/10</strong>
                  </div>
                  <div>
                    <span>Daily function score</span>
                    <strong>{Number(item.function_score).toFixed(2)}</strong>
                  </div>
                </div>

                <details className="daily-history-details">
                  <summary>View more reported symptoms</summary>
                  <dl>
                    <dt>Sleep difficulty</dt>
                    <dd>{item.symptoms.sleep_diff}/3</dd>

                    <dt>Anxiety</dt>
                    <dd>{item.symptoms.anxiety}/3</dd>

                    <dt>Flares in the past 30 days</dt>
                    <dd>{item.symptoms.flares_30d}</dd>

                    <dt>Daily activity minutes</dt>
                    <dd>{item.symptoms.activity_min}</dd>

                    <dt>Heart rate</dt>
                    <dd>
                      {item.symptoms.heart_rate > 0
                        ? `${item.symptoms.heart_rate} bpm`
                        : 'Not provided'}
                    </dd>
                  </dl>
                </details>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  )
}
