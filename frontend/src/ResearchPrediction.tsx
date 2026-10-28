
import { useState, type FormEvent } from 'react'
import {
  predictResearchSymptomWorsening,
  type ResearchPredictionInput,
  type ResearchPredictionResult,
} from './services/api'

const initialValues = {
  past_mean_severity: '',
  past_std_severity: '',
  past_trend: '',
  past_last_severity: '',
  past_engagement: '',
  personal_baseline: '',
  age: '',
  sex: '',
}

type NumericFeature =
  | 'past_mean_severity'
  | 'past_std_severity'
  | 'past_trend'
  | 'past_last_severity'
  | 'past_engagement'
  | 'personal_baseline'
  | 'age'

const featureFields: {
  name: NumericFeature
  label: string
  description: string
  min?: number
  step: number
}[] = [
  {
    name: 'past_mean_severity',
    label: 'Past mean severity',
    description: 'Average recorded symptom severity in the past window.',
    min: 0,
    step: 0.1,
  },
  {
    name: 'past_std_severity',
    label: 'Past severity variation',
    description: 'Standard deviation of recorded symptom severity.',
    min: 0,
    step: 0.1,
  },
  {
    name: 'past_trend',
    label: 'Past severity trend',
    description: 'Historical trend value; negative values are allowed.',
    step: 0.1,
  },
  {
    name: 'past_last_severity',
    label: 'Most recent severity',
    description: 'Severity value from the most recent available record.',
    min: 0,
    step: 0.1,
  },
  {
    name: 'past_engagement',
    label: 'Past engagement',
    description: 'Engagement feature calculated by the dataset pipeline.',
    min: 0,
    step: 1,
  },
  {
    name: 'personal_baseline',
    label: 'Personal baseline',
    description: 'Historical personal severity baseline.',
    min: 0,
    step: 0.1,
  },
  {
    name: 'age',
    label: 'Age',
    description: 'Age in years.',
    min: 1,
    step: 1,
  },
]

export default function ResearchPrediction() {
  const [values, setValues] = useState(initialValues)
  const [result, setResult] = useState<ResearchPredictionResult | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setResult(null)

    const input: ResearchPredictionInput = {
      past_mean_severity: Number(values.past_mean_severity),
      past_std_severity: Number(values.past_std_severity),
      past_trend: Number(values.past_trend),
      past_last_severity: Number(values.past_last_severity),
      past_engagement: Number(values.past_engagement),
      personal_baseline: Number(values.personal_baseline),
      age: Number(values.age),
      sex: values.sex.trim(),
    }

    if (!input.sex) {
      setError('Please enter the sex category expected by your dataset.')
      return
    }

    if (
      !Number.isFinite(input.age) ||
      input.age < 1 ||
      input.age > 120
    ) {
      setError('Enter an age between 1 and 120.')
      return
    }

    if (
      input.past_mean_severity < 0 ||
      input.past_std_severity < 0 ||
      input.past_last_severity < 0 ||
      input.past_engagement < 0 ||
      input.personal_baseline < 0
    ) {
      setError('Severity, engagement, and baseline values cannot be negative.')
      return
    }

    setBusy(true)

    try {
      const response = await predictResearchSymptomWorsening(input)
      setResult(response)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to run the research prediction.',
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="panel secondary-page research-prediction-page">
      <div className="research-intro">
        <span className="eyebrow">EXPERIMENTAL ML MODEL</span>
        <h2>Research symptom-worsening prediction</h2>
        <p>
          Enter historical summary features to test the saved Logistic
          Regression research model. These fields are not automatically
          populated from your regular symptom assessment.
        </p>
      </div>

      <div className="research-warning" role="note">
        <strong>Research use only</strong>
        <p>
          This model estimates a dataset-defined symptom-worsening proxy.
          It does not predict a confirmed clinical RA flare and is not a
          diagnosis or medical advice.
        </p>
      </div>

      <form className="research-form" onSubmit={handleSubmit}>
        <div className="research-field-grid">
          {featureFields.map((field) => (
            <label className="research-field" key={field.name}>
              <span>{field.label}</span>
              <input
                type="number"
                required
                min={field.min}
                step={field.step}
                value={values[field.name]}
                onChange={(event) => {
                  setValues((current) => ({
                    ...current,
                    [field.name]: event.target.value,
                  }))
                }}
              />
              <small>{field.description}</small>
            </label>
          ))}

          <label className="research-field">
            <span>Sex category</span>
            <input
              type="text"
              required
              value={values.sex}
              onChange={(event) =>
                setValues((current) => ({
                  ...current,
                  sex: event.target.value,
                }))
              }
              placeholder="Enter dataset category"
            />
            <small>
              Use the category format expected by the model's training data.
            </small>
          </label>
        </div>

        {error && (
          <p className="research-error" role="alert">
            {error}
          </p>
        )}

        <button
          type="submit"
          className="primary-button"
          disabled={busy}
        >
          {busy ? 'Running model…' : 'Run research prediction'}
        </button>
      </form>

      {result && (
        <section className="research-result" aria-live="polite">
          <span className="eyebrow">MODEL OUTPUT · NOT A CLINICAL RESULT</span>
          <h3>
            {result.prediction.class === 1
              ? 'Symptom-worsening proxy predicted'
              : 'No symptom-worsening proxy predicted'}
          </h3>

          <p>
            Experimental positive-class model score:{' '}
            <strong>
              {(result.prediction.positive_class_model_score * 100).toFixed(1)}%
            </strong>
          </p>

          <p>
            Model: <strong>{result.model.name}</strong>
          </p>

          <p>
            Evaluation split: {result.model.evaluation_split ?? 'Not specified'}
          </p>

          <p className="research-result-note">
            This percentage is a model score, not a calibrated clinical
            probability or the probability of having an RA flare.
          </p>

          <h3>Limitations</h3>
          <ul>
            {result.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </section>
      )}
    </section>
  )
}
