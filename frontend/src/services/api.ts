
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? (import.meta.env.PROD ? '' : 'http://localhost:8000')

// ============================================================
// TYPES
// ============================================================

export interface RegisterData {
  username: string
  password: string
  display_name: string
}

export interface LoginData {
  username: string
  password: string
}

export interface AuthUser {
  authenticated: boolean
  username: string
  display_name: string
}

interface AuthResponse {
  success: boolean
  message: string
  username?: string
  display_name?: string
  expires_at?: string
}

export interface AssessmentData {
  age: number
  symptom_change: string
  city: string
  fn_vals: Record<string, number>
  pain: number
  global_estimate: number
  fatigue: number
  flares_30d: number
  duration_years: number
  sleep_diff: number
  anxiety: number
  activity_min: number
  heart_rate: number
}

export interface AssessmentResult {
  success: boolean
  message: string
  assessment: {
    username: string
    city: string
    created_at: string
    rapid3: number
    rapid3_category: string
    function_score: number
    pain: number
    global_estimate: number
    experimental_score: number
    risk_level: string
    risk_label: string
    risk_color: string
    peak_experimental_score: number
    peak_risk_level: string
    contributions: Record<string, number>
    reported_inputs?: {
      symptom_change: 'better' | 'same' | 'worse'
      fatigue: number
      sleep_difficulty: number
      anxiety: number
      flares_30d: number
      activity_min: number
      heart_rate_bpm: number | null
    }
  }
  weather: {
    city: string
    temperature: number
    humidity: number
    pressure: number
    description: string
    is_live: boolean
  }
  forecast: Array<{
    day: number
    pain: number
    fatigue: number
    anxiety: number
    sleep_difficulty: number
    experimental_score: number
    risk_level: string
  }>
  limitations: string[]
}

export interface HistoryItem {
  created_at: string
  city: string | null
  rapid3: number | null
  peak_experimental_score: number | null
  risk_level: string | null
}

export interface AssessmentHistory {
  success: boolean
  username: string
  count: number
  assessments: HistoryItem[]
}

interface ApiError {
  detail?: string | ApiValidationIssue[]
  message?: string
}

interface ApiValidationIssue {
  loc?: Array<string | number>
  msg?: string
}

function getApiErrorMessage(data: unknown, status: number): string {
  if (typeof data === 'object' && data !== null) {
    const errorData = data as ApiError

    if (typeof errorData.detail === 'string') {
      return errorData.detail
    }

    if (Array.isArray(errorData.detail)) {
      const validationMessages = errorData.detail.flatMap((issue) => {
        if (!issue || typeof issue !== 'object') {
          return []
        }

        const location = Array.isArray(issue.loc)
          ? issue.loc.filter((part) => part !== 'body').join('.')
          : ''
        const message = typeof issue.msg === 'string' ? issue.msg : ''

        return message
          ? [`${location ? `${location}: ` : ''}${message}`]
          : []
      })

      if (validationMessages.length > 0) {
        return validationMessages.join('; ')
      }
    }

    if (typeof errorData.message === 'string') {
      return errorData.message
    }
  }

  return `Request failed (${status}).`
}

// ============================================================
// COMMON REQUEST HELPER
// ============================================================

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      credentials: 'include',
      headers: {
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
  } catch {
    throw new Error(
      'Cannot connect to the FlareGuard API. Check that the backend is running.',
    )
  }

  const contentType = response.headers.get('content-type') || ''
  let data: unknown

  if (contentType.includes('application/json')) {
    data = await response.json()
  } else {
    data = await response.text()
  }

  if (!response.ok) {
    throw new Error(getApiErrorMessage(data, response.status))
  }

  return data as T
}

// ============================================================
// AUTHENTICATION
// ============================================================

export function register(data: RegisterData) {
  return request<AuthResponse>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function login(data: LoginData) {
  return request<AuthResponse>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function getCurrentUser() {
  return request<AuthUser>('/api/auth/me')
}

export function logout() {
  return request<{ success: boolean; message: string }>(
    '/api/auth/logout',
    {
      method: 'POST',
    },
  )
}

// ============================================================
// HEALTH CHECK
// ============================================================

export function checkApiHealth() {
  return request<{
    status: string
    application: string
    mode: string
  }>('/api/health')
}

// ============================================================
// ASSESSMENTS
// ============================================================

export function submitAssessment(data: AssessmentData) {
  return request<AssessmentResult>('/api/assessments', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

/** Calculate a guest assessment without requiring login or saving account history. */
export function submitGuestAssessment(data: AssessmentData) {
  return request<AssessmentResult>('/api/assessments/guest', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

// ============================================================
// ASSESSMENT HISTORY
// ============================================================

export function getAssessmentHistory(limit?: number) {
  if (limit === undefined) {
    return request<AssessmentHistory>('/api/assessments')
  }

  const safeLimit = Math.min(100, Math.max(1, Math.floor(limit)))

  return request<AssessmentHistory>(
    `/api/assessments?limit=${safeLimit}`,
  )
}


export interface ResearchPredictionInput {
  past_mean_severity: number
  past_std_severity: number
  past_trend: number
  past_last_severity: number
  past_engagement: number
  personal_baseline: number
  age: number
  sex: string
}

export interface ResearchPredictionResult {
  success: boolean
  research_only: boolean
  target: string
  prediction: {
    class: number
    label: 'symptom_worsening_proxy' | 'no_symptom_worsening_proxy'
    positive_class_model_score: number
  }
  model: {
    name: string
    evaluation_split: string | null
  }
  limitations: string[]
}

export function predictResearchSymptomWorsening(
  data: ResearchPredictionInput,
) {
  return request<ResearchPredictionResult>(
    '/api/research/symptom-worsening-prediction',
    {
      method: 'POST',
      body: JSON.stringify(data),
    },
  )
}

/* ============================================================
   DAILY SYMPTOM CHECK-IN HISTORY
   ============================================================ */

export interface DailyCheckinSymptoms {
  age: number
  symptom_change: 'better' | 'same' | 'worse'
  city: string
  fn_vals: Record<string, number>
  pain: number
  global_estimate: number
  fatigue: number
  flares_30d: number
  duration_years: number
  sleep_diff: number
  anxiety: number
  activity_min: number
  heart_rate: number
}

export interface DailyCheckin {
  date: string
  created_at: string
  symptoms: DailyCheckinSymptoms
  rapid3: number
  function_score: number
  experimental_score: number
  risk_level: string
}

export interface DailyCheckinHistory {
  success: boolean
  count: number
  checkins: DailyCheckin[]
}

export function getDailyCheckins(limit = 30) {
  const safeLimit = Math.min(100, Math.max(1, Math.floor(limit)))

  return request<DailyCheckinHistory>(
    `/api/daily-checkins?limit=${safeLimit}`,
  )
}
