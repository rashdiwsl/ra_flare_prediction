
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
  detail?: string
  message?: string
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
    const errorData = data as ApiError

    const message =
      typeof errorData?.detail === 'string'
        ? errorData.detail
        : typeof errorData?.message === 'string'
          ? errorData.message
          : `Request failed (${response.status}).`

    throw new Error(message)
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

// ============================================================
// ASSESSMENT HISTORY
// ============================================================

export function getAssessmentHistory(limit = 10) {
  const safeLimit = Math.min(100, Math.max(1, limit))

  return request<AssessmentHistory>(
    `/api/assessments?limit=${safeLimit}`,
  )
}
