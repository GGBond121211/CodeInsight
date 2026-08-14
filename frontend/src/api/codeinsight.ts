import type { AutoAnswerRequest, AutoAnswerResponse } from './types'

const API_ROOT = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')

async function postJson<TResponse>(path: string, request: object): Promise<TResponse> {
  const response = await fetch(`${API_ROOT}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) message = payload.detail
    } catch {
      // Keep the status-based fallback when the server did not return JSON.
    }
    throw new Error(message)
  }
  return (await response.json()) as TResponse
}

/* Disabled public client wiring; retained for a future explicit restoration.
export function searchRepository(request: SearchRequest): Promise<SearchResponse> {
  return postJson<SearchResponse>('/search', request)
}

export function answerRepository(request: AnswerRequest): Promise<AnswerResponse> {
  return postJson<AnswerResponse>('/answer', request)
}

export function agentAnswerRepository(request: AnswerRequest): Promise<AgentAnswerResponse> {
  return postJson<AgentAnswerResponse>('/agent/answer', request)
}
*/

export function autoAnswerRepository(request: AutoAnswerRequest): Promise<AutoAnswerResponse> {
  return postJson<AutoAnswerResponse>('/auto/answer', request)
}
