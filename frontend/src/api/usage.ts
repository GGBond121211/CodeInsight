import type { UsageCallsResponse, UsageSummary } from './types'

const API_ROOT = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')

async function getJson<TResponse>(path: string): Promise<TResponse> {
  const response = await fetch(`${API_ROOT}${path}`, { cache: 'no-store' })
  if (!response.ok) {
    let message = `监控请求失败，状态码为 ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string | { message?: string } }
      if (typeof payload.detail === 'string') message = payload.detail
      if (payload.detail && typeof payload.detail === 'object' && payload.detail.message) {
        message = payload.detail.message
      }
    } catch {
      // 保留基于状态码的回退信息。
    }
    throw new Error(message)
  }
  return (await response.json()) as TResponse
}

export function usageSummary(): Promise<UsageSummary> {
  return getJson<UsageSummary>('/usage/summary')
}

export function usageCalls(limit = 20): Promise<UsageCallsResponse> {
  return getJson<UsageCallsResponse>(`/usage/calls?limit=${limit}`)
}
