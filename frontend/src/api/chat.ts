import type {
  ChatEvent,
  ChatSessionResponse,
  ChatTurnResponse,
  DebugReasoningEvent,
} from './types'

const API_ROOT = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')
const CHAT_ROOT = API_ROOT.replace(/\/api\/v1$/, '/api/v2/chat')

async function requestJson<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const response = await fetch(`${CHAT_ROOT}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  })
  if (!response.ok) {
    let message = `对话请求失败，状态码为 ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) message = payload.detail
    } catch {
      // 服务端没有返回 JSON 时保留状态码回退信息。
    }
    throw new Error(message)
  }
  return (await response.json()) as TResponse
}

export function createChatSession(repositoryRoot: string): Promise<ChatSessionResponse> {
  return requestJson<ChatSessionResponse>('/sessions', {
    method: 'POST',
    body: JSON.stringify({ repository_root: repositoryRoot }),
  })
}

export function submitChatTurn(request: {
  session_id: string
  repository_root: string
  message: string
  client_turn_id?: string
  limit?: number
  validation_profile?: string
  show_debug_reasoning?: boolean
}): Promise<ChatTurnResponse> {
  return requestJson<ChatTurnResponse>('/turns', {
    method: 'POST',
    body: JSON.stringify({
      limit: 5,
      validation_profile: 'python_compile',
      show_debug_reasoning: false,
      ...request,
    }),
  })
}

export function getChatTurn(turnId: string): Promise<ChatTurnResponse> {
  return requestJson<ChatTurnResponse>(`/turns/${encodeURIComponent(turnId)}`)
}

export function approveChatTurn(turnId: string): Promise<ChatTurnResponse> {
  return requestJson<ChatTurnResponse>(`/turns/${encodeURIComponent(turnId)}/approve`, {
    method: 'POST',
  })
}

export async function streamChatEvents(
  turnId: string,
  onEvent: (event: ChatEvent) => void,
  onReasoning: (event: DebugReasoningEvent) => void,
  options: { afterSequence?: number; signal?: AbortSignal } = {},
): Promise<void> {
  const afterSequence = options.afterSequence || 0
  const response = await fetch(
    `${CHAT_ROOT}/turns/${encodeURIComponent(turnId)}/events?after_sequence=${afterSequence}`,
    { headers: { Accept: 'text/event-stream' }, signal: options.signal },
  )
  if (!response.ok || !response.body) {
    throw new Error(`实时事件连接失败，状态码为 ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let eventName = 'message'
  let dataLines: string[] = []

  function dispatch() {
    if (dataLines.length === 0) return
    const data = dataLines.join('\n')
    try {
      if (eventName === 'debug_reasoning') {
        onReasoning(JSON.parse(data) as DebugReasoningEvent)
      } else {
        onEvent(JSON.parse(data) as ChatEvent)
      }
    } finally {
      eventName = 'message'
      dataLines = []
    }
  }

  while (true) {
    const chunk = await reader.read()
    if (chunk.done) {
      buffer += decoder.decode()
      break
    }
    buffer += decoder.decode(chunk.value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() || ''
    for (const line of lines) {
      if (line === '') {
        dispatch()
      } else if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart())
      }
    }
  }
  if (buffer === '') dispatch()
}
