import { useRef, useState } from 'react'

import {
  approveChatTurn,
  createChatSession,
  getChatTurn,
  streamChatEvents,
  submitChatTurn,
} from './api/chat'
import type {
  ChatEvent,
  ChatTurnResponse,
  ChatTurnStatus,
  DebugReasoningEvent,
} from './api/types'
import { ObservabilityDashboard } from './components/ObservabilityDashboard'
import './App.css'

interface ChatMessage {
  id: string
  turnId: string
  role: 'user' | 'assistant'
  content: string
  taskType?: 'general_chat' | 'scope_redirect' | 'clarify' | 'explain' | 'change'
  result?: Record<string, unknown> | null
}

const EVENT_LABELS: Record<string, string> = {
  run_started: 'Run 已创建',
  turn_accepted: '请求已接收，进入后台队列',
  session_loaded: '已恢复 Session 与历史记忆',
  intent_classified: '业务编排已选择执行路线',
  context_assembled: '已装配本轮模型上下文',
  context_compacted: '长对话已压缩旧轮次并保留摘要',
  retrieval_started: '正在扫描并检索仓库证据',
  retrieval_finished: '仓库证据检索完成',
  model_generating: '模型回答已完成',
  model_called: '正在调用供应商模型',
  model_result: '已收到模型结果与 usage',
  step_started: 'Agent 正在执行下一阶段',
  tool_call_requested: '模型请求调用工具',
  tool_call_validated: '工具请求已通过策略校验',
  tool_dispatched: '工具已分发执行',
  tool_result_committed: '工具结果已提交回上下文',
  patch_generated: '修改预览已生成',
  approval_requested: '等待用户审批修改',
  approval_granted: '用户已批准修改',
  validation_started: '正在运行固定校验',
  validation_finished: '固定校验已返回',
  validation_preflight: '正在检查 Sandbox 校验环境',
  patch_rejected: '补丁被拒绝：没有产生实际变化',
  state_transitioned: 'Run 状态已推进',
  run_finished: 'Run 已结束',
}

const MODEL_ERROR_DETAILS: Record<string, string> = {
  structured_output_invalid_json: '结构化输出不是有效 JSON',
  structured_output_truncated: '结构化输出达到 max_tokens 后被截断',
  structured_output_not_object: '结构化输出不是 JSON object',
  tool_response_empty: '工具路线返回空响应',
  tool_call_missing_function: 'tool_call 缺少 function',
  tool_call_arguments_invalid_json: 'tool_call 参数不是有效 JSON',
  tool_call_arguments_not_object: 'tool_call 参数不是 JSON object',
}

// 还在跑的状态：排队、执行、等隔离校验都算。界面靠它决定「还要不要继续接收事件」，
// 少写一个就会出现「校验中却显示已结束」。
const IN_FLIGHT_STATUSES: ReadonlySet<ChatTurnStatus> = new Set([
  'QUEUED',
  'RUNNING',
  'WAITING_VALIDATION',
])

// 必须有人过问的两种说法：明确要求人工对账，以及 Worker 中途退出后没人能说清结果的。
const ATTENTION_STATUSES: ReadonlySet<ChatTurnStatus> = new Set(['MANUAL_REQUIRED', 'UNKNOWN'])

function isInFlight(turn: ChatTurnResponse | null): boolean {
  return turn ? IN_FLIGHT_STATUSES.has(turn.status) : false
}

function needsAttention(status: ChatTurnStatus | undefined): boolean {
  return status ? ATTENTION_STATUSES.has(status) : false
}

const FALLBACK_REASON_LABELS: Record<string, string> = {
  INVALID_RESPONSE: '主模型响应格式不合格',
  RATE_LIMIT: '主模型触发限流',
  TIMEOUT: '主模型超时',
  NETWORK: '主模型网络错误',
  UPSTREAM_5XX: '主模型服务错误',
  CIRCUIT_OPEN: '主模型熔断',
}

function eventLabel(event: ChatEvent): string {
  if (event.event_type === 'state_transitioned' && event.payload.to_status) {
    return `状态：${event.payload.from_status} → ${event.payload.to_status}`
  }
  if (event.event_type === 'model_called') {
    const model = event.payload.model || '未知模型'
    const fallbackReason = event.payload.fallback_reason
    if (fallbackReason && fallbackReason !== 'none') {
      const reason = FALLBACK_REASON_LABELS[fallbackReason] || fallbackReason
      return `正在调用降级模型 ${model} · ${reason}`
    }
    return `正在调用模型 ${model}`
  }
  if (event.event_type === 'intent_classified' && event.payload.task_type === 'scope_redirect') {
    return '当前话题与代码业务无关，正在自然引导回 CodeInsight'
  }
  if (event.event_type === 'model_generating' && event.payload.status === 'started') {
    return `模型正在生成 ${event.payload.route === 'general_chat' ? '普通对话' : '代码回答'}`
  }
  if (event.event_type === 'model_result' && event.payload.outcome === 'error') {
    const detail = event.payload.error_detail
      ? MODEL_ERROR_DETAILS[event.payload.error_detail] || event.payload.error_detail
      : event.payload.error_class || '未分类错误'
    const model = event.payload.model ? ` ${event.payload.model} ` : ''
    return `模型${model}调用失败 · ${detail}`
  }
  if (event.event_type === 'model_result' && event.payload.cache_hit_ratio) {
    const model = event.payload.model ? ` ${event.payload.model} ` : ''
    return `模型${model}已返回 · 本次 cache hit ${(Number(event.payload.cache_hit_ratio) * 100).toFixed(1)}%`
  }
  if (event.event_type === 'validation_started' && event.payload.mode === 'development_skipped') {
    return '开发模式：跳过 Docker 固定校验'
  }
  if (event.event_type === 'validation_finished' && event.payload.skipped === 'true') {
    return '开发模式：固定校验未执行'
  }
  return EVENT_LABELS[event.event_type] || event.event_type
}

function formatTime(epochMilliseconds: number): string {
  return new Date(epochMilliseconds).toLocaleTimeString('zh-CN', { hour12: false })
}

function textValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback
}

function statusClass(status: ChatTurnResponse['status']): string {
  return status.toLowerCase().replace(/_/g, '-')
}

const STATUS_LABELS: Record<ChatTurnResponse['status'], string> = {
  QUEUED: '排队中',
  RUNNING: '正在运行',
  WAITING_APPROVAL: '等待审批',
  WAITING_VALIDATION: '等待隔离校验',
  COMPLETED: '已完成',
  FAILED: '执行失败',
  CANCELLED: '已取消',
  UNKNOWN: '结果不明',
  MANUAL_REQUIRED: '需要人工确认',
}

function statusLabel(status: ChatTurnResponse['status']): string {
  return STATUS_LABELS[status] || status
}

const EXAMPLE_QUESTIONS = [
  'checkout 如何校验输入？',
  '这次请求从哪里进入系统？',
  '把这个校验提取成独立函数。',
]

function ChatResult({
  result,
  approvalPending,
  onApprove,
  approving,
}: {
  result: Record<string, unknown> | null | undefined
  approvalPending: boolean
  onApprove: () => void
  approving: boolean
}) {
  if (!result) return null
  const kind = textValue(result.kind)
  if (kind === 'change_preview') {
    const preview = (result.preview || {}) as Record<string, unknown>
    const development = (preview.development_mode || {}) as Record<string, unknown>
    const autoApprove = development.auto_approve_changes === true
    return (
      <div className="chat-result change-preview-card">
        <div className="chat-result-heading">
          <span className="result-tag">CHANGE PREVIEW</span>
          <strong>{textValue(preview.path, '待修改文件')}</strong>
        </div>
        <p>
          {autoApprove
            ? '开发模式将自动确认此预览；修改仍只会写入受控隔离 workspace。'
            : '修改只会写入受控隔离 workspace；确认 diff 后，点击批准才会应用。'}
        </p>
        <pre className="diff-preview">{textValue(preview.diff, '没有返回 diff')}</pre>
        {approvalPending && (
          <button className="primary-action approval-action" type="button" onClick={onApprove} disabled={approving}>
            {approving ? '正在应用并校验…' : '批准并应用修改'}
          </button>
        )}
      </div>
    )
  }
  if (kind === 'change_result') {
    const validation = (result.validation || {}) as Record<string, unknown>
    const validationSkipped = validation.skipped === true
    return (
      <div className="chat-result change-result-card">
        <div className="chat-result-heading">
          <span className="result-tag">CHANGE RESULT</span>
          <strong>{textValue(result.status, 'UNKNOWN')}</strong>
        </div>
        <p>{textValue(result.reason, '修改流程已结束。')}</p>
        {textValue(validation.error_class) && (
          <span className="result-meta">失败类型：{textValue(validation.error_class)}</span>
        )}
        {textValue(validation.message_excerpt) && (
          <p className="validation-excerpt">{textValue(validation.message_excerpt)}</p>
        )}
        <span className="result-meta">
          校验：{validationSkipped
            ? '开发模式已跳过'
            : validation.passed === true
              ? '通过'
              : validation.passed === false
                ? '未通过'
                : '未返回'}
        </span>
      </div>
    )
  }
  if (kind === 'validation_blocked' || kind === 'change_blocked') {
    const validation = (result.validation || {}) as Record<string, unknown>
    const blockedReason = kind === 'validation_blocked'
      ? '固定校验环境当前不可用，修改尚未开始。'
      : '本轮没有生成新的有效修改。'
    return (
      <div className="chat-result change-result-card">
        <div className="chat-result-heading">
          <span className="result-tag">CHANGE BLOCKED</span>
          <strong>{textValue(validation.error_class, 'ACTION_REQUIRED')}</strong>
        </div>
        <p>{blockedReason}</p>
        <p>{textValue(validation.message_excerpt, textValue(result.error, '请查看运行时间线中的具体原因。'))}</p>
        <span className="result-meta">
          {textValue(validation.next_action, '请修复校验环境后重新提交。')}
        </span>
      </div>
    )
  }
  if (kind === 'general_chat') {
    const observability = (result.observability || {}) as Record<string, unknown>
    return (
      <div className="chat-result chat-meta-result">
        <details className="chat-details">
          <summary>普通对话 · 查看模型与用量</summary>
          <span className="result-meta">
            生效模型：{textValue(result.model, '未返回')} · Route：{textValue(result.route, 'general_chat')}
          </span>
          <span className="result-meta">
            本轮 cache read/miss：{textValue(observability.cache_read_tokens, '0')} /{' '}
            {textValue(observability.cache_miss_tokens, '0')} · 命中率：
            {(Number(observability.cache_hit_ratio || 0) * 100).toFixed(1)}%
          </span>
        </details>
      </div>
    )
  }
  if (kind === 'scope_redirect') {
    return (
      <div className="chat-result chat-meta-result">
        <div className="chat-result-heading">
          <span className="result-tag">SCOPE REDIRECT</span>
          <strong>回到 CodeInsight</strong>
        </div>
        <span className="result-meta">本轮未调用模型，也未访问仓库；已保留当前 Session 上下文。</span>
      </div>
    )
  }
  if (kind === 'clarify') {
    return (
      <div className="chat-result chat-meta-result">
        <span className="result-tag">CLARIFY</span>
        <span className="result-meta">本轮未访问仓库，也未调用模型。</span>
      </div>
    )
  }
  const citations = Array.isArray(result.citations) ? result.citations : []
  const observability = (result.observability || {}) as Record<string, unknown>
  return (
    <div className="chat-result answer-result-card">
      <div className="chat-result-heading">
        <span className="result-tag">CODE ANSWER</span>
        <strong>{textValue(result.outcome, 'answered')}</strong>
      </div>
      <details className="chat-details">
        <summary>查看本轮证据与运行详情</summary>
        {citations.length > 0 && (
          <div className="citation-chips" aria-label="回答引用">
            {citations.map((item, index) => {
              const citation = (item || {}) as Record<string, unknown>
              return (
                <span className="citation-chip" key={`${textValue(citation.evidence_id)}-${index}`}>
                  {textValue(citation.evidence_id, `E${index + 1}`)} · {textValue(citation.relative_path)}:
                  {textValue(citation.start_line)}-{textValue(citation.end_line)}
                </span>
              )
            })}
          </div>
        )}
        <span className="result-meta">
          生效模型：{textValue(result.model, '未返回')} · Router：{textValue(result.router_model, '未返回')}
        </span>
        <span className="result-meta">
          本轮 cache read/miss：{textValue(observability.cache_read_tokens, '0')} /{' '}
          {textValue(observability.cache_miss_tokens, '0')} · 命中率：
          {(Number(observability.cache_hit_ratio || 0) * 100).toFixed(1)}%
        </span>
      </details>
    </div>
  )
}

function App() {
  const [activeView, setActiveView] = useState<'conversation' | 'observability'>('conversation')
  const [repositoryRoot, setRepositoryRoot] = useState('tests/fixtures/sample_repo')
  const [question, setQuestion] = useState('checkout 如何校验输入？')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeTurn, setActiveTurn] = useState<ChatTurnResponse | null>(null)
  const [events, setEvents] = useState<ChatEvent[]>([])
  const [reasoning, setReasoning] = useState<DebugReasoningEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [approving, setApproving] = useState(false)
  const [showDebugReasoning, setShowDebugReasoning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [usageRefreshToken, setUsageRefreshToken] = useState(0)
  const eventCursor = useRef(0)

  const running = isInFlight(activeTurn)
  const waitingValidation = activeTurn?.status === 'WAITING_VALIDATION'
  const pendingApproval = activeTurn?.status === 'WAITING_APPROVAL'
  const failed = activeTurn?.status === 'FAILED'
  const manualAttention = needsAttention(activeTurn?.status)
  const lastUserQuestion = [...messages].reverse().find((message) => message.role === 'user')?.content || ''

  function addAssistantMessage(turn: ChatTurnResponse) {
    const assistantMessage = turn.assistant_message
    if (!assistantMessage) return
    const resultKind = textValue(turn.result?.kind, turn.status.toLowerCase())
    const messageId = `assistant-${turn.turn_id}-${resultKind}`
    setMessages((current) => {
      if (current.some((message) => message.id === messageId)) return current
      return [
        ...current,
        {
          id: messageId,
          turnId: turn.turn_id,
          role: 'assistant',
          content: assistantMessage,
          taskType: turn.task_type,
          result: turn.result,
        },
      ]
    })
  }

  async function watchTurn(turn: ChatTurnResponse, afterSequence = 0): Promise<ChatTurnResponse> {
    setActiveTurn(turn)
    await streamChatEvents(
      turn.turn_id,
      (event) => {
        eventCursor.current = Math.max(eventCursor.current, event.sequence)
        setEvents((current) => (
          current.some((item) => item.event_id === event.event_id) ? current : [...current, event]
        ))
      },
      (event) => setReasoning((current) => [...current, event]),
      { afterSequence },
    )
    const completed = await getChatTurn(turn.turn_id)
    setActiveTurn(completed)
    addAssistantMessage(completed)
    return completed
  }

  async function submit() {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || loading || pendingApproval) return
    setLoading(true)
    setError(null)
    setEvents([])
    setReasoning([])
    eventCursor.current = 0
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, turnId: 'pending', role: 'user', content: trimmedQuestion },
    ])
    try {
      const session = sessionId
        ? { session_id: sessionId }
        : await createChatSession(repositoryRoot)
      if (!sessionId) setSessionId(session.session_id)
      const accepted = await submitChatTurn({
        session_id: session.session_id,
        repository_root: repositoryRoot,
        message: trimmedQuestion,
        client_turn_id: `client-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        show_debug_reasoning: showDebugReasoning,
      })
      setMessages((current) => {
        const last = current[current.length - 1]
        if (!last || last.role !== 'user') return current
        return [...current.slice(0, -1), { ...last, turnId: accepted.turn_id }]
      })
      await watchTurn(accepted)
      setUsageRefreshToken((current) => current + 1)
      setQuestion('')
    } catch (caught) {
      setActiveTurn((current) => (
        isInFlight(current) ? null : current
      ))
      setError(caught instanceof Error ? caught.message : '请求失败。')
    } finally {
      setLoading(false)
    }
  }

  async function approve() {
    if (!activeTurn || !pendingApproval || approving) return
    setApproving(true)
    setError(null)
    setReasoning([])
    try {
      const resumed = await approveChatTurn(activeTurn.turn_id)
      const completed = await watchTurn(resumed, eventCursor.current)
      if (completed.status === 'FAILED' || needsAttention(completed.status)) {
        setError(completed.error || '修改应用失败。')
      }
      setUsageRefreshToken((current) => current + 1)
    } catch (caught) {
      setActiveTurn((current) => (
        isInFlight(current) ? null : current
      ))
      setError(caught instanceof Error ? caught.message : '审批或应用失败。')
    } finally {
      setApproving(false)
    }
  }

  return (
    <main className="app-shell">
      <div className="workspace-shell">
        <aside className="app-sidebar" aria-label="CodeInsight 工作台">
          <div className="sidebar-brand">
            <div className="brand-mark" aria-hidden="true">CI</div>
            <div>
              <strong>CodeInsight</strong>
              <span>local workspace</span>
            </div>
          </div>

          <nav className="sidebar-nav" aria-label="页面导航">
            <button
              className={`sidebar-nav-item ${activeView === 'conversation' ? 'active' : ''}`}
              type="button"
              aria-label="Conversation"
              aria-current={activeView === 'conversation' ? 'page' : undefined}
              onClick={() => setActiveView('conversation')}
            >
              <span className="nav-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" role="presentation"><path d="M5 6.5h14M5 12h9M5 17.5h14" /></svg>
              </span>
              <span className="nav-copy"><strong>会话</strong><small>Conversation</small></span>
            </button>
            <button
              className={`sidebar-nav-item ${activeView === 'observability' ? 'active' : ''}`}
              type="button"
              aria-label="调用监控"
              aria-current={activeView === 'observability' ? 'page' : undefined}
              onClick={() => setActiveView('observability')}
            >
              <span className="nav-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" role="presentation"><path d="M6 5v14M6 12h7M13 12V7h5M13 12v5h5" /></svg>
              </span>
              <span className="nav-copy"><strong>调用监控</strong><small>Observability</small></span>
            </button>
          </nav>

          <div className="sidebar-divider" />
          <div className="sidebar-session-card">
            <div className="sidebar-session-top">
              <span className={`session-dot ${sessionId ? 'connected' : ''}`} />
              <span>{sessionId ? 'ACTIVE' : 'READY'}</span>
            </div>
            <strong>{sessionId ? `Session ${sessionId.slice(0, 12)}…` : '尚未创建 Session'}</strong>
            <span>一个用户 · 一个连续上下文</span>
          </div>
          <div className="sidebar-footer">
            <span className="local-dot" />
            <span><strong>LOCAL WORKSPACE</strong><small>events and traces stay here</small></span>
          </div>
        </aside>

        <section className="workspace-content">
          <header className="workspace-topbar">
            <h1>{activeView === 'conversation' ? '代码会话' : '调用监控'}</h1>
            <div className="topbar-meta">
              <span className="topbar-mode"><span className="pulse" /> 本地模式</span>
              <span className="topbar-session">{sessionId ? `Session ${sessionId.slice(0, 12)}…` : '未创建 Session'}</span>
            </div>
          </header>

          {activeView === 'conversation' ? (
            <div className="conversation-page">
              <section className="conversation-card" aria-label="CodeInsight 对话" aria-busy={running || pendingApproval}>
                <div className="conversation-header">
                  <div>
                    <div className="conversation-state"><span className={`live-dot ${activeTurn ? 'is-active' : 'is-idle'}`} /> {activeTurn ? 'Agent Run' : 'Ready for a question'}</div>
                    <h2>代码对话</h2>
                  </div>
                  {activeTurn && (
                    <span className={`status-pill ${statusClass(activeTurn.status)}`} role="status" aria-live="polite">
                      {statusLabel(activeTurn.status)}
                    </span>
                  )}
                </div>

                <div className="message-list" aria-live="polite" aria-busy={running}>
                  {messages.length === 0 && !activeTurn && (
                    <div className="empty-state conversation-empty">
                      <div className="empty-marker" aria-hidden="true"><span className="empty-marker-dot" /></div>
                      <h2>从一个代码问题开始</h2>
                      <p>下一轮会自动带上这一轮的公开对话上下文；理解和修改不再是两个孤立页面。</p>
                      <div className="empty-hints" aria-label="示例问题">
                        {EXAMPLE_QUESTIONS.map((example) => (
                          <button type="button" key={example} onClick={() => setQuestion(example)}>
                            {example}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {messages.map((message) => (
                    <article className={`message-bubble ${message.role}`} key={message.id}>
                      <div className="message-author">{message.role === 'user' ? '你' : 'CodeInsight'}</div>
                      <p>{message.content}</p>
                      {message.role === 'assistant' && (
                        <ChatResult
                          result={message.result}
                          approvalPending={pendingApproval && activeTurn?.turn_id === message.turnId}
                          onApprove={() => void approve()}
                          approving={approving}
                        />
                      )}
                    </article>
                  ))}
                </div>

                {activeTurn && (
                  running || pendingApproval || failed || manualAttention || reasoning.length > 0 || (showDebugReasoning && events.length > 0)
                ) && (
                  <section className="run-progress" aria-label="实时运行状态" aria-live="polite">
                    <div className="run-progress-heading">
                      <div className={`spinner ${running || pendingApproval ? '' : 'done'}`} aria-hidden="true"><span /></div>
                      <div>
                        <strong>
                          {pendingApproval
                            ? '等待你的审批'
                            : waitingValidation
                              ? '等待隔离校验返回'
                              : running
                                ? 'Agent 正在运行'
                                : manualAttention
                                  ? '这一轮需要人工确认'
                                  : failed
                                    ? '本轮执行失败'
                                    : '本轮已结束'}
                        </strong>
                        <span className="run-id">{activeTurn.run_id}</span>
                      </div>
                      <span className="run-stage">{events.length ? `${events.length} events` : 'waiting for events'}</span>
                    </div>
                    <ol className="event-timeline">
                      {events.slice(-8).map((event) => (
                        <li key={event.event_id}>
                          <span className="timeline-dot" />
                          <span>{eventLabel(event)}</span>
                          <time>{formatTime(event.occurred_at_epoch_ms)}</time>
                        </li>
                      ))}
                    </ol>
                    {showDebugReasoning && (
                      <details className="reasoning-panel" open={reasoning.length > 0}>
                        <summary>供应商原生 reasoning（仅本地调试实时显示）</summary>
                        {reasoning.length > 0 ? (
                          reasoning.map((item, index) => (
                            <div className="reasoning-block" key={`${item.occurred_at_epoch_ms}-${index}`}>
                              <span>{item.model}</span>
                              <p>{item.content}</p>
                            </div>
                          ))
                        ) : (
                          <p className="reasoning-empty">当前 Provider 尚未返回 reasoning_content/thinking。</p>
                        )}
                      </details>
                    )}
                  </section>
                )}

                {error && (
                  <div className="error-state inline-error" role="alert">
                    <div>
                      <strong>请求失败</strong>
                      <span>{error}</span>
                    </div>
                    {lastUserQuestion && (
                      <button className="text-action" type="button" onClick={() => setQuestion(lastUserQuestion)}>
                        带回输入框
                      </button>
                    )}
                  </div>
                )}

                <form
                  className="chat-composer"
                  onSubmit={(event) => {
                    event.preventDefault()
                    void submit()
                  }}
                >
                  <div className="composer-heading">
                    <span className="composer-title">下一轮对话</span>
                    <span className="composer-hint">Enter 发送一轮新的对话</span>
                  </div>
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="例如：刚才的 checkout 校验在哪里？或者：把这个校验提取成独立函数。"
                    rows={3}
                    disabled={loading || pendingApproval}
                    required
                  />
                  <div className="composer-footer">
                    <span>
                      {running
                        ? '正在接收实时事件，请稍候…'
                        : pendingApproval
                          ? '请先处理上面的修改审批。'
                          : manualAttention
                            ? '这一轮需要人工确认，请查看运行详情。'
                            : '支持中文、英文或中英混合问题'}
                    </span>
                    <button className="primary-action" disabled={loading || pendingApproval || !question.trim()} type="submit">
                      {loading ? '处理中…' : '发送消息'}
                    </button>
                  </div>
                </form>
              </section>

              <aside className="environment-card" aria-label="环境信息">
                <div className="environment-heading">
                  <h2>环境信息</h2>
                </div>
                <label>
                  <span>仓库路径</span>
                  <input
                    value={repositoryRoot}
                    onChange={(event) => setRepositoryRoot(event.target.value)}
                    placeholder="C:\\repos\\project"
                    aria-label="仓库路径"
                    aria-describedby="repository-path-hint"
                    maxLength={260}
                    required
                    disabled={running || pendingApproval}
                  />
                  <span id="repository-path-hint" className="field-hint">后端可访问的本地路径；默认示例可直接运行。</span>
                </label>
                <div className="session-status">
                  <span className={`session-dot ${sessionId ? 'connected' : ''}`} />
                  {sessionId ? `Session ${sessionId.slice(0, 18)}…` : '首次发送时自动创建 Session'}
                </div>
                <button
                  className={`debug-mode-button ${showDebugReasoning ? 'enabled' : ''}`}
                  type="button"
                  aria-pressed={showDebugReasoning}
                  disabled={running || pendingApproval}
                  onClick={() => setShowDebugReasoning((current) => !current)}
                >
                  <span className="debug-mode-dot" aria-hidden="true" />
                  <span className="debug-mode-copy">
                    <strong>{showDebugReasoning ? '调试模式' : '正常使用模式'}</strong>
                    <small>{showDebugReasoning ? '显示模型、降级链和原生 reasoning' : '只显示用户可读结果'}</small>
                  </span>
                  <span className="debug-mode-switch" aria-hidden="true"><span /></span>
                </button>
                <p className="environment-note">只有 API 明确返回 reasoning_content/thinking 时才会显示；未返回时不会本地编造。</p>
                <div className="environment-routes">
                  <strong>同一 Session 的两条路径</strong>
                  <span>代码理解 → 只读检索与引用回答</span>
                  <span>修改请求 → MCP 探索 → diff 预览 → 用户审批 → 隔离校验</span>
                </div>
              </aside>
            </div>
          ) : (
            <div className="monitoring-page">
              <ObservabilityDashboard refreshToken={usageRefreshToken} />
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
