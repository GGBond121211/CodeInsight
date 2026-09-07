import { useRef, useState } from 'react'

import {
  approveChatTurn,
  createChatSession,
  getChatTurn,
  streamChatEvents,
  submitChatTurn,
} from './api/chat'
import type { ChatEvent, ChatTurnResponse, DebugReasoningEvent } from './api/types'
import { ObservabilityDashboard } from './components/ObservabilityDashboard'
import './App.css'

interface ChatMessage {
  id: string
  turnId: string
  role: 'user' | 'assistant'
  content: string
  taskType?: 'general_chat' | 'clarify' | 'explain' | 'change'
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
  const [repositoryRoot, setRepositoryRoot] = useState('tests/fixtures/sample_repo')
  const [question, setQuestion] = useState('checkout 如何校验输入？')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeTurn, setActiveTurn] = useState<ChatTurnResponse | null>(null)
  const [events, setEvents] = useState<ChatEvent[]>([])
  const [reasoning, setReasoning] = useState<DebugReasoningEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [approving, setApproving] = useState(false)
  const [showDebugReasoning, setShowDebugReasoning] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [usageRefreshToken, setUsageRefreshToken] = useState(0)
  const eventCursor = useRef(0)

  const running = activeTurn?.status === 'QUEUED' || activeTurn?.status === 'RUNNING'
  const pendingApproval = activeTurn?.status === 'WAITING_APPROVAL'
  const failed = activeTurn?.status === 'FAILED'

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
        current?.status === 'QUEUED' || current?.status === 'RUNNING' ? null : current
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
      if (completed.status === 'FAILED') setError(completed.error || '修改应用失败。')
      setUsageRefreshToken((current) => current + 1)
    } catch (caught) {
      setActiveTurn((current) => (
        current?.status === 'QUEUED' || current?.status === 'RUNNING' ? null : current
      ))
      setError(caught instanceof Error ? caught.message : '审批或应用失败。')
    } finally {
      setApproving(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="brand-mark" aria-hidden="true">CI</div>
        <div>
          <span className="eyebrow">统一 Session · 代码理解与修改</span>
          <h1>CodeInsight</h1>
          <p>在一个对话中连续理解代码、追问上下文，并在确认 diff 后安全修改。</p>
        </div>
        <div className="local-badge"><span className="pulse" /> 本地调试模式</div>
      </header>

      <div className="conversation-layout">
        <aside className="question-card conversation-controls">
          <div className="mode-tabs" aria-label="对话模式">
            <button className="mode-tab active" type="button" aria-current="page">Conversation</button>
          </div>
          <label>
            仓库路径
            <input
              value={repositoryRoot}
              onChange={(event) => setRepositoryRoot(event.target.value)}
              placeholder="C:\\repos\\project"
              required
              disabled={running || pendingApproval}
            />
          </label>
          <div className="session-status">
            <span className={`session-dot ${sessionId ? 'connected' : ''}`} />
            {sessionId ? `Session ${sessionId.slice(0, 18)}…` : '尚未创建 Session，首次发送时自动创建'}
          </div>
          <label className="debug-toggle">
            <input
              type="checkbox"
              checked={showDebugReasoning}
              onChange={(event) => setShowDebugReasoning(event.target.checked)}
            />
            <span>调试显示模型、降级链和供应商原生 reasoning</span>
          </label>
          <p className="field-hint">
            只有 API 明确返回 reasoning_content/thinking 时才会显示；未返回时会明确标记，不会本地编造。
          </p>
          <div className="control-note">
            <strong>同一 Session 的两条路径</strong>
            <span>普通聊天 → 自然语言回答，不访问仓库</span>
            <span>代码理解 → 只读检索与引用回答</span>
            <span>修改请求 → MCP 探索 → diff 预览 → 开发模式自动审批/生产模式人工审批 → 隔离校验</span>
          </div>
        </aside>

        <section className="conversation-card" aria-label="CodeInsight 对话">
          <div className="conversation-header">
            <div>
              <span className="section-label">Live Agent Run</span>
              <h2>代码对话</h2>
            </div>
            {activeTurn && <span className={`status-pill ${pendingApproval ? 'warning' : ''}`}>{activeTurn.status}</span>}
          </div>

          <div className="message-list" aria-live="polite">
            {messages.length === 0 && !activeTurn && (
              <div className="empty-state conversation-empty">
                <span className="empty-glyph">⌁</span>
                <h2>从一个代码问题开始</h2>
                <p>下一轮会自动带上这一轮的公开对话上下文；理解和修改不再是两个孤立页面。</p>
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
            running || pendingApproval || failed || reasoning.length > 0 || (showDebugReasoning && events.length > 0)
          ) && (
            <section className="run-progress" aria-label="实时运行状态">
              <div className="run-progress-heading">
                <div className={`spinner ${running || pendingApproval ? '' : 'done'}`} aria-hidden="true" />
                <div>
                  <strong>
                    {pendingApproval
                      ? '等待你的审批'
                      : running
                        ? 'Agent 正在运行'
                        : failed
                          ? '本轮失败，请查看运行详情'
                          : '本轮已结束'}
                  </strong>
                  <span>{activeTurn.run_id}</span>
                </div>
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

          {error && <div className="error-state inline-error" role="alert"><strong>请求失败</strong><span>{error}</span></div>}

          <form
            className="chat-composer"
            onSubmit={(event) => {
              event.preventDefault()
              void submit()
            }}
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="例如：刚才的 checkout 校验在哪里？或者：把这个校验提取成独立函数。"
              rows={3}
              disabled={loading || pendingApproval}
              required
            />
            <div className="composer-footer">
              <span>{running ? '正在接收实时事件，请稍候…' : pendingApproval ? '请先处理上面的修改审批。' : 'Enter 发送一轮新的对话'}</span>
              <button className="primary-action" disabled={loading || pendingApproval || !question.trim()} type="submit">
                {loading ? '处理中…' : '发送消息'}
              </button>
            </div>
          </form>
        </section>
      </div>

      <ObservabilityDashboard refreshToken={usageRefreshToken} />
    </main>
  )
}

export default App
