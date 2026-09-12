import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'

import App from './App'
import {
  createChatSession,
  getChatTurn,
  streamChatEvents,
  submitChatTurn,
} from './api/chat'
import type { ChatTurnResponse } from './api/types'
import { usageCalls, usageSummary } from './api/usage'

vi.mock('./api/chat', () => ({
  approveChatTurn: vi.fn(),
  createChatSession: vi.fn(),
  getChatTurn: vi.fn(),
  streamChatEvents: vi.fn(),
  submitChatTurn: vi.fn(),
}))
vi.mock('./api/usage', () => ({ usageCalls: vi.fn(), usageSummary: vi.fn() }))

const completedTurn: ChatTurnResponse = {
  turn_id: 'turn-1',
  session_id: 'session-1',
  run_id: 'run-1',
  task_type: 'explain',
  status: 'COMPLETED',
  user_message: 'checkout 如何校验输入？',
  assistant_message: '校验逻辑有证据支持。',
  result: {
    kind: 'code_answer',
    outcome: 'answered',
    citations: [],
    observability: {
      cache_read_tokens: 20,
      cache_miss_tokens: 4,
      cache_hit_ratio: 0.83,
    },
  },
  error: null,
  reasoning_available: false,
  created_at_epoch_ms: 1,
  updated_at_epoch_ms: 2,
}

beforeEach(() => {
  vi.mocked(createChatSession).mockReset()
  vi.mocked(getChatTurn).mockReset()
  vi.mocked(streamChatEvents).mockReset()
  vi.mocked(submitChatTurn).mockReset()
  vi.mocked(usageSummary).mockResolvedValue({
    requests: 0,
    provider_attempts: 0,
    successful_attempts: 0,
    semantic_cache_hits: 0,
    input_tokens: 0,
    cache_read_tokens: 0,
    cache_miss_tokens: 0,
    cache_write_tokens: null,
    output_tokens: 0,
    cache_hit_ratio: 0,
    estimated_cost_stars: '0',
    unknown_usage_count: 0,
    usage_coverage: 0,
  })
  vi.mocked(usageCalls).mockResolvedValue({ object: 'list', data: [] })
})

it('将统一 Conversation 作为唯一产品入口公开', () => {
  render(<App />)

  expect(screen.getByRole('button', { name: 'Conversation' })).toBeInTheDocument()
  expect(screen.getByRole('textbox', { name: '仓库路径' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '发送消息' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Smart Answer' })).not.toBeInTheDocument()
})

it('提交一轮消息并在实时事件结束后展示回答', async () => {
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...completedTurn, status: 'QUEUED', assistant_message: null, result: null })
  vi.mocked(streamChatEvents).mockImplementation(async (_turnId, onEvent) => {
    onEvent({
      event_id: 'run-1:1',
      run_id: 'run-1',
      sequence: 1,
      event_type: 'retrieval_started',
      occurred_at_epoch_ms: 1,
      payload: { route: 'auto' },
    })
  })
  vi.mocked(getChatTurn).mockResolvedValue(completedTurn)

  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findAllByText('校验逻辑有证据支持。')).toHaveLength(1)
  expect(screen.getByText('查看本轮证据与运行详情')).toBeInTheDocument()
  expect(screen.getByText('代码对话')).toBeInTheDocument()
  expect(submitChatTurn).toHaveBeenCalledWith(
    expect.objectContaining({
      session_id: 'session-1',
      repository_root: 'tests/fixtures/sample_repo',
      message: 'checkout 如何校验输入？',
      show_debug_reasoning: false,
    }),
  )
})

it('普通聊天只展示 assistant_message，不显示 CODE ANSWER 卡片', async () => {
  const generalTurn: ChatTurnResponse = {
    ...completedTurn,
    task_type: 'general_chat',
    user_message: '你好',
    assistant_message: '你好！我是 CodeInsight。',
    result: {
      kind: 'general_chat',
      outcome: 'answered',
      route: 'general_chat',
      model: 'deepseek-v4-flash',
      observability: { cache_hit_ratio: 0 },
    },
  }
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...generalTurn, status: 'QUEUED', assistant_message: null, result: null })
  vi.mocked(getChatTurn).mockResolvedValue(generalTurn)

  const user = userEvent.setup()
  render(<App />)
  const composer = screen.getByPlaceholderText('例如：刚才的 checkout 校验在哪里？或者：把这个校验提取成独立函数。')
  await user.clear(composer)
  await user.type(composer, '你好')
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('你好！我是 CodeInsight。')).toBeInTheDocument()
  expect(screen.queryByText('CODE ANSWER')).not.toBeInTheDocument()
  expect(screen.getByText('普通对话 · 查看模型与用量')).toBeInTheDocument()
})

it('业务外话题自然引导回 CodeInsight，并展示未调用模型', async () => {
  const redirectTurn: ChatTurnResponse = {
    ...completedTurn,
    task_type: 'scope_redirect',
    user_message: '我喜欢打篮球',
    assistant_message: '收到，我理解你是在分享一个日常话题。',
    result: {
      kind: 'scope_redirect',
      outcome: 'redirected',
      route: 'scope_redirect',
      reason: 'out_of_scope',
      model_called: false,
      prompt_version: 'scope-redirect-v1',
    },
  }
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...redirectTurn, status: 'QUEUED', assistant_message: null, result: null })
  vi.mocked(getChatTurn).mockResolvedValue(redirectTurn)

  const user = userEvent.setup()
  render(<App />)
  const composer = screen.getByPlaceholderText('例如：刚才的 checkout 校验在哪里？或者：把这个校验提取成独立函数。')
  await user.clear(composer)
  await user.type(composer, '我喜欢打篮球')
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('收到，我理解你是在分享一个日常话题。')).toBeInTheDocument()
  expect(screen.getByText('SCOPE REDIRECT')).toBeInTheDocument()
  expect(screen.getByText('本轮未调用模型，也未访问仓库；已保留当前 Session 上下文。')).toBeInTheDocument()
})

it('失败后仍保留模型失败原因和实时事件', async () => {
  const failedTurn: ChatTurnResponse = {
    ...completedTurn,
    status: 'FAILED',
    assistant_message: '本轮执行失败：结构化输出不是有效 JSON',
    result: null,
    error: '结构化输出不是有效 JSON',
  }
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...failedTurn, status: 'QUEUED', assistant_message: null })
  vi.mocked(streamChatEvents).mockImplementation(async (_turnId, onEvent) => {
    onEvent({
      event_id: 'run-1:1',
      run_id: 'run-1',
      sequence: 1,
      event_type: 'model_called',
      occurred_at_epoch_ms: 1,
      payload: {
        model: 'gpt-5.4-mini',
        fallback_reason: 'INVALID_RESPONSE',
        max_output_tokens: '40960',
      },
    })
    onEvent({
      event_id: 'run-1:2',
      run_id: 'run-1',
      sequence: 2,
      event_type: 'model_result',
      occurred_at_epoch_ms: 1,
      payload: {
        model: 'deepseek-v4-flash',
        outcome: 'error',
        error_class: 'INVALID_RESPONSE',
        error_detail: 'structured_output_invalid_json',
      },
    })
  })
  vi.mocked(getChatTurn).mockResolvedValue(failedTurn)

  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('正在调用降级模型 gpt-5.4-mini · 主模型响应格式不合格')).toBeInTheDocument()
  expect(screen.getByText('模型 deepseek-v4-flash 调用失败 · 结构化输出不是有效 JSON')).toBeInTheDocument()
  expect(screen.getByText('本轮执行失败')).toBeInTheDocument()
})

it('修改前 Sandbox 不可用时展示具体阻塞原因和恢复动作', async () => {
  const blockedTurn: ChatTurnResponse = {
    ...completedTurn,
    status: 'FAILED',
    assistant_message: '修改前无法运行固定校验：SANDBOX_PERMISSION_DENIED',
    result: {
      kind: 'validation_blocked',
      validation: {
        error_class: 'SANDBOX_PERMISSION_DENIED',
        message_excerpt: 'docker_engine: Access is denied',
        next_action: '请先恢复 Sandbox，再重新提交修改。',
      },
    },
    error: '修改前无法运行固定校验：SANDBOX_PERMISSION_DENIED',
  }
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...blockedTurn, status: 'QUEUED', assistant_message: null })
  vi.mocked(streamChatEvents).mockImplementation(async (_turnId, onEvent) => {
    onEvent({
      event_id: 'run-1:1',
      run_id: 'run-1',
      sequence: 1,
      event_type: 'validation_preflight',
      occurred_at_epoch_ms: 1,
      payload: {
        available: 'false',
        error_class: 'SANDBOX_PERMISSION_DENIED',
      },
    })
  })
  vi.mocked(getChatTurn).mockResolvedValue(blockedTurn)

  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('CHANGE BLOCKED')).toBeInTheDocument()
  expect(screen.getByText('docker_engine: Access is denied')).toBeInTheDocument()
  expect(screen.getByText('请先恢复 Sandbox，再重新提交修改。')).toBeInTheDocument()
  expect(screen.getByText('正在检查 Sandbox 校验环境')).toBeInTheDocument()
})

it('等待隔离校验的轮次仍算进行中，不显示成已结束', async () => {
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...completedTurn, status: 'QUEUED', assistant_message: null, result: null })
  vi.mocked(streamChatEvents).mockResolvedValue(undefined)
  vi.mocked(getChatTurn).mockResolvedValue({
    ...completedTurn,
    status: 'WAITING_VALIDATION',
    assistant_message: null,
    result: null,
  })

  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('等待隔离校验')).toBeInTheDocument()
  expect(screen.getByText('等待隔离校验返回')).toBeInTheDocument()
  expect(screen.getByText('正在接收实时事件，请稍候…')).toBeInTheDocument()
  expect(screen.queryByText('本轮已结束')).not.toBeInTheDocument()
})

it('需要人工确认的轮次给出明确提示，而不是当成普通失败', async () => {
  vi.mocked(createChatSession).mockResolvedValue({
    session_id: 'session-1',
    repo_id: 'repo-1',
    index_version: 'conversation-scan-v1',
    status: 'READY',
    summary: null,
    compacted_through_sequence: 0,
    active_goal: null,
    recent_turns: [],
    cache_hit: false,
    cache_fallback: false,
  })
  vi.mocked(submitChatTurn).mockResolvedValue({ ...completedTurn, status: 'QUEUED', assistant_message: null, result: null })
  vi.mocked(streamChatEvents).mockResolvedValue(undefined)
  vi.mocked(getChatTurn).mockResolvedValue({
    ...completedTurn,
    status: 'MANUAL_REQUIRED',
    assistant_message: null,
    result: null,
    error: 'DISPATCH_UNKNOWN',
  })

  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: '发送消息' }))

  expect(await screen.findByText('需要人工确认')).toBeInTheDocument()
  expect(screen.getByText('这一轮需要人工确认')).toBeInTheDocument()
  expect(screen.getByText('这一轮需要人工确认，请查看运行详情。')).toBeInTheDocument()
})
