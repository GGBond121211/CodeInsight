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
    kind: 'auto_answer',
    outcome: 'answered',
    answer: '校验逻辑有证据支持。',
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

  expect((await screen.findAllByText('校验逻辑有证据支持。')).length).toBeGreaterThanOrEqual(1)
  expect(screen.getByText('代码对话')).toBeInTheDocument()
  expect(submitChatTurn).toHaveBeenCalledWith(
    expect.objectContaining({
      session_id: 'session-1',
      repository_root: 'tests/fixtures/sample_repo',
      message: 'checkout 如何校验输入？',
      show_debug_reasoning: true,
    }),
  )
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
  expect(screen.getByText('本轮失败，请查看运行详情')).toBeInTheDocument()
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
