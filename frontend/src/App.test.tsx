import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'

import App from './App'
import { autoAnswerRepository } from './api/codeinsight'

vi.mock('./api/codeinsight', () => ({ autoAnswerRepository: vi.fn() }))

beforeEach(() => {
  vi.mocked(autoAnswerRepository).mockReset()
})

it('将 Smart Answer 作为唯一产品模式公开', () => {
  render(<App />)

  expect(screen.getByRole('button', { name: 'Smart Answer' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '基于证据的回答' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '已核验的 Agent' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '检查检索' })).not.toBeInTheDocument()
})

it('通过 Auto Answer 提交仓库问题', async () => {
  vi.mocked(autoAnswerRepository).mockResolvedValue({
    outcome: 'partially_answered',
    answer: '校验逻辑有证据支持；支付逻辑没有证据支持。',
    citations: [],
    retrieval_mode: 'auto',
    model: 'fake-model',
    prompt_version: 'code-answer-v2',
    usage: { input_tokens: 20, output_tokens: 8 },
    plan: {
      original_question: 'checkout 和支付？',
      language: 'mixed',
      normalized_question: '解释 checkout 和支付。',
      subquestions: [
        { question: 'checkout 如何校验？', intent: 'implementation', retrieval_mode: 'bm25' },
      ],
      retrieval_modes: ['bm25'],
      execution_route: 'linear',
      confidence: 0.91,
      fallback_reason: null,
    },
    subquestions: [
      {
        question: 'checkout 如何校验？',
        intent: 'implementation',
        retrieval_mode: 'bm25',
        outcome: 'answered',
        answer: '它调用 validate_request。',
        citations: [],
      },
    ],
    router_model: 'fake-router',
    router_usage: { input_tokens: 7, output_tokens: 3 },
    router_elapsed_milliseconds: 2,
    embedding_input_tokens: 20,
    fallback_reason: null,
    events: [{ sequence: 1, step: 'route', summary: '已准备 1 个子问题。' }],
  })
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: '运行 Smart Answer' }))

  expect(await screen.findByText('Router 引导的解释')).toBeInTheDocument()
  expect(screen.getByText('公开执行计划')).toBeInTheDocument()
  expect(autoAnswerRepository).toHaveBeenCalledWith(
    expect.objectContaining({ repository_root: 'tests/fixtures/sample_repo', limit: 5 }),
  )
})
