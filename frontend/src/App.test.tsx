import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'

import App from './App'
import { autoAnswerRepository } from './api/codeinsight'

vi.mock('./api/codeinsight', () => ({ autoAnswerRepository: vi.fn() }))

beforeEach(() => {
  vi.mocked(autoAnswerRepository).mockReset()
})

it('exposes Smart Answer as the only product mode', () => {
  render(<App />)

  expect(screen.getByRole('button', { name: 'Smart Answer' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Grounded answer' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Verified Agent' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Inspect retrieval' })).not.toBeInTheDocument()
})

it('submits the repository question through Auto Answer', async () => {
  vi.mocked(autoAnswerRepository).mockResolvedValue({
    outcome: 'partially_answered',
    answer: 'Validation is supported; payment is not evidenced.',
    citations: [],
    retrieval_mode: 'auto',
    model: 'fake-model',
    prompt_version: 'code-answer-v2',
    usage: { input_tokens: 20, output_tokens: 8 },
    plan: {
      original_question: 'checkout and payment?',
      language: 'mixed',
      normalized_question: 'Explain checkout and payment.',
      subquestions: [
        { question: 'How does checkout validate?', intent: 'implementation', retrieval_mode: 'bm25' },
      ],
      retrieval_modes: ['bm25'],
      execution_route: 'linear',
      confidence: 0.91,
      fallback_reason: null,
    },
    subquestions: [
      {
        question: 'How does checkout validate?',
        intent: 'implementation',
        retrieval_mode: 'bm25',
        outcome: 'answered',
        answer: 'It calls validate_request.',
        citations: [],
      },
    ],
    router_model: 'fake-router',
    router_usage: { input_tokens: 7, output_tokens: 3 },
    router_elapsed_milliseconds: 2,
    embedding_input_tokens: 20,
    fallback_reason: null,
    events: [{ sequence: 1, step: 'route', summary: 'Prepared one subquestion.' }],
  })
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: 'Run Smart Answer' }))

  expect(await screen.findByText('Router-guided explanation')).toBeInTheDocument()
  expect(screen.getByText('Public route plan')).toBeInTheDocument()
  expect(autoAnswerRepository).toHaveBeenCalledWith(
    expect.objectContaining({ repository_root: 'tests/fixtures/sample_repo', limit: 5 }),
  )
})
