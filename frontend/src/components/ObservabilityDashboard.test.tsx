import { render, screen } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import { usageCalls, usageSummary } from '../api/usage'
import { ObservabilityDashboard } from './ObservabilityDashboard'

vi.mock('../api/usage', () => ({ usageCalls: vi.fn(), usageSummary: vi.fn() }))

beforeEach(() => {
  vi.mocked(usageSummary).mockResolvedValue({
    requests: 2,
    provider_attempts: 2,
    successful_attempts: 2,
    semantic_cache_hits: 0,
    input_tokens: 100,
    cache_read_tokens: 70,
    cache_miss_tokens: 30,
    cache_write_tokens: null,
    output_tokens: 12,
    cache_hit_ratio: 0.7,
    estimated_cost_stars: '0.0184',
    unknown_usage_count: 0,
    usage_coverage: 1,
  })
  vi.mocked(usageCalls).mockResolvedValue({
    object: 'list',
    data: [
      {
        recorded_at_epoch_ms: 1_700_000_000_000,
        request_id: 'request-1234567890',
        attempt_id: 'attempt-1234567890',
        scene: 'explain',
        prompt_version: 'prompt-v1',
        provider: 'deepseek',
        model_tier: 'high',
        model: 'deepseek-v4-flash',
        status: 'ok',
        input_tokens: 100,
        cache_write_tokens: null,
        cache_read_tokens: 70,
        cache_miss_tokens: 30,
        output_tokens: 12,
        cache_hit_ratio: 0.7,
        estimated_cost_stars: '0.0184',
        latency_milliseconds: 420,
        fallback_reason: null,
        error_class: null,
        usage_source: 'provider_native',
        request_fingerprint: 'a'.repeat(64),
        stable_prefix_fingerprint: 'b'.repeat(64),
      },
    ],
  })
})

it('展示汇总、供应商缓存来源和调用明细', async () => {
  render(<ObservabilityDashboard refreshToken={0} />)

  expect(await screen.findByText('70.0%')).toBeInTheDocument()
  expect(screen.getAllByText('☆ 0.0184')).toHaveLength(2)
  expect(screen.getByText('供应商原生')).toBeInTheDocument()
  expect(screen.getByText('deepseek-v4-flash')).toBeInTheDocument()
  expect(screen.getAllByText('—').length).toBeGreaterThan(0)
})
