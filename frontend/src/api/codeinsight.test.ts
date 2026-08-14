import { afterEach, describe, expect, it, vi } from 'vitest'

import { autoAnswerRepository } from './codeinsight'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('CodeInsight API client', () => {
  it('posts only to the Auto Answer product endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ outcome: 'answered' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await autoAnswerRepository({
      repository_root: 'repo',
      question: 'Where is checkout?',
      limit: 5,
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/auto/answer',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('surfaces the API detail message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'model request failed' }), {
          status: 502,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    await expect(
      autoAnswerRepository({ repository_root: 'repo', question: 'question', limit: 5 }),
    ).rejects.toThrow('model request failed')
  })
})
