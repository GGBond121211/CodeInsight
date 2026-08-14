import { afterEach, describe, expect, it, vi } from 'vitest'

import { autoAnswerRepository } from './codeinsight'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('CodeInsight API 客户端', () => {
  it('只向 Auto Answer 产品接口发送请求', async () => {
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

  it('向调用方传递 API detail 错误信息', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: '模型请求失败' }), {
          status: 502,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    await expect(
      autoAnswerRepository({ repository_root: 'repo', question: 'question', limit: 5 }),
    ).rejects.toThrow('模型请求失败')
  })
})
