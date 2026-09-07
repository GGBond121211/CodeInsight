import { afterEach, expect, it, vi } from 'vitest'

import { streamChatEvents } from './chat'

afterEach(() => {
  vi.unstubAllGlobals()
})

it('解析阶段 SSE 和临时 reasoning 事件', async () => {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      controller.enqueue(
        encoder.encode(
          'id: run-1:1\nevent: retrieval_started\ndata: {"sequence":1,"event_type":"retrieval_started"}\n\n',
        ),
      )
      controller.enqueue(
        encoder.encode(
          'event: debug_reasoning\ndata: {"model":"fake","content":"正在分析"}\n\n',
        ),
      )
      controller.close()
    },
  })
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status: 200 })))

  const events: string[] = []
  const reasoning: string[] = []
  await streamChatEvents(
    'turn-1',
    (event) => events.push(event.event_type),
    (event) => reasoning.push(event.content),
  )

  expect(events).toEqual(['retrieval_started'])
  expect(reasoning).toEqual(['正在分析'])
})
