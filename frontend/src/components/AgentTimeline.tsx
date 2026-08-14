import type { AgentEvent } from '../api/types'

interface AgentTimelineProps {
  events: AgentEvent[]
  revisions: number
}

export function AgentTimeline({ events, revisions }: AgentTimelineProps) {
  return (
    <section className="result-card timeline-card" aria-labelledby="timeline-heading">
      <header className="timeline-header">
        <div>
          <span className="eyebrow">LangGraph 执行</span>
          <h2 id="timeline-heading">公开 Agent 时间线</h2>
        </div>
        <span className="revision-count">{revisions} 次修订</span>
      </header>
      <ol className="timeline-list">
        {events.map((event) => (
          <li key={event.sequence}>
            <span className="timeline-index">{event.sequence}</span>
            <div>
              <strong>{event.step.replace('_', ' ')}</strong>
              <p>{event.summary}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
