import type { AutoAnswerResponse } from '../api/types'
import { EvidenceList } from './EvidenceList'

interface AutoAnswerPanelProps {
  result: AutoAnswerResponse
}

export function AutoAnswerPanel({ result }: AutoAnswerPanelProps) {
  const status = result.outcome === 'insufficient_evidence' ? 'INSUFFICIENT' : result.outcome

  return (
    <div className="agent-result-stack">
      <section className="result-card answer-card" aria-live="polite">
        <header className="result-header">
          <div>
            <span className="eyebrow">Smart Answer</span>
            <h2>Router-guided explanation</h2>
          </div>
          <span className="status-pill">{status.toUpperCase()}</span>
        </header>
        <p className="answer-copy">{result.answer}</p>
        <EvidenceList citations={result.citations} />
        <footer className="metadata-row">
          <span>{result.model ?? 'model not called'}</span>
          <span>{result.plan.execution_route}</span>
          <span>{result.plan.confidence.toFixed(2)} confidence</span>
          <span>
            router {result.router_usage.input_tokens}/{result.router_usage.output_tokens}
          </span>
          <span>embedding {result.embedding_input_tokens} input tokens</span>
        </footer>
      </section>

      <section className="result-card" aria-label="Router plan">
        <header className="result-header">
          <div>
            <span className="eyebrow">Public route plan</span>
            <h2>{result.plan.subquestions.length} subquestions</h2>
          </div>
          <span className="status-pill">{result.plan.retrieval_modes.join(' + ') || 'none'}</span>
        </header>
        <p className="field-hint">{result.fallback_reason ?? 'Router completed without fallback.'}</p>
        <div className="search-results">
          {result.subquestions.map((item, index) => (
            <article className="search-row" key={`${item.question}:${index}`}>
              <span className="rank">#{index + 1}</span>
              <div>
                <strong>{item.outcome}</strong>
                <span className="symbol-path">{item.intent}</span>
                <p>{item.question}</p>
                <p>{item.answer}</p>
                <EvidenceList citations={item.citations} />
              </div>
              <span className="line-range">{item.retrieval_mode}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
