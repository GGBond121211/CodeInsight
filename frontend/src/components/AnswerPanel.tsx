import type { AnswerResponse } from '../api/types'
import { EvidenceList } from './EvidenceList'

interface AnswerPanelProps {
  result: AnswerResponse
}

export function AnswerPanel({ result }: AnswerPanelProps) {
  const insufficient = result.outcome === 'insufficient_evidence'

  return (
    <section className="result-card answer-card" aria-live="polite">
      <header className="result-header">
        <div>
          <span className="eyebrow">Repository answer</span>
          <h2>{insufficient ? 'Evidence is insufficient' : 'Grounded explanation'}</h2>
        </div>
        <span className={insufficient ? 'status-pill warning' : 'status-pill'}>
          {insufficient ? 'INSUFFICIENT' : 'ANSWERED'}
        </span>
      </header>

      <p className="answer-copy">{result.answer}</p>
      <EvidenceList citations={result.citations} />

      <footer className="metadata-row">
        <span>{result.model ?? 'model not called'}</span>
        <span>{result.prompt_version}</span>
        <span>{result.retrieval_mode}</span>
        <span>
          {result.usage.input_tokens} in / {result.usage.output_tokens} out
        </span>
      </footer>
    </section>
  )
}
