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
          <span className="eyebrow">仓库回答</span>
          <h2>{insufficient ? '证据不足' : '基于证据的解释'}</h2>
        </div>
        <span className={insufficient ? 'status-pill warning' : 'status-pill'}>
          {insufficient ? 'INSUFFICIENT' : 'ANSWERED'}
        </span>
      </header>

      <p className="answer-copy">{result.answer}</p>
      <EvidenceList citations={result.citations} />

      <footer className="metadata-row">
        <span>{result.model ?? '未调用模型'}</span>
        <span>{result.prompt_version}</span>
        <span>{result.retrieval_mode}</span>
        <span>
          {result.usage.input_tokens} in / {result.usage.output_tokens} out
        </span>
      </footer>
    </section>
  )
}
