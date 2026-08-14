import type { Citation } from '../api/types'

interface EvidenceListProps {
  citations: Citation[]
}

export function EvidenceList({ citations }: EvidenceListProps) {
  if (!citations.length) return null

  return (
    <section className="evidence-section" aria-labelledby="evidence-heading">
      <div className="section-label" id="evidence-heading">
        Verified evidence
      </div>
      <div className="evidence-list">
        {citations.map((citation) => (
          <article className="evidence-row" key={citation.evidence_id}>
            <span className="evidence-id">{citation.evidence_id}</span>
            <code>{citation.relative_path}</code>
            <span className="line-range">
              L{citation.start_line}–L{citation.end_line}
            </span>
          </article>
        ))}
      </div>
    </section>
  )
}
