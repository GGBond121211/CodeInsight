import type { StructuredEvidenceRow } from '../api/types'

interface StructuredEvidenceTableProps {
  rows: StructuredEvidenceRow[]
  truncated?: boolean
}

function locationLabel(row: StructuredEvidenceRow): string {
  if (row.start_line === null) return row.path
  if (row.end_line === null || row.end_line === row.start_line) {
    return `${row.path}:L${row.start_line}`
  }
  return `${row.path}:L${row.start_line}-L${row.end_line}`
}

// Q-012 U4：结构化结果区。每一行都由后端从真实工具结果映射，模型写不出也改不动，
// 所以这里展示的是「可核验事实」，与正文里的自然语言解释分开呈现。
export function StructuredEvidenceTable({ rows, truncated = false }: StructuredEvidenceTableProps) {
  if (!rows.length) return null

  return (
    <section className="structured-evidence" aria-labelledby="structured-evidence-heading">
      <h4 className="structured-evidence-heading" id="structured-evidence-heading">
        结构化事实
      </h4>
      <table className="structured-evidence-table">
        <thead>
          <tr>
            <th scope="col">类型</th>
            <th scope="col">位置</th>
            <th scope="col">符号</th>
            <th scope="col">来源工具</th>
            <th scope="col">证据</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.kind}-${row.path}-${row.start_line}-${index}`}>
              <td>{row.kind_label}</td>
              <td>
                <code>{locationLabel(row)}</code>
              </td>
              <td>{row.symbol || '—'}</td>
              <td>{row.source_tool}</td>
              <td>{row.evidence_id || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {truncated && (
        <p className="structured-evidence-note">只显示前若干行；完整列表已截断。</p>
      )}
    </section>
  )
}
