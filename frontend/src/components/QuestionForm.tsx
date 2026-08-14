import type { FormEvent } from 'react'

interface QuestionFormProps {
  repositoryRoot: string
  question: string
  loading: boolean
  onRepositoryRootChange: (value: string) => void
  onQuestionChange: (value: string) => void
  onSubmit: () => void
}

export function QuestionForm({
  repositoryRoot,
  question,
  loading,
  onRepositoryRootChange,
  onQuestionChange,
  onSubmit,
}: QuestionFormProps) {
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onSubmit()
  }

  return (
    <form className="question-card" onSubmit={submit}>
      <div className="mode-tabs" aria-label="查询模式">
        <button className="mode-tab active" type="button" aria-current="page">
          Smart Answer
        </button>
        {/* 为将来明确恢复而保留的停用入口概念：
            基于证据的回答 -> answer
            已核验的 Agent -> agent
            检查检索 -> search */}
      </div>

      <label>
        仓库路径
        <input
          value={repositoryRoot}
          onChange={(event) => onRepositoryRootChange(event.target.value)}
          placeholder="C:\\repos\\project"
          required
        />
      </label>

      <label>
        问题
        <textarea
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="这个仓库如何校验传入请求？"
          rows={4}
          required
        />
      </label>
      <p className="field-hint">
        HTTPX 演示：使用 ../work/benchmarks/httpx/httpx，并询问顶层 get helper 如何分发请求。
      </p>

      <div className="form-footer">
        <button className="primary-action" disabled={loading} type="submit">
          {loading ? '处理中…' : '运行 Smart Answer'}
        </button>
      </div>
    </form>
  )
}
