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
      <div className="mode-tabs" aria-label="Query mode">
        <button className="mode-tab active" type="button" aria-current="page">
          Smart Answer
        </button>
        {/* Disabled entry controls retained conceptually for explicit restoration:
            Grounded answer -> answer
            Verified Agent -> agent
            Inspect retrieval -> search */}
      </div>

      <label>
        Repository path
        <input
          value={repositoryRoot}
          onChange={(event) => onRepositoryRootChange(event.target.value)}
          placeholder="C:\\repos\\project"
          required
        />
      </label>

      <label>
        Question
        <textarea
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="How does this repository validate incoming requests?"
          rows={4}
          required
        />
      </label>
      <p className="field-hint">
        HTTPX demo: use ../work/benchmarks/httpx/httpx and ask how the top-level get helper
        dispatches a request.
      </p>

      <div className="form-footer">
        <button className="primary-action" disabled={loading} type="submit">
          {loading ? 'Working…' : 'Run Smart Answer'}
        </button>
      </div>
    </form>
  )
}
