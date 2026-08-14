import { useState } from 'react'

import { autoAnswerRepository } from './api/codeinsight'
import type { AutoAnswerResponse } from './api/types'
import { AutoAnswerPanel } from './components/AutoAnswerPanel'
import { QuestionForm } from './components/QuestionForm'
import './App.css'

type ViewResult = { kind: 'auto'; value: AutoAnswerResponse }

function App() {
  const [repositoryRoot, setRepositoryRoot] = useState('tests/fixtures/sample_repo')
  const [question, setQuestion] = useState('How does checkout validate input?')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ViewResult | null>(null)

  async function submit() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult({
        kind: 'auto',
        value: await autoAnswerRepository({ repository_root: repositoryRoot, question, limit: 5 }),
      })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The request failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="brand-mark" aria-hidden="true">
          CI
        </div>
        <div>
          <span className="eyebrow">Local repository intelligence</span>
          <h1>CodeInsight</h1>
          <p>
            Ask how unfamiliar code works, then verify every conclusion against repository paths
            and line ranges.
          </p>
        </div>
        <div className="local-badge">
          <span className="pulse" /> Local read-only demo
        </div>
      </header>

      <div className="workspace-grid">
        <QuestionForm
          repositoryRoot={repositoryRoot}
          question={question}
          loading={loading}
          onRepositoryRootChange={setRepositoryRoot}
          onQuestionChange={setQuestion}
          onSubmit={submit}
        />

        <section className="result-area">
          {!result && !error && !loading && (
            <div className="empty-state">
              <span className="empty-glyph">⌁</span>
              <h2>Evidence will appear here</h2>
              <p>CodeInsight never asks the model to invent a filename or line number.</p>
            </div>
          )}
          {loading && <div className="loading-state">Scanning, ranking, and grounding…</div>}
          {error && (
            <div className="error-state" role="alert">
              <strong>Request failed</strong>
              <span>{error}</span>
            </div>
          )}
          {result?.kind === 'auto' && <AutoAnswerPanel result={result.value} />}
        </section>
      </div>
    </main>
  )
}

export default App
