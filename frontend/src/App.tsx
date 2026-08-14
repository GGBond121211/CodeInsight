import { useState } from 'react'

import { autoAnswerRepository } from './api/codeinsight'
import type { AutoAnswerResponse } from './api/types'
import { AutoAnswerPanel } from './components/AutoAnswerPanel'
import { QuestionForm } from './components/QuestionForm'
import './App.css'

type ViewResult = { kind: 'auto'; value: AutoAnswerResponse }

function App() {
  const [repositoryRoot, setRepositoryRoot] = useState('tests/fixtures/sample_repo')
  const [question, setQuestion] = useState('checkout 如何校验输入？')
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
      setError(caught instanceof Error ? caught.message : '请求失败。')
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
          <span className="eyebrow">本地代码仓库理解</span>
          <h1>CodeInsight</h1>
          <p>
            询问陌生代码如何工作，再通过仓库路径和行号范围核验每个结论。
          </p>
        </div>
        <div className="local-badge">
          <span className="pulse" /> 本地只读演示
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
              <h2>证据会显示在这里</h2>
              <p>CodeInsight 不会要求模型凭空编造文件名或行号。</p>
            </div>
          )}
          {loading && <div className="loading-state">正在扫描、排序并建立证据依据…</div>}
          {error && (
            <div className="error-state" role="alert">
              <strong>请求失败</strong>
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
