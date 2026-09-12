import { useCallback, useEffect, useMemo, useState } from 'react'

import { usageCalls, usageSummary } from '../api/usage'
import type { UsageCall, UsageSummary } from '../api/types'
import './ObservabilityDashboard.css'

interface ObservabilityDashboardProps {
  refreshToken: number
}

interface DashboardState {
  summary: UsageSummary | null
  calls: UsageCall[]
  loading: boolean
  error: string | null
  updatedAtEpochMs: number | null
}

const EMPTY_STATE: DashboardState = {
  summary: null,
  calls: [],
  loading: true,
  error: null,
  updatedAtEpochMs: null,
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat('zh-CN').format(value)
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatCost(value: string): string {
  return Number(value).toFixed(4)
}

function formatLatency(value: number): string {
  if (value < 1_000) return `${value.toFixed(0)}ms`
  return `${(value / 1_000).toFixed(1)}s`
}

function formatTime(epochMilliseconds: number | null): string {
  if (!epochMilliseconds) return '—'
  return new Date(epochMilliseconds).toLocaleTimeString('zh-CN', { hour12: false })
}

function shorten(value: string, length = 12): string {
  return value.length <= length ? value : `${value.slice(0, length)}…`
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) {
    return <span className="sparkline-empty">等待更多调用</span>
  }
  const width = 220
  const height = 54
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width
      const y = height - ((value - min) / range) * (height - 8) - 4
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="最近调用趋势">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function MetricCard({ label, value, caption, trend, trendLabel }: { label: string; value: string; caption: string; trend?: number[]; trendLabel?: string }) {
  return (
    <article className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-caption">{caption}</span>
      {trend && (
        <div className="metric-trend">
          <span className="metric-trend-label">{trendLabel || '趋势'}</span>
          <Sparkline values={trend} />
        </div>
      )}
    </article>
  )
}

function usageSourceLabel(source: string): string {
  if (source === 'provider_native') return '供应商原生'
  if (source === 'openai_compatible') return '兼容字段'
  if (source === 'inferred_uncached') return '本地推断未命中'
  if (source === 'unavailable') return '未返回'
  if (source === 'synthetic') return 'Fake Provider'
  return source
}

export function ObservabilityDashboard({ refreshToken }: ObservabilityDashboardProps) {
  const [state, setState] = useState<DashboardState>(EMPTY_STATE)

  const load = useCallback(async () => {
    setState((current) => ({ ...current, loading: true, error: null }))
    try {
      const [summary, calls] = await Promise.all([usageSummary(), usageCalls()])
      setState({ summary, calls: calls.data, loading: false, error: null, updatedAtEpochMs: Date.now() })
    } catch (caught) {
      setState((current) => ({
        ...current,
        loading: false,
        error: caught instanceof Error ? caught.message : '监控数据暂时不可用。',
      }))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshToken])

  const latencyTrend = useMemo(
    () => state.calls
      .slice()
      .reverse()
      .map((call) => call.latency_milliseconds)
      .slice(-20),
    [state.calls],
  )

  return (
    <section className="observability-card" aria-labelledby="observability-title">
      <div className="dashboard-header">
        <div>
          <h2 id="observability-title">调用监控</h2>
          <p>这里展示实际处理 Auto Answer 的同一 Gateway 进程记录。</p>
        </div>
        <div className="dashboard-header-meta">
          <button className="secondary-action" type="button" onClick={() => void load()} disabled={state.loading}>
            {state.loading ? '刷新中…' : '刷新监控'}
          </button>
          <span className="dashboard-updated">
            {state.updatedAtEpochMs ? `数据截至 ${formatTime(state.updatedAtEpochMs)}` : '正在读取数据'}
          </span>
        </div>
      </div>

      {state.error && <div className="dashboard-notice" role="status">{state.error}</div>}

      {state.summary && (
        <>
          <div className="metric-grid">
            <MetricCard
              label="调用次数"
              value={formatInteger(state.summary.requests)}
              caption={`${formatInteger(state.summary.provider_attempts)} 次供应商 attempt`}
              trend={latencyTrend}
              trendLabel="最近延迟趋势"
            />
            <MetricCard
              label="总 input / output"
              value={formatInteger(state.summary.input_tokens + state.summary.output_tokens)}
              caption={`${formatInteger(state.summary.input_tokens)} / ${formatInteger(state.summary.output_tokens)}`}
            />
            <MetricCard
              label="缓存命中率"
              value={formatPercent(state.summary.cache_hit_ratio)}
              caption={`${formatInteger(state.summary.cache_read_tokens)} read · ${formatInteger(state.summary.cache_miss_tokens)} miss`}
            />
            <MetricCard
              label="估算成本"
              value={formatCost(state.summary.estimated_cost_stars)}
              caption="按实际 usage 与价格版本计算"
            />
            <MetricCard
              label="usage 覆盖率"
              value={formatPercent(state.summary.usage_coverage)}
              caption={`${formatInteger(state.summary.unknown_usage_count)} 次 usage 未返回`}
            />
            <MetricCard
              label="cache write"
              value={state.summary.cache_write_tokens === null ? '—' : formatInteger(state.summary.cache_write_tokens)}
              caption="DeepSeek API 未提供该字段，不做本地推断"
            />
          </div>

          <div className="dashboard-explanation">
            <span>缓存 read/miss 来自供应商响应；时长由 Gateway 在本地测量；request_id、attempt_id 和错误归因由本地链路记录。</span>
          </div>

          <div className="calls-section">
            <div className="calls-heading">
              <div>
                <h3>调用明细</h3>
              </div>
              <span className="calls-count">最近 {state.calls.length} 条</span>
            </div>
            {state.calls.length === 0 ? (
              <div className="calls-empty">提交一次自然语言问题后，这里会出现对应的调用明细。</div>
            ) : (
              <div className="usage-table-wrap">
                <table className="usage-table">
                  <thead>
                    <tr>
                      <th>时间</th>
                      <th>Agent / route</th>
                      <th>model</th>
                      <th>provider</th>
                      <th>tier</th>
                      <th>状态</th>
                      <th>时长</th>
                      <th>input</th>
                      <th>cache write</th>
                      <th>cache read</th>
                      <th>output</th>
                      <th>成本</th>
                      <th>来源</th>
                      <th>request / attempt</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.calls.map((call) => (
                      <tr key={call.attempt_id}>
                        <td>{formatTime(call.recorded_at_epoch_ms)}</td>
                        <td>{call.scene}</td>
                        <td>{call.model}</td>
                        <td>{call.provider}</td>
                        <td>{call.model_tier}</td>
                        <td><span className={`call-status ${call.status === 'ok' ? 'ok' : 'error'}`}>{call.status}</span></td>
                        <td>{formatLatency(call.latency_milliseconds)}</td>
                        <td>{formatInteger(call.input_tokens)}</td>
                        <td>{call.cache_write_tokens === null ? '—' : formatInteger(call.cache_write_tokens)}</td>
                        <td>{formatInteger(call.cache_read_tokens)}</td>
                        <td>{formatInteger(call.output_tokens)}</td>
                        <td>{formatCost(call.estimated_cost_stars)}</td>
                        <td>{usageSourceLabel(call.usage_source)}</td>
                        <td className="id-cell" title={`${call.request_id} / ${call.attempt_id}`}>
                          {shorten(call.request_id)} / {shorten(call.attempt_id)}
                          {(call.fallback_reason || call.error_class) && (
                            <small>{call.fallback_reason || call.error_class}</small>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </section>
  )
}
