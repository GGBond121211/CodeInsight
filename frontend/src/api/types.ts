export type RetrievalMode = 'lexical' | 'bm25' | 'hybrid'
export type RetrievalReason =
  | 'direct_match'
  | 'semantic_match'
  | 'hybrid_match'

export interface SearchRequest {
  repository_root: string
  question: string
  limit: number
  retrieval_mode: RetrievalMode
}

export type AnswerRequest = SearchRequest

export interface SearchHit {
  rank: number
  score: number
  relative_path: string
  start_line: number
  end_line: number
  excerpt: string
  symbol_path: string | null
  retrieval_reason: RetrievalReason
}

export interface SearchResponse {
  retrieval_mode: RetrievalMode
  results: SearchHit[]
}

export interface Citation {
  evidence_id: string
  relative_path: string
  start_line: number
  end_line: number
}

export interface TokenUsage {
  input_tokens: number
  output_tokens: number
}

export interface UsageSummary {
  requests: number
  provider_attempts: number
  successful_attempts: number
  semantic_cache_hits: number
  input_tokens: number
  cache_read_tokens: number
  cache_miss_tokens: number
  cache_write_tokens: number | null
  output_tokens: number
  cache_hit_ratio: number
  estimated_cost_stars: string
  unknown_usage_count: number
  usage_coverage: number
}

export interface UsageCall {
  recorded_at_epoch_ms: number | null
  request_id: string
  attempt_id: string
  scene: string
  prompt_version: string
  provider: string
  model_tier: string
  model: string
  status: 'ok' | 'error'
  input_tokens: number
  cache_write_tokens: number | null
  cache_read_tokens: number
  cache_miss_tokens: number
  output_tokens: number
  cache_hit_ratio: number
  estimated_cost_stars: string
  latency_milliseconds: number
  fallback_reason: string | null
  error_class: string | null
  usage_source: string
  request_fingerprint: string | null
  stable_prefix_fingerprint: string | null
}

export interface UsageCallsResponse {
  object: 'list'
  data: UsageCall[]
}

export interface AnswerResponse {
  outcome: 'answered' | 'insufficient_evidence'
  answer: string
  citations: Citation[]
  retrieval_mode: RetrievalMode
  model: string | null
  prompt_version: string
  usage: TokenUsage
}

export interface AgentEvent {
  sequence: number
  step: string
  summary: string
}

export interface AgentAnswerResponse extends AnswerResponse {
  revisions: number
  events: AgentEvent[]
}

export interface AutoAnswerRequest {
  repository_root: string
  question: string
  limit: number
  force_route?: 'linear' | 'agent'
}

export interface QueryPlanSubQuestion {
  question: string
  intent: string
  retrieval_mode: RetrievalMode
}

export interface QueryPlan {
  original_question: string
  language: string
  normalized_question: string
  subquestions: QueryPlanSubQuestion[]
  retrieval_modes: RetrievalMode[]
  execution_route: 'linear' | 'agent' | 'insufficient'
  confidence: number
  fallback_reason: string | null
}

export interface AutoSubQuestion {
  question: string
  intent: string
  retrieval_mode: RetrievalMode
  outcome: 'answered' | 'insufficient_evidence'
  answer: string
  citations: Citation[]
}

export interface AutoAnswerResponse {
  outcome: 'answered' | 'partially_answered' | 'insufficient_evidence'
  answer: string
  citations: Citation[]
  retrieval_mode: string
  model: string | null
  prompt_version: string
  usage: TokenUsage
  plan: QueryPlan
  subquestions: AutoSubQuestion[]
  router_model: string | null
  router_usage: TokenUsage
  router_elapsed_milliseconds: number
  embedding_input_tokens: number
  fallback_reason: string | null
  events: AgentEvent[]
}

export interface ChatSessionResponse {
  session_id: string
  repo_id: string
  index_version: string
  status: string
  summary: string | null
  compacted_through_sequence: number
  active_goal: {
    goal_id: string
    task_type: string
    mode: string
    status: string
  } | null
  recent_turns: Array<{ sequence: number; role: 'user' | 'assistant'; content: string }>
  cache_hit: boolean
  cache_fallback: boolean
}

export interface ChatTurnResponse {
  turn_id: string
  session_id: string
  run_id: string
  task_type: 'explain' | 'change'
  status: 'QUEUED' | 'RUNNING' | 'WAITING_APPROVAL' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
  user_message: string
  assistant_message: string | null
  result: Record<string, unknown> | null
  error: string | null
  reasoning_available: boolean
  created_at_epoch_ms: number
  updated_at_epoch_ms: number
}

export interface ChatEvent {
  event_id: string
  run_id: string
  sequence: number
  event_type: string
  occurred_at_epoch_ms: number
  payload: Record<string, string>
}

export interface DebugReasoningEvent {
  run_id: string
  model: string
  content: string
  occurred_at_epoch_ms: number
}
