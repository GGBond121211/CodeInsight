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
