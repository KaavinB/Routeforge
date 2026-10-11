// Typed client for the RouteForge admin/analytics API. The admin key is kept in
// sessionStorage (per-tab, cleared when the tab closes) and sent as a Bearer token.

const KEY_STORAGE = "routeforge.adminKey";

export function getAdminKey(): string {
  try {
    return sessionStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setAdminKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable (private mode): key lives only in memory for this page */
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${getAdminKey()}`,
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = body?.error?.message ?? msg;
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------- types
export interface Percentiles {
  p50: number | null;
  p95: number | null;
  p99: number | null;
}

export interface Summary {
  window_hours: number;
  total_requests: number;
  successful_requests: number;
  success_rate: number | null;
  estimated_spend_usd: number;
  cost_per_successful_request_usd: number | null;
  cache_hits: number;
  cache_lookups: number;
  cache_hit_rate: number | null;
  cache_saved_usd: number;
  fallback_requests: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: Percentiles;
  avg_routing_ms: number | null;
  simulated_requests: number;
  active_models: string[];
  strategy_distribution: Record<string, number>;
  model_distribution: Record<string, number>;
  mode: string;
  simulated_mode: boolean;
  pricing: { version: string; sources: Record<string, string>; disclaimer: string };
  router_artifact: string | null;
  default_strategy: string;
  breakers: Record<string, { state: string; opened_count: number }>;
}

export interface TimeBucket {
  start: string;
  requests: number;
  errors: number;
  cost_usd: number;
  cache_hits: number;
  p95_latency_ms: number | null;
}

export interface ModelRow {
  model: string;
  requests: number;
  success_rate: number | null;
  error_rate: number | null;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  cache_hits: number;
  avg_provider_ms: number | null;
  latency_ms: Percentiles;
}

export interface ModelsResponse {
  models: ModelRow[];
  registry: Record<string, { family: string; tier: number; input_cost_per_mtok: number; output_cost_per_mtok: number; enabled: boolean }>;
  breakers: Record<string, { state: string; opened_count: number }>;
}

export interface RequestRow {
  request_id: string;
  trace_id: string | null;
  created_at: string;
  strategy: string;
  selected_model: string | null;
  final_model: string | null;
  routing_reason: string | null;
  predicted_task: string | null;
  predicted_quality: number | null;
  cache_status: string;
  status: string;
  error_type: string | null;
  http_status: number;
  attempts: number;
  fallback_used: boolean;
  stream: boolean;
  simulated: boolean;
  latency_ms: number;
  routing_ms: number | null;
  ttft_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface Candidate {
  model_id: string;
  eligible: boolean;
  ineligible_reason: string | null;
  p_quality: number | null;
  p_quality_adjusted: number | null;
  expected_output_tokens: number | null;
  expected_cost_usd: number | null;
  expected_latency_ms: number | null;
  meets_quality: boolean | null;
  meets_latency: boolean | null;
}

export interface RoutingDetail {
  strategy: string;
  selected_model: string;
  fallback_chain: string[];
  reason: string;
  confidence: number | null;
  predicted_task: string | null;
  quality_threshold: number | null;
  latency_budget_ms: number | null;
  uncertain: boolean;
  decision_ms: number;
  candidates: Candidate[];
  task_probs?: Record<string, number>;
}

export interface RequestDetail extends RequestRow {
  routing_detail: RoutingDetail | null;
  attempt_detail: { model_id: string; attempt: number; outcome: string; latency_s: number }[] | null;
  quality_threshold: number | null;
  queue_ms: number | null;
  provider_ms: number | null;
}

export interface EvalRun {
  id: string;
  name: string;
  dataset: string;
  dataset_version: string;
  split: string;
  target: string;
  status: string;
  config: Record<string, unknown>;
  summary: EvalSummary | null;
  error: string | null;
  git_sha: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface EvalBlock {
  n: number;
  acceptable: number;
  acceptable_rate: number | null;
  acceptable_rate_ci95: [number, number];
  mean_score: number | null;
  errors: number;
  total_cost_usd: number;
  cost_per_request_usd: number | null;
  cost_per_acceptable_usd: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  model_distribution: Record<string, number>;
  simulated: boolean;
}

export interface EvalSummary {
  overall: EvalBlock;
  by_category: Record<string, EvalBlock>;
}

export interface BenchmarkListItem {
  name: string;
  benchmark: string | null;
  experiment_id: string | null;
  timestamp: string | null;
  git_sha: string | null;
  mode: string | null;
  status: string;
}

// ------------------------------------------------------------- formatting
export const fmt = {
  int: (n: number | null | undefined) => (n == null ? "—" : Math.round(n).toLocaleString()),
  pct: (n: number | null | undefined, digits = 1) => (n == null ? "—" : `${(100 * n).toFixed(digits)}%`),
  ms: (n: number | null | undefined) => (n == null ? "—" : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 2 : 0)} ms`),
  usd: (n: number | null | undefined) => {
    if (n == null) return "—";
    if (n === 0) return "$0";
    if (Math.abs(n) < 0.01) return `$${n.toPrecision(2)}`;
    return `$${n.toFixed(n < 100 ? 2 : 0)}`;
  },
  time: (iso: string) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }),
};
