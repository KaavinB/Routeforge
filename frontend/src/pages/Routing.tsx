import { CheckCircle2, CircleSlash, XCircle } from "lucide-react";
import { useState } from "react";

import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader, Td, Th } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, fmt, type RequestRow, type RoutingDetail } from "../services/api";

const STRATEGIES = ["learned", "heuristic", "premium", "economy", "random", "learned-no-cost", "learned-task-only", "learned-no-task", "learned-full"];

export function RoutingPage() {
  const [prompt, setPrompt] = useState("Write a Python function that returns the n-th Fibonacci number using memoization.");
  const [strategy, setStrategy] = useState("learned");
  const [threshold, setThreshold] = useState(0.8);
  const [budget, setBudget] = useState("");
  const [preview, setPreview] = useState<RoutingDetail & { features?: Record<string, number> } | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const recent = useApi<{ items: RequestRow[] }>("/api/analytics/requests?limit=12&hours=24", 10000);

  async function run() {
    setError(null);
    try {
      const qs = new URLSearchParams({ prompt, strategy, quality_threshold: String(threshold) });
      if (budget) qs.set("latency_budget_ms", budget);
      setPreview(await api(`/api/analytics/routing/preview?${qs}`));
    } catch (e) {
      setError(e as Error);
    }
  }

  return (
    <>
      <PageHeader
        title="Routing intelligence"
        description="Why a model was chosen: each candidate's calibrated probability of an acceptable answer, safety-adjusted probability, expected cost and latency. Dry-run any prompt below — no model is called."
      />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <Card title="Routing preview">
          <div className="space-y-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              className="w-full rounded-lg border border-line bg-surface p-2 font-mono text-xs focus:outline-2 focus:outline-accent"
            />
            <div className="grid grid-cols-3 gap-2 text-xs">
              <label className="space-y-1">
                <div className="text-ink-2">Strategy</div>
                <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="w-full rounded-md border border-line bg-panel px-2 py-1">
                  {STRATEGIES.map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </label>
              <label className="space-y-1">
                <div className="text-ink-2">Quality threshold τ = {threshold.toFixed(2)}</div>
                <input type="range" min={0.3} max={0.99} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-[#2a78d6]" />
              </label>
              <label className="space-y-1">
                <div className="text-ink-2">Latency budget (ms)</div>
                <input value={budget} onChange={(e) => setBudget(e.target.value.replace(/\D/g, ""))} placeholder="none" className="w-full rounded-md border border-line bg-panel px-2 py-1" />
              </label>
            </div>
            <button onClick={run} className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90">
              Explain routing
            </button>
            {error && <ErrorState error={error} />}
          </div>
        </Card>
        <Card title="Decision" subtitle={preview ? `${preview.strategy} · ${fmt.ms(preview.decision_ms)} decision time` : "Run a preview to see candidate scores"}>
          {preview ? <DecisionView d={preview} /> : <EmptyState title="No decision yet" />}
        </Card>
      </div>
      <div className="mt-4">
        <Card title="Recent routed requests" subtitle="Open a request in the Requests view for its full decision record">
          {recent.error && <ErrorState error={recent.error} />}
          {recent.loading && !recent.data && <Loading />}
          {recent.data && recent.data.items.length === 0 && <EmptyState title="No requests yet" />}
          {recent.data && recent.data.items.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-line">
                    <Th>Time</Th><Th>Strategy</Th><Th>Task</Th><Th>Selected</Th><Th>Served by</Th><Th right>P(ok)</Th><Th>Reason</Th>
                  </tr>
                </thead>
                <tbody>
                  {recent.data.items.map((r) => (
                    <tr key={r.request_id} className="border-b border-line/60 last:border-0">
                      <Td>{fmt.time(r.created_at)}</Td>
                      <Td mono>{r.strategy}</Td>
                      <Td>{r.predicted_task ?? "—"}</Td>
                      <Td mono>{r.selected_model ?? "—"}</Td>
                      <Td mono>
                        {r.final_model ?? "—"} {r.fallback_used && <Badge tone="warn">fallback</Badge>}
                      </Td>
                      <Td right>{r.predicted_quality == null ? "—" : r.predicted_quality.toFixed(2)}</Td>
                      <Td>
                        <span className="text-xs text-ink-2">{r.routing_reason}</span>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

export function DecisionView({ d }: { d: RoutingDetail }) {
  const taskTop = d.task_probs ? Object.entries(d.task_probs).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-ink-2">Selected</span>
        <span className="font-mono font-semibold">{d.selected_model}</span>
        <Badge tone="accent">{d.reason}</Badge>
        {d.uncertain && <Badge tone="warn">out-of-distribution: extra margin applied</Badge>}
      </div>
      {taskTop.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs text-ink-2">
          Predicted task:
          {taskTop.map(([t, p]) => (
            <span key={t} className="font-mono">
              {t} {fmt.pct(p, 0)}
            </span>
          ))}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-line">
              <Th>Model</Th><Th right>P(ok)</Th><Th right>Adjusted</Th><Th right>Exp. cost</Th><Th right>Exp. latency</Th><Th>Quality</Th><Th>Latency</Th>
            </tr>
          </thead>
          <tbody>
            {d.candidates.map((c) => (
              <tr key={c.model_id} className={`border-b border-line/60 last:border-0 ${c.model_id === d.selected_model ? "bg-accent-soft/60" : ""}`}>
                <Td mono>
                  {c.model_id}
                  {!c.eligible && <span className="ml-1 text-xs text-bad">({c.ineligible_reason})</span>}
                </Td>
                <Td right>{c.p_quality == null ? "—" : c.p_quality.toFixed(3)}</Td>
                <Td right>{c.p_quality_adjusted == null ? "—" : c.p_quality_adjusted.toFixed(3)}</Td>
                <Td right>{fmt.usd(c.expected_cost_usd)}</Td>
                <Td right>{fmt.ms(c.expected_latency_ms)}</Td>
                <Td><Check ok={c.meets_quality} /></Td>
                <Td><Check ok={c.meets_latency} /></Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        τ = {d.quality_threshold ?? "—"} · latency budget {d.latency_budget_ms ? fmt.ms(d.latency_budget_ms) : "none"} · fallback chain:{" "}
        <span className="font-mono">{d.fallback_chain.join(" → ") || "none"}</span>. Adjusted = P(ok) − calibration margin (− OOD margin when flagged).
      </p>
    </div>
  );
}

function Check({ ok }: { ok: boolean | null }) {
  if (ok == null) return <CircleSlash className="h-4 w-4 text-ink-3" aria-label="not evaluated" />;
  return ok ? <CheckCircle2 className="h-4 w-4 text-good" aria-label="meets" /> : <XCircle className="h-4 w-4 text-bad" aria-label="fails" />;
}
