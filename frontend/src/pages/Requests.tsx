import { X } from "lucide-react";
import { useState } from "react";

import { Badge, CacheBadge, Card, EmptyState, ErrorState, Loading, PageHeader, Select, StatusBadge, Td, Th } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { fmt, type RequestDetail, type RequestRow } from "../services/api";
import { DecisionView } from "./Routing";

const PAGE = 25;

export function RequestsPage() {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [cache, setCache] = useState("");
  const [strategy, setStrategy] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  const qs = new URLSearchParams({ hours: "168", limit: String(PAGE), offset: String(offset) });
  if (search.trim()) qs.set("search", search.trim());
  if (status) qs.set("status", status);
  if (cache) qs.set("cache", cache);
  if (strategy) qs.set("strategy", strategy);
  const { data, error, loading } = useApi<{ total: number; items: RequestRow[] }>(`/api/analytics/requests?${qs}`, 10000);

  return (
    <>
      <PageHeader
        title="Request explorer"
        description="Recent requests (last 7 days). Prompts and completions are never stored or shown by default; use the trace ID to find a request in Jaeger or Langfuse."
      />
      <Card
        title={data ? `${fmt.int(data.total)} requests` : "Requests"}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setOffset(0);
              }}
              placeholder="request or trace ID"
              className="w-56 rounded-md border border-line bg-surface px-2 py-1 font-mono text-xs focus:outline-2 focus:outline-accent"
            />
            <Select label="Status" value={status} onChange={(v) => { setStatus(v); setOffset(0); }} options={[{ value: "", label: "all" }, { value: "success", label: "success" }, { value: "error", label: "error" }]} />
            <Select label="Cache" value={cache} onChange={(v) => { setCache(v); setOffset(0); }} options={["", "hit", "miss", "bypass", "nondeterministic", "disabled"].map((v) => ({ value: v, label: v || "all" }))} />
            <Select label="Strategy" value={strategy} onChange={(v) => { setStrategy(v); setOffset(0); }} options={["", "learned", "heuristic", "premium", "economy", "random", "explicit"].map((v) => ({ value: v, label: v || "all" }))} />
          </div>
        }
      >
        {error && <ErrorState error={error} />}
        {loading && !data && <Loading />}
        {data && data.items.length === 0 && <EmptyState title="No matching requests" />}
        {data && data.items.length > 0 && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-line">
                    <Th>Time</Th><Th>Request ID</Th><Th>Model</Th><Th>Strategy</Th><Th right>Cost</Th><Th right>Latency</Th><Th>Cache</Th><Th>Status</Th><Th>Trace ID</Th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((r) => (
                    <tr key={r.request_id} onClick={() => setSelected(r.request_id)} className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-sunken">
                      <Td>{fmt.time(r.created_at)}</Td>
                      <Td mono>{r.request_id.slice(0, 12)}…</Td>
                      <Td mono>
                        {r.final_model ?? "—"} {r.fallback_used && <Badge tone="warn">fallback</Badge>} {r.simulated && <Badge tone="warn">sim</Badge>}
                      </Td>
                      <Td mono>{r.strategy}</Td>
                      <Td right>{fmt.usd(r.cost_usd)}</Td>
                      <Td right>{fmt.ms(r.latency_ms)}</Td>
                      <Td><CacheBadge status={r.cache_status} /></Td>
                      <Td>
                        <StatusBadge status={r.status} /> {r.error_type && <span className="text-xs text-ink-3">{r.error_type}</span>}
                      </Td>
                      <Td mono>{r.trace_id ? `${r.trace_id.slice(0, 10)}…` : "—"}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-ink-2">
              <span>
                {offset + 1}–{Math.min(offset + PAGE, data.total)} of {fmt.int(data.total)}
              </span>
              <div className="flex gap-2">
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} className="rounded-md border border-line px-2 py-1 disabled:opacity-40">
                  Previous
                </button>
                <button disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)} className="rounded-md border border-line px-2 py-1 disabled:opacity-40">
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </Card>
      {selected && <RequestDrawer id={selected} onClose={() => setSelected(null)} />}
    </>
  );
}

function RequestDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const { data, error, loading } = useApi<RequestDetail>(`/api/analytics/requests/${encodeURIComponent(id)}`);
  return (
    <div className="fixed inset-0 z-20 flex justify-end bg-black/20" onClick={onClose}>
      <aside className="h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-panel p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-sm font-semibold">Request detail</h2>
            <div className="font-mono text-xs text-ink-2">{id}</div>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-md p-1 hover:bg-sunken">
            <X className="h-4 w-4" />
          </button>
        </div>
        {error && <ErrorState error={error} />}
        {loading && <Loading />}
        {data && (
          <div className="space-y-4 text-sm">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
              {[
                ["Trace ID", data.trace_id ?? "—"],
                ["Status", `${data.status} (HTTP ${data.http_status})${data.error_type ? ` · ${data.error_type}` : ""}`],
                ["Served by", data.final_model ?? "—"],
                ["Cache", data.cache_status],
                ["End-to-end latency", fmt.ms(data.latency_ms)],
                ["Provider latency", fmt.ms(data.provider_ms)],
                ["Routing decision", fmt.ms(data.routing_ms)],
                ["Queue wait", fmt.ms(data.queue_ms)],
                ["Time to first token", fmt.ms(data.ttft_ms)],
                ["Tokens (in / out)", `${fmt.int(data.input_tokens)} / ${fmt.int(data.output_tokens)}`],
                ["Estimated cost", fmt.usd(data.cost_usd)],
                ["Streaming", data.stream ? "yes" : "no"],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="font-mono">{v}</dd>
                </div>
              ))}
            </dl>
            {data.attempt_detail && data.attempt_detail.length > 0 && (
              <div>
                <h3 className="mb-1 text-xs font-semibold text-ink-2 uppercase">Upstream attempts</h3>
                <ol className="space-y-1 text-xs">
                  {data.attempt_detail.map((a, i) => (
                    <li key={i} className="flex justify-between rounded-md bg-sunken px-2 py-1 font-mono">
                      <span>
                        {i + 1}. {a.model_id} (try {a.attempt})
                      </span>
                      <span className={a.outcome === "success" ? "text-good" : "text-bad"}>
                        {a.outcome} · {fmt.ms(a.latency_s * 1000)}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {data.routing_detail && (
              <div>
                <h3 className="mb-1 text-xs font-semibold text-ink-2 uppercase">Routing decision</h3>
                <DecisionView d={data.routing_detail} />
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
