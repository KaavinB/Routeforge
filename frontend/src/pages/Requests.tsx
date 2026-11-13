import { ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { useEffect, useState } from "react";

import { Badge, Button, Card, EmptyState, ErrorState, ModelChip, PageHeader, Select, SkeletonRows, StatusBadge, Table, Td, Th, rowClass, seriesColor } from "../components/ui";
import { useApi, useModelOrder } from "../hooks/useApi";
import { fmt, type RequestDetail, type RequestRow } from "../services/api";
import { strategyName } from "../services/labels";
import { DecisionView } from "./Routing";

const PAGE = 25;

export function RequestsPage() {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [strategy, setStrategy] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const order = useModelOrder();

  const qs = new URLSearchParams({ hours: "168", limit: String(PAGE), offset: String(offset) });
  if (search.trim()) qs.set("search", search.trim());
  if (status) qs.set("status", status);
  if (strategy) qs.set("strategy", strategy);
  const { data, error, loading } = useApi<{ total: number; items: RequestRow[] }>(`/api/analytics/requests?${qs}`, 10000);

  return (
    <>
      <PageHeader
        title="Requests"
        description="The last 7 days of traffic. Click a row to see which model answered, why, and where the time went. Prompts and answers are never stored."
      />
      <Card
        title={data ? `${fmt.int(data.total)} request${data.total === 1 ? "" : "s"}` : "Requests"}
        actions={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <label className="relative">
              <Search className="pointer-events-none absolute top-1/2 left-3 h-3.5 w-3.5 -translate-y-1/2 text-ink-3" aria-hidden />
              <input
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setOffset(0);
                }}
                aria-label="Search by request or trace ID"
                placeholder="request or trace ID"
                className="w-56 rounded-full border border-line bg-surface py-1.5 pr-3 pl-8 font-mono text-xs text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none"
              />
            </label>
            <Select label="Status" value={status} onChange={(v) => { setStatus(v); setOffset(0); }} options={[{ value: "", label: "all" }, { value: "success", label: "success" }, { value: "error", label: "error" }]} />
            <Select label="Policy" value={strategy} onChange={(v) => { setStrategy(v); setOffset(0); }} options={["", "learned", "heuristic", "premium", "economy", "random", "explicit"].map((v) => ({ value: v, label: v ? strategyName(v) : "all" }))} />
          </div>
        }
      >
        {error && <ErrorState error={error} />}
        {loading && !data && <SkeletonRows n={8} />}
        {data && data.items.length === 0 && <EmptyState title="No matching requests" icon={Search}>Try loosening the filters.</EmptyState>}
        {data && data.items.length > 0 && (
          <>
            <Table>
              <thead>
                <tr>
                  <Th>Time</Th><Th>Model</Th><Th>Policy</Th><Th right>Cost</Th><Th right>Latency</Th><Th>Status</Th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((r) => (
                  <tr
                    key={r.request_id}
                    onClick={() => setSelected(r.request_id)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSelected(r.request_id)}
                    tabIndex={0}
                    className={`${rowClass} cursor-pointer focus:bg-sunken focus:outline-none`}
                  >
                    <Td>{fmt.time(r.created_at)}</Td>
                    <Td>
                      <span className="inline-flex items-center gap-1.5">
                        <ModelChip id={r.final_model} color={r.final_model ? seriesColor(r.final_model, order) : undefined} />
                        {r.fallback_used && <Badge tone="warn">fallback</Badge>}
                        {r.simulated && <Badge tone="warn">sim</Badge>}
                        {r.cache_status === "hit" && <Badge tone="good">cached</Badge>}
                      </span>
                    </Td>
                    <Td>{strategyName(r.strategy)}</Td>
                    <Td right>{fmt.usd(r.cost_usd)}</Td>
                    <Td right>{fmt.ms(r.latency_ms)}</Td>
                    <Td>
                      <StatusBadge status={r.status} />
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <div className="mt-4 flex items-center justify-between text-xs text-ink-2">
              <span className="tabular">
                {offset + 1}–{Math.min(offset + PAGE, data.total)} of {fmt.int(data.total)}
              </span>
              <div className="flex gap-2">
                <Button variant="ghost" className="!px-3 !py-1.5 !text-xs" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                  <ChevronLeft className="h-3.5 w-3.5" aria-hidden /> Previous
                </Button>
                <Button variant="ghost" className="!px-3 !py-1.5 !text-xs" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>
                  Next <ChevronRight className="h-3.5 w-3.5" aria-hidden />
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
      {selected && <RequestDrawer id={selected} order={order} onClose={() => setSelected(null)} />}
    </>
  );
}

function RequestDrawer({ id, order, onClose }: { id: string; order: string[]; onClose: () => void }) {
  const { data, error, loading } = useApi<RequestDetail>(`/api/analytics/requests/${encodeURIComponent(id)}`);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-anvil/40 backdrop-blur-[2px]" onClick={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Request detail"
        className="h-full w-full max-w-2xl animate-[rise_0.3s_ease] overflow-y-auto border-l border-line bg-surface shadow-lift"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 z-10 flex items-start justify-between border-b border-line bg-surface/90 px-6 py-4 backdrop-blur">
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold">Request detail</h2>
            <div className="truncate font-mono text-xs text-ink-3">{id}</div>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-full p-1.5 text-ink-2 hover:bg-sunken hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-5 p-6">
          {error && <ErrorState error={error} />}
          {loading && <SkeletonRows n={8} />}
          {data && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge status={data.status} />
                {data.fallback_used && <Badge tone="warn">rescued by fallback</Badge>}
                {data.cache_status === "hit" && <Badge tone="good">cached</Badge>}
                {data.error_type && <Badge tone="bad">{data.error_type}</Badge>}
              </div>

              <section className="rounded-2xl border border-line bg-panel p-4 shadow-card">
                <h3 className="mb-3 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">Where the time went</h3>
                <Waterfall d={data} />
              </section>

              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 rounded-2xl border border-line bg-panel p-4 text-xs shadow-card sm:grid-cols-3">
                {[
                  ["Served by", <ModelChip key="m" id={data.final_model} color={data.final_model ? seriesColor(data.final_model, order) : undefined} />],
                  ["Policy", strategyName(data.strategy)],
                  ["Estimated cost", fmt.usd(data.cost_usd)],
                  ["Tokens in / out", `${fmt.int(data.input_tokens)} / ${fmt.int(data.output_tokens)}`],
                  ["HTTP status", String(data.http_status)],
                  ["Trace ID", <span key="t" className="font-mono break-all">{data.trace_id ?? "—"}</span>],
                ].map(([k, v]) => (
                  <div key={String(k)} className="min-w-0">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="mt-0.5 font-medium text-ink">{v}</dd>
                  </div>
                ))}
              </dl>

              {data.attempt_detail && data.attempt_detail.length > 0 && (
                <section>
                  <h3 className="mb-2 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">Upstream attempts</h3>
                  <ol className="relative space-y-2 border-l-2 border-line pl-4">
                    {data.attempt_detail.map((a, i) => {
                      const ok = a.outcome === "success";
                      return (
                        <li key={i} className="relative">
                          <span className={`absolute top-2.5 -left-[23px] h-3 w-3 rounded-full border-2 border-surface ${ok ? "bg-good" : "bg-bad"}`} />
                          <div className="flex items-center justify-between rounded-xl border border-line bg-panel px-3 py-2 text-xs">
                            <span className="flex items-center gap-2">
                              <ModelChip id={a.model_id} color={seriesColor(a.model_id, order)} />
                              <span className="text-ink-3">try {a.attempt}</span>
                            </span>
                            <span className={`font-mono ${ok ? "text-good" : "text-bad"}`}>
                              {a.outcome} · {fmt.ms(a.latency_s * 1000)}
                            </span>
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                </section>
              )}
              {data.routing_detail && (
                <section>
                  <h3 className="mb-2 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">Routing decision</h3>
                  <div className="rounded-2xl border border-line bg-panel p-4 shadow-card">
                    <DecisionView d={data.routing_detail} order={order} />
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

/** Stacked bar of the request's end-to-end latency: queue, routing, provider, gateway. */
function Waterfall({ d }: { d: RequestDetail }) {
  const total = d.latency_ms || 0;
  const parts = [
    { k: "Waiting in queue", v: d.queue_ms ?? 0, c: "#a39b8f" },
    { k: "Choosing a model", v: d.routing_ms ?? 0, c: "#e8590c" },
    { k: "Model thinking", v: d.provider_ms ?? 0, c: "#2a78d6" },
  ];
  const known = parts.reduce((a, p) => a + p.v, 0);
  parts.push({ k: "Gateway overhead", v: Math.max(0, total - known), c: "#1baf7a" });
  if (total <= 0) return <p className="text-xs text-ink-3">No timing recorded.</p>;
  // Replay mode reports the originally recorded provider latency without waiting for it,
  // so the parts can add up to more than the measured end-to-end time.
  const scale = Math.max(total, known);
  return (
    <div>
      <div className="mb-3 flex items-baseline gap-2">
        <span className="tabular font-display text-2xl font-semibold">{fmt.ms(total)}</span>
        <span className="text-xs text-ink-3">end to end</span>
      </div>
      <div className="flex h-3 overflow-hidden rounded-full bg-sunken">
        {parts.map((p) => (p.v > 0 ? <div key={p.k} title={`${p.k}: ${fmt.ms(p.v)}`} style={{ width: `${Math.max(1, (100 * p.v) / scale)}%`, background: p.c }} /> : null))}
      </div>
      <ul className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        {parts.map((p) => (
          <li key={p.k} className="flex items-start gap-1.5">
            <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: p.c }} />
            <span>
              <span className="block text-ink-3">{p.k}</span>
              <span className="tabular font-medium text-ink">{fmt.ms(p.v)}</span>
            </span>
          </li>
        ))}
      </ul>
      {known > total * 1.05 && (
        <p className="mt-3 text-[11px] leading-relaxed text-ink-3">
          The parts add up to more than the end-to-end time: the provider time is the latency recorded when this answer was originally collected (replay mode serves it instantly).
        </p>
      )}
    </div>
  );
}
