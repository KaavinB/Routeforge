import { ShieldAlert, ShieldCheck, ShieldHalf } from "lucide-react";
import { useState } from "react";

import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader, Segmented, Select, TRAFFIC_OPTIONS, Table, Td, Th, rowClass, seriesColor } from "../components/ui";
import { useApi, useTrafficFilter } from "../hooks/useApi";
import { fmt, type ModelsResponse } from "../services/api";

export function ModelsPage() {
  const [hours, setHours] = useState("24");
  const [traffic, setTraffic] = useTrafficFilter();
  const { data, error, loading } = useApi<ModelsResponse>(`/api/analytics/models?hours=${hours}&traffic=${traffic}`, 15000);
  const order = data ? Object.entries(data.registry).sort((a, b) => a[1].tier - b[1].tier || a[0].localeCompare(b[0])).map(([k]) => k) : [];
  const usage = new Map(data?.models.map((m) => [m.model, m]) ?? []);

  return (
    <>
      <PageHeader
        title="Models"
        description="Every model RouteForge can choose from: what it costs, how much it's used, and whether it's healthy."
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <Select label="Traffic" value={traffic} onChange={setTraffic} options={TRAFFIC_OPTIONS} />
            <Segmented label="Time window" value={hours} onChange={setHours} options={[{ value: "1", label: "1 h" }, { value: "24", label: "24 h" }, { value: "168", label: "7 d" }]} />
          </div>
        }
      />
      {error && <ErrorState error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <div className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {order.map((m) => {
              const r = data.registry[m];
              const b = data.breakers[m];
              const u = usage.get(m);
              const color = seriesColor(m, order);
              const state = b?.state ?? "closed";
              const BreakerIcon = state === "closed" ? ShieldCheck : state === "half_open" ? ShieldHalf : ShieldAlert;
              return (
                <div key={m} className="group animate-rise relative overflow-hidden rounded-2xl border border-line bg-panel p-5 shadow-card transition-shadow hover:shadow-lift">
                  <div className="absolute inset-y-0 left-0 w-1" style={{ background: color }} />
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-[14px] font-semibold">{m}</div>
                      <div className="mt-0.5 text-xs text-ink-3">
                        {r.family} · tier {r.tier}
                      </div>
                    </div>
                    <Badge tone={state === "closed" ? "good" : state === "half_open" ? "warn" : "bad"}>
                      <BreakerIcon className="h-3 w-3" aria-hidden />
                      {state === "closed" ? "healthy" : state === "half_open" ? "recovering" : "benched"}
                    </Badge>
                  </div>
                  <p className="mt-3 text-xs text-ink-3">
                    <span className="tabular font-medium text-ink">${r.input_cost_per_mtok.toFixed(2)} / ${r.output_cost_per_mtok.toFixed(2)}</span> per 1M tokens in / out
                  </p>
                  <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-line pt-3 text-xs">
                    <div>
                      <dt className="text-ink-3">Requests</dt>
                      <dd className="tabular font-display text-base font-semibold">{fmt.int(u?.requests ?? 0)}</dd>
                    </div>
                    <div>
                      <dt className="text-ink-3">Typical speed</dt>
                      <dd className="tabular font-display text-base font-semibold">{fmt.ms(u?.latency_ms.p50)}</dd>
                    </div>
                    <div>
                      <dt className="text-ink-3">Spend</dt>
                      <dd className="tabular font-display text-base font-semibold">{fmt.usd(u?.cost_usd ?? 0)}</dd>
                    </div>
                  </dl>
                  {b && b.opened_count > 0 && <p className="mt-2 text-[11px] text-warn">Benched {b.opened_count}× after repeated failures.</p>}
                </div>
              );
            })}
          </div>
          {data.models.length === 0 ? (
            <Card>
              <EmptyState title="No model traffic in this window" />
            </Card>
          ) : (
            <>
              <details className="group rounded-2xl border border-line bg-panel shadow-card">
                <summary className="cursor-pointer list-none px-5 py-4 text-sm font-medium text-ink-2 select-none hover:text-ink">
                  <span className="mr-1 inline-block transition-transform group-open:rotate-90">›</span> Show all numbers
                </summary>
                <div className="px-5 pb-4">
                <Table>
                  <thead>
                    <tr>
                      <Th>Model</Th><Th right>Requests</Th><Th right>Success</Th><Th right>Typical time</Th><Th right>Slowest 5%</Th><Th right>Model time</Th><Th right>Tokens in</Th><Th right>Tokens out</Th><Th right>Cache hits</Th><Th right>Cost</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.models.map((m) => (
                      <tr key={m.model} className={rowClass}>
                        <Td mono>
                          <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(m.model, order) }} />
                          {m.model}
                        </Td>
                        <Td right>{fmt.int(m.requests)}</Td>
                        <Td right>{fmt.pct(m.success_rate)}</Td>
                        <Td right>{fmt.ms(m.latency_ms.p50)}</Td>
                        <Td right>{fmt.ms(m.latency_ms.p95)}</Td>
                        <Td right>{fmt.ms(m.avg_provider_ms)}</Td>
                        <Td right>{fmt.int(m.input_tokens)}</Td>
                        <Td right>{fmt.int(m.output_tokens)}</Td>
                        <Td right>{fmt.int(m.cache_hits)}</Td>
                        <Td right>{fmt.usd(m.cost_usd)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
                </div>
              </details>
            </>
          )}
        </div>
      )}
    </>
  );
}
