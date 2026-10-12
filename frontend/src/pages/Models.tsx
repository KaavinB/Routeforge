import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader, Select, TRAFFIC_OPTIONS, Td, Th, seriesColor } from "../components/ui";
import { useApi, useTrafficFilter } from "../hooks/useApi";
import { fmt, type ModelsResponse } from "../services/api";

export function ModelsPage() {
  const [hours, setHours] = useState("24");
  const [traffic, setTraffic] = useTrafficFilter();
  const { data, error, loading } = useApi<ModelsResponse>(`/api/analytics/models?hours=${hours}&traffic=${traffic}`, 15000);
  const order = data ? Object.entries(data.registry).sort((a, b) => a[1].tier - b[1].tier).map(([k]) => k) : [];

  return (
    <>
      <PageHeader
        title="Models"
        description="Per-model traffic, reliability, latency, token usage and estimated cost, plus the registry's list prices and circuit-breaker state."
        actions={
          <div className="flex gap-3">
            <Select label="Traffic" value={traffic} onChange={setTraffic} options={TRAFFIC_OPTIONS} />
            <Select label="Window" value={hours} onChange={setHours} options={[{ value: "1", label: "1 h" }, { value: "24", label: "24 h" }, { value: "168", label: "7 d" }]} />
          </div>
        }
      />
      {error && <ErrorState error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <div className="space-y-4">
          <Card title="Model registry" subtitle="List prices per 1M tokens (USD) and live circuit-breaker state">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-line">
                    <Th>Model</Th><Th>Family</Th><Th right>Tier</Th><Th right>Input $/M</Th><Th right>Output $/M</Th><Th>Breaker</Th>
                  </tr>
                </thead>
                <tbody>
                  {order.map((m) => {
                    const r = data.registry[m];
                    const b = data.breakers[m];
                    return (
                      <tr key={m} className="border-b border-line/60 last:border-0">
                        <Td mono>
                          <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(m, order) }} />
                          {m}
                        </Td>
                        <Td>{r.family}</Td>
                        <Td right>{r.tier}</Td>
                        <Td right>{r.input_cost_per_mtok.toFixed(2)}</Td>
                        <Td right>{r.output_cost_per_mtok.toFixed(2)}</Td>
                        <Td>
                          <Badge tone={!b || b.state === "closed" ? "good" : b.state === "half_open" ? "warn" : "bad"}>{b?.state ?? "closed"}</Badge>
                          {b && b.opened_count > 0 && <span className="ml-2 text-xs text-ink-3">opened {b.opened_count}×</span>}
                        </Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
          {data.models.length === 0 ? (
            <Card><EmptyState title="No model traffic in this window" /></Card>
          ) : (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <MetricBars title="Requests" data={data.models.map((m) => ({ model: m.model, v: m.requests }))} order={order} format={fmt.int} />
                <MetricBars title="Estimated cost (USD)" data={data.models.map((m) => ({ model: m.model, v: m.cost_usd }))} order={order} format={fmt.usd} />
                <MetricBars title="p95 latency" data={data.models.map((m) => ({ model: m.model, v: m.latency_ms.p95 ?? 0 }))} order={order} format={fmt.ms} />
                <MetricBars title="Error rate" data={data.models.map((m) => ({ model: m.model, v: m.error_rate ?? 0 }))} order={order} format={(v) => fmt.pct(v)} />
              </div>
              <Card title="Per-model detail">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-line">
                        <Th>Model</Th><Th right>Requests</Th><Th right>Success</Th><Th right>p50</Th><Th right>p95</Th><Th right>Avg provider</Th><Th right>Input tok</Th><Th right>Output tok</Th><Th right>Cache hits</Th><Th right>Cost</Th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.models.map((m) => (
                        <tr key={m.model} className="border-b border-line/60 last:border-0">
                          <Td mono>{m.model}</Td>
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
                  </table>
                </div>
              </Card>
            </>
          )}
        </div>
      )}
    </>
  );
}

function MetricBars({ title, data, order, format }: { title: string; data: { model: string; v: number }[]; order: string[]; format: (v: number) => string }) {
  return (
    <Card title={title}>
      <ResponsiveContainer width="100%" height={Math.max(120, 36 * data.length + 30)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40 }}>
          <CartesianGrid horizontal={false} stroke="#e4e3df" />
          <XAxis type="number" tickFormatter={(v) => format(Number(v))} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="model" width={140} axisLine={false} tickLine={false} />
          <Tooltip formatter={(v) => format(Number(v))} cursor={{ fill: "#f4f4f2" }} />
          <Bar dataKey="v" name={title} radius={[0, 4, 4, 0]} barSize={16}>
            {data.map((d) => (
              <Cell key={d.model} fill={seriesColor(d.model, order)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
