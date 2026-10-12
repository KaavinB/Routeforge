import { useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, EmptyState, ErrorState, Loading, PageHeader, Select, SimulatedBanner, StatTile, TRAFFIC_OPTIONS } from "../components/ui";
import { useApi, useTrafficFilter } from "../hooks/useApi";
import { fmt, type Summary, type TimeBucket } from "../services/api";

const WINDOWS = [
  { value: "1", label: "Last hour" },
  { value: "24", label: "Last 24 h" },
  { value: "168", label: "Last 7 days" },
];

export function OverviewPage() {
  const [hours, setHours] = useState("24");
  const [traffic, setTraffic] = useTrafficFilter();
  const summary = useApi<Summary>(`/api/analytics/summary?hours=${hours}&traffic=${traffic}`, 10000);
  const series = useApi<{ buckets: TimeBucket[] }>(`/api/analytics/timeseries?hours=${hours}&buckets=48&traffic=${traffic}`, 10000);
  const s = summary.data;

  return (
    <>
      <PageHeader
        title="Overview"
        description="Live operational metrics from the gateway's request log. Costs are estimated from provider-reported token usage and list prices."
        actions={
          <div className="flex gap-3">
            <Select label="Traffic" value={traffic} onChange={setTraffic} options={TRAFFIC_OPTIONS} />
            <Select label="Window" value={hours} onChange={setHours} options={WINDOWS} />
          </div>
        }
      />
      {summary.error && <ErrorState error={summary.error} />}
      {summary.loading && !s && <Loading />}
      {s && (
        <div className="space-y-4">
          <SimulatedBanner show={s.simulated_mode || s.simulated_requests > 0}>
            {s.simulated_requests > 0 && !s.simulated_mode
              ? `${s.simulated_requests} of ${s.total_requests} requests in this window were served by simulated (mock) providers.`
              : undefined}
          </SimulatedBanner>
          {s.total_requests === 0 ? (
            <Card>
              <EmptyState title="No requests in this window">
                Send traffic to <code className="font-mono">POST /v1/chat/completions</code> (any OpenAI-compatible client) and it will appear here.
              </EmptyState>
            </Card>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <StatTile label="Total requests" value={fmt.int(s.total_requests)} hint={`${fmt.int(s.successful_requests)} successful`} />
                <StatTile
                  label="Success rate"
                  value={fmt.pct(s.success_rate, 2)}
                  tone={s.success_rate != null && s.success_rate < 0.99 ? (s.success_rate < 0.95 ? "bad" : "warn") : undefined}
                  hint={`${fmt.int(s.fallback_requests)} recovered via fallback`}
                />
                <StatTile label="Estimated spend" value={fmt.usd(s.estimated_spend_usd)} hint={`${fmt.usd(s.cost_per_successful_request_usd)} per successful request`} />
                <StatTile label="p95 latency" value={fmt.ms(s.latency_ms.p95)} hint={`p50 ${fmt.ms(s.latency_ms.p50)} · p99 ${s.latency_ms.p99 == null ? "n<200" : fmt.ms(s.latency_ms.p99)}`} />
                <StatTile label="Cache hit rate" value={fmt.pct(s.cache_hit_rate)} hint={`${fmt.int(s.cache_hits)} / ${fmt.int(s.cache_lookups)} eligible lookups · saved ${fmt.usd(s.cache_saved_usd)}`} />
                <StatTile label="Routing overhead" value={fmt.ms(s.avg_routing_ms)} hint="mean decision time" />
                <StatTile label="Active models" value={s.active_models.length} hint={s.active_models.join(", ") || "—"} />
                <StatTile label="Tokens" value={fmt.int(s.input_tokens + s.output_tokens)} hint={`${fmt.int(s.input_tokens)} in · ${fmt.int(s.output_tokens)} out`} />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Requests over time" subtitle="Successful vs. failed per bucket">
                  {series.data && <TrafficChart buckets={series.data.buckets} />}
                </Card>
                <Card title="p95 latency over time" subtitle="Successful requests only">
                  {series.data && <LatencyChart buckets={series.data.buckets} />}
                </Card>
                <Card title="Routing policy distribution" subtitle="Requests by strategy">
                  <DistributionBars data={s.strategy_distribution} />
                </Card>
                <Card title="Model distribution" subtitle="Requests by serving model">
                  <DistributionBars data={s.model_distribution} />
                </Card>
              </div>
              <p className="text-xs text-ink-3">
                Pricing {s.pricing.version}. {s.pricing.disclaimer} Router artifact: <span className="font-mono">{s.router_artifact ?? "none"}</span>.
              </p>
            </>
          )}
        </div>
      )}
    </>
  );
}

function tick(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function TrafficChart({ buckets }: { buckets: TimeBucket[] }) {
  const data = buckets.map((b) => ({ t: b.start, ok: b.requests - b.errors, errors: b.errors }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} barCategoryGap={1}>
        <CartesianGrid vertical={false} stroke="#e4e3df" />
        <XAxis dataKey="t" tickFormatter={tick} minTickGap={40} axisLine={false} tickLine={false} />
        <YAxis allowDecimals={false} axisLine={false} tickLine={false} width={40} />
        <Tooltip labelFormatter={(l) => fmt.time(String(l))} />
        <Bar dataKey="ok" name="successful" stackId="a" fill="#2a78d6" />
        <Bar dataKey="errors" name="failed" stackId="a" fill="#e34948" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function LatencyChart({ buckets }: { buckets: TimeBucket[] }) {
  const data = buckets.map((b) => ({ t: b.start, p95: b.p95_latency_ms }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data}>
        <CartesianGrid vertical={false} stroke="#e4e3df" />
        <XAxis dataKey="t" tickFormatter={tick} minTickGap={40} axisLine={false} tickLine={false} />
        <YAxis tickFormatter={(v) => fmt.ms(Number(v))} axisLine={false} tickLine={false} width={56} />
        <Tooltip labelFormatter={(l) => fmt.time(String(l))} formatter={(v) => fmt.ms(Number(v))} />
        <Area type="monotone" dataKey="p95" name="p95 latency" stroke="#2a78d6" strokeWidth={2} fill="#2a78d6" fillOpacity={0.08} connectNulls />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function DistributionBars({ data }: { data: Record<string, number> }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (!rows.length) return <EmptyState title="No data" />;
  return (
    <div className="space-y-2">
      {rows.map(([k, v]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between text-xs">
            <span className="font-mono">{k}</span>
            <span className="tabular text-ink-2">
              {fmt.int(v)} · {fmt.pct(v / total)}
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded bg-sunken">
            <div className="h-full rounded bg-accent" style={{ width: `${(100 * v) / total}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
