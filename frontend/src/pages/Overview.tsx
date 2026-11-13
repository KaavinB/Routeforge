import { Zap } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, ChartTooltip, EmptyState, ErrorState, Loading, PageHeader, Ring, Segmented, Select, SimulatedBanner, StatTile, TRAFFIC_OPTIONS, seriesColor } from "../components/ui";
import { useApi, useModelOrder, useTrafficFilter } from "../hooks/useApi";
import { useChartColors } from "../hooks/useTheme";
import { fmt, type Summary, type TimeBucket } from "../services/api";

const WINDOWS = [
  { value: "1", label: "1 hour" },
  { value: "24", label: "24 hours" },
  { value: "168", label: "7 days" },
];
const WINDOW_WORDS: Record<string, string> = { "1": "In the last hour", "24": "In the last 24 hours", "168": "In the last 7 days" };


export function OverviewPage() {
  const [hours, setHours] = useState("24");
  const [traffic, setTraffic] = useTrafficFilter();
  const summary = useApi<Summary>(`/api/analytics/summary?hours=${hours}&traffic=${traffic}`, 10000);
  const series = useApi<{ buckets: TimeBucket[] }>(`/api/analytics/timeseries?hours=${hours}&buckets=48&traffic=${traffic}`, 10000);
  const order = useModelOrder();
  const s = summary.data;
  const b = series.data?.buckets ?? [];

  return (
    <>
      <PageHeader
        title="Overview"
        description="How much your AI traffic costs, how fast it is, and which models are doing the work."
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <Select label="Traffic" value={traffic} onChange={setTraffic} options={TRAFFIC_OPTIONS} />
            <Segmented label="Time window" value={hours} onChange={setHours} options={WINDOWS} />
          </div>
        }
      />
      {summary.error && <ErrorState error={summary.error} />}
      {summary.loading && !s && <Loading />}
      {s && (
        <div className="space-y-5">
          <SimulatedBanner show={s.simulated_mode || s.simulated_requests > 0}>
            {s.simulated_requests > 0 && !s.simulated_mode
              ? `${s.simulated_requests} of ${s.total_requests} requests in this window were served by simulated (mock) providers.`
              : undefined}
          </SimulatedBanner>
          {s.total_requests === 0 ? (
            <Card>
              <EmptyState title="No requests yet" icon={Zap}>
                Send your app's AI requests to RouteForge and they'll show up here. Apps built for OpenAI's API work as-is: just point them at RouteForge's address.
              </EmptyState>
            </Card>
          ) : (
            <>
              <Hero s={s} window={WINDOW_WORDS[hours]} />

              <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                <StatTile label="Requests" value={fmt.int(s.total_requests)} hint={`${fmt.int(s.fallback_requests)} rescued by a fallback`} info="Every call an app made to the gateway in this window, including ones served from cache." />
                <StatTile label="Estimated spend" value={fmt.usd(s.estimated_spend_usd)} hint={`${fmt.usd(s.cost_per_successful_request_usd)} per answer`} info="Token counts reported by each provider × their public list prices. An estimate, not an invoice." />
                <StatTile
                  label="Response time"
                  value={fmt.ms(s.latency_ms.p50)}
                  hint={`slowest 5% took over ${fmt.ms(s.latency_ms.p95)}`}
                  info="How long a typical request takes to answer. The line below shows how slow the slowest 5% of requests were."
                />
                <StatTile label="Answered from cache" value={fmt.pct(s.cache_hit_rate)} hint={`saved ${fmt.usd(s.cache_saved_usd)}`} info="When the exact same deterministic request comes in twice, the stored answer is returned instantly and for free." />
              </div>

              <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
                <Card title="Traffic" subtitle="Requests over time; red marks failures">
                  {b.length > 0 && <TrafficChart buckets={b} />}
                </Card>
                <Card title="Who did the work" subtitle="Share of requests per model">
                  <ModelDonut data={s.model_distribution} order={order} />
                </Card>
              </div>
              <p className="text-xs leading-relaxed text-ink-3">
                Costs are estimates based on each provider's public prices (as of {s.pricing.version}), not actual bills.
              </p>
            </>
          )}
        </div>
      )}
    </>
  );
}

function Hero({ s, window }: { s: Summary; window: string }) {
  const rate = s.success_rate ?? 0;
  const ringColor = rate >= 0.99 ? "#4cc489" : rate >= 0.95 ? "#f2b24a" : "#ff6b5e";
  const n = s.active_models.length;
  return (
    <section className="animate-rise relative overflow-hidden rounded-3xl bg-anvil text-white shadow-lift">
      <div className="absolute -top-24 -right-16 h-72 w-72 rounded-full bg-[#ff6a1f]/25 blur-3xl" />
      <div className="relative flex flex-col gap-6 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
        <div>
          <p className="text-[13px] text-white/50">{window}</p>
          <p className="mt-2 max-w-2xl font-display text-[22px] leading-snug font-medium tracking-tight text-white/90 sm:text-[26px]">
            RouteForge handled <Hl>{fmt.int(s.total_requests)} requests</Hl> across <Hl>{n} model{n === 1 ? "" : "s"}</Hl> for about{" "}
            <Hl>{fmt.usd(s.estimated_spend_usd)}</Hl>.
          </p>
        </div>
        <Ring value={rate} color={ringColor} size={104}>
          <span className="tabular font-display text-xl font-semibold">{fmt.pct(rate, rate === 1 ? 0 : 1)}</span>
          <span className="text-[10px] tracking-wider text-white/50 uppercase">success</span>
        </Ring>
      </div>
    </section>
  );
}

function Hl({ children }: { children: ReactNode }) {
  return <span className="bg-linear-to-r from-[#ffc46b] to-[#ff7a2e] bg-clip-text font-semibold text-transparent">{children}</span>;
}

function fmtCompact(n: number) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

function tick(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function TrafficChart({ buckets }: { buckets: TimeBucket[] }) {
  const c = useChartColors();
  const data = buckets.map((b) => ({ t: b.start, ok: b.requests - b.errors, errors: b.errors }));
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} barCategoryGap={2}>
        <defs>
          <linearGradient id="ember-bar" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff8a3d" />
            <stop offset="100%" stopColor={c.accent} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} stroke={c.grid} strokeDasharray="3 4" />
        <XAxis dataKey="t" tickFormatter={tick} minTickGap={40} axisLine={false} tickLine={false} />
        <YAxis allowDecimals={false} axisLine={false} tickLine={false} width={36} />
        <Tooltip cursor={{ fill: c.cursor }} content={<ChartTooltip labelFormat={(l) => fmt.time(String(l))} valueFormat={fmt.int} />} />
        <Bar dataKey="ok" name="answered" stackId="a" fill="url(#ember-bar)" radius={[0, 0, 0, 0]} />
        <Bar dataKey="errors" name="failed" stackId="a" fill={c.bad} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ModelDonut({ data, order }: { data: Record<string, number>; order: string[] }) {
  const rows = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }));
  const total = rows.reduce((a, r) => a + r.value, 0);
  if (!rows.length) return <EmptyState title="No data" />;
  return (
    <div className="flex flex-col items-center gap-6 sm:flex-row">
      <div className="relative h-44 w-44 shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={rows} dataKey="value" nameKey="name" innerRadius="66%" outerRadius="100%" paddingAngle={2} cornerRadius={4} stroke="none" isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.name} fill={seriesColor(r.name, order)} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip valueFormat={(v) => `${fmt.int(v)} · ${fmt.pct(v / total)}`} />} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="tabular font-display text-2xl font-semibold">{fmtCompact(total)}</span>
          <span className="text-[11px] text-ink-3">requests</span>
        </div>
      </div>
      <ul className="w-full min-w-0 space-y-2">
        {rows.map((r) => (
          <li key={r.name} className="flex items-center gap-2 text-[13px]">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: seriesColor(r.name, order) }} />
            <span className="truncate font-mono text-xs">{r.name}</span>
            <span className="tabular ml-auto text-ink-2">{fmt.pct(r.value / total)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

