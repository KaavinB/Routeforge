import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { BarChart3, FlaskConical } from "lucide-react";

import { Badge, Card, ChartTooltip, EmptyState, ErrorState, Loading, PageHeader, SERIES, Table, Td, Th, rowClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { useChartColors } from "../hooks/useTheme";
import { fmt, type BenchmarkListItem } from "../services/api";
import { benchmarkTitle } from "../services/labels";

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyObj = Record<string, any>;

export function BenchmarksPage() {
  const list = useApi<{ benchmarks: BenchmarkListItem[] }>("/api/benchmarks");
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    const all = list.data?.benchmarks ?? [];
    // Open on the headline experiment (live routing on the latest model pool) when present.
    if (!selected && all.length) setSelected((all.find((x) => x.name.includes("routing_live_latest")) ?? all[0]).name);
  }, [list.data, selected]);
  const detail = useApi<AnyObj>(selected ? `/api/benchmarks/${encodeURIComponent(selected)}` : null);

  return (
    <>
      <PageHeader
        title="Benchmarks"
        description="Recorded experiments behind every claim: does routing save money, does the cache help, does it survive outages? Each result is saved with the exact code version and settings it ran with."
      />
      {list.error && <ErrorState error={list.error} />}
      {list.loading && !list.data && <Loading />}
      {list.data && list.data.benchmarks.length === 0 && (
        <Card>
          <EmptyState title="No benchmark results found" icon={BarChart3}>
            Run <code className="font-mono">make replay</code> or <code className="font-mono">make benchmarks</code>; results appear in <code className="font-mono">benchmarks/results/</code>.
          </EmptyState>
        </Card>
      )}
      {list.data && list.data.benchmarks.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]">
          <nav aria-label="Experiments" className="space-y-1 lg:sticky lg:top-6 lg:self-start">
            <div className="mb-2 px-1 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">Experiments</div>
            {list.data.benchmarks.map((b) => {
              const t = benchmarkTitle(b.name);
              const on = selected === b.name;
              return (
                <button
                  key={b.name}
                  onClick={() => setSelected(b.name)}
                  aria-current={on ? "true" : undefined}
                  className={`flex w-full items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-all ${on ? "border-accent/40 bg-panel shadow-card" : "border-transparent hover:bg-panel/70"}`}
                >
                  <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg font-display text-sm font-bold ${on ? "bg-linear-to-br from-[#ff8a3d] to-accent text-white" : "bg-sunken text-ink-2"}`}>{t.letter}</span>
                  <span className="min-w-0">
                    <span className={`block text-[13px] font-medium ${on ? "text-ink" : "text-ink-2"}`}>{t.title}</span>
                    <span className="block truncate text-[11px] text-ink-3">
                      {t.detail || b.name} {b.status !== "complete" && <Badge tone="warn">{b.status}</Badge>}
                    </span>
                  </span>
                </button>
              );
            })}
          </nav>
          <div className="min-w-0 space-y-5">
            {detail.error && <ErrorState error={detail.error} />}
            {detail.loading && !detail.data && <Loading />}
            {detail.data && <BenchmarkView data={detail.data} />}
          </div>
        </div>
      )}
    </>
  );
}

function BenchmarkView({ data }: { data: AnyObj }) {
  const meta = data.metadata ?? {};
  return (
    <>
      {(meta.mode === "mock" || data.simulated) && (
        <div className="flex items-start gap-3 rounded-2xl border border-warn/25 bg-warn-soft px-4 py-3 text-sm text-warn">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <div>
          <strong>Simulated providers.</strong> This benchmark measures the gateway itself (routing, caching, reliability, throughput) against mock models. It says nothing about real model quality or provider latency.
          </div>
        </div>
      )}
      {data.headline && <Headline items={data.headline} />}
      {data.policies && <PolicyView data={data} />}
      {data.levels && <LoadView levels={data.levels} />}
      {data.scenarios && <ScenarioView scenarios={data.scenarios} />}
      {data.cache_modes && <CacheView modes={data.cache_modes} />}
      {data.overhead && <OverheadView overhead={data.overhead} />}
      {data.limitations && (
        <Card title="Limitations" >
          <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-ink-2 marker:text-accent">
            {data.limitations.map((l: string) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Card>
      )}
      <details className="group rounded-2xl border border-line bg-panel shadow-card">
        <summary className="cursor-pointer list-none px-5 py-4 text-sm font-medium text-ink-2 select-none hover:text-ink">
          <span className="mr-1 inline-block transition-transform group-open:rotate-90">›</span> Run details (commit, dataset, pricing, environment)
        </summary>
        <pre className="mx-5 mb-5 max-h-80 overflow-auto rounded-xl bg-anvil p-4 font-mono text-[11px] leading-relaxed text-[#f3d9c4]">{JSON.stringify(meta, null, 2)}</pre>
      </details>
    </>
  );
}

function Headline({ items }: { items: AnyObj[] }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {items.map((h, i) => (
        <div
          key={h.label}
          className={`animate-rise relative overflow-hidden rounded-2xl border px-4 py-4 shadow-card ${i === 0 ? "border-transparent bg-anvil text-white" : "border-line bg-panel"}`}
        >
          {i === 0 && <div className="absolute -top-10 -right-10 h-32 w-32 rounded-full bg-[#ff6a1f]/40 blur-2xl" />}
          <div className={`relative text-xs font-medium ${i === 0 ? "text-white/60" : "text-ink-2"}`}>{h.label}</div>
          <div className={`tabular relative mt-1.5 font-display text-[28px] leading-none font-semibold tracking-tight ${i === 0 ? "bg-linear-to-r from-[#ffc46b] to-[#ff7a2e] bg-clip-text text-transparent" : ""}`}>{h.value}</div>
          {h.detail && <div className={`relative mt-1.5 text-xs ${i === 0 ? "text-white/50" : "text-ink-3"}`}>{h.detail}</div>}
        </div>
      ))}
    </div>
  );
}

function PolicyView({ data }: { data: AnyObj }) {
  const pol: AnyObj = data.policies;
  const names = Object.keys(pol);
  const points: AnyObj[] = data.pareto?.points ?? [];
  const sweep = points.filter((p) => p.kind === "learned_sweep").map((p) => ({ x: 1000 * p.cost_per_request_usd, y: 100 * p.acceptable_rate, label: p.label }));
  const c = useChartColors();
  const refKinds: [string, string][] = [["fixed", c.neutral], ["heuristic", SERIES[1]], ["random", SERIES[2]], ["oracle", c.ink2]];
  return (
    <>
      {points.length > 0 && (
        <Card title="Quality vs. cost" subtitle="Each dot is a policy on the same prompts. Up and left is better. The blue line is the learned router at different quality bars.">
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid stroke={c.grid} strokeDasharray="3 4" />
              <XAxis
                type="number"
                dataKey="x"
                scale="log"
                domain={["auto", "auto"]}
                ticks={LOG_TICKS}
                name="$ / 1k requests"
                tickFormatter={(v) => `$${Number(v)}`}
                axisLine={false}
                tickLine={false}
                label={{ value: "estimated $ per 1,000 requests (log scale)", position: "insideBottom", offset: -4, fontSize: 11, fill: c.axis }}
              />
              <YAxis type="number" dataKey="y" name="acceptable %" unit="%" domain={["auto", "auto"]} axisLine={false} tickLine={false} width={48} />
              <ZAxis range={[110, 110]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ payload }) => {
                  const p = payload?.[0]?.payload;
                  if (!p) return null;
                  return (
                    <div className="rounded-lg border border-line bg-panel px-2 py-1 text-xs shadow-sm">
                      <div className="font-mono font-medium">{p.label}</div>
                      <div>{p.y.toFixed(1)}% acceptable · ${p.x.toFixed(4)} / 1k req</div>
                    </div>
                  );
                }}
              />
              <Legend />
              <Scatter name="learned router (τ sweep)" data={sweep} fill={SERIES[0]} line={{ stroke: SERIES[0], strokeWidth: 2 }} />
              {refKinds.map(([kind, color]) => {
                const pts = points.filter((p) => p.kind === kind).map((p) => ({ x: 1000 * p.cost_per_request_usd, y: 100 * p.acceptable_rate, label: p.label }));
                return pts.length ? <Scatter key={kind} name={kind === "fixed" ? "single model" : kind} data={pts} fill={color} shape={kind === "fixed" ? "square" : kind === "oracle" ? "star" : "diamond"} /> : null;
              })}
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      )}
      <Card title="Policy comparison" subtitle={data.metadata?.methodology}>
        <Table>
            <thead>
              <tr>
                <Th>Policy</Th><Th right>Acceptable</Th><Th right>$ / 1k req</Th><Th right>Typical latency</Th><Th>Model mix</Th>
              </tr>
            </thead>
            <tbody>
              {names.map((n) => {
                const b = pol[n];
                const mix = Object.entries(b.model_distribution ?? {}) as [string, number][];
                return (
                  <tr key={n} className={rowClass}>
                    <Td mono>{n}</Td>
                    <Td right>
                      <span title={b.acceptable_rate_ci95 ? `95% CI ${fmt.pct(b.acceptable_rate_ci95[0])} – ${fmt.pct(b.acceptable_rate_ci95[1])}` : undefined}>{fmt.pct(b.acceptable_rate)}</span>
                    </Td>
                    <Td right>{fmt.usd(b.cost_per_1k_requests_usd ?? (b.cost_per_request_usd ?? 0) * 1000)}</Td>
                    <Td right>{fmt.ms(b.latency_p50_ms)}</Td>
                    <Td>
                      <div className="flex h-2 w-32 overflow-hidden rounded bg-sunken" title={mix.map(([m, s]) => `${m} ${(100 * s).toFixed(0)}%`).join(", ")}>
                        {mix.map(([m, s], i) => (
                          <div key={m} style={{ width: `${100 * s}%`, background: SERIES[i % SERIES.length] }} />
                        ))}
                      </div>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
        </Table>
      </Card>
      {data.comparisons && (
        <Card title="Head-to-head" subtitle="Paired bootstrap (resampling prompts) · 95% intervals. If an interval crosses zero, the difference isn't conclusive.">
          <Table>
              <thead>
                <tr>
                  <Th>Comparison (candidate vs baseline)</Th><Th right>Cost reduction</Th><Th right>Quality Δ (pp)</Th><Th right>Quality retention</Th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.comparisons as AnyObj).map(([k, c]) => (
                  <tr key={k} className={rowClass}>
                    <Td mono>{k}</Td>
                    <Td right>{ci(c.cost_reduction_pct, "%")}</Td>
                    <Td right>{ci(c.quality_diff_pp, "")}</Td>
                    <Td right>{ci(c.quality_retention_pct, "%")}</Td>
                  </tr>
                ))}
              </tbody>
          </Table>
        </Card>
      )}
    </>
  );
}

const LOG_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];

function ci(x: AnyObj | undefined, unit: string) {
  if (!x) return "—";
  const f = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(1);
  return `${f(x.estimate)}${unit} [${f(x.ci_low)}, ${f(x.ci_high)}]`;
}

function LoadView({ levels }: { levels: AnyObj[] }) {
  const c = useChartColors();
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Throughput by concurrency" subtitle="Successful requests per second as more clients pile on">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={levels}>
            <CartesianGrid stroke={c.grid} strokeDasharray="3 4" vertical={false} />
            <XAxis dataKey="users" axisLine={false} tickLine={false} />
            <YAxis axisLine={false} tickLine={false} width={44} />
            <Tooltip content={<ChartTooltip labelFormat={(l) => `${l} clients`} valueFormat={(v) => v.toFixed(0)} />} />
            <Line dataKey="success_rps" name="successful req/s" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Latency by concurrency" subtitle="Client-observed end-to-end latency">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={levels}>
            <CartesianGrid stroke={c.grid} strokeDasharray="3 4" vertical={false} />
            <XAxis dataKey="users" axisLine={false} tickLine={false} />
            <YAxis axisLine={false} tickLine={false} width={52} tickFormatter={(v) => fmt.ms(Number(v))} />
            <Tooltip content={<ChartTooltip labelFormat={(l) => `${l} clients`} valueFormat={fmt.ms} />} />
            <Legend />
            <Line dataKey="p50_ms" name="typical" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
            <Line dataKey="p95_ms" name="slowest 5%" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 4 }} />
            <Line dataKey="p99_ms" name="slowest 1%" stroke={SERIES[2]} strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Load levels" className="lg:col-span-2">
        <GenericTable rows={levels} />
      </Card>
    </div>
  );
}

function ScenarioView({ scenarios }: { scenarios: AnyObj[] }) {
  return (
    <Card title="Failure-injection scenarios" subtitle="Recovery = request succeeded despite the injected fault; graceful degradation is reported separately from full recovery.">
      <GenericTable rows={scenarios} />
    </Card>
  );
}

function CacheView({ modes }: { modes: AnyObj }) {
  return (
    <Card title="Cache effectiveness">
      <GenericTable rows={Object.entries(modes).map(([k, v]) => ({ workload: k, ...(v as AnyObj) }))} />
    </Card>
  );
}

function OverheadView({ overhead }: { overhead: AnyObj }) {
  return (
    <Card title="Router overhead">
      <GenericTable rows={Object.entries(overhead).map(([k, v]) => ({ measurement: k, ...(v as AnyObj) }))} />
    </Card>
  );
}

function GenericTable({ rows }: { rows: AnyObj[] }) {
  if (!rows.length) return <EmptyState title="No rows" />;
  const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).filter((c) => typeof rows[0][c] !== "object" || rows[0][c] === null);
  const show = (v: unknown) =>
    typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2)) : v == null ? "—" : String(v);
  return (
    <Table>
        <thead>
          <tr>
            {cols.map((c) => (
              <Th key={c} right={typeof rows[0][c] === "number"}>
                {c.replaceAll("_", " ")}
              </Th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={rowClass}>
              {cols.map((c) => (
                <Td key={c} right={typeof r[c] === "number"} mono={typeof r[c] === "string"}>
                  {show(r[c])}
                </Td>
              ))}
            </tr>
          ))}
        </tbody>
    </Table>
  );
}
