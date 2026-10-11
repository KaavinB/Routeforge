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

import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader, SERIES, Td, Th } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { fmt, type BenchmarkListItem } from "../services/api";

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyObj = Record<string, any>;

export function BenchmarksPage() {
  const list = useApi<{ benchmarks: BenchmarkListItem[] }>("/api/benchmarks");
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    if (!selected && list.data?.benchmarks.length) setSelected(list.data.benchmarks[0].name);
  }, [list.data, selected]);
  const detail = useApi<AnyObj>(selected ? `/api/benchmarks/${encodeURIComponent(selected)}` : null);

  return (
    <>
      <PageHeader
        title="Benchmark reports"
        description="Results written by the benchmarks/ suite (machine-readable summary.json per experiment). Every number here comes from a recorded run; open the metadata to see commit, dataset, pricing and environment."
      />
      {list.error && <ErrorState error={list.error} />}
      {list.loading && !list.data && <Loading />}
      {list.data && list.data.benchmarks.length === 0 && (
        <Card>
          <EmptyState title="No benchmark results found">
            Run <code className="font-mono">make replay</code> or <code className="font-mono">make benchmarks</code>; results appear in <code className="font-mono">benchmarks/results/</code>.
          </EmptyState>
        </Card>
      )}
      {list.data && list.data.benchmarks.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
          <Card title="Experiments">
            <ul className="-m-2 space-y-0.5">
              {list.data.benchmarks.map((b) => (
                <li key={b.name}>
                  <button
                    onClick={() => setSelected(b.name)}
                    className={`w-full rounded-md px-2 py-1.5 text-left text-xs ${selected === b.name ? "bg-accent-soft text-accent" : "hover:bg-sunken"}`}
                  >
                    <div className="font-mono font-medium">{b.name}</div>
                    <div className="text-ink-3">
                      {b.timestamp ? fmt.time(b.timestamp) : "—"} · {b.mode ?? "?"} {b.status !== "complete" && <Badge tone="warn">{b.status}</Badge>}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          <div className="min-w-0 space-y-4">
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
        <div className="rounded-lg border border-warn/30 bg-warn-soft px-3 py-2 text-sm text-warn">
          <strong>Simulated providers.</strong> This benchmark measures the gateway itself (routing, caching, reliability, throughput) against mock models. It says nothing about real model quality or provider latency.
        </div>
      )}
      {data.headline && <Headline items={data.headline} />}
      {data.policies && <PolicyView data={data} />}
      {data.levels && <LoadView levels={data.levels} />}
      {data.scenarios && <ScenarioView scenarios={data.scenarios} />}
      {data.cache_modes && <CacheView modes={data.cache_modes} />}
      {data.overhead && <OverheadView overhead={data.overhead} />}
      {data.limitations && (
        <Card title="Limitations">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            {data.limitations.map((l: string) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Card>
      )}
      <Card title="Experiment metadata">
        <pre className="max-h-80 overflow-auto rounded-lg bg-sunken p-3 font-mono text-[11px] leading-relaxed">{JSON.stringify(meta, null, 2)}</pre>
      </Card>
    </>
  );
}

function Headline({ items }: { items: AnyObj[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {items.map((h) => (
        <div key={h.label} className="rounded-xl border border-line bg-panel px-4 py-3">
          <div className="text-xs font-medium text-ink-2">{h.label}</div>
          <div className="tabular mt-1 text-xl font-semibold">{h.value}</div>
          {h.detail && <div className="mt-0.5 text-xs text-ink-3">{h.detail}</div>}
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
  const refKinds: [string, string][] = [["fixed", "#9a9893"], ["heuristic", SERIES[1]], ["random", SERIES[2]], ["oracle", "#52514e"]];
  return (
    <>
      {points.length > 0 && (
        <Card title="Quality vs. cost (Pareto frontier)" subtitle="Each point is a policy on the same held-out prompts. Up and to the left is better; the blue line sweeps the learned router's quality threshold.">
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid stroke="#e4e3df" />
              <XAxis type="number" dataKey="x" scale="log" domain={["auto", "auto"]} name="$ / 1k requests" tickFormatter={(v) => `$${Number(v).toPrecision(2)}`} axisLine={false} tickLine={false} />
              <YAxis type="number" dataKey="y" name="acceptable %" unit="%" domain={["auto", "auto"]} axisLine={false} tickLine={false} width={48} />
              <ZAxis range={[60, 60]} />
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
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-line">
                <Th>Policy</Th><Th right>Acceptable</Th><Th right>95% CI</Th><Th right>$ / 1k req</Th><Th right>$ / acceptable</Th><Th right>p50</Th><Th right>p95</Th><Th right>Violations</Th><Th>Model mix</Th>
              </tr>
            </thead>
            <tbody>
              {names.map((n) => {
                const b = pol[n];
                const mix = Object.entries(b.model_distribution ?? {}) as [string, number][];
                return (
                  <tr key={n} className="border-b border-line/60 last:border-0">
                    <Td mono>{n}</Td>
                    <Td right>{fmt.pct(b.acceptable_rate)}</Td>
                    <Td right>
                      {b.acceptable_rate_ci95 ? `${fmt.pct(b.acceptable_rate_ci95[0])} – ${fmt.pct(b.acceptable_rate_ci95[1])}` : "—"}
                    </Td>
                    <Td right>{fmt.usd(b.cost_per_1k_requests_usd ?? (b.cost_per_request_usd ?? 0) * 1000)}</Td>
                    <Td right>{fmt.usd(b.cost_per_acceptable_usd)}</Td>
                    <Td right>{fmt.ms(b.latency_p50_ms)}</Td>
                    <Td right>{fmt.ms(b.latency_p95_ms)}</Td>
                    <Td right>{b.quality_violation_rate == null ? "—" : fmt.pct(b.quality_violation_rate)}</Td>
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
          </table>
        </div>
      </Card>
      {data.comparisons && (
        <Card title="Paired comparisons" subtitle="Paired bootstrap (resampling prompts) · 95% intervals">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-line">
                  <Th>Comparison (candidate vs baseline)</Th><Th right>Cost reduction</Th><Th right>Quality Δ (pp)</Th><Th right>Quality retention</Th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.comparisons as AnyObj).map(([k, c]) => (
                  <tr key={k} className="border-b border-line/60 last:border-0">
                    <Td mono>{k}</Td>
                    <Td right>{ci(c.cost_reduction_pct, "%")}</Td>
                    <Td right>{ci(c.quality_diff_pp, "")}</Td>
                    <Td right>{ci(c.quality_retention_pct, "%")}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}

function ci(x: AnyObj | undefined, unit: string) {
  if (!x) return "—";
  const f = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(1);
  return `${f(x.estimate)}${unit} [${f(x.ci_low)}, ${f(x.ci_high)}]`;
}

function LoadView({ levels }: { levels: AnyObj[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Throughput by concurrency" subtitle="Successful requests per second">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={levels}>
            <CartesianGrid stroke="#e4e3df" vertical={false} />
            <XAxis dataKey="users" axisLine={false} tickLine={false} />
            <YAxis axisLine={false} tickLine={false} width={44} />
            <Tooltip />
            <Line dataKey="success_rps" name="successful req/s" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Latency by concurrency" subtitle="Client-observed end-to-end latency">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={levels}>
            <CartesianGrid stroke="#e4e3df" vertical={false} />
            <XAxis dataKey="users" axisLine={false} tickLine={false} />
            <YAxis axisLine={false} tickLine={false} width={52} tickFormatter={(v) => fmt.ms(Number(v))} />
            <Tooltip formatter={(v) => fmt.ms(Number(v))} />
            <Legend />
            <Line dataKey="p50_ms" name="p50" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
            <Line dataKey="p95_ms" name="p95" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 4 }} />
            <Line dataKey="p99_ms" name="p99" stroke={SERIES[2]} strokeWidth={2} dot={{ r: 4 }} />
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
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            {cols.map((c) => (
              <Th key={c} right={typeof rows[0][c] === "number"}>
                {c.replaceAll("_", " ")}
              </Th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0">
              {cols.map((c) => (
                <Td key={c} right={typeof r[c] === "number"} mono={typeof r[c] === "string"}>
                  {show(r[c])}
                </Td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
