import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader, SimulatedBanner, StatusBadge, Td, Th } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, fmt, type EvalRun } from "../services/api";

const DEFAULT_TARGETS = "routeforge-learned, routeforge-heuristic, routeforge-premium, routeforge-economy";

export function EvaluationsPage() {
  const runs = useApi<{ runs: EvalRun[] }>("/api/evaluations/results?limit=100", 8000);
  const [targets, setTargets] = useState(DEFAULT_TARGETS);
  const [limit, setLimit] = useState("20");
  const [judge, setJudge] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<Error | null>(null);

  async function start() {
    setErr(null);
    setMsg(null);
    try {
      const body = {
        targets: targets.split(",").map((t) => t.trim()).filter(Boolean),
        split: "test",
        limit: limit ? Number(limit) : null,
        use_judge: judge,
      };
      const r = await api<{ run_ids: string[] }>("/api/evaluations/run", { method: "POST", body: JSON.stringify(body) });
      setMsg(`Started ${r.run_ids.length} run(s).`);
      runs.reload();
    } catch (e) {
      setErr(e as Error);
    }
  }

  const done = (runs.data?.runs ?? []).filter((r) => r.status === "done" && r.summary);
  const latestByTarget = new Map<string, EvalRun>();
  for (const r of done) if (!latestByTarget.has(r.target)) latestByTarget.set(r.target, r);
  const compare = [...latestByTarget.values()].map((r) => {
    const o = r.summary!.overall;
    const rate = o.acceptable_rate ?? 0;
    return {
      target: r.target.replace("routeforge-", ""),
      rate: 100 * rate,
      err: [100 * (rate - o.acceptable_rate_ci95[0]), 100 * (o.acceptable_rate_ci95[1] - rate)],
      costPer1k: 1000 * (o.cost_per_request_usd ?? 0),
      simulated: o.simulated,
    };
  });

  return (
    <>
      <PageHeader
        title="Evaluations"
        description="Run the versioned benchmark suite through the gateway (cache bypassed) and compare models and routing policies. Offline policy replays and the full methodology live under Benchmarks."
      />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]">
        <Card title="Start an evaluation" subtitle="Dataset routeforge-bench-v1 · held-out test split">
          <div className="space-y-3 text-xs">
            <label className="block space-y-1">
              <span className="text-ink-2">Targets (model IDs or routeforge-&lt;strategy&gt;)</span>
              <input value={targets} onChange={(e) => setTargets(e.target.value)} className="w-full rounded-md border border-line bg-surface px-2 py-1 font-mono" />
            </label>
            <div className="flex items-center gap-4">
              <label className="space-y-1">
                <span className="text-ink-2">Cases per target</span>
                <input value={limit} onChange={(e) => setLimit(e.target.value.replace(/\D/g, ""))} className="w-24 rounded-md border border-line bg-surface px-2 py-1" placeholder="all" />
              </label>
              <label className="mt-4 flex items-center gap-1.5">
                <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} /> LLM judge for summaries (paid)
              </label>
            </div>
            <button onClick={start} className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90">
              Run evaluation
            </button>
            {msg && <div className="text-good">{msg}</div>}
            {err && <ErrorState error={err} />}
            <p className="text-ink-3">In hosted mode this calls paid APIs. Without the judge, summaries are scored by a ROUGE-L fallback that is not used for headline metrics.</p>
          </div>
        </Card>
        <Card title="Latest result per target" subtitle="Acceptable-answer rate with 95% Wilson interval">
          {compare.length === 0 ? (
            <EmptyState title="No completed evaluation runs">Start one on the left, or run benchmarks from the CLI.</EmptyState>
          ) : (
            <>
              <SimulatedBanner show={compare.some((c) => c.simulated)}>Some runs used simulated (mock) providers; their quality numbers are meaningless.</SimulatedBanner>
              <ResponsiveContainer width="100%" height={Math.max(160, 40 * compare.length + 40)}>
                <BarChart data={compare} layout="vertical" margin={{ left: 8, right: 48 }}>
                  <CartesianGrid horizontal={false} stroke="#e4e3df" />
                  <XAxis type="number" domain={[0, 100]} unit="%" axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="target" width={150} axisLine={false} tickLine={false} />
                  <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} cursor={{ fill: "#f4f4f2" }} />
                  <Bar dataKey="rate" name="acceptable" fill="#2a78d6" radius={[0, 4, 4, 0]} barSize={16}>
                    <ErrorBar dataKey="err" width={4} stroke="#52514e" direction="x" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>
      <div className="mt-4">
        <Card title="Evaluation history">
          {runs.error && <ErrorState error={runs.error} />}
          {runs.loading && !runs.data && <Loading />}
          {runs.data && runs.data.runs.length === 0 && <EmptyState title="No evaluation runs yet" />}
          {runs.data && runs.data.runs.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-line">
                    <Th>Started</Th><Th>Target</Th><Th>Dataset</Th><Th right>Cases</Th><Th right>Acceptable</Th><Th right>Cost / acceptable</Th><Th right>p50 latency</Th><Th>Status</Th><Th>Commit</Th>
                  </tr>
                </thead>
                <tbody>
                  {runs.data.runs.map((r) => (
                    <tr key={r.id} className="border-b border-line/60 last:border-0">
                      <Td>{fmt.time(r.created_at)}</Td>
                      <Td mono>{r.target}</Td>
                      <Td mono>
                        {r.dataset}@{r.dataset_version}:{r.split}
                      </Td>
                      <Td right>{fmt.int(r.summary?.overall.n)}</Td>
                      <Td right>
                        {fmt.pct(r.summary?.overall.acceptable_rate)} {r.summary?.overall.simulated && <Badge tone="warn">sim</Badge>}
                      </Td>
                      <Td right>{fmt.usd(r.summary?.overall.cost_per_acceptable_usd)}</Td>
                      <Td right>{fmt.ms(r.summary?.overall.latency_p50_ms)}</Td>
                      <Td>
                        <StatusBadge status={r.status} />
                        {r.error && <span className="ml-1 text-xs text-bad">{r.error.slice(0, 60)}</span>}
                      </Td>
                      <Td mono>{r.git_sha?.slice(0, 7) ?? "—"}</Td>
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
