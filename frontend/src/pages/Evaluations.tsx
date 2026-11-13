import { Play } from "lucide-react";
import { useState } from "react";

import { Badge, Button, Card, EmptyState, ErrorState, PageHeader, SimulatedBanner, SkeletonRows, StatusBadge, Table, Td, Th, Toggle, inputClass, rowClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, fmt, type EvalRun } from "../services/api";
import { strategyName } from "../services/labels";

const PRESETS = ["routeforge-learned", "routeforge-heuristic", "routeforge-premium", "routeforge-economy", "routeforge-random"];
const DEFAULT_TARGETS = PRESETS.slice(0, 4).join(", ");

const targetName = (t: string) => (t.startsWith("routeforge-") ? strategyName(t.replace("routeforge-", "")) : t);

export function EvaluationsPage() {
  const runs = useApi<{ runs: EvalRun[] }>("/api/evaluations/results?limit=100", 8000);
  const [targets, setTargets] = useState(DEFAULT_TARGETS);
  const [limit, setLimit] = useState("20");
  const [judge, setJudge] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<Error | null>(null);
  const [busy, setBusy] = useState(false);

  const list = targets.split(",").map((t) => t.trim()).filter(Boolean);
  const toggleTarget = (t: string) => setTargets((list.includes(t) ? list.filter((x) => x !== t) : [...list, t]).join(", "));

  async function start() {
    setErr(null);
    setMsg(null);
    setBusy(true);
    try {
      const body = { targets: list, split: "test", limit: limit ? Number(limit) : null, use_judge: judge };
      const r = await api<{ run_ids: string[] }>("/api/evaluations/run", { method: "POST", body: JSON.stringify(body) });
      setMsg(`Started ${r.run_ids.length} run(s). Results appear below as they finish.`);
      runs.reload();
    } catch (e) {
      setErr(e as Error);
    } finally {
      setBusy(false);
    }
  }

  const done = (runs.data?.runs ?? []).filter((r) => r.status === "done" && r.summary);
  const latestByTarget = new Map<string, EvalRun>();
  for (const r of done) if (!latestByTarget.has(r.target)) latestByTarget.set(r.target, r);
  const scores = [...latestByTarget.values()]
    .map((r) => ({ target: r.target, o: r.summary!.overall }))
    .sort((a, b) => (b.o.acceptable_rate ?? 0) - (a.o.acceptable_rate ?? 0));

  return (
    <>
      <PageHeader
        title="Evaluations"
        description="Run a fixed set of held-out test prompts through the gateway and grade every answer automatically, to see which routing policy gets the most right and at what price."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Card title="Run an evaluation">
          <div className="space-y-4 text-xs">
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((t) => {
                const on = list.includes(t);
                return (
                  <button
                    key={t}
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggleTarget(t)}
                    className={`rounded-full border px-3 py-1.5 font-medium transition-colors ${on ? "border-accent bg-accent-soft text-accent-ink" : "border-line text-ink-2 hover:border-ink-3"}`}
                  >
                    {targetName(t)}
                  </button>
                );
              })}
            </div>
            <details className="group">
              <summary className="cursor-pointer list-none font-medium text-ink-2 select-none hover:text-ink">
                <span className="inline-block transition-transform group-open:rotate-90">›</span> Advanced
              </summary>
              <div className="mt-3 space-y-3">
                <label className="block space-y-1.5">
                  <span className="text-ink-2">Targets: routeforge-&lt;policy&gt; or exact model IDs, comma-separated</span>
                  <input value={targets} onChange={(e) => setTargets(e.target.value)} className={`${inputClass} font-mono text-xs`} />
                </label>
                <label className="block space-y-1.5">
                  <span className="text-ink-2">Prompts per target (blank = all 161)</span>
                  <input value={limit} onChange={(e) => setLimit(e.target.value.replace(/\D/g, ""))} className={`${inputClass} w-28`} placeholder="all" />
                </label>
                <Toggle checked={judge} onChange={setJudge} label={<span>Grade summaries with an LLM judge <Badge tone="warn">paid</Badge></span>} />
              </div>
            </details>
            <Button onClick={start} disabled={busy || list.length === 0}>
              <Play className="h-4 w-4" aria-hidden /> {busy ? "Starting…" : "Run evaluation"}
            </Button>
            {msg && <div className="text-good">{msg}</div>}
            {err && <ErrorState error={err} />}
            <p className="text-ink-3">In hosted mode this calls paid APIs.</p>
          </div>
        </Card>
        <Card title="Scoreboard" subtitle="Latest run per target">
          {scores.length === 0 ? (
            <EmptyState title="No finished runs yet">Start one on the left.</EmptyState>
          ) : (
            <div className="space-y-4">
              <SimulatedBanner show={scores.some((x) => x.o.simulated)}>Some runs used simulated (mock) providers, so their quality numbers are meaningless.</SimulatedBanner>
              <ul className="space-y-3">
                {scores.map(({ target, o }, i) => {
                  const rate = o.acceptable_rate ?? 0;
                  return (
                    <li key={target} title={`95% interval ${fmt.pct(o.acceptable_rate_ci95[0])} – ${fmt.pct(o.acceptable_rate_ci95[1])} · ${o.n} prompts`}>
                      <div className="mb-1 flex items-baseline justify-between gap-3 text-[13px]">
                        <span className="font-medium text-ink">{targetName(target)}</span>
                        <span className="tabular text-xs text-ink-2">
                          <span className="font-semibold text-ink">{fmt.pct(rate)}</span> right · {fmt.usd(o.cost_per_acceptable_usd)} each
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-sunken">
                        <div className={`h-full rounded-full ${i === 0 ? "bg-accent" : "bg-ink-3/50"}`} style={{ width: `${rate * 100}%` }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
              <p className="text-[11px] text-ink-3">"Right" = answer graded acceptable. "Each" = cost per acceptable answer. Hover a row for its 95% confidence interval.</p>
            </div>
          )}
        </Card>
      </div>
      <div className="mt-5">
        <Card title="History">
          {runs.error && <ErrorState error={runs.error} />}
          {runs.loading && !runs.data && <SkeletonRows />}
          {runs.data && runs.data.runs.length === 0 && <EmptyState title="No evaluation runs yet" />}
          {runs.data && runs.data.runs.length > 0 && (
            <Table>
              <thead>
                <tr>
                  <Th>Started</Th><Th>Target</Th><Th right>Prompts</Th><Th right>Acceptable</Th><Th right>Cost / acceptable</Th><Th>Status</Th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((r) => (
                  <tr key={r.id} className={rowClass} title={`${r.dataset}@${r.dataset_version}:${r.split}${r.git_sha ? ` · commit ${r.git_sha.slice(0, 7)}` : ""}`}>
                    <Td>{fmt.time(r.created_at)}</Td>
                    <Td>{targetName(r.target)}</Td>
                    <Td right>{fmt.int(r.summary?.overall.n)}</Td>
                    <Td right>
                      <span className="inline-flex items-center gap-1.5">
                        {r.summary?.overall.simulated && <Badge tone="warn">sim</Badge>}
                        {fmt.pct(r.summary?.overall.acceptable_rate)}
                      </span>
                    </Td>
                    <Td right>{fmt.usd(r.summary?.overall.cost_per_acceptable_usd)}</Td>
                    <Td>
                      <StatusBadge status={r.status} />
                      {r.error && <span className="ml-1 text-xs text-bad">{r.error.slice(0, 60)}</span>}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>
      </div>
    </>
  );
}
