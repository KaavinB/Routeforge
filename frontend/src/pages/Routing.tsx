import { AlertTriangle, Braces, CheckCircle2, CircleSlash, Code2, Crown, MessageCircle, Tags, XCircle, type LucideIcon } from "lucide-react";
import { useEffect, useState, type CSSProperties } from "react";

import { Badge, Card, EmptyState, ErrorState, InfoTip, PageHeader, SkeletonRows, inputClass, seriesColor } from "../components/ui";
import { useModelOrder } from "../hooks/useApi";
import { api, fmt, type Candidate, type RoutingDetail } from "../services/api";
import { INELIGIBLE, STRATEGY_INFO, explainReason, strategyName, taskName } from "../services/labels";

const STRATEGIES = ["learned", "heuristic", "premium", "economy", "random", "learned-no-cost", "learned-task-only", "learned-no-task", "learned-full"];

// Example prompts use the same instruction style as the benchmark the router was trained on.
const EXAMPLES: { id: string; label: string; icon: LucideIcon; prompt: string }[] = [
  {
    id: "classify",
    label: "Sort a headline",
    icon: Tags,
    prompt: "Classify the news article into exactly one topic: World, Sports, Business, or Sci/Tech. Reply with the topic label only.\n\nArticle: Apple shares jumped 4% after the company reported record iPhone sales in Asia.",
  },
  {
    id: "json",
    label: "Fill a form",
    icon: Braces,
    prompt: 'Extract the requested fields from the text into a JSON object that conforms exactly to the given JSON Schema. Return only the JSON object.\n\nJSON Schema: {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}}}\n\nText: Priya Raman, 34, moved to Toronto last year.',
  },
  { id: "code", label: "Write code", icon: Code2, prompt: "Write a Python function that returns the n-th Fibonacci number using memoization." },
  { id: "chat", label: "Something unusual", icon: MessageCircle, prompt: "hey can you help me write a birthday message for my grandma" },
];

export function RoutingPage() {
  const [prompt, setPrompt] = useState(EXAMPLES[0].prompt);
  const [strategy, setStrategy] = useState("learned");
  const [threshold, setThreshold] = useState(0.8);
  const [budget, setBudget] = useState("");
  const [preview, setPreview] = useState<RoutingDetail | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [busy, setBusy] = useState(false);
  const order = useModelOrder();

  async function run(p = prompt) {
    if (!p.trim()) return;
    setError(null);
    setBusy(true);
    try {
      const qs = new URLSearchParams({ prompt: p, strategy, quality_threshold: String(threshold) });
      if (budget) qs.set("latency_budget_ms", budget);
      setPreview(await api(`/api/analytics/routing/preview?${qs}`));
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  }

  // Decide straight away, and again whenever a knob changes. Typing waits for a pause.
  useEffect(() => {
    const t = setTimeout(() => void run(), 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt, strategy, threshold, budget]);

  const activeExample = EXAMPLES.find((e) => e.prompt === prompt)?.id;

  return (
    <>
      <PageHeader
        title="Why this model?"
        description="RouteForge sends each request to the cheapest model that's likely to answer it well. Try a prompt to see the decision. Nothing is actually sent to a model."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Card>
          <div className="space-y-5">
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map(({ id, label, icon: Icon, prompt: p }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setPrompt(p)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                    activeExample === id ? "border-accent bg-accent-soft text-accent-ink" : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" aria-hidden />
                  {label}
                </button>
              ))}
            </div>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={6} aria-label="Prompt" className={`${inputClass} resize-y font-mono text-xs leading-relaxed`} />
            <label className="block space-y-2 text-xs">
              <span className="flex items-center justify-between font-medium text-ink-2">
                <span className="flex items-center gap-1.5">
                  Quality bar
                  <InfoTip label="the quality bar">The minimum predicted chance of an acceptable answer. Higher is safer but costs more; lower is cheaper but riskier.</InfoTip>
                </span>
                <span className="tabular font-semibold text-ink">{fmt.pct(threshold, 0)}</span>
              </span>
              <input
                type="range"
                min={0.3}
                max={0.99}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="forge-range w-full"
                style={{ "--fill": `${((threshold - 0.3) / 0.69) * 100}%` } as CSSProperties}
              />
              <span className="flex justify-between text-[11px] text-ink-3">
                <span>cheaper</span>
                <span>safer</span>
              </span>
            </label>
            <details className="group text-xs">
              <summary className="cursor-pointer list-none font-medium text-ink-2 select-none hover:text-ink">
                <span className="inline-block transition-transform group-open:rotate-90">›</span> More options
              </summary>
              <div className="mt-3 grid gap-4 sm:grid-cols-2">
                <label className="block space-y-1.5">
                  <span className="text-ink-2">Routing policy</span>
                  <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className={inputClass} title={STRATEGY_INFO[strategy]?.blurb}>
                    {STRATEGIES.map((s) => (
                      <option key={s} value={s}>
                        {strategyName(s)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block space-y-1.5">
                  <span className="text-ink-2">Latency budget (ms)</span>
                  <input value={budget} onChange={(e) => setBudget(e.target.value.replace(/\D/g, ""))} placeholder="no limit" className={inputClass} />
                </label>
              </div>
            </details>
            {error && <ErrorState error={error} />}
          </div>
        </Card>
        <Card>{preview ? <DecisionView d={preview} order={order} /> : busy ? <SkeletonRows n={6} /> : <EmptyState title="Type a prompt to see a decision" />}</Card>
      </div>
    </>
  );
}

export function DecisionView({ d, order }: { d: RoutingDetail; order?: string[] }) {
  const task = d.task_probs ? Object.entries(d.task_probs).sort((a, b) => b[1] - a[1])[0] : null;
  const colorOrder = order?.length ? order : d.candidates.map((c) => c.model_id);
  // Cheapest first, so "the first model over the bar wins" reads top-down.
  const ladder = [...d.candidates].sort((a, b) => (a.expected_cost_usd ?? Infinity) - (b.expected_cost_usd ?? Infinity));
  const tau = d.quality_threshold;

  return (
    <div className="space-y-5">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-ink-3">Sent to</span>
          <span className="inline-flex items-center gap-2 font-mono text-base font-semibold">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: seriesColor(d.selected_model, colorOrder) }} />
            {d.selected_model}
          </span>
          {d.uncertain && (
            <Badge tone="warn">
              <AlertTriangle className="h-3 w-3" aria-hidden /> unusual prompt, extra caution
            </Badge>
          )}
        </div>
        <p className="mt-2 text-[14px] leading-relaxed text-ink-2">
          {explainReason(d.reason, d.selected_model, { threshold: tau, budget: d.latency_budget_ms })}
          {task && task[1] >= 0.5 && <> Looks like a <span className="text-ink">{taskName(task[0]).toLowerCase()}</span> task.</>}
        </p>
        <p className="mt-1 font-mono text-[11px] text-ink-3" title={`decided in ${fmt.ms(d.decision_ms)}`}>
          {d.reason}
        </p>
      </div>

      <div>
        <div className="mb-2 grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_4.5rem] gap-x-4 px-3 text-[11px] text-ink-3">
          <span>Cheapest first</span>
          <span>Chance of a good answer{tau != null && ` (bar: ${fmt.pct(tau, 0)})`}</span>
          <span className="text-right">Cost / 1k</span>
        </div>
        <ul className="space-y-1.5">
          {ladder.map((c) => (
            <CandidateRow key={c.model_id} c={c} chosen={c.model_id === d.selected_model} color={seriesColor(c.model_id, colorOrder)} tau={tau} />
          ))}
        </ul>
      </div>
    </div>
  );
}

function CandidateRow({ c, chosen, color, tau }: { c: Candidate; chosen: boolean; color: string; tau: number | null }) {
  const p = c.p_quality_adjusted ?? c.p_quality;
  return (
    <li
      className={`grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_4.5rem] items-center gap-x-4 rounded-xl px-3 py-2 ${chosen ? "bg-accent-soft ring-1 ring-accent/30" : ""} ${c.eligible ? "" : "opacity-50"}`}
      title={c.p_quality != null ? `raw ${c.p_quality.toFixed(3)}, after safety margin ${c.p_quality_adjusted?.toFixed(3) ?? "—"} · expected latency ${fmt.ms(c.expected_latency_ms)}` : undefined}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="truncate font-mono text-xs font-medium">{c.model_id}</span>
        {chosen && <Crown className="h-3.5 w-3.5 shrink-0 text-accent" aria-label="chosen" />}
        {!c.eligible && <span className="truncate text-[11px] text-bad">{INELIGIBLE[c.ineligible_reason ?? ""] ?? c.ineligible_reason}</span>}
        {c.eligible && c.meets_latency === false && <span className="shrink-0 text-[11px] text-warn">too slow</span>}
      </div>
      <div className="flex items-center gap-2">
        {p == null ? (
          <span className="text-xs text-ink-3">not scored by this policy</span>
        ) : (
          <>
            <div className="relative h-2 flex-1 rounded-full bg-sunken">
              <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${p * 100}%`, background: color }} />
              {tau != null && <div className="absolute -top-1 -bottom-1 w-0.5 rounded bg-ink" style={{ left: `${tau * 100}%` }} />}
            </div>
            <span className="tabular w-9 text-right text-xs">{fmt.pct(p, 0)}</span>
            <Check ok={c.meets_quality} />
          </>
        )}
      </div>
      <div className="tabular text-right text-xs text-ink-2">{c.expected_cost_usd == null ? "—" : fmt.usd(c.expected_cost_usd * 1000)}</div>
    </li>
  );
}

function Check({ ok }: { ok: boolean | null }) {
  if (ok == null) return <CircleSlash className="h-4 w-4 shrink-0 text-ink-3" aria-label="not evaluated" />;
  return ok ? <CheckCircle2 className="h-4 w-4 shrink-0 text-good" aria-label="meets the bar" /> : <XCircle className="h-4 w-4 shrink-0 text-bad" aria-label="below the bar" />;
}
