// Human-friendly names and explanations for the gateway's internal vocabulary, so the
// dashboard reads well to people who don't know the technical terms.

export const STRATEGY_INFO: Record<string, { name: string; blurb: string }> = {
  learned: { name: "Learned router", blurb: "Predicts which models will answer well, then picks the cheapest one that clears the bar." },
  heuristic: { name: "Rule of thumb", blurb: "Hand-written rules: harder-looking tasks go to bigger models." },
  premium: { name: "Always premium", blurb: "Sends everything to the most capable model. The expensive baseline." },
  economy: { name: "Always cheapest", blurb: "Sends everything to the cheapest model. The budget baseline." },
  random: { name: "Random", blurb: "Picks a model at random — a sanity-check baseline." },
  explicit: { name: "Caller's choice", blurb: "The application asked for a specific model by name." },
  unresolved: { name: "Rejected early", blurb: "Stopped before routing (e.g. rate limit or bad request)." },
  "learned-no-cost": { name: "Learned, ignoring cost", blurb: "Experiment: picks any model that clears the bar, ignoring price." },
  "learned-task-only": { name: "Learned, task only", blurb: "Experiment: the router only knows what kind of task it is." },
  "learned-no-task": { name: "Learned, no task", blurb: "Experiment: the router is not told what kind of task it is." },
  "learned-full": { name: "Learned, all features", blurb: "Experiment: the router uses every signal, even weak ones." },
};

export const strategyName = (s: string) => STRATEGY_INFO[s]?.name ?? s;

export const TASK_NAMES: Record<string, string> = {
  classification: "Classification",
  code: "Coding",
  extraction: "Extraction",
  factual_qa: "Quick fact",
  json: "Structured JSON",
  reasoning: "Math & reasoning",
  summarization: "Summarising",
};

export const taskName = (t: string | null | undefined) => (t ? TASK_NAMES[t] ?? t.replaceAll("_", " ") : "—");

/** Turn a routing reason code into a sentence. */
export function explainReason(reason: string, model: string, opts: { threshold?: number | null; budget?: number | null } = {}): string {
  const bar = opts.threshold != null ? `${Math.round(opts.threshold * 100)}%` : "the quality bar";
  const withBudget = opts.budget ? " and fits the latency budget" : "";
  if (reason === "cheapest_meeting_constraints")
    return `${model} is the cheapest model predicted to give an acceptable answer at least ${bar} of the time${withBudget}.`;
  if (reason === "random_meeting_constraints") return `${model} was picked at random from the models that clear ${bar} (cost ignored).`;
  if (reason.startsWith("no_model_met_quality"))
    return opts.budget
      ? `No model is both fast enough for the latency budget and likely enough to clear ${bar}, so RouteForge chose the most reliable model that fits the budget: ${model}.`
      : `No model was likely enough to clear ${bar}, so RouteForge chose the most reliable one: ${model}.`;
  if (reason.startsWith("no_model_met_constraints"))
    return `Nothing met every constraint, so RouteForge played it safe with the model most likely to answer well: ${model}.`;
  if (reason === "fixed_target") return `This policy always sends requests to ${model}.`;
  if (reason.startsWith("fixed_target_unavailable")) return `The usual model was unavailable (${reason.split(":")[1] ?? "unknown"}), so ${model} stepped in.`;
  if (reason === "uniform_random") return `${model} was chosen at random — this policy is a baseline for comparison.`;
  if (reason === "client_pinned_model") return `The caller asked for ${model} by name, so routing was skipped.`;
  if (reason.startsWith("task=")) {
    const parts = Object.fromEntries(reason.split(";").map((p) => (p.includes("=") ? p.split("=") : [p, "1"])));
    const bumps = [parts.long_context && "a long prompt", parts.large_max_tokens && "a long answer", parts.json_requested && "JSON output"].filter(Boolean);
    return `Rule of thumb: ${taskName(parts.task).toLowerCase()} tasks need capability level ${parts.level} of 3${bumps.length ? `, raised for ${bumps.join(" and ")}` : ""}. ${model} is the cheapest model at that level.`;
  }
  return reason;
}

export const INELIGIBLE: Record<string, string> = {
  circuit_open: "temporarily benched (failing)",
  context_window: "prompt too long",
};

/** Friendly title for a benchmark result folder like "A_routing_live_latest". */
export function benchmarkTitle(name: string): { letter: string; title: string; detail: string } {
  const [letter, ...rest] = name.split("_");
  const key = rest.join("_");
  const known: Record<string, [string, string]> = {
    routing_live_latest: ["Routing, live", "latest models, real API calls"],
    routing_live_hosted: ["Routing, live", "earlier GPT-4.1 pool"],
    routing_offline_latest: ["Routing, offline replay", "latest models"],
    routing_offline_hosted: ["Routing, offline replay", "earlier GPT-4.1 pool"],
    cache_hosted: ["Response cache", "real provider"],
    capacity_replay: ["Capacity", "how fast can one worker go"],
    load_mock_realistic_latency: ["Load test", "simulated provider latency"],
    faults_mock: ["Fault injection", "outages, timeouts, 500s"],
    router_overhead: ["Router overhead", "cost of thinking"],
    regression_gate_validation_latest: ["Regression gate", "does CI catch bad changes?"],
    judge_agreement: ["Judge agreement", "do two LLM judges agree?"],
  };
  const [title, detail] = known[key] ?? [key.replaceAll("_", " "), ""];
  return { letter: letter.length === 1 ? letter : "•", title, detail };
}
