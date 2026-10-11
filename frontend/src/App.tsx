import { Activity, BarChart3, Boxes, FlaskConical, GitBranch, KeyRound, ListTree } from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";

import { Badge } from "./components/ui";
import { useApi } from "./hooks/useApi";
import { BenchmarksPage } from "./pages/Benchmarks";
import { EvaluationsPage } from "./pages/Evaluations";
import { ModelsPage } from "./pages/Models";
import { OverviewPage } from "./pages/Overview";
import { RequestsPage } from "./pages/Requests";
import { RoutingPage } from "./pages/Routing";
import { getAdminKey, setAdminKey } from "./services/api";

interface ReadyInfo {
  status: string;
  simulated: boolean;
  default_strategy: string;
  components: Record<string, string>;
}

const PAGES: { id: string; label: string; icon: ComponentType<{ className?: string }>; Component: ComponentType }[] = [
  { id: "overview", label: "Overview", icon: Activity, Component: OverviewPage },
  { id: "models", label: "Models", icon: Boxes, Component: ModelsPage },
  { id: "routing", label: "Routing", icon: GitBranch, Component: RoutingPage },
  { id: "evaluations", label: "Evaluations", icon: FlaskConical, Component: EvaluationsPage },
  { id: "requests", label: "Requests", icon: ListTree, Component: RequestsPage },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3, Component: BenchmarksPage },
];

function currentPage(): string {
  const id = window.location.hash.replace("#/", "").split("?")[0];
  return PAGES.some((p) => p.id === id) ? id : "overview";
}

export default function App() {
  const [page, setPage] = useState(currentPage);
  const [key, setKey] = useState(getAdminKey);
  const [draft, setDraft] = useState("");
  const ready = useApi<ReadyInfo>("/ready", 15000);

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const Active = PAGES.find((p) => p.id === page)!.Component;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-line bg-panel md:flex">
        <div className="flex items-center gap-2 px-4 py-4">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm font-bold text-white">R</div>
          <div>
            <div className="text-sm font-semibold">RouteForge</div>
            <div className="text-[11px] text-ink-3">LLM gateway console</div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 px-2">
          {PAGES.map(({ id, label, icon: Icon }) => (
            <a
              key={id}
              href={`#/${id}`}
              className={`flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm ${
                page === id ? "bg-accent-soft font-medium text-accent" : "text-ink-2 hover:bg-sunken hover:text-ink"
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </a>
          ))}
        </nav>
        <div className="mt-auto space-y-3 border-t border-line p-3 text-xs">
          {ready.data && (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-ink-2">Gateway</span>
                <Badge tone={ready.data.status === "ready" ? "good" : ready.data.status === "degraded" ? "warn" : "bad"}>{ready.data.status}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-2">Mode</span>
                <Badge tone={ready.data.simulated ? "warn" : "accent"}>{ready.data.components.models?.split("(")[1]?.replace(")", "") ?? "?"}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-2">Default policy</span>
                <span className="font-mono">{ready.data.default_strategy}</span>
              </div>
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setAdminKey(draft.trim());
              setKey(draft.trim());
              setDraft("");
              window.location.reload();
            }}
            className="space-y-1.5"
          >
            <label className="flex items-center gap-1 text-ink-2">
              <KeyRound className="h-3.5 w-3.5" /> Admin API key {key ? <Badge tone="good">set</Badge> : <Badge tone="warn">missing</Badge>}
            </label>
            <input
              type="password"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={key ? "replace key…" : "rf-admin-…"}
              autoComplete="off"
              className="w-full rounded-md border border-line bg-surface px-2 py-1 font-mono text-xs focus:outline-2 focus:outline-accent"
            />
          </form>
        </div>
      </aside>
      <div className="min-w-0 flex-1">
        <div className="flex gap-1 overflow-x-auto border-b border-line bg-panel px-3 py-2 md:hidden">
          {PAGES.map(({ id, label }) => (
            <a key={id} href={`#/${id}`} className={`rounded-md px-2 py-1 text-xs ${page === id ? "bg-accent-soft text-accent" : "text-ink-2"}`}>
              {label}
            </a>
          ))}
        </div>
        <main className="mx-auto max-w-7xl px-4 py-6 md:px-8">
          <Active />
        </main>
      </div>
    </div>
  );
}
