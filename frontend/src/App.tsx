import { Activity, BarChart3, Boxes, Check, FlaskConical, GitBranch, KeyRound, ListTree, Monitor, Moon, Sun, type LucideIcon } from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";

import { LogoMark } from "./components/Logo";
import { useApi } from "./hooks/useApi";
import { setThemePref, useTheme, type ThemePref } from "./hooks/useTheme";
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

const PAGES: { id: string; label: string; blurb: string; icon: LucideIcon; Component: ComponentType }[] = [
  { id: "overview", label: "Overview", blurb: "How it's going", icon: Activity, Component: OverviewPage },
  { id: "routing", label: "Routing", blurb: "Why each model was picked", icon: GitBranch, Component: RoutingPage },
  { id: "models", label: "Models", blurb: "Prices, speed, health", icon: Boxes, Component: ModelsPage },
  { id: "requests", label: "Requests", blurb: "Every call, inspected", icon: ListTree, Component: RequestsPage },
  { id: "evaluations", label: "Evaluations", blurb: "Is quality holding up?", icon: FlaskConical, Component: EvaluationsPage },
  { id: "benchmarks", label: "Benchmarks", blurb: "The receipts", icon: BarChart3, Component: BenchmarksPage },
];

function currentPage(): string {
  const id = window.location.hash.replace("#/", "").split("?")[0];
  return PAGES.some((p) => p.id === id) ? id : "overview";
}

export default function App() {
  const [page, setPage] = useState(currentPage);
  const ready = useApi<ReadyInfo>("/ready", 15000);

  useEffect(() => {
    const onHash = () => {
      setPage(currentPage());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const active = PAGES.find((p) => p.id === page)!;
  useEffect(() => {
    document.title = `${active.label} · RouteForge`;
  }, [active.label]);
  const Active = active.Component;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col overflow-y-auto bg-anvil text-white md:flex">
        <a href="#/overview" className="relative flex items-center gap-2.5 px-5 pt-6 pb-6">
          <LogoMark size={30} />
          <span className="font-display text-[17px] font-semibold tracking-tight">RouteForge</span>
        </a>
        <nav className="relative flex flex-col gap-1 px-3" aria-label="Main">
          {PAGES.map(({ id, label, blurb, icon: Icon }) => {
            const on = page === id;
            return (
              <a
                key={id}
                href={`#/${id}`}
                aria-current={on ? "page" : undefined}
                title={blurb}
                className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 transition-colors ${
                  on ? "bg-white/[0.08] text-white" : "text-white/60 hover:bg-white/[0.04] hover:text-white"
                }`}
              >
                {on && <span className="absolute top-2 bottom-2 -left-3 w-1 rounded-r-full bg-linear-to-b from-[#ffb547] to-[#ff6a1f]" />}
                <Icon className={`h-4 w-4 ${on ? "text-[#ff8a3d]" : ""}`} aria-hidden />
                <span className="text-[13.5px] font-medium">{label}</span>
              </a>
            );
          })}
        </nav>
        <div className="relative mt-auto space-y-3 p-4">
          {ready.data && <GatewayStatus info={ready.data} />}
          <KeyForm />
          <ThemeSwitch compact />
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 border-b border-anvil-line bg-anvil text-white md:hidden">
          <div className="flex items-center justify-between px-4 py-3">
            <a href="#/overview" className="flex items-center gap-2">
              <LogoMark size={28} />
              <span className="font-display text-base font-semibold">RouteForge</span>
            </a>
            <ThemeSwitch compact />
          </div>
          <nav className="flex gap-1 overflow-x-auto px-3 pb-2" aria-label="Main">
            {PAGES.map(({ id, label, icon: Icon }) => (
              <a
                key={id}
                href={`#/${id}`}
                aria-current={page === id ? "page" : undefined}
                className={`flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${page === id ? "bg-linear-to-b from-[#ff8a3d] to-[#e8590c] text-white" : "text-white/60"}`}
              >
                <Icon className="h-3.5 w-3.5" aria-hidden />
                {label}
              </a>
            ))}
          </nav>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 sm:py-8 md:px-8 lg:px-10">
          {!getAdminKey() && (
            <div className="mb-6 md:hidden">
              <KeyForm light />
            </div>
          )}
          <Active key={page} />
        </main>
      </div>
    </div>
  );
}

function GatewayStatus({ info }: { info: ReadyInfo }) {
  const ok = info.status === "ready";
  const color = ok ? "bg-[#4cc489]" : info.status === "degraded" ? "bg-[#f2b24a]" : "bg-[#ff6b5e]";
  const mode = info.components.models?.split("(")[1]?.replace(")", "") ?? "?";
  return (
    <div className="flex items-center gap-2 text-xs text-white/60" title={`Default policy: ${info.default_strategy}`}>
      <span className={`h-2 w-2 rounded-full ${color}`} />
      <span>Gateway {info.status}</span>
      <span className={`ml-auto font-mono text-[11px] ${info.simulated ? "text-[#f2b24a]" : "text-white/40"}`}>{mode}</span>
    </div>
  );
}

function KeyForm({ light = false }: { light?: boolean }) {
  const [key, setKey] = useState(getAdminKey);
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(!key);
  const muted = light ? "text-ink-2" : "text-white/55";
  if (!editing) {
    return (
      <div className={`flex items-center gap-1.5 text-xs ${muted}`}>
        <KeyRound className="h-3.5 w-3.5" aria-hidden /> Admin key
        <Check className="h-3 w-3 text-[#4cc489]" aria-label="set" />
        <button type="button" onClick={() => setEditing(true)} className="ml-auto underline-offset-2 hover:underline">
          change
        </button>
      </div>
    );
  }
  return (
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
      <label htmlFor={light ? "key-m" : "key"} className={`flex items-center gap-1.5 text-xs ${muted}`}>
        <KeyRound className="h-3.5 w-3.5" aria-hidden /> Admin API key
      </label>
      <input
        id={light ? "key-m" : "key"}
        type="password"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="paste rf-admin-… and press Enter"
        autoComplete="off"
        autoFocus={Boolean(key)}
        onBlur={() => key && !draft && setEditing(false)}
        className={
          light
            ? "w-full rounded-xl border border-line bg-panel px-3 py-2 font-mono text-xs text-ink focus:border-accent focus:outline-none"
            : "w-full rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 font-mono text-xs text-white placeholder:text-white/30 focus:border-[#ff7a2e] focus:outline-none"
        }
      />
    </form>
  );
}

const THEMES: { value: ThemePref; label: string; icon: LucideIcon }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Monitor },
  { value: "dark", label: "Dark", icon: Moon },
];

function ThemeSwitch({ compact = false }: { compact?: boolean }) {
  const { pref } = useTheme();
  return (
    <div role="radiogroup" aria-label="Colour theme" className={`flex rounded-full border border-white/10 bg-white/[0.04] p-0.5 ${compact ? "" : "w-full"}`}>
      {THEMES.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={pref === value}
          title={label}
          onClick={() => setThemePref(value)}
          className={`flex flex-1 items-center justify-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] transition-colors ${pref === value ? "bg-white/15 text-white" : "text-white/45 hover:text-white"}`}
        >
          <Icon className="h-3.5 w-3.5" aria-hidden />
          {!compact && label}
        </button>
      ))}
    </div>
  );
}
