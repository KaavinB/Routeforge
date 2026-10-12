import { AlertTriangle, FlaskConical, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { ApiError } from "../services/api";

// Categorical series colours in fixed order (validated palette). Index by a stable key
// (e.g. the model's position in the registry), never by rank in the current view.
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

export function seriesColor(key: string, order: string[]): string {
  const i = order.indexOf(key);
  return i >= 0 && i < SERIES.length ? SERIES[i] : "#9a9893";
}

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-panel ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-4 py-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "good" | "warn" | "bad" }) {
  const toneClass = tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel px-4 py-3">
      <div className="text-xs font-medium text-ink-2">{label}</div>
      <div className={`tabular mt-1 text-2xl font-semibold tracking-tight ${toneClass}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

const BADGE_TONES = {
  neutral: "bg-sunken text-ink-2 border-line",
  good: "bg-good-soft text-good border-transparent",
  warn: "bg-warn-soft text-warn border-transparent",
  bad: "bg-bad-soft text-bad border-transparent",
  accent: "bg-accent-soft text-accent border-transparent",
} as const;

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: keyof typeof BADGE_TONES }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${BADGE_TONES[tone]}`}>
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={status === "success" || status === "done" ? "good" : status === "running" ? "accent" : "bad"}>{status}</Badge>;
}

export function CacheBadge({ status }: { status: string }) {
  return <Badge tone={status === "hit" ? "good" : status === "miss" ? "accent" : "neutral"}>{status}</Badge>;
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <Inbox className="h-6 w-6 text-ink-3" aria-hidden />
      <div className="text-sm font-medium text-ink">{title}</div>
      {children && <div className="max-w-md text-xs text-ink-2">{children}</div>}
    </div>
  );
}

export function Loading() {
  return (
    <div className="flex items-center gap-2 px-2 py-8 text-sm text-ink-2">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Loading…
    </div>
  );
}

export function ErrorState({ error }: { error: Error }) {
  const auth = error instanceof ApiError && (error.status === 401 || error.status === 403);
  return (
    <div className="flex items-start gap-2 rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        {auth ? "An admin API key is required. Enter it in the sidebar." : error.message}
      </div>
    </div>
  );
}

export function SimulatedBanner({ show, children }: { show: boolean; children?: ReactNode }) {
  if (!show) return null;
  return (
    <div role="status" className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn-soft px-3 py-2 text-sm text-warn">
      <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        <strong>Simulated data.</strong>{" "}
        {children ?? "The gateway is in mock mode: responses, latencies and token counts come from simulated providers and are not evidence of real model performance."}
      </div>
    </div>
  );
}

export function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return <th className={`whitespace-nowrap px-3 py-2 text-[11px] font-semibold tracking-wide text-ink-2 uppercase ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

export function Td({ children, right, mono }: { children: ReactNode; right?: boolean; mono?: boolean }) {
  return <td className={`whitespace-nowrap px-3 py-2 text-[13px] ${right ? "tabular text-right" : ""} ${mono ? "font-mono text-xs" : ""}`}>{children}</td>;
}

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string }) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-ink-2">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line bg-panel px-2 py-1 text-xs text-ink focus:outline-2 focus:outline-accent"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-2">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

export const TRAFFIC_OPTIONS = [
  { value: "real", label: "Real providers" },
  { value: "simulated", label: "Simulated (mock)" },
  { value: "all", label: "All traffic" },
];
