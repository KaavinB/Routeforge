import { AlertTriangle, ChevronDown, FlaskConical, HelpCircle, Inbox, type LucideIcon } from "lucide-react";
import { useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";

import { ApiError } from "../services/api";

// Categorical series colours in fixed order (validated palette). Index by a stable key
// (e.g. the model's position in the registry), never by rank in the current view.
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

export function seriesColor(key: string, order: string[]): string {
  const i = order.indexOf(key);
  return i >= 0 && i < SERIES.length ? SERIES[i] : "#9a9893";
}

const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

// ------------------------------------------------------------------ layout

export function PageHeader({ eyebrow, icon: Icon, title, description, actions }: {
  eyebrow?: string;
  icon?: LucideIcon;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && (
          <div className="mb-2 inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.14em] text-accent uppercase">
            {Icon && <Icon className="h-3.5 w-3.5" aria-hidden />}
            {eyebrow}
          </div>
        )}
        <h1 className="font-display text-[28px] leading-tight font-semibold tracking-tight text-ink sm:text-[32px]">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-[14px] leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Card({ title, subtitle, actions, icon: Icon, children, className = "", bodyClassName = "" }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("animate-rise rounded-2xl border border-line bg-panel shadow-card", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 px-5 pt-4">
          <div className="flex min-w-0 items-start gap-2.5">
            {Icon && (
              <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sunken text-ink-2">
                <Icon className="h-4 w-4" aria-hidden />
              </span>
            )}
            <div className="min-w-0">
              {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
              {subtitle && <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{subtitle}</p>}
            </div>
          </div>
          {actions}
        </header>
      )}
      <div className={cx("p-5", title || actions ? "pt-4" : "", bodyClassName)}>{children}</div>
    </section>
  );
}

// ------------------------------------------------------------------ numbers

const TONE_TEXT = { good: "text-good", warn: "text-warn", bad: "text-bad" } as const;

export function StatTile({ label, value, hint, tone, info, icon: Icon, spark, sparkColor, iconClass = "bg-accent-soft text-accent" }: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "good" | "warn" | "bad";
  info?: ReactNode;
  icon?: LucideIcon;
  spark?: (number | null)[];
  sparkColor?: string;
  iconClass?: string;
}) {
  return (
    <div className="group/tile animate-rise relative flex flex-col overflow-hidden rounded-2xl border border-line bg-panel px-4 pt-4 pb-3 shadow-card transition-shadow hover:shadow-lift">
      <div className="flex items-center gap-2">
        {Icon && (
          <span className={cx("flex h-7 w-7 items-center justify-center rounded-lg", iconClass)}>
            <Icon className="h-4 w-4" aria-hidden />
          </span>
        )}
        <span className="text-[13px] font-medium text-ink-2">{label}</span>
        {info && <InfoTip label={label}>{info}</InfoTip>}
      </div>
      <div className={cx("tabular mt-2 font-display text-[28px] leading-none font-semibold tracking-tight", tone ? TONE_TEXT[tone] : "text-ink")}>{value}</div>
      {hint && <div className="mt-1.5 text-xs leading-snug text-ink-3">{hint}</div>}
      {spark && spark.filter((v) => v != null && v > 0).length >= 3 && <Sparkline data={spark} color={sparkColor} />}
    </div>
  );
}

export function Sparkline({ data, color = "#e8590c", height = 36 }: { data: (number | null)[]; color?: string; height?: number }) {
  const id = useId().replace(/:/g, "");
  const rows = data.map((v, i) => ({ i, v }));
  return (
    <div className="-mx-4 -mb-3 mt-3" aria-hidden>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={rows} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`sp-${id}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.28} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis hide domain={[0, "dataMax"]} />
          <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.75} fill={`url(#sp-${id})`} connectNulls isAnimationActive={false} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Ring gauge for a 0..1 value. */
export function Ring({ value, size = 112, stroke = 10, color = "#4cc489", track = "rgb(255 255 255 / 0.1)", children }: {
  value: number;
  size?: number;
  stroke?: number;
  color?: string;
  track?: string;
  children?: ReactNode;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} stroke={track} strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${c * v} ${c}`}
          style={{ transition: "stroke-dasharray 0.8s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  );
}

// ------------------------------------------------------------------ small pieces

export function InfoTip({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <span className="group/tip relative inline-flex">
      <button type="button" aria-label={label ? `What is ${label}?` : "More info"} className="rounded-full text-ink-3 transition-colors hover:text-accent focus:text-accent focus:outline-none">
        <HelpCircle className="h-3.5 w-3.5" aria-hidden />
      </button>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute top-full left-1/2 z-30 mt-2 w-60 -translate-x-1/2 rounded-xl border border-line bg-panel px-3 py-2 text-xs leading-relaxed font-normal text-ink-2 opacity-0 shadow-lift transition-opacity group-focus-within/tip:visible group-focus-within/tip:opacity-100 group-hover/tip:visible group-hover/tip:opacity-100"
      >
        {children}
      </span>
    </span>
  );
}

const BADGE_TONES = {
  neutral: "bg-sunken text-ink-2 ring-line",
  good: "bg-good-soft text-good ring-good/15",
  warn: "bg-warn-soft text-warn ring-warn/15",
  bad: "bg-bad-soft text-bad ring-bad/15",
  accent: "bg-accent-soft text-accent-ink ring-accent/15",
} as const;

const DOT_TONES = { neutral: "bg-ink-3", good: "bg-good", warn: "bg-warn", bad: "bg-bad", accent: "bg-accent" } as const;

export function Badge({ children, tone = "neutral", dot }: { children: ReactNode; tone?: keyof typeof BADGE_TONES; dot?: boolean }) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap ring-1 ring-inset", BADGE_TONES[tone])}>
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", DOT_TONES[tone])} />}
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge dot tone={status === "success" || status === "done" ? "good" : status === "running" ? "accent" : "bad"}>
      {status}
    </Badge>
  );
}

export function CacheBadge({ status }: { status: string }) {
  return <Badge tone={status === "hit" ? "good" : status === "miss" ? "accent" : "neutral"}>{status}</Badge>;
}

export function ModelChip({ id, color }: { id: string | null | undefined; color?: string }) {
  if (!id) return <span className="text-ink-3">—</span>;
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs text-ink">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color ?? "var(--color-ink-3)" }} />
      {id}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <code className="rounded-md border border-line bg-sunken px-1.5 py-0.5 font-mono text-[12px] text-ink">{children}</code>;
}

// ------------------------------------------------------------------ states

export function EmptyState({ title, children, icon: Icon = Inbox }: { title: string; children?: ReactNode; icon?: LucideIcon }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <div className="relative mb-1">
        <div className="absolute inset-0 rounded-2xl bg-accent/20 blur-xl" />
        <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl border border-line bg-panel text-accent shadow-card">
          <Icon className="h-5 w-5" aria-hidden />
        </div>
      </div>
      <div className="text-sm font-semibold text-ink">{title}</div>
      {children && <div className="max-w-md text-[13px] leading-relaxed text-ink-2">{children}</div>}
    </div>
  );
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3 py-2" role="status" aria-live="polite">
      <span className="sr-only">Loading…</span>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="skeleton h-24 rounded-2xl" />
        ))}
      </div>
      {Array.from({ length: rows - 1 }, (_, i) => (
        <div key={i} className="skeleton h-40 rounded-2xl" />
      ))}
    </div>
  );
}

export function SkeletonRows({ n = 5 }: { n?: number }) {
  return (
    <div className="space-y-2" role="status">
      <span className="sr-only">Loading…</span>
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className="skeleton h-8 rounded-lg" />
      ))}
    </div>
  );
}

export function ErrorState({ error }: { error: Error }) {
  const auth = error instanceof ApiError && (error.status === 401 || error.status === 403);
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-bad/25 bg-bad-soft px-4 py-3 text-sm text-bad">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        {auth ? (
          <>
            <strong className="font-semibold">You're not signed in.</strong> Paste the admin API key (<span className="font-mono text-xs">ROUTEFORGE_BOOTSTRAP_ADMIN_KEY</span> in <span className="font-mono text-xs">.env</span>) into the box in the sidebar.
          </>
        ) : (
          error.message
        )}
      </div>
    </div>
  );
}

export function SimulatedBanner({ show, children }: { show: boolean; children?: ReactNode }) {
  if (!show) return null;
  return (
    <div role="status" className="flex items-start gap-3 rounded-2xl border border-warn/25 bg-warn-soft px-4 py-3 text-sm text-warn">
      <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        <strong>Simulated data.</strong>{" "}
        {children ?? "The gateway is in mock mode: responses, latencies and token counts come from simulated providers and are not evidence of real model performance."}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ tables

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="-mx-5 overflow-x-auto">
      <table className="w-full">{children}</table>
    </div>
  );
}

export function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return (
    <th className={cx("border-b border-line px-3 pb-2 text-[11px] font-semibold tracking-wider whitespace-nowrap text-ink-3 uppercase first:pl-5 last:pr-5", right ? "text-right" : "text-left")}>
      {children}
    </th>
  );
}

export function Td({ children, right, mono }: { children: ReactNode; right?: boolean; mono?: boolean }) {
  return <td className={cx("px-3 py-2.5 text-[13px] whitespace-nowrap first:pl-5 last:pr-5", right && "tabular text-right", mono && "font-mono text-xs")}>{children}</td>;
}

export const rowClass = "border-b border-line/60 last:border-0 transition-colors hover:bg-sunken/60";

// ------------------------------------------------------------------ inputs

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string }) {
  return (
    <label className="flex items-center gap-2 text-xs font-medium text-ink-2">
      {label}
      <span className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="appearance-none rounded-full border border-line bg-panel py-1.5 pr-8 pl-3 text-xs font-medium text-ink shadow-card transition-colors hover:border-ink-3 focus:outline-2 focus:outline-offset-1 focus:outline-accent"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute top-1/2 right-2.5 h-3.5 w-3.5 -translate-y-1/2 text-ink-3" aria-hidden />
      </span>
    </label>
  );
}

export function Segmented({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-full border border-line bg-panel p-0.5 shadow-card">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-full px-3 py-1 text-xs font-medium transition-colors",
            value === o.value ? "bg-ink text-panel shadow-sm" : "text-ink-2 hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2.5 text-[13px] text-ink-2 select-none">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx("relative h-5 w-9 shrink-0 rounded-full transition-colors", checked ? "bg-accent" : "bg-line")}
      >
        <span className={cx("absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform", checked && "translate-x-4")} />
      </button>
      {label}
    </label>
  );
}

export function Button({ variant = "primary", className = "", children, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" }) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary"
          ? "bg-linear-to-b from-[#ff7a2e] to-accent text-white shadow-[0_6px_18px_-6px_rgb(232_89_12/0.7),inset_0_1px_0_rgb(255_255_255/0.25)] hover:brightness-110 active:translate-y-px"
          : "border border-line bg-panel text-ink-2 hover:border-ink-3 hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}

export const inputClass =
  "w-full rounded-xl border border-line bg-surface px-3 py-2 text-[13px] text-ink placeholder:text-ink-3 transition-colors focus:border-accent focus:bg-panel focus:outline-none focus:ring-4 focus:ring-accent/15";

// ------------------------------------------------------------------ charts

export function ChartTooltip({ active, payload, label, labelFormat, valueFormat }: {
  active?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: readonly any[];
  label?: unknown;
  labelFormat?: (l: unknown) => string;
  valueFormat?: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-panel px-3 py-2 text-xs shadow-lift">
      {label != null && <div className="mb-1 font-medium text-ink">{labelFormat ? labelFormat(label) : String(label)}</div>}
      {payload.map((p) => (
        <div key={String(p.dataKey ?? p.name)} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color ?? p.fill ?? p.stroke }} />
          <span>{p.name}</span>
          <span className="tabular ml-auto pl-3 font-medium text-ink">{valueFormat ? valueFormat(Number(p.value)) : String(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

export const TRAFFIC_OPTIONS = [
  { value: "real", label: "Real providers" },
  { value: "simulated", label: "Simulated (mock)" },
  { value: "all", label: "All traffic" },
];
