import { useSyncExternalStore } from "react";

// Light / dark theme. The preference ("system" follows the OS) is a per-viewer
// convenience kept in localStorage; storage failures fall back to "system".

export type ThemePref = "light" | "dark" | "system";
export type Theme = "light" | "dark";

const KEY = "routeforge.theme";
const listeners = new Set<() => void>();

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

let pref: ThemePref = typeof window === "undefined" ? "system" : readPref();

function systemDark(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolved(): Theme {
  return pref === "system" ? (systemDark() ? "dark" : "light") : pref;
}

function apply() {
  if (typeof document !== "undefined") document.documentElement.dataset.theme = resolved();
  listeners.forEach((l) => l());
}

if (typeof window !== "undefined") {
  apply();
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", apply);
}

export function setThemePref(next: ThemePref) {
  pref = next;
  try {
    if (next === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, next);
  } catch {
    /* storage unavailable: preference lasts for this page only */
  }
  apply();
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useTheme(): { theme: Theme; pref: ThemePref } {
  const theme = useSyncExternalStore(subscribe, resolved, () => "light" as Theme);
  const p = useSyncExternalStore(subscribe, () => pref, () => "system" as ThemePref);
  return { theme, pref: p };
}

/** Concrete colours for SVG charts (Recharts needs real values, not CSS variables). */
export function useChartColors() {
  const { theme } = useTheme();
  return theme === "dark"
    ? { grid: "#2c2621", axis: "#877e72", ink2: "#bcb3a6", cursor: "rgba(255,255,255,0.04)", accent: "#ff7a2e", accent2: "#f59f00", bad: "#ff6b5e", neutral: "#6f675d" }
    : { grid: "#ece7df", axis: "#8f877b", ink2: "#5d564c", cursor: "rgba(40,30,20,0.04)", accent: "#e8590c", accent2: "#f59f00", bad: "#e34948", neutral: "#a39b8f" };
}
