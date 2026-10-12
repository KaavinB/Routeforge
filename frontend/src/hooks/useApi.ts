import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "../services/api";

export interface ApiState<T> {
  data: T | null;
  error: ApiError | Error | null;
  loading: boolean;
  reload: () => void;
}

/** Fetches ``path`` and optionally re-polls every ``intervalMs``. Pass ``null`` to skip. */
export function useApi<T>(path: string | null, intervalMs?: number): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(Boolean(path));
  const [tick, setTick] = useState(0);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setLoading(true);
    api<T>(path)
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path, tick]);

  useEffect(() => {
    if (!path || !intervalMs) return;
    const id = setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => clearInterval(id);
  }, [path, intervalMs]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** Default traffic filter: real traffic unless the gateway itself runs in mock mode.
 *  Simulated requests carry simulated latency/cost and must not be mixed silently with
 *  real measurements. */
export function useTrafficFilter(): [string, (v: string) => void] {
  const ready = useApi<{ simulated: boolean }>("/ready");
  const [choice, setChoice] = useState<string | null>(null);
  const fallback = ready.data ? (ready.data.simulated ? "all" : "real") : "real";
  return [choice ?? fallback, setChoice];
}
