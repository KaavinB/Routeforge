import { useId } from "react";

/** RouteForge mark: one request in, forked to the model that fits. */
export function LogoMark({ size = 32 }: { size?: number }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id={`lg-${id}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ffb547" />
          <stop offset="55%" stopColor="#ff6a1f" />
          <stop offset="100%" stopColor="#d9364f" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill={`url(#lg-${id})`} />
      <path d="M6.5 16h6.5c3 0 4-7.5 8-7.5h3M13 16h11.5M13 16c3 0 4 7.5 8 7.5h3" stroke="#fff" strokeWidth="2.3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="6.5" cy="16" r="2.4" fill="#fff" />
      <circle cx="25.5" cy="16" r="2.1" fill="#fff" />
      <circle cx="25.5" cy="8.5" r="1.6" fill="#fff" fillOpacity="0.75" />
      <circle cx="25.5" cy="23.5" r="1.6" fill="#fff" fillOpacity="0.75" />
    </svg>
  );
}
