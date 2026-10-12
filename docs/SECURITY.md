# Security notes and threat model

RouteForge is built with production-style security hygiene. It is not a claim of formal
compliance (SOC 2, ISO 27001, etc.).

## Assets

- Provider API keys (OpenAI, Anthropic) — highest value: direct spend.
- RouteForge API keys and the admin key — gateway access and spend on our behalf.
- Prompts and completions — may contain user data.
- Request logs, evaluation data, router artifacts.

## Threats and mitigations

| Threat | Mitigation |
|---|---|
| Provider key exfiltration | Keys only in environment / `.env` (gitignored, `chmod 600`); never logged (redaction filter on every log line as a backstop); never returned in responses; upstream error messages are sanitized before reaching clients |
| Stolen database → usable API keys | Only HMAC-SHA256 digests stored, keyed with a server-side pepper; raw keys shown once at creation |
| Unauthorized use / abuse | Bearer-key auth on every inference and admin route; separate `admin` role for analytics, evaluations, key management, cache flush, breakers and fault injection; per-key token-bucket rate limits and concurrency caps; global concurrency cap |
| Cost amplification via retries | Max 2 attempts × 3 models per request, request deadline, global retry budget (≤20% of successful volume), circuit breakers skip failing providers |
| Oversized / malformed requests | 512 KB body limit enforced in ASGI middleware (also for chunked bodies); message-count and `max_tokens` caps; strict schema (`extra="forbid"`) |
| Cache poisoning / cross-tenant leakage | Cache keys include tenant, resolved model, registry fingerprint and all generation params; no prompt text in keys; TTL bounded; `X-RouteForge-No-Store` / `routeforge.cache=false` opt-outs |
| Sensitive data at rest | Prompts/completions are not stored by default (`ROUTEFORGE_LOG_PROMPTS=false`): the request log stores a message digest, routing metadata, tokens, cost, latency. Langfuse receives no content unless prompt logging is enabled |
| SQL injection | SQLAlchemy parameterized queries only |
| Least privilege in the database | Runtime role `routeforge_app` has only SELECT/INSERT/UPDATE/DELETE; DDL is done by a separate migration job using the owner role (verified: `CREATE TABLE` as the app role → permission denied) |
| Executing model-generated code | Never on the host. Code-generation scoring runs in a throwaway Docker container (`--network none`, read-only root, tmpfs, `--cap-drop ALL`, `no-new-privileges`, uid 65534, CPU/memory/PID limits) with an inner per-program subprocess (`python -I -S`, empty environment, CPU/file/fd rlimits, timeout) after a static AST policy check. Inside the API container the subprocess backend is used — see residual risks |
| Malicious router artifact | `router.joblib` (pickle) is loaded only after its SHA-256 matches the committed digest; artifacts must come from the trusted training pipeline |
| Fault injection misuse | Disabled unless `ROUTEFORGE_FAULT_INJECTION_ENABLED=true`; admin-only endpoints |
| Supply-chain compromise | Every Python/npm dependency pinned to an exact version published ≥14 days before pinning (cooldown against freshly compromised releases, cf. the 2026 LiteLLM PyPI incident); `uv.lock`/`package-lock.json` committed; `pip-audit` and `npm audit` run in CI (0 known vulnerabilities at time of writing) |
| Browser attacks on the dashboard | nginx sets CSP, `X-Frame-Options: DENY`, `nosniff`, no-referrer; admin key kept in `sessionStorage` (tab-scoped) |
| Exposed infrastructure | Compose binds Postgres, Redis, Prometheus, Grafana and Jaeger to 127.0.0.1 only; CORS restricted to the dashboard origins |
| CI spending money | No provider credentials exist in CI; all CI evaluation runs use mock or replay providers |

## Residual risks / known gaps

- The `subprocess` sandbox used *inside* the API container shares that container's kernel
  and user; the AST policy blocks file and process access but is not a security boundary.
  Prefer running code-category evaluations offline with the Docker sandbox.
- `/metrics` is unauthenticated by default (standard for internal scraping); set
  `ROUTEFORGE_METRICS_REQUIRE_AUTH=true` if the port is reachable from untrusted networks.
- Rate limits and breakers are per process for concurrency; token buckets are shared via
  Redis.
- No TLS termination is configured (expected to sit behind a load balancer / ingress).
- Langfuse and OTLP exporters send telemetry to whatever endpoint is configured; review
  before enabling prompt logging.
