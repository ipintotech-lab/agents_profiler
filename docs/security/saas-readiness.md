# SaaS readiness: security review and plan

Status: pre-production. This file records what exists today, what is missing
before customers upload their own files, and what this branch implements.
The model: **the browser and every uploaded file are hostile; the backend and
storage are the security boundary.**

## 1. Current architecture (as inspected)

| Area | What exists | Where |
|---|---|---|
| Frontend | One static page, vanilla JS, no build step, no third-party assets | `src/api/static/index.html` (served by `root()` in `src/api/main.py`) |
| Backend | FastAPI, single process (`uvicorn api.main:app`) | `src/api/main.py`, `railway.json` |
| Authentication | **None** | every route in `src/api/main.py` |
| Authorization | **None**, no users, tenants or roles | — |
| Database | One SQLite file for opt-in behavior history, global (no owner column) | `src/profiler/history.py` (`BEHAVIOR_DB`, default `data/behavior.sqlite`; Railway volume `/data`) |
| File upload | Multipart on `/analyze` and `/finops`; whole file read into memory, parsed and analyzed inside the request; nothing written to disk except history | `_load_events`, `_read_aggregate`, `_parse_trace_events` in `src/api/main.py` |
| File storage | None for uploads; history rows hold full event JSON (inputs and outputs) | `history.add` |
| Processing | In the web process. `/analyze` calls `run_pipeline` **directly on the event loop** (blocks every other request while it runs) | `analyze()` in `src/api/main.py` |
| Background workers | None | — |
| Parsers of customer content | `json` (stdlib), `csv` (stdlib), pydantic `TraceEvent`, gzip/zlib/zstd for Hugging Face files. No XML, PDF, Office or archive parsers | `src/ingest/loader.py`, `src/ingest/normalize.py`, `src/ingest/usage.py`, `src/ingest/huggingface.py` |
| External services | Hugging Face Hub (server-side fetch, fixed host `HF_ENDPOINT`) | `src/ingest/huggingface.py` |
| AI/LLM integrations | **None.** No customer data is sent to any model. The optional clingo solver runs locally | `src/profiler/asp.py` |
| Secrets | `HF_TOKEN` env var, or `~/.cache/huggingface/token` on the server. No secrets in the repo or in `index.html` (checked: no keys, tokens or credentials in tracked files) | `resolve_token` in `src/ingest/huggingface.py` |
| Security headers / CORS | None set; no CORS middleware (same-origin only by default) | `src/api/main.py` |

## 2. Security gaps (current code)

P0 unless noted. Each one names the code it comes from.

1. **No authentication on any route.** Anyone on the internet can call every
   endpoint in `src/api/main.py`.
2. **Global, unauthenticated history: cross-customer read and delete.**
   `GET /behavior` and `GET /behavior/history` return everything every
   uploader chose to keep (`history.load`, `history.summary`), and
   `DELETE /behavior/history` (`behavior_history_clear` → `history.clear`)
   deletes all of it. The `events` table has no tenant column.
3. **Server credential lent to anonymous callers (confused deputy).**
   `/analyze`, `/finops` and `/huggingface/files` fall back to the server's
   `HF_TOKEN` (or the host's saved `hf auth login` token) via
   `resolve_token`, so any caller can read gated datasets the operator's
   account can see.
4. **No upload limits.** `_load_events` and `_read_aggregate` call
   `await upload.read()` with no size, count or quota limit; one request can
   exhaust memory. Starlette's multipart parser spools large parts to temp
   disk first.
5. **Processing in the web request, on the event loop.** `analyze()` runs
   `run_pipeline` synchronously inside `async def`; one large upload stalls
   the whole server. No CPU, memory or time limits on processing.
6. **Unauthenticated expensive endpoints (DoS).** `/run_sweep`, `/poc/run`,
   `/solve`, `/generate`, `/validate_seed_range` are CPU-heavy QA or demo
   tools open to everyone.
7. **Decompression without an output limit (P1: Hugging Face path only).**
   `iter_records` calls `gzip.decompress(raw)` on up to 200 MB of compressed
   input with no cap on the output, and `_iter_lines` buffers a line with no
   length cap.
8. **No browser security headers.** No CSP, `frame-ancestors`,
   `X-Content-Type-Options` or HSTS (`src/api/main.py`). The page renders
   customer data with `innerHTML`; escaping via `esc()` is consistent where
   checked, but a CSP is the missing second layer.
9. **No tenant, file, job or result model, no audit log.** Nothing to
   attach ownership, retention or deletion to.
10. **Error details echo parser messages** (`could not parse {name}: {exc}`).
    Pydantic and JSON errors can quote fragments of the uploaded content back
    to the uploader (P2: same user, but keep it out of logs).

Not found (no action needed now): SQL injection (parameterized queries in
`history.py`), path traversal (no customer filename touches the filesystem),
SSRF (`HF_ENDPOINT` is fixed; the dataset link only fills the path), secrets in
the frontend, LLM or prompt-injection exposure (no model is called).

## 3. Target architecture

```text
                      Internet
                          |
                     CDN / WAF            (infra; Railway edge today)
                          |
                 FastAPI web (api.main)
       security headers, CSRF origin check, size limits
                          |
             +------------+--------------+
             |                           |
     Authentication (saas.auth)    Static page (no secrets)
     OIDC code + PKCE -> server
     session, HttpOnly cookie
             |
     Principal = user + active tenant + role   (derived server-side, never from the request)
             |
   /api/files  /api/jobs  /api/results  /api/audit   (saas.routes)
     every query: WHERE id = ? AND tenant_id = principal.tenant_id
             |                     |
   Metadata DB (saas.db)      Job queue (jobs table)
   users, tenants,                 |
   memberships, files,      Worker process (python -m saas.worker)
   jobs, results, audit       re-reads file + tenant from DB
             |                     |
             |              Sandbox child per job: no env secrets,
             |              no DB, rlimit CPU/memory, wall-clock timeout
             |                     |
   ObjectStorage (saas.storage) <--+
     tenants/<tenant>/uploads/<random id>     (quarantine -> ready)
     tenants/<tenant>/results/<job id>
   MalwareScanner (saas.scanner): ClamAV (clamd) or none in dev
```

Design rules:

- Tenant context comes only from the session. No route accepts `tenant_id`,
  `user_id`, `role` or an owner field from the client.
- A missing object and another tenant's object both return **404**, so IDs
  can't be probed.
- Upload allowlist = what the product reads: JSON, JSON Lines, CSV (UTF-8
  text). Archives and compressed uploads are **rejected**, which removes
  zip-bomb and path-traversal risk instead of mitigating it. Revisit only if
  customers need archives.
- The customer's filename is metadata only; storage keys are random.
- No LLM is in the processing path. If one is added: send only the minimum
  excerpt, one request per tenant job with no shared memory, customer text in
  the user turn (never the system prompt), and tools authorized by the
  backend, not by the model.

## 4. Backlog

### P0: before any customer uploads

| # | Item | This branch |
|---|---|---|
| 1 | Security headers: CSP (script hash), `frame-ancestors 'none'`, nosniff, referrer policy, HSTS | yes |
| 2 | Authentication: generic OIDC (authorization code + PKCE), server-side sessions, HttpOnly/Secure/SameSite cookie, CSRF origin check, logout | yes |
| 3 | Tenant model: users, tenants, memberships, roles; principal derived from session | yes |
| 4 | Tenant-scoped files, jobs, results; 404 across tenants | yes |
| 5 | Secure upload API: streamed size limit, count limit, tenant storage and daily quotas, allowlist by content, random storage keys, SHA-256, explicit status lifecycle | yes |
| 6 | Private storage behind an `ObjectStorage` interface (local disk under a volume now; S3/GCS adapter later) | yes (local) |
| 7 | Processing out of the web process: job queue + worker + sandboxed child with CPU, memory and time limits and a scrubbed environment | yes |
| 8 | Lock down legacy routes in SaaS mode: login for `/analyze`, `/finops`, `/huggingface/files`, `/behavior*`; admin for QA tools; history scoped by tenant; no server `HF_TOKEN` for customer requests | yes |
| 9 | Size limits on legacy upload routes | yes |
| 10 | Cross-tenant and upload security test suites | yes |
| 11 | Deletion: file, its results and stored objects; audited | yes |
| 12 | Audit log for login, upload, reject, download, delete, job start/finish | yes |

### P1: before GA

- ClamAV in its own container wired to `MalwareScanner` (interface and clamd
  client are here; the service is infrastructure).
- S3-compatible `ObjectStorage` with a private bucket, SSE, and short-lived
  presigned URLs for one object and one operation.
- Postgres instead of SQLite (several web replicas, row-level security as a
  second layer).
- Rate limiting per user and per tenant (login, upload, jobs) in a shared
  store (Redis); the app enforces quotas today, not request rates.
- Worker in a separate container with no network egress and no access to the
  web service's secrets or the cloud metadata endpoint.
- Retention policy (automatic purge after N days) and per-tenant export.
- Invitations and membership management UI; tenant switcher.
- Output cap on gzip/zstd decompression in `src/ingest/huggingface.py`.
- Security monitoring and alerting on audit events.

### P2: defense in depth

- Generic error messages for parse failures (no content echoed).
- Subresource-integrity-free bundle stays third-party-free; keep it that way.
- Anomaly detection on upload and download patterns.
- Customer-managed encryption keys.

## 5. Implementation plan (this branch)

Small commits, each with its tests:

1. `security: add browser security headers`
2. `security: add SaaS config, tenant model and audit log`
3. `security: add OIDC login and server-side sessions`
4. `security: add private object storage and upload validation`
5. `security: add tenant-scoped files API with quotas`
6. `security: move file analysis to a sandboxed worker`
7. `security: require auth on legacy routes in SaaS mode`
8. `security: add cross-tenant and upload test suites` (tests also land with each step)

Modes (`AUTH_MODE`):

- `demo` (default): today's public demo, unchanged, and the new `/api/*`
  routes are off. **Not for customer data.**
- `oidc`: production. Every customer-data route needs a session.
- `dev`: local only. `/auth/dev-login` signs in as a test user. It refuses
  to start unless `APP_ORIGIN` is a localhost address.

## 6. Needs infrastructure (not in the repo)

- An OIDC provider (Auth0, Okta, Entra ID, Google Workspace, Keycloak…):
  `OIDC_ISSUER`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, redirect URI
  `<APP_ORIGIN>/auth/callback`.
- `SESSION_SECRET` is not needed: sessions are random tokens stored hashed
  in the DB.
- Railway: the volume at `/data` must hold `SAAS_DB` and `STORAGE_DIR`, or
  data is lost on deploy. A second Railway service runs
  `python -m saas.worker` against the same volume. Railway volumes attach to
  one service, so production needs Postgres + object storage (P1) before the
  worker can be its own service; until then the worker runs in the same
  container as a separate process.
- ClamAV (`CLAMD_HOST`, `CLAMD_PORT`) for malware scanning.
