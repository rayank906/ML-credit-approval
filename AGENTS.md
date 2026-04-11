# AGENTS.md

## Backend (FastAPI) Guidelines
- Use FastAPI for backend services that handle auth, mailbox access, and OTP ingestion.
- Prefer `async`/`await` throughout request handlers and I/O paths.
- Keep all backend code typed.
- Use type hints for functions and variables where they improve clarity.
- Use Pydantic models for request/response contracts and validation.
- Reuse the same logging standards in this file: centralized logger abstraction, structured logs, consistent levels, no sensitive data in logs.
- Reuse the same error handling standards in this file: bubble by default, catch at boundaries, add context only when useful, avoid noisy catch-and-rethrow.
- Add rate limiting on sensitive endpoints (auth, code retrieval, provider callbacks, polling endpoints).
- Add authentication and authorization checks on every protected endpoint.
- Validate and sanitize all external input.
- Keep secrets in environment/secret storage only; never hardcode credentials or tokens.
- Use HTTPS in all non-local environments.
- Apply CORS restrictions to known trusted origins.
- Prefer idempotent endpoints and safe retry behavior for ingestion flows.

## Secret Handling
- Store local runtime secrets under `backend/secrets/`.
- Never read files under `backend/secrets/` unless the user explicitly asks for a secret-management task.
- Never print, diff, summarize, or quote secret values unless the user explicitly requests that exact action.

## Implementation Defaults
- Write explicit types for public APIs and shared boundaries.
- Prefer small, testable units of behavior.
- Keep side effects isolated and easy to track.
- Make async flows predictable, with explicit error handling and user-safe fallbacks.

## Error Handling Strategy
- Bubble errors by default.
- Catch errors only where they can be handled meaningfully.
- Low-level/domain functions should usually throw, not catch-and-rethrow.
- Catch at boundaries (request handlers, jobs, CLI entrypoints, and other service edges).
- Catch locally only for meaningful behavior: retry, fallback, cleanup, error translation, or adding useful context.
- Avoid inline `try/catch` blocks that only log and rethrow with no added value.
- Treat global error handling as a safety net for containment, not the primary design.
- Prefer meaningful domain/service errors and translate them near boundaries into user-safe responses.
- Log once near the top boundary; avoid repeated logging of the same error across layers.
- Define custom error types where useful to preserve intent and handling clarity.

## Logging Guidelines
- Use a centralized logger abstraction, not scattered `console.log` calls.
- Keep logging backend-agnostic so output can be routed to console, storage, telemetry, or remote sinks later.
- Prefer structured logs (message + metadata) over plain string logs.
- Use consistent log levels (`debug`, `info`, `warn`, `error`) with clear intent.
- Include context fields that help debugging (module, operation, request/message id), but never log secrets or sensitive user data.
- Log errors once at handling boundaries with enough context to diagnose failures.
