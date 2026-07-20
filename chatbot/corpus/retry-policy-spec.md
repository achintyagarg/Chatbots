# Zarnex Retry Policy Specification v3.2

Internal specification for the Zarnex ingestion service. This document is
deliberately fictional: none of it appears in any model's training data, so a
correct answer about it can only come from retrieval.

## Backoff behaviour

The Zarnex client retries failed requests using truncated exponential backoff.
The base delay is 340 milliseconds and the multiplier is 1.8. Delays are capped
at 47 seconds regardless of attempt count.

A request is retried at most 6 times before the client raises
`ZarnexExhaustedError`. The total time budget across all attempts is 210
seconds; once exceeded, remaining retries are abandoned even if the attempt
count has not been reached.

## Jitter

Full jitter is applied to every delay after the second attempt. The jitter
window is plus or minus 22 percent of the computed delay. Jitter is disabled
entirely when the environment variable `ZARNEX_DETERMINISTIC` is set to `1`,
which exists only for reproducing failures in tests.

## Retryable conditions

Only these conditions are retried:

- HTTP 429 responses, honouring `Retry-After` when present
- HTTP 503 and 504 responses
- Connection resets and DNS resolution failures
- The application-level error code `ZX-1180` (shard temporarily migrating)

Notably, HTTP 500 is **not** retried by default, because Zarnex treats it as a
deterministic server bug rather than a transient fault. Setting
`ZARNEX_RETRY_500=true` overrides this, but the specification recommends
against it in production.

## Idempotency

Retries require an idempotency key supplied in the `X-Zarnex-Idempotency`
header. Keys expire after 26 hours. A retry that reuses an expired key is
rejected with `ZX-1206` and is not retried further.

## Circuit breaker

After 14 consecutive failures to the same shard, the client opens a circuit
breaker for that shard for 90 seconds. While open, requests to that shard fail
immediately with `ZX-1194` without consuming the retry budget.
