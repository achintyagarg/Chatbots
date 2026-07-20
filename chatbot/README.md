# Grounded Assistants (Google ADK)

Two chatbots on one Google ADK harness, both built to answer from **retrieved
evidence rather than training memory**, and to ask a human before changing
anything:

- **GitHub assistant** — live repository state over the GitHub MCP server,
  plus a document corpus you control.
- **Finance research assistant** — live market data over two MCP servers
  (yfinance and, optionally, Alpha Vantage), your own research notes as a
  corpus, and locally computed quant analytics. Data and analysis only —
  it is designed never to give buy/sell advice.

```
                        ┌──────────────────┐
  live data over MCP ── │                  │
  (GitHub / market)     │   LlmAgent  x2   │ ──> answer + citations
                        │                  │
  your documents ────── │  SafetyPlugin    │ ──> write? ──> human approval
  (Chroma, per-agent)   │  Observability   │
  quant skills ──────── │                  │
                        └──────────────────┘
```

Everything below the agent definitions is shared: one ingestion pipeline, one
safety plugin, one server, one frontend with an agent picker. A bug fixed once
is fixed for both.

---

## Quick start

```bash
cd chatbot
python -m venv .venv && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in GOOGLE_API_KEY and GITHUB_PERSONAL_ACCESS_TOKEN
python -m ingestion.cli ingest ./corpus
python -m ingestion.cli ingest ./corpus_finance --collection finance
python -m uvicorn server.main:app --reload
```

Open <http://127.0.0.1:8000/ui/> and pick an agent from the header dropdown.

Two keys are needed: a Gemini key from <https://aistudio.google.com/apikey>, and
a GitHub token from <https://github.com/settings/tokens>. The MCP toolset is
read-only, so a token with `public_repo` (or read-only Contents/Issues/Metadata)
is enough unless you deliberately enable writes.

The finance agent works with no extra keys (market data via yfinance, which the
first call fetches through `uvx`). Adding a free `ALPHAVANTAGE_API_KEY` extends
it with deeper fundamentals and news sentiment — but that free tier is ~25
requests/day, so the prompt treats it as scarce.

`adk web` also works if you prefer ADK's own dev UI — run it from this directory
and pick either agent.

---

## What to try

The sample corpus is deliberately fictional, so a correct answer can only come
from retrieval — the model cannot have memorized any of it.

| Ask | What should happen |
|---|---|
| "How many open issues does google/adk-python have right now?" | Calls a GitHub MCP tool, cites the exact count and issue number |
| "What is the Zarnex base retry delay?" | Calls `search_corpus`, answers 340 ms, cites file + heading |
| "What does error ZX-2071 mean?" | Retrieves from the sample **PDF**, cites page 2 |
| "What's the price of a used Civic in Ohio?" | Says it doesn't know — no source covers it |
| "What do the vendor notes say about ticket #8812?" | Reports the planted injection **as content**, does not obey it |
| "Open an issue on X about the flaky test" | Pauses for approval; blocked outright unless X is allowlisted |

And on the finance agent:

| Ask | What should happen |
|---|---|
| "Is the market down right now?" | Live index-ETF quote with as-of time and the 15-minute-delay caveat |
| "Is it a good time to buy NVDA?" | No yes/no. Valuation + trend + news data, then a one-line not-advice reminder |
| "What does my momentum framework say about position sizing?" | Cites `momentum-framework.md` — your notes, not generic advice |
| "Sharpe and correlation for SPY + QQQ over 3 years?" | Calls `analyze_portfolio`; exact numbers, conventions disclosed |
| "How would a 50/200 SMA cross have done on SPY?" | Calls `backtest_sma_cross`; strategy vs buy-and-hold **with caveats** |
| "Add NVDA to my watchlist" | Pauses for your approval before touching the file |

Watch the **Tool calls** panel while you do this. It is the point: an answer
with no tool call behind it is an answer you should not trust.

---

## How each capability is implemented

### Grounding — two sources, cite or abstain

Live GitHub state comes from the official remote MCP server via `McpToolset`.
Documents come from `search_corpus`, which returns each passage together with
its `source`, `page`, and `heading`.

Returning provenance *inside the tool payload* is what makes citation
enforceable rather than aspirational — the model cannot cite a page number it
was never handed.

`search_corpus` filters on a similarity floor and returns `no_match` below it,
which is what lets the agent say "the corpus doesn't cover this" instead of
citing the least-irrelevant chunk it could find. **That threshold is calibrated,
not guessed, and per collection** (`ingestion/corpus_tools.py`): Gemini
embeddings have a high floor, where unrelated queries still score ~0.45–0.59
while genuine matches score 0.62–0.84. An intuitive-looking 0.35 never fires and
silently disables the whole no-match path.

Each agent has its own Chroma collection, and each collection gets its own
measured threshold, because the number does **not** transfer across domains:

| Collection | Relevant min | Irrelevant max | Threshold |
|---|---|---|---|
| `corpus` (GitHub) | 0.625 | 0.570 | `0.60` |
| `finance` | 0.713 | 0.588 | `0.65` (`CORPUS_MIN_SIMILARITY_FINANCE`) |

Separate collections also mean retrieval never crosses agents: a finance query
misses cleanly instead of surfacing the least-irrelevant GitHub spec chunk.
Re-run the calibration if you change the embedding model or corpus contents.

### Chunking — `ingestion/`

Structure first, offsets never. Loaders recover headings and page numbers; the
chunker packs sections into a ~512-token budget with ~64 tokens of overlap,
splitting on sentence boundaries and only hard-splitting as a last resort.

Three details that matter more than they look:

- The heading breadcrumb is prepended to the **embedded** text (not the
  displayed text), so a chunk retrieved in isolation still says what it is about.
- `chunk_id` is `sha256(source + ordinal)`, so re-ingesting an edited file
  upserts instead of stacking duplicates that compete at query time.
- Packing happens *within* a heading group, so a chunk never straddles two
  unrelated topics.

PDF heading detection is heuristic — extraction discards font size — and uses a
one-line lookahead: a short capitalized line whose *next* line begins lowercase
is a wrapped body line, not a heading.

Inspect what actually got stored, which is the fastest way to catch a bad
chunker:

```bash
python -m ingestion.cli inspect --source retry-policy-spec.md
```

### Human in the loop — `tools/github_write.py`

Writes are local `FunctionTool`s rather than MCP tools, because
`require_confirmation` is a FunctionTool mechanism — owning the function is what
makes the gate possible. The MCP toolset stays read-only, so no MCP tool added
later can become an ungated write path.

Two gate styles:

- `add_comment` — a yes/no decision, via `require_confirmation`.
- `create_issue` — uses `tool_context.request_confirmation(hint, payload)`, so
  the approver gets the title and body as **editable fields** and can amend the
  action, not just accept or refuse it.

Approval is the second gate. `SafetyPlugin` has already rejected writes to
non-allowlisted repos before any human is asked.

### Context engineering — `prompts.py`, `agent.py`

All standing guidance — persona, grounding rules, safety policy, skill
fragments — lives in `static_instruction`, and **`instruction` is deliberately
empty**.

That is not a style choice, and getting it wrong breaks the agent completely
while every individual piece still looks correct. When both are set, ADK does
not put `instruction` in the system prompt; it appends it to `contents` as a
*user message after the user's question*
(`flows/llm_flows/instructions.py`). Standing guidance placed there becomes the
most recent message in the conversation, so the model answers **it** instead of
you — replying "I understand the instructions, I am ready to assist!" and never
calling a tool.

An `InstructionProvider` returning `""` does not rescue this: the callable is
truthy, so ADK appends an empty user turn instead. The branch is only skipped
when `instruction` is falsy.

To add per-session context later, don't reintroduce `instruction`. Either drop
`static_instruction` (then `instruction` becomes the system prompt and supports
`{key?}` state templating), or inject from a `before_model_callback` where you
control the position. `tests/test_prompt_wiring.py` guards the invariant and
fails if it regresses.

`EventsCompactionConfig` summarizes long conversations instead of overflowing.
Context caching is **opt-in** via `ENABLE_CONTEXT_CACHE=1`: the Gemini free
tier permits zero cached-content storage, so enabling it there 429s on every
turn — harmlessly, since caching degrades gracefully, but noisily and for no
benefit. Worth turning on with a paid key, where the ~3.5k-token static prefix
is exactly what you want cached. Note Gemini also enforces a hard 4096-token
floor on `min_tokens`, so lower values are silently no-ops.

Retrieved chunks enter as **tool output, never as appended system prompt**, which
keeps provenance attached and the cacheable prefix stable.

### Harness engineering — `agent.py`, `server/main.py`

`agents/github_agent/agent.py` exports `app = App(...)`, not a bare
`root_agent`. ADK's loader checks for `app` first, so one definition gives
plugins and caching to **both** `adk web` and the FastAPI server. Exporting
`root_agent` instead would silently drop the safety plugin under one of the two
entry points.

The server is built on ADK's `get_fast_api_app`, which already provides
`/run_sse`, sessions, artifacts, and the confirmation round-trip. Sessions
persist to SQLite, so a pending approval survives a restart.

### MCP integration

Remote GitHub MCP server over streamable HTTP with a bearer token — no Docker,
no local process. `tool_filter` keeps the exposed surface small and read-only.
The session is closed on shutdown through the FastAPI `lifespan` hook.

### Safety — `plugins/safety.py`

Four layers, registered once on the `App` so a newly added agent or skill cannot
opt out. Listed by how much they actually buy:

1. **`after_tool` — untrusted-data marking.** The most valuable layer. Issue
   bodies, PR descriptions and document text are attacker-writable input flowing
   into a model that can also write to GitHub. Free-text fields are wrapped in
   explicit markers, and the prompt states that content inside them is data,
   never instructions. Content is marked, never deleted — the agent still has to
   be able to report what an issue says.
2. **`before_tool` — deterministic policy.** Repo allowlist, argument bounds. This
   is code, not persuasion, so a jailbroken model still cannot get past it.
3. **`after_model` — credential redaction** before text reaches the user.
4. **`before_model` — input screening** for obvious injection phrasing.

Layer 4 is pattern matching and is the weakest of the four; it catches casual
attempts, not determined ones. The real guarantees are layers 1 and 2. Plus
Gemini `safety_settings` on the agent and the approval gate on every write.

Writes are **disabled by default** — `GITHUB_WRITE_ALLOWLIST` is empty, and an
empty allowlist blocks everything.

### Skills — `skills/`

A skill is a module declaring `TOOLS` and an optional `INSTRUCTION` fragment.
`load_skills()` discovers them at import, so adding a capability means dropping
in one file with no edit to `agent.py`. A skill that fails to import is logged
and skipped rather than taking down the agent.

`pdf_skill.py` is the reference implementation: `parse_pdf` reads an uploaded
PDF for the current conversation, `ingest_pdf` makes it permanently searchable.
Both reuse `ingestion/loaders.py`, so an uploaded PDF and a batch-ingested one
chunk identically.

To add a skill, copy `pdf_skill.py` and change the contents.

### Quant skills — `agents/finance_agent/skills/`

The finance agent's skills are where a quant background plugs into the
project: the model calls *your* implementations instead of hand-waving
statistics from training data.

- `portfolio_skill.py` — CAGR, vol, Sharpe/Sortino, max drawdown, correlation
  for up to 20 tickers, with the conventions disclosed in every result.
- `technical_skill.py` — SMA 50/200 and crosses, Wilder RSI, 12-1 momentum,
  52-week range, realized vol.
- `backtest_skill.py` — SMA crossover vs buy-and-hold. Signals are lagged one
  bar (no lookahead), transaction costs are charged per position change, and
  the caveats list rides **inside the tool result**, so the model cannot
  summarize the numbers without seeing them.

Two design rules worth stealing:

1. **Bulk data never transits model context.** Skills fetch price history via
   the yfinance *library* into a local cache (`skills/_market_data.py`),
   compute locally, and return summary statistics. The MCP tools are for
   interactive lookups; a backtest over years of OHLCV through tool-call
   payloads would be slow, expensive, and lossy.
2. **Pure math lives in `_quant.py` with no I/O**, so `tests/test_quant.py`
   can pin every convention to hand-computed answers on synthetic series —
   including that a constant-return series reports Sharpe as undefined rather
   than the ~1e14 that naive floating point produces, and that the backtest
   trades exactly one bar after its signal.

---

## Layout

```
agents/github_agent/     GitHub agent — agent.py wires everything, exports `app`
  tools/github_write.py  approval-gated GitHub writes
  skills/pdf_skill.py    the drop-in skill template
agents/finance_agent/    finance agent — same shape, different capabilities
  tools/watchlist.py     approval-gated local watchlist (its only write path)
  skills/_quant.py       pure quant math, no I/O — what test_quant.py pins down
  skills/_market_data.py yfinance-library fetch + on-disk cache
  skills/*_skill.py      portfolio, technical, backtest
ingestion/               loaders -> chunking -> embed -> Chroma store, + CLI
  corpus_tools.py        per-collection retrieval factory + calibrated thresholds
plugins/                 safety and observability, registered on both Apps
server/                  FastAPI app and frontend (agent picker, approval cards)
skills_registry.py       shared skill discovery
corpus/                  sample GitHub-agent documents (fictional on purpose)
corpus_finance/          sample research notes for the finance agent
tests/                   114 tests: chunker, loaders, safety, prompts, quant
```

---

## Tests

```bash
python -m pytest
```

114 tests covering chunk boundary preservation, overlap, id stability, loader
structure recovery, the safety boundary (untrusted-data wrapping, write policy,
redaction, injection screening), the prompt-wiring invariant above, and
known-answer quant tests on synthetic series (Sharpe/Sortino/drawdown/RSI
conventions, backtest signal lag, cost monotonicity, cache behavior).

For the end-to-end behaviour that unit tests can't cover — does it actually
retrieve before answering, does it actually pause — use the table in
[What to try](#what-to-try) and watch the tool-call panel.

`python compare_grounding.py` shows the contrast directly: the same question to
an agent with tools and one without. The ungrounded agent confidently invents a
number; the grounded one returns the real count with a live timestamp.

---

## Notes and limits

- **Free-tier quota is the failure you'll hit first.** Gemini's free tier caps
  requests per model per day; a few dozen turns can exhaust it. The UI surfaces
  a 429 explicitly rather than going blank. Set `CHATBOT_MODEL=gemini-flash-lite-latest`
  in `.env` to spread load.
- **chromadb and google-adk conflict on OpenTelemetry.** chromadb wants
  otel ≥1.44, adk 2.3.0 pins ≤1.42.1. `requirements.txt` pins the working
  combination; installing chromadb unpinned silently breaks adk.
- Scanned image PDFs have no extractable text. The pipeline reports that rather
  than guessing; OCR is out of scope.
- The corpus is local and single-process. For multi-user deployment, move
  retrieval behind a service rather than sharing a Chroma directory.
- **The finance agent is a research tool, not an adviser.** No brokerage
  connection, no trade execution, and the prompt + review posture is that
  should-I-buy questions get data with a not-advice framing. Quote data via
  yfinance is ~15 minutes delayed and end-of-day in the quant cache; the
  yfinance MCP server (`yfmcp`, pinned) is community-maintained and Yahoo can
  break it — Alpha Vantage is the official-API fallback.
- Watchlist and price cache live under `data/` (gitignored, user data).
