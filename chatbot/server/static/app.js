/*
 * Frontend for the grounded GitHub assistant.
 *
 * Three things this client has to get right:
 *
 *  1. The ADK run API is camelCase (appName / userId / sessionId / newMessage).
 *     Some published examples show snake_case; that is stale and 422s here.
 *  2. Human-in-the-loop resume: when the agent pauses, it emits a function call
 *     named `adk_request_confirmation`. Approving means POSTing a
 *     functionResponse whose `id` matches that call's id exactly. A mismatched
 *     id leaves the agent waiting forever with no visible error.
 *  3. Tool calls belong on screen. The trace panel is what lets a user check
 *     that an answer was actually retrieved rather than recalled.
 */

const CONFIRM_TOOL = "adk_request_confirmation";

const state = {
  appName: null,
  userId: "web-user",
  sessionId: null,
  busy: false,
};

const el = {
  status: document.getElementById("status"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  trace: document.getElementById("trace"),
  corpus: document.getElementById("corpus"),
  drop: document.getElementById("drop"),
  file: document.getElementById("file"),
  uploadStatus: document.getElementById("upload-status"),
};

/* ---------------------------------------------------------------- setup */

async function init() {
  try {
    const config = await fetchJSON("/api/config");
    state.appName = config.agent_name;

    const session = await fetchJSON(
      `/apps/${state.appName}/users/${state.userId}/sessions`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
    );
    state.sessionId = session.id;

    setStatus(`session ${session.id.slice(0, 8)}`, "ready");
    refreshCorpus();
  } catch (err) {
    setStatus(`startup failed: ${err.message}`, "error");
  }
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

function setStatus(text, cls = "") {
  el.status.textContent = text;
  el.status.className = `status ${cls}`;
}

/* -------------------------------------------------------------- sending */

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value.trim();
  if (!text || state.busy) return;
  el.input.value = "";
  autosize();
  addMessage("user", text);
  run({ role: "user", parts: [{ text }] });
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});
el.input.addEventListener("input", autosize);

function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 144)}px`;
}

async function run(newMessage) {
  setBusy(true);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("spinner");
  let gotText = false;

  try {
    const response = await fetch("/run_sse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        appName: state.appName,
        userId: state.userId,
        sessionId: state.sessionId,
        newMessage,
        streaming: false,
      }),
    });

    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);

    for await (const event of readSSE(response)) {
      if (handleEvent(event, bubble)) gotText = true;
    }

    if (!gotText && !bubble.textContent) {
      // A run that ends with no text is normal when the agent paused for
      // approval -- the approval card carries the state instead.
      bubble.closest(".msg").remove();
    }
  } catch (err) {
    bubble.textContent = `Error: ${err.message}`;
    bubble.closest(".msg").classList.add("error");
  } finally {
    bubble.classList.remove("spinner");
    setBusy(false);
    scrollDown();
  }
}

function setBusy(busy) {
  state.busy = busy;
  el.send.disabled = busy;
  el.input.disabled = busy;
  if (!busy) el.input.focus();
}

/* Parse an SSE body incrementally: split on blank lines, take `data:` field. */
async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      const payload = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");

      if (!payload) continue;
      try {
        yield JSON.parse(payload);
      } catch {
        /* keepalive or partial frame */
      }
    }
  }
}

/* Returns true if the event contributed assistant text. */
function handleEvent(event, bubble) {
  // A failed run still returns HTTP 200 and reports the failure *inside* the
  // stream, so `response.ok` says nothing about it. Without this branch a
  // quota error or model failure leaves the UI silently blank, which is the
  // single most confusing thing this client could do.
  if (event?.errorCode || event?.errorMessage) {
    const message = event.errorMessage || event.errorCode;
    bubble.textContent = `${friendlyError(event.errorCode, message)}`;
    bubble.closest(".msg").classList.add("error");
    return true;
  }

  const parts = event?.content?.parts || [];
  let producedText = false;

  for (const part of parts) {
    if (part.text) {
      bubble.textContent += part.text;
      producedText = true;
      scrollDown();
    }

    if (part.functionCall) {
      const call = part.functionCall;
      if (call.name === CONFIRM_TOOL) {
        renderApproval(call);
      } else {
        addTrace(call.name, call.args, "call");
      }
    }

    if (part.functionResponse) {
      const fr = part.functionResponse;
      if (fr.name !== CONFIRM_TOOL) {
        addTrace(fr.name, summarizeResult(fr.response), "response");
      }
    }
  }
  return producedText;
}

/* Quota exhaustion is by far the most common failure on a free API key, and
 * the raw message is a wall of JSON. Translate the ones worth translating. */
function friendlyError(code, message) {
  const text = String(message || "");
  if (code === "RESOURCE_EXHAUSTED" || text.includes("RESOURCE_EXHAUSTED")) {
    const retry = text.match(/retry in ([\d.]+)s/i);
    return (
      "Gemini API quota exhausted." +
      (retry ? ` Retry in about ${Math.ceil(Number(retry[1]))}s.` : "") +
      " The free tier caps requests per model per day — set" +
      " CHATBOT_MODEL=gemini-flash-lite-latest in .env to spread the load."
    );
  }
  return `Run failed: ${text.slice(0, 300)}`;
}

function summarizeResult(response) {
  if (!response || typeof response !== "object") return response;
  const status = response.status;
  if (status === "blocked") return { status, error: response.error };
  if (Array.isArray(response.results)) {
    return { status, results: response.results.length };
  }
  return { status: status ?? "ok" };
}

/* ------------------------------------------------------------- approval */

function renderApproval(call) {
  const original = call.args?.originalFunctionCall || {};
  const confirmation = call.args?.toolConfirmation || {};
  const payload = confirmation.payload || {};
  const editable = Object.keys(payload).length > 0;

  const card = document.createElement("div");
  card.className = "approval";

  const title = document.createElement("h3");
  title.textContent = `Approval required — ${original.name || "action"}`;
  card.appendChild(title);

  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent =
    confirmation.hint || `Run ${original.name} with these arguments?`;
  card.appendChild(hint);

  // Editable payload fields, so approving can also mean amending.
  const inputs = {};
  if (editable) {
    for (const [key, value] of Object.entries(payload)) {
      const label = document.createElement("label");
      label.textContent = key;
      card.appendChild(label);

      const multiline = typeof value === "string" && value.length > 60;
      const field = document.createElement(multiline ? "textarea" : "input");
      field.value = value ?? "";
      card.appendChild(field);
      inputs[key] = field;
    }
  } else {
    const args = document.createElement("pre");
    args.className = "args";
    args.textContent = JSON.stringify(original.args || {}, null, 2);
    card.appendChild(args);
  }

  const actions = document.createElement("div");
  actions.className = "actions";

  const approve = document.createElement("button");
  approve.className = "primary";
  approve.textContent = editable ? "Approve with these values" : "Approve";

  const reject = document.createElement("button");
  reject.className = "danger";
  reject.textContent = "Reject";

  actions.append(approve, reject);
  card.appendChild(actions);

  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.appendChild(card);
  el.messages.appendChild(wrapper);
  scrollDown();

  const resolve = (confirmed) => {
    const edited = {};
    for (const [key, field] of Object.entries(inputs)) {
      const original_value = payload[key];
      edited[key] =
        typeof original_value === "number" ? Number(field.value) : field.value;
    }

    actions.remove();
    const resolved = document.createElement("div");
    resolved.className = `resolved ${confirmed ? "approved" : "rejected"}`;
    resolved.textContent = confirmed ? "✓ Approved" : "✗ Rejected";
    card.appendChild(resolved);

    for (const field of Object.values(inputs)) field.disabled = true;

    // The id must be the confirmation call's id, not the original tool call's.
    run({
      role: "user",
      parts: [
        {
          functionResponse: {
            id: call.id,
            name: CONFIRM_TOOL,
            response: { confirmed, payload: editable ? edited : payload },
          },
        },
      ],
    });
  };

  approve.addEventListener("click", () => resolve(true));
  reject.addEventListener("click", () => resolve(false));
}

/* ---------------------------------------------------------------- trace */

function addTrace(name, detail, kind) {
  if (el.trace.querySelector(".muted")) el.trace.innerHTML = "";

  const blocked = detail && detail.status === "blocked";
  const item = document.createElement("div");
  item.className = `call${blocked ? " blocked" : ""}`;

  const tag = document.createElement("div");
  tag.className = `tag${blocked ? " blocked" : kind === "response" ? " ok" : ""}`;
  tag.textContent = blocked ? "blocked by policy" : kind;
  item.appendChild(tag);

  const label = document.createElement("div");
  label.className = "name";
  label.textContent = name;
  item.appendChild(label);

  if (detail && Object.keys(detail).length) {
    const args = document.createElement("div");
    args.className = "args";
    args.textContent = JSON.stringify(detail, null, 1);
    item.appendChild(args);
  }

  el.trace.appendChild(item);
  el.trace.scrollTop = el.trace.scrollHeight;
}

/* --------------------------------------------------------------- corpus */

async function refreshCorpus() {
  try {
    const stats = await fetchJSON("/api/corpus/stats");
    el.corpus.className = "corpus";
    if (!stats.chunks) {
      el.corpus.innerHTML =
        '<span class="muted">Empty. Index a document to enable corpus grounding.</span>';
      return;
    }
    const list = stats.sources.map((s) => `<li>${escapeHTML(s)}</li>`).join("");
    el.corpus.innerHTML = `<strong>${stats.documents}</strong> document(s), <strong>${stats.chunks}</strong> chunks<ul>${list}</ul>`;
  } catch (err) {
    el.corpus.className = "corpus muted";
    el.corpus.textContent = `stats unavailable: ${err.message}`;
  }
}

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

el.drop.addEventListener("click", () => el.file.click());
el.file.addEventListener("change", () => {
  if (el.file.files[0]) upload(el.file.files[0]);
});

["dragenter", "dragover"].forEach((name) =>
  el.drop.addEventListener(name, (event) => {
    event.preventDefault();
    el.drop.classList.add("over");
  })
);
["dragleave", "drop"].forEach((name) =>
  el.drop.addEventListener(name, (event) => {
    event.preventDefault();
    el.drop.classList.remove("over");
  })
);
el.drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) upload(file);
});

async function upload(file) {
  el.uploadStatus.className = "upload-status";
  el.uploadStatus.textContent = `Indexing ${file.name}… (chunking and embedding)`;

  const form = new FormData();
  form.append("file", file);

  try {
    const result = await fetchJSON("/api/ingest", { method: "POST", body: form });
    el.uploadStatus.className = "upload-status ok";
    el.uploadStatus.textContent = `Indexed ${result.filename} — ${result.chunks_indexed} chunks.`;
    refreshCorpus();
  } catch (err) {
    el.uploadStatus.className = "upload-status error";
    el.uploadStatus.textContent = `Failed: ${err.message}`;
  } finally {
    el.file.value = "";
  }
}

/* -------------------------------------------------------------- helpers */

function addMessage(role, text) {
  const empty = el.messages.querySelector(".empty");
  if (empty) empty.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;

  wrapper.append(label, body);
  el.messages.appendChild(wrapper);
  scrollDown();
  return body;
}

function scrollDown() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

init();
