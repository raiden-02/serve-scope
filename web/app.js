const $ = (id) => document.getElementById(id);

function fmt(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "Unavailable";
  return `${value}${suffix}`;
}

function setText(id, value) {
  $(id).textContent = value;
}

function drawChart(history) {
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) return;
  const maxY = Math.max(1, ...history.map((p) => Math.max(p.vllm_waiting || 0, p.servescope_pending || 0)));
  const t0 = history[0].t_s;
  const t1 = history[history.length - 1].t_s;
  const span = Math.max(0.001, t1 - t0);
  const series = [
    { key: "vllm_waiting", color: "#8a2b1e" },
    { key: "servescope_pending", color: "#2b4c7e" },
  ];
  series.forEach((s) => {
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    history.forEach((p, i) => {
      const x = ((p.t_s - t0) / span) * (w - 16) + 8;
      const y = h - 12 - ((p[s.key] || 0) / maxY) * (h - 24);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#5c574e";
  ctx.fillText("vLLM waiting", 8, 14);
  ctx.fillStyle = "#2b4c7e";
  ctx.fillText("ServeScope deferred", 110, 14);
}

async function loadEvidence() {
  const res = await fetch("/api/evidence");
  const data = await res.json();
  $("evidence-note").textContent = data.note || "";
  renderEvidence("p3", data.p3);
  renderEvidence("p4", data.p4);
}

function renderEvidence(prefix, block) {
  if (!block || !block.available) {
    $(`${prefix}-title`).textContent = "Evidence unavailable";
    $(`${prefix}-body`).textContent = "The accepted comparison artifact is missing.";
    $(`${prefix}-path`).textContent = "";
    return;
  }
  $(`${prefix}-title`).textContent = block.label;
  const lines = (block.rows || []).map((row) => {
    const bits = [`${row.name}: burst p95 ${row.burst_p95}`];
    if (row.waiting != null) bits.push(`runtime waiting ${row.waiting}`);
    if (row.local_pending != null) bits.push(`local pending ${row.local_pending}`);
    if (row.background_p95) bits.push(`background p95 ${row.background_p95}`);
    if (row.output_goodput) bits.push(row.output_goodput);
    return bits.join(". ");
  });
  if (block.claim_note) lines.push(block.claim_note);
  lines.push(block.session);
  $(`${prefix}-body`).textContent = lines.join("\n\n");
  $(`${prefix}-path`).textContent = block.path;
}

function applyLive(data) {
  const connected = data.server === "connected";
  const pill = $("server-pill");
  pill.textContent = connected ? "vLLM connected" : "Disconnected";
  pill.className = `pill ${connected ? "connected" : "disconnected"}`;
  setText("model", fmt(data.model));
  setText("gpu-name", fmt(data.gpu_name));
  setText("gpu-util", data.gpu_util_pct == null ? "Unavailable" : `${data.gpu_util_pct}%`);
  if (data.vram_used_mib == null || data.vram_total_mib == null) {
    setText("vram", "Unavailable");
  } else {
    setText("vram", `${Math.round(data.vram_used_mib)} / ${Math.round(data.vram_total_mib)} MiB`);
  }
  setText("vllm-running", data.vllm_running == null ? "Unavailable" : String(data.vllm_running));
  setText("vllm-waiting", data.vllm_waiting == null ? "Unavailable" : String(data.vllm_waiting));
  setText("burst-state", data.burst_state);
  setText("bg-offered", data.background_offered);
  setText("bg-admitted", data.background_admitted);
  setText("bg-running", data.background_running);
  setText("bg-pending", data.mode === "backpressure" ? data.background_pending : 0);
  setText("bg-completed", data.background_completed);
  setText("bg-failed", data.background_failed);
  setText("queue-runtime", data.vllm_waiting == null ? "Unavailable" : String(data.vllm_waiting));
  setText("queue-local", data.mode === "backpressure" ? String(data.background_pending) : "0 (native submits immediately)");
  if (data.mode === "backpressure") {
    setText(
      "admission-line",
      `Background cap: ${data.controller_limit} · last action: ${data.controller_action}`,
    );
  } else {
    setText("admission-line", "Native priority submits background jobs immediately. No local defer queue.");
  }
  if (data.burst_preset) {
    const p = data.burst_preset;
    $("burst-hint").textContent =
      `Demo burst: ${p.rps} jobs/s × ${p.duration_s} s = ${p.jobs} jobs. These are real model requests, not a P4 benchmark.`;
  }
  $("mode-native").classList.toggle("active", data.mode === "native");
  $("mode-backpressure").classList.toggle("active", data.mode === "backpressure");
  $("burst").disabled = !data.burst_allowed;
  $("mode-native").disabled = !data.mode_switch_allowed;
  $("mode-backpressure").disabled = !data.mode_switch_allowed;
  drawChart(data.history || []);
}

async function poll() {
  try {
    const res = await fetch("/api/live");
    applyLive(await res.json());
  } catch {
    $("server-pill").textContent = "Disconnected";
    $("server-pill").className = "pill disconnected";
  }
}

async function setMode(mode) {
  const res = await fetch("/api/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    $("chat-status").textContent = await res.text();
    return;
  }
  applyLive(await res.json());
}

async function startBurst() {
  const res = await fetch("/api/burst", { method: "POST" });
  if (!res.ok) {
    $("chat-status").textContent = await res.text();
    return;
  }
  applyLive(await res.json());
}

async function sendChat(event) {
  event.preventDefault();
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  $("reply").textContent = "";
  $("ttft").textContent = "Live UI TTFT: waiting";
  $("chat-status").textContent = "";
  const t0 = performance.now();
  let first = false;
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    $("chat-status").textContent = (await res.text()) || "Chat failed";
    $("ttft").textContent = "Live UI TTFT: -";
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((row) => row.startsWith("data: "));
      if (!line) continue;
      const msg = JSON.parse(line.slice(6));
      if (msg.type === "token") {
        if (!first) {
          first = true;
          $("ttft").textContent = `Live UI TTFT: ${Math.round(performance.now() - t0)} ms`;
        }
        $("reply").textContent += msg.text;
      } else if (msg.type === "error") {
        $("chat-status").textContent = msg.message;
      }
    }
  }
}

$("chat-form").addEventListener("submit", sendChat);
$("send").addEventListener("click", sendChat);
$("burst").addEventListener("click", startBurst);
$("mode-native").addEventListener("click", () => setMode("native"));
$("mode-backpressure").addEventListener("click", () => setMode("backpressure"));
loadEvidence();
poll();
setInterval(poll, 500);
