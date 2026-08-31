const $ = (id) => document.getElementById(id);

const BURST_LABEL = {
  idle: "Ready",
  injecting: "Adding jobs",
  draining: "Finishing background work",
  complete: "Finished",
};

function fmt(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "Unavailable";
  return `${value}${suffix}`;
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function shortGpu(name) {
  if (!name) return "GPU unknown";
  return name.replace("NVIDIA GeForce ", "");
}

function shortModel(name) {
  if (!name) return "Model unknown";
  return String(name).split("/").pop();
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
    { key: "vllm_waiting", color: "#e08b6b" },
    { key: "servescope_pending", color: "#d7b06a" },
  ];
  series.forEach((s) => {
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    history.forEach((p, i) => {
      const x = ((p.t_s - t0) / span) * (w - 16) + 8;
      const y = h - 12 - ((p[s.key] || 0) / maxY) * (h - 24);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function setUnavailable(prefix) {
  const missing = $(`${prefix}-unavailable`);
  if (missing) missing.hidden = false;
  if (prefix === "p4") {
    setText("p4-native-ttft", "Evidence unavailable");
    setText("p4-ss-ttft", "Evidence unavailable");
  }
}

async function loadEvidence() {
  const res = await fetch("/api/evidence");
  const data = await res.json();
  setText("evidence-note", data.note || "");
  renderP4(data.p4);
  renderP3(data.p3);
}

function renderP4(block) {
  if (!block || !block.available) {
    setUnavailable("p4");
    return;
  }
  $("p4-unavailable").hidden = true;
  const native = block.rows[0];
  const gated = block.rows[1];
  setText("p4-native-ttft", native.burst_p95);
  setText("p4-ss-ttft", gated.burst_p95);
  setText("p4-native-wait", native.waiting == null ? "Unavailable" : native.waiting);
  setText("p4-ss-wait", gated.waiting == null ? "Unavailable" : gated.waiting);
  if (block.ttft_reduction_pct != null) {
    setText("p4-reduction", `${block.ttft_reduction_pct}% lower p95 first-token delay`);
  }
  const from = native.background_p95 || "-";
  const to = gated.background_p95 || "-";
  setText("p4-tradeoff", `Background jobs finished later: ${from} → ${to} p95`);
  setText("p4-path", block.path || "");
  if (block.jobs_completed != null) {
    setText("p4-jobs", `All ${block.jobs_completed} jobs still completed.`);
  }
}

function renderP3(block) {
  if (!block || !block.available) {
    $("p3-unavailable").hidden = false;
    return;
  }
  $("p3-unavailable").hidden = true;
  setText("p3-fcfs", block.rows[0].burst_p95);
  setText("p3-native", block.rows[1].burst_p95);
  setText("p3-path", block.path || "");
}

function applyLive(data) {
  const connected = data.server === "connected";
  document.body.classList.toggle("server-live", connected);
  document.body.classList.toggle("server-offline", !connected);
  document.body.classList.toggle("mode-native", data.mode === "native");
  document.body.classList.toggle("mode-backpressure", data.mode === "backpressure");
  document.body.classList.remove("burst-idle", "burst-injecting", "burst-draining", "burst-complete");
  document.body.classList.add(`burst-${data.burst_state || "idle"}`);

  $("offline-banner").hidden = connected;
  setText("live-status", connected ? "Live" : "Offline");
  setText("live-gpu", connected ? shortGpu(data.gpu_name) : "GPU unavailable");
  setText("live-model", connected ? shortModel(data.model) : "Model unavailable");

  setText("model", fmt(data.model));
  setText("gpu-name", fmt(data.gpu_name));
  setText("gpu-util", data.gpu_util_pct == null ? "Unavailable" : `${data.gpu_util_pct}%`);
  if (data.vram_used_mib == null || data.vram_total_mib == null) {
    setText("vram", "Unavailable");
  } else {
    setText("vram", `${Math.round(data.vram_used_mib)} / ${Math.round(data.vram_total_mib)} MiB`);
  }
  const waiting = data.vllm_waiting;
  const pending = data.mode === "backpressure" ? data.background_pending : 0;
  setText("vllm-running", data.vllm_running == null ? "Unavailable" : String(data.vllm_running));
  setText("vllm-waiting", waiting == null ? "Unavailable" : String(waiting));
  setText("burst-state", BURST_LABEL[data.burst_state] || data.burst_state);
  setText("bg-offered", data.background_offered);
  setText("bg-admitted", data.background_admitted);
  setText("bg-running", data.background_running);
  setText("bg-completed", data.background_completed);
  setText("bg-failed", data.background_failed);
  setText("queue-runtime", waiting == null ? "Unavailable" : String(waiting));
  setText("queue-local", String(pending));
  setText("flow-held", String(pending));
  setText("flow-ss-wait", waiting == null ? "Unavailable" : String(waiting));
  setText(
    "flow-native-wait",
    waiting == null ? "Unavailable waiting inside server" : `${waiting} waiting inside server`,
  );

  const target = data.burst_target || (data.burst_preset && data.burst_preset.jobs) || 0;
  const finished = (data.background_completed || 0) + (data.background_failed || 0);
  const denom = Math.max(target, data.background_offered || 0, 1);
  $("progress-fill").style.width = `${Math.min(100, (finished / denom) * 100)}%`;
  setText("progress-label", `${finished} / ${target || data.background_offered || 0} finished`);

  if (data.mode === "backpressure") {
    setText(
      "admission-line",
      `Background jobs allowed in: ${data.controller_limit} (concurrency limit) · last action: ${data.controller_action}`,
    );
    setText("hood-cap", String(data.controller_limit));
    setText("hood-action", data.controller_action);
    setText("flow-caption", "ServeScope can hold extra background jobs before they reach the model server.");
  } else {
    setText("admission-line", "Native vLLM sends background jobs immediately. Nothing is held by ServeScope.");
    setText("hood-cap", "n/a in native mode");
    setText("hood-action", "n/a in native mode");
    setText("flow-caption", "Native vLLM sends background jobs straight to the model server.");
  }

  if (data.burst_preset) {
    const p = data.burst_preset;
    $("burst-hint").textContent =
      `${p.rps} jobs/s for ${p.duration_s} seconds · ${p.jobs} real model requests. Shorter than the recorded 60-second mixed workload.`;
  }

  $("mode-native").classList.toggle("active", data.mode === "native");
  $("mode-backpressure").classList.toggle("active", data.mode === "backpressure");
  $("burst").disabled = !data.burst_allowed || !connected;
  $("send").disabled = !connected;
  $("mode-native").disabled = !data.mode_switch_allowed;
  $("mode-backpressure").disabled = !data.mode_switch_allowed;
  drawChart(data.history || []);
}

async function poll() {
  try {
    const res = await fetch("/api/live");
    applyLive(await res.json());
  } catch {
    document.body.classList.add("server-offline");
    document.body.classList.remove("server-live");
    $("offline-banner").hidden = false;
    setText("live-status", "Offline");
    $("burst").disabled = true;
    $("send").disabled = true;
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
  $("ttft-value").textContent = "waiting";
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
    $("ttft-value").textContent = "-";
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
          $("ttft-value").textContent = `${Math.round(performance.now() - t0)} ms`;
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
