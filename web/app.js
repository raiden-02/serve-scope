const $ = (id) => document.getElementById(id);

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

function formatSeconds(value) {
  if (value === null || value === undefined) return "-";
  const seconds = Number(value);
  if (Number.isNaN(seconds)) return "-";
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  return `${Math.round(seconds * 1000)} ms`;
}

function formatCount(value) {
  if (value === null || value === undefined) return "-";
  return String(value);
}

function drawChart(canvasId, history, yMax) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) return;
  const maxY = Math.max(
    1,
    yMax || 0,
    ...history.map((p) => Math.max(p.vllm_waiting || 0, p.servescope_pending || 0)),
  );
  const t0 = history[0].t_s || 0;
  const t1 = history[history.length - 1].t_s || t0;
  const span = Math.max(0.001, t1 - t0);
  const series = [
    { key: "vllm_waiting", color: "#e08b6b" },
    { key: "servescope_pending", color: "#d7b06a" },
  ];
  series.forEach((s) => {
    if (!history.some((p) => (p[s.key] || 0) > 0) && s.key === "servescope_pending") return;
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
    setText("p4-jobs", `All ${block.jobs_completed} jobs completed.`);
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

function applySide(prefix, side, sharedY) {
  if (!side) return;
  const m = side.metrics || {};
  setText(`${prefix}-status`, side.status_label || side.state || "-");
  setText(`${prefix}-phase`, side.phase || "-");
  const elapsed = side.elapsed_s || 0;
  const pct = Math.min(100, (elapsed / 60) * 100);
  const bar = $(`${prefix === "native" ? "native" : "ss"}-timeline`);
  if (bar) bar.style.width = `${pct}%`;
  const p95 = formatSeconds(m.burst_p95_ttft_s);
  setText(`${prefix}-p95`, p95);
  setText(
    `${prefix}-p95-note`,
    side.final ? "First-token p95" : `p95 so far · ${m.burst_p95_sample_count || 0} burst samples`,
  );
  setText(`${prefix}-wait`, formatCount(m.vllm_waiting));
  setText(`${prefix}-wait-max`, formatCount(m.max_vllm_waiting));
  setText(`${prefix}-ix-offered`, formatCount(m.interactive_target != null ? m.interactive_target : m.interactive_offered));
  setText(`${prefix}-ix-done`, formatCount(m.interactive_completed));
  setText(`${prefix}-ix-fail`, formatCount(m.interactive_failed));
  setText(`${prefix}-bg-offered`, formatCount(m.background_target != null ? m.background_target : m.background_offered));
  setText(`${prefix}-bg-admitted`, formatCount(m.background_admitted));
  setText(`${prefix}-bg-done`, formatCount(m.background_completed));
  setText(`${prefix}-bg-fail`, formatCount(m.background_failed));
  if (prefix === "ss") {
    setText("ss-pending", formatCount(m.local_pending));
    setText("ss-pending-max", formatCount(m.peak_local_pending));
  }
  if (side.final && m.background_e2e_p95_s != null) {
    const extra = m.background_output_goodput_tps != null
      ? ` · ${Math.round(m.background_output_goodput_tps)} tok/s`
      : "";
    setText(`${prefix}-tradeoff`, `Background p95 total E2E ${formatSeconds(m.background_e2e_p95_s)}${extra}`);
  }
  drawChart(prefix === "native" ? "native-chart" : "ss-chart", side.history || [], sharedY);
}

function applyComparison(cmp) {
  if (!cmp) return;
  setText("comparison-state", cmp.overall_state || "idle");
  const wl = cmp.workload || {};
  if (wl.interactive) {
    setText(
      "workload-line",
      `Live workload: ${wl.interactive.offered_rps} interactive requests/s for ${wl.scenario_duration_s} seconds, with ${wl.background.offered_rps} background requests/s during the ${wl.pre_burst_end_s}–${wl.burst_end_s} second burst.`,
    );
  }
  const nativeHist = (cmp.native && cmp.native.history) || [];
  const ssHist = (cmp.servescope && cmp.servescope.history) || [];
  const sharedY = Math.max(
    1,
    ...nativeHist.map((p) => p.vllm_waiting || 0),
    ...ssHist.map((p) => Math.max(p.vllm_waiting || 0, p.servescope_pending || 0)),
  );
  applySide("native", cmp.native, sharedY);
  applySide("ss", cmp.servescope, sharedY);

  const done = ["complete", "invalid", "failed", "cancelled"].includes(cmp.overall_state);
  $("live-result").hidden = !done;
  if (!done) return;

  const native = (cmp.native && (cmp.native.final || cmp.native.metrics)) || {};
  const gated = (cmp.servescope && (cmp.servescope.final || cmp.servescope.metrics)) || {};
  setText("live-native-p95", formatSeconds(native.burst_p95_ttft_s));
  setText("live-ss-p95", formatSeconds(gated.burst_p95_ttft_s));
  setText("live-native-wait", formatCount(native.max_vllm_waiting));
  setText("live-ss-wait", formatCount(gated.max_vllm_waiting));
  setText("live-native-pending", "0");
  setText("live-ss-pending", formatCount(gated.peak_local_pending));
  setText("live-native-e2e", formatSeconds(native.background_e2e_p95_s));
  setText("live-ss-e2e", formatSeconds(gated.background_e2e_p95_s));
  setText(
    "live-native-bg",
    `${formatCount(native.background_completed)}/${formatCount(native.background_offered)}`,
  );
  setText(
    "live-ss-bg",
    `${formatCount(gated.background_completed)}/${formatCount(gated.background_offered)}`,
  );
  setText(
    "live-native-fail",
    String((native.interactive_failed || 0) + (native.background_failed || 0)),
  );
  setText(
    "live-ss-fail",
    String((gated.interactive_failed || 0) + (gated.background_failed || 0)),
  );

  const result = cmp.comparison || {};
  if (result.available) {
    setText("live-headline", result.headline || "");
    $("live-headline").classList.remove("invalid");
    setText("live-validity", "");
  } else {
    setText("live-headline", result.headline || "Live run was not valid for comparison.");
    $("live-headline").classList.add("invalid");
    setText("live-validity", result.reason ? `Reason: ${result.reason}` : "");
  }
}

function applyLive(data) {
  const connected = data.server === "connected";
  document.body.classList.toggle("server-live", connected);
  document.body.classList.toggle("server-offline", !connected);
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
  setText("vllm-running", data.vllm_running == null ? "Unavailable" : String(data.vllm_running));
  setText("vllm-waiting", data.vllm_waiting == null ? "Unavailable" : String(data.vllm_waiting));

  const cmp = data.comparison;
  applyComparison(cmp);
  const running = cmp && cmp.overall_state === "running";
  $("run-comparison").disabled = !connected || running;
  $("send").disabled = !connected || Boolean(data.chat_locked);
  $("chat-lock").hidden = !data.chat_locked;
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
    $("run-comparison").disabled = true;
    $("send").disabled = true;
  }
}

async function startComparison() {
  const res = await fetch("/api/comparison/start", { method: "POST" });
  if (!res.ok) {
    $("chat-status").textContent = await res.text();
    return;
  }
  applyComparison(await res.json());
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
      } else if (msg.type === "first" && msg.backend_ttft_ms != null && !first) {
        first = true;
        $("ttft-value").textContent = `${Math.round(msg.backend_ttft_ms)} ms`;
      } else if (msg.type === "error") {
        $("chat-status").textContent = msg.message;
      }
    }
  }
}

$("chat-form").addEventListener("submit", sendChat);
$("send").addEventListener("click", sendChat);
$("run-comparison").addEventListener("click", startComparison);
loadEvidence();
poll();
setInterval(poll, 500);
