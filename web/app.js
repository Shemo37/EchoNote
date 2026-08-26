/* EchoNote SPA — vanilla JS, no build step. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = {
  get: (url) => fetch(url).then(r => r.json()),
  del: (url) => fetch(url, { method: "DELETE" }).then(r => r.json()),
  post: (url, body) => fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  }),
  put: (url, body) => fetch(url, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json()),
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtDur(s) {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtDate(t) {
  return new Date(t * 1000).toLocaleString(undefined,
    { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/* citations: [mm:ss] / [h:mm:ss] become clickable seeks */
function linkCitations(html) {
  return html.replace(/\[(\d{1,2}:)?(\d{1,2}):(\d{2})\]/g, (m, h, mm, ss) => {
    const secs = (parseInt(h || "0") * 3600) + parseInt(mm) * 60 + parseInt(ss);
    return `<span class="cite" data-seek="${secs}">${m}</span>`;
  });
}

/* tiny markdown renderer: headers, bold, checkboxes, bullets, paragraphs */
function renderMarkdown(md) {
  const lines = esc(md).split("\n");
  let html = "", inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const line of lines) {
    let m;
    if ((m = line.match(/^#{1,3}\s+(.*)/))) { closeList(); html += `<h3>${m[1]}</h3>`; }
    else if ((m = line.match(/^\s*[-*]\s*\[([ xX]?)\]\s*(.*)/))) {
      if (!inList) { html += "<ul>"; inList = true; }
      const checked = m[1].trim() ? "checked" : "";
      html += `<li><input type="checkbox" disabled ${checked}>${m[2]}</li>`;
    }
    else if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${m[1]}</li>`;
    }
    else if (line.trim() === "") closeList();
    else { closeList(); html += `<p>${line}</p>`; }
  }
  closeList();
  html = html.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  return linkCitations(html);
}

/* SSE POST reader: onDelta(text), resolves with {error?} */
async function streamPost(url, body, onDelta) {
  const resp = await api.post(url, body);
  if (!resp.ok && resp.headers.get("content-type")?.includes("json")) {
    const err = await resp.json();
    return { error: err.detail || err.error || `HTTP ${resp.status}` };
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "", result = {};
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      if (!chunk.startsWith("data: ")) continue;
      const payload = JSON.parse(chunk.slice(6));
      if (payload.delta) onDelta(payload.delta);
      if (payload.error) result.error = payload.error;
      if (payload.done) result.done = true;
    }
  }
  return result;
}

/* ---------------- router ---------------- */

const views = ["dashboard", "library", "recording", "settings"];
let currentRecording = null;
let pollTimer = null;

function show(view) {
  views.forEach(v => { $(`#view-${v}`).hidden = v !== view; });
  document.querySelectorAll("#sidebar nav a").forEach(a =>
    a.classList.toggle("active", a.dataset.view === view));
  if (view === "dashboard") loadDashboard();
  if (view === "library") loadLibrary();
  if (view === "settings") loadSettings();
}

function route() {
  const hash = location.hash.slice(1) || "dashboard";
  const recMatch = hash.match(/^recording\/(\d+)$/);
  if (recMatch) { openRecording(parseInt(recMatch[1])); return; }
  if (views.includes(hash)) show(hash);
}
window.addEventListener("hashchange", route);

/* ---------------- health ---------------- */

async function loadHealth() {
  try {
    const h = await api.get("/api/health");
    $("#health").innerHTML =
      `<span class="${h.ollama ? "ok" : "off"}">Ollama ${h.ollama ? "connected" : "not running"}</span>` +
      `<span class="${h.diarization_configured ? "ok" : "off"}">Speakers ${h.diarization_configured ? "on" : "off"}</span>` +
      `<span class="${h.ffmpeg ? "ok" : "warn-dot"}">ffmpeg ${h.ffmpeg ? "found" : "missing"}</span>`;
  } catch { /* server restarting */ }
}

/* ---------------- dashboard ---------------- */

async function loadDashboard() {
  const d = await api.get("/api/dashboard");
  const s = d.stats;
  const openItems = d.action_items.filter(a => !a.done).length;
  $("#tiles").innerHTML = [
    [s.recordings, "recordings"],
    [(s.total_seconds / 3600).toFixed(1) + "h", "transcribed"],
    [s.this_week, "this week"],
    [openItems, "open action items"],
  ].map(([v, l]) => `<div class="tile"><div class="value">${v}</div><div class="label">${l}</div></div>`).join("");

  const max = Math.max(1, ...s.activity_14d);
  const today = new Date();
  $("#activity").innerHTML = s.activity_14d.map((n, i) => {
    const day = new Date(today); day.setDate(today.getDate() - (13 - i));
    const label = i % 2 ? "" : day.toLocaleDateString(undefined, { day: "numeric" });
    const h = n ? Math.max(8, Math.round(n / max * 64)) : 2;
    return `<div class="col" title="${day.toLocaleDateString()}: ${n} recording${n === 1 ? "" : "s"}">
      <div class="bar ${n ? "" : "zero"}" style="height:${h}px"></div><div class="day">${label}</div></div>`;
  }).join("");

  $("#recent").innerHTML = d.recent.length
    ? d.recent.map(recItemHtml).join("")
    : `<div class="empty">No recordings yet — record or upload one in the Library.</div>`;
  $("#action-items").innerHTML = d.action_items.length
    ? d.action_items.map(a => `
      <div class="action ${a.done ? "done" : ""}">
        <input type="checkbox" ${a.done ? "checked" : ""} data-item="${a.id}">
        <label>${esc(a.text)} <span class="from">— ${esc(a.recording_title || "")}</span></label>
      </div>`).join("")
    : `<div class="empty">Action items from your summaries appear here.</div>`;
}

$("#action-items").addEventListener("change", async (e) => {
  const id = e.target.dataset.item;
  if (id) { await api.post(`/api/action_items/${id}/toggle`); loadDashboard(); }
});

/* ---------------- library ---------------- */

function statusChip(r) {
  if (r.status === "ready") return `<span class="chip ready">ready</span>`;
  if (r.status === "error") return `<span class="chip error" title="${esc(r.error)}">error</span>`;
  let extra = "";
  if (r.progress && r.progress.duration && r.progress.seconds_done)
    extra = ` ${Math.round(r.progress.seconds_done / r.progress.duration * 100)}%`;
  return `<span class="chip busy">${esc(r.status)}${extra}</span>`;
}

function recItemHtml(r) {
  return `<div class="rec-item" data-rec="${r.id}">
    <div class="info">
      <div class="title">${esc(r.title)}</div>
      ${r.gist ? `<div class="gist">${esc(r.gist)}</div>` : ""}
      <div class="meta">${fmtDate(r.created_at)} · ${fmtDur(r.duration_s)}${r.language ? " · " + esc(r.language) : ""}</div>
    </div>
    ${statusChip(r)}
  </div>`;
}

async function loadLibrary() {
  const q = $("#search").value.trim();
  const recs = await api.get(`/api/recordings${q ? `?q=${encodeURIComponent(q)}` : ""}`);
  $("#library-list").innerHTML = recs.length
    ? recs.map(recItemHtml).join("")
    : `<div class="empty">Nothing here yet.</div>`;
  schedulePoll(recs);
}

function schedulePoll(recs) {
  clearTimeout(pollTimer);
  if (recs.some(r => !["ready", "error"].includes(r.status)))
    pollTimer = setTimeout(() => {
      if (!$("#view-library").hidden) loadLibrary();
      else if (!$("#view-dashboard").hidden) loadDashboard();
    }, 2000);
}

$("#search").addEventListener("input", () => { clearTimeout(pollTimer); pollTimer = setTimeout(loadLibrary, 300); });
document.addEventListener("click", (e) => {
  const item = e.target.closest(".rec-item");
  if (item) location.hash = `recording/${item.dataset.rec}`;
});

/* upload */
async function uploadFiles(files) {
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    await fetch("/api/recordings", { method: "POST", body: fd });
  }
  location.hash = "library";
  loadLibrary();
}
$("#btn-upload").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => uploadFiles(e.target.files));
const dz = $("#dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("hover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("hover"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("hover");
  uploadFiles(e.dataTransfer.files);
});

/* ---------------- record modal ---------------- */

let mediaRecorder = null, recChunks = [], recStart = 0, recTimer = null, audioCtx = null;

$("#btn-record").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recChunks.push(e.data);
    mediaRecorder.start();
    recStart = Date.now();
    $("#record-modal").hidden = false;
    recTimer = setInterval(() => {
      const s = Math.floor((Date.now() - recStart) / 1000);
      $("#rec-time").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 250);
    drawMeter(stream);
  } catch (err) {
    alert("Microphone access failed: " + err.message);
  }
});

function drawMeter(stream) {
  audioCtx = new AudioContext();
  const src = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  const canvas = $("#rec-meter"), ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  (function frame() {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = css.getPropertyValue("--accent");
    const bars = 48, step = Math.floor(data.length / bars);
    for (let i = 0; i < bars; i++) {
      const v = data[i * step] / 255;
      const h = Math.max(2, v * canvas.height);
      ctx.fillRect(i * (canvas.width / bars), canvas.height - h, canvas.width / bars - 2, h);
    }
    requestAnimationFrame(frame);
  })();
}

function stopRecording(save) {
  clearInterval(recTimer);
  $("#record-modal").hidden = true;
  if (!mediaRecorder) return;
  const rec = mediaRecorder; mediaRecorder = null;
  rec.onstop = async () => {
    rec.stream.getTracks().forEach(t => t.stop());
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (save && recChunks.length) {
      const blob = new Blob(recChunks, { type: rec.mimeType || "audio/webm" });
      const fd = new FormData();
      const name = `Recording ${new Date().toLocaleString()}`;
      fd.append("file", blob, "recording.webm");
      fd.append("title", name);
      await fetch("/api/recordings", { method: "POST", body: fd });
      loadLibrary();
    }
  };
  rec.stop();
}
$("#rec-stop").addEventListener("click", () => stopRecording(true));
$("#rec-cancel").addEventListener("click", () => stopRecording(false));

/* ---------------- recording view ---------------- */

async function openRecording(id) {
  currentRecording = id;
  show("recording");
  const r = await api.get(`/api/recordings/${id}`);
  $("#rec-title").textContent = r.title;
  $("#rec-meta").textContent =
    `${fmtDate(r.created_at)} · ${fmtDur(r.duration_s)}` +
    (r.language ? ` · ${r.language}` : "") +
    (r.status !== "ready" ? ` · ${r.status}${r.error ? ": " + r.error : ""}` : "");
  $("#player").src = `/api/recordings/${id}/audio`;

  const tb = $("#tab-transcript");
  if (r.segments.length) {
    tb.innerHTML = r.segments.map(s => `
      <div class="segment">
        <span class="ts" data-seek="${s.start_s}">${s.ts}</span>
        ${s.speaker ? `<span class="spk" style="background:${s.color}">${esc(s.speaker)}</span>` : ""}
        <span class="txt">${esc(s.text)}</span>
      </div>`).join("");
  } else {
    tb.innerHTML = `<div class="empty">${r.status === "error"
      ? "Processing failed: " + esc(r.error)
      : "Transcript will appear when processing finishes (" + esc(r.status) + "…)"}</div>`;
    if (!["ready", "error"].includes(r.status))
      pollTimer = setTimeout(() => { if (currentRecording === id) openRecording(id); }, 2000);
  }

  const latest = r.summaries[0];
  $("#summary-content").innerHTML = latest ? renderMarkdown(latest.content)
    : `<div class="empty">Pick a template and press Generate.</div>`;

  $("#chat-log").innerHTML = r.chats.map(c =>
    `<div class="msg ${c.role}">${c.role === "assistant" ? linkCitations(esc(c.content)) : esc(c.content)}</div>`).join("");
}

$("#btn-delete").addEventListener("click", async () => {
  if (!confirm("Delete this recording and its transcript?")) return;
  await api.del(`/api/recordings/${currentRecording}`);
  location.hash = "library";
});

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
  document.querySelectorAll(".tab-body").forEach(b => b.hidden = true);
  $(`#tab-${t.dataset.tab}`).hidden = false;
}));

/* click-to-seek for timestamps + citations */
document.addEventListener("click", (e) => {
  const seek = e.target.dataset?.seek;
  if (seek !== undefined) {
    const player = $("#player");
    player.currentTime = parseFloat(seek);
    player.play();
  }
});

/* summaries */
async function loadTemplates() {
  const names = await api.get("/api/templates");
  $("#template-select").innerHTML =
    names.map(n => `<option value="${n}" ${n === "meeting-minutes" ? "selected" : ""}>${n.replace(/-/g, " ")}</option>`).join("");
}
$("#btn-summarize").addEventListener("click", async () => {
  const btn = $("#btn-summarize"), out = $("#summary-content");
  btn.disabled = true; out.textContent = "";
  let acc = "";
  const res = await streamPost(
    `/api/recordings/${currentRecording}/summarize?template=${$("#template-select").value}`,
    null,
    (d) => { acc += d; out.innerHTML = renderMarkdown(acc); });
  if (res.error) out.innerHTML = `<div class="msg error">${esc(res.error)}</div>` + out.innerHTML;
  btn.disabled = false;
});

$("#btn-pdf").addEventListener("click", () => {
  window.open(`/api/recordings/${currentRecording}/summary.pdf?template=${$("#template-select").value}`, "_blank");
});

/* ask */
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#ask-input"), log = $("#chat-log");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  log.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}</div>`);
  const msg = document.createElement("div");
  msg.className = "msg assistant"; msg.textContent = "…";
  log.appendChild(msg);
  let acc = "";
  const res = await streamPost(`/api/recordings/${currentRecording}/ask`,
    { question: q }, (d) => { acc += d; msg.innerHTML = linkCitations(esc(acc)); });
  if (res.error) { msg.className = "msg error"; msg.textContent = res.error; }
  log.scrollTop = log.scrollHeight;
});

/* ---------------- settings ---------------- */

async function loadSettings() {
  const s = await api.get("/api/settings");
  const form = $("#settings-form");
  for (const el of form.elements) {
    if (el.name && s[el.name] !== undefined) el.value = s[el.name] ?? "";
  }
  form.elements.hf_token.placeholder = s.hf_token_set ? "saved — leave blank to keep" : "hf_...";
}
$("#settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target, changes = {};
  for (const el of form.elements) if (el.name) changes[el.name] = el.value;
  await api.put("/api/settings", changes);
  $("#settings-status").textContent = "Saved ✓";
  setTimeout(() => { $("#settings-status").textContent = ""; }, 2000);
  loadHealth();
});

/* ---------------- boot ---------------- */
loadHealth();
loadTemplates();
setInterval(loadHealth, 15000);
route();
