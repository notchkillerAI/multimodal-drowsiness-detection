// Dashboard page for one mode (EAR only, DL only, EOG only, multimodal). Display only: all detection runs in Python on
// the server. Live camera: this page opens the camera and sends one JPEG frame at a time. Recordings and a live
// electrode sensor are read by the server, and this page asks for the newest result.
"use strict";

const $ = (id) => document.getElementById(id);
const SEND_WIDTH = 640; // camera frames are shrunk to this width before sending
const MIN_GAP_MS = 33; // at most about 30 frames per second
const WINDOW_S = 60; // seconds shown in the score charts
const SIGNAL_S = 30; // seconds of electrode signal shown
const KEEP_ROWS = 60000; // rows kept for the CSV
const STATUS = {
  "ALERT": ["good", "✓"], "EARLY WARNING": ["warning", "⚠"], "DROWSY": ["serious", "⚠"],
  "MICROSLEEP": ["critical", "⛔"], "SLEEP": ["critical", "⛔"], "UNRESPONSIVE": ["critical", "⛔"],
  "SENSORS UNRELIABLE": ["none", "?"],
};
const ALARM_GAP_MS = { "DROWSY": 900, "MICROSLEEP": 450, "SLEEP": 300, "UNRESPONSIVE": 200 };
const BRANCH = {
  ear: { name: "EAR", colour: "s1", about: "eye aspect ratio from 68 face landmarks" },
  cnn: { name: "DL", colour: "s2", about: "eye-state neural network on the two eye crops" },
  eog: { name: "EOG", colour: "s3", about: "eye events in the electrode signal, against the person's normal" },
};

const video = $("video"), picture = $("picture"), overlay = $("overlay"), work = document.createElement("canvas");
let cfg = null, names = [], session = null, running = false, stream = null, info = null, replay = false;
let rows = [], signal = [], events = [], lastReply = null, fps = 0, lastDone = 0, audio = null, lastBeep = 0, settingsTimer = null;
let charts = [];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, digits = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(digits));
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- server calls
async function post(path, body, type) {
  const response = await fetch(`${path}${session ? `?s=${session}` : ""}`, {
    method: "POST", body, headers: type ? { "Content-Type": type } : undefined,
  });
  return { status: response.status, data: await response.json().catch(() => ({})) };
}

async function openSession(source) {
  session = null;
  const { status, data } = await post("/api/session", JSON.stringify({ source, speed: +$("speed").value }), "application/json");
  if (status !== 200) throw new Error(data.error || "The server did not start a session.");
  session = data.session;
  replay = data.replay;
  showSettings(data);
  for (const [name, why] of Object.entries(data.unavailable || {})) {
    const body = document.querySelector(`[data-branch="${name}"] .branch-body`);
    if (body) body.innerHTML = `<p class="muted">This branch could not start: ${escapeHtml(why)}</p>`;
  }
}

// ---------------------------------------------------------------- page set-up from the mode
function build() {
  names = [...cfg.video, ...(cfg.eog ? ["eog"] : [])];
  $("title").textContent = cfg.title;
  document.title = cfg.title;
  const groups = { live: "Live", recorded: "Pre-recorded" };
  $("source").innerHTML = Object.entries(groups).map(([group, title]) => {
    const options = cfg.sources.filter((s) => s.group === group).map((s) =>
      `<option value="${escapeHtml(s.id)}" ${s.available ? "" : "disabled"}>${escapeHtml(s.label)}${s.available ? "" : " (not connected)"}</option>`);
    return options.length ? `<optgroup label="${title}">${options.join("")}</optgroup>` : "";
  }).join("");
  const first = cfg.sources.find((s) => s.available);
  if (first) $("source").value = first.id;
  $("start").disabled = !first;

  $("branches").innerHTML = names.map((name) => `
    <article class="card branch" data-branch="${name}">
      <div class="branch-head"><span class="swatch ${BRANCH[name].colour}"></span><h2>${BRANCH[name].name}</h2><span class="muted">${BRANCH[name].about}</span>
        ${names.length > 1 ? `<label class="switch"><input type="checkbox" data-enable="${name}" checked><span>on</span></label>` : ""}</div>
      <div class="branch-body"><p class="muted">Waiting for the first result.</p></div>
    </article>`).join("");
  const hasCamera = cfg.video.length > 0;
  $("camera-card").hidden = !hasCamera;
  $("main").classList.toggle("no-camera", !hasCamera);
  $("recalibrate").hidden = !cfg.video.includes("ear");
  $("ear-chart-card").hidden = !cfg.video.includes("ear");
  $("cnn-chart-card").hidden = !cfg.video.includes("cnn");
  $("eye-charts").hidden = !hasCamera;
  $("signal-card").hidden = !cfg.eog;
  for (const name of names) document.querySelector(`[data-weight="${name}"]`).hidden = names.length < 2;
  $("formula").textContent = names.length > 1
    ? "fused = Σ weight × confidence × score ÷ Σ weight × confidence. A closure of 1 s or more seen by a confident camera branch alarms at once, whatever the score."
    : cfg.eog ? "The level follows the smoothed EOG score. The length of an electrode event is not the length of an eye closure, so the EOG alone never raises MICROSLEEP."
      : "The level follows the smoothed score of this one branch. A closure of 1 s or more alarms at once, whatever the score.";

  const series = [...names.map((name) => ({ key: `${name}_score`, label: BRANCH[name].name, colour: `--${BRANCH[name].colour}` })),
    { key: "smoothed", label: names.length > 1 ? "fused (smoothed)" : "smoothed", colour: "--text", width: 2.5 }];
  $("score-legend").innerHTML = [...series].reverse().map((s) => `<span><i class="line" style="background:var(${s.colour})"></i>${s.label}</span>`).join("");
  charts = [new LineChart($("chart-scores"), { rows: () => rows, windowS: WINDOW_S, range: () => [0, 1], series,
    references: () => (info ? [{ y: info.drowsy_on, label: "DROWSY above" }, { y: info.warn_on, label: "EARLY WARNING above" }] : []) })];
  if (cfg.video.includes("ear")) {
    charts.push(new LineChart($("chart-ear"), { rows: () => rows, windowS: WINDOW_S, range: () => [0, 0.5], series: [{ key: "ear", label: "EAR", colour: "--s1" }],
      references: () => { const b = lastReply && lastReply.branches.ear; return b && b.frame ? [{ y: b.frame.threshold, label: "closed below" }] : []; } }));
  }
  if (cfg.video.includes("cnn")) {
    charts.push(new LineChart($("chart-cnn"), { rows: () => rows, windowS: WINDOW_S, range: () => [0, 1], series: [{ key: "p_closed", label: "p(closed)", colour: "--s2" }],
      references: () => [{ y: 0.5, label: "closed above" }] }));
  }
  if (cfg.eog) {
    charts.push(new LineChart($("chart-signal"), { rows: () => signal, windowS: SIGNAL_S, series: [{ key: "v", label: "signal", colour: "--s3", width: 1.5 }],
      bands: () => events, digits: 0,
      range: () => { // symmetric around zero for a signed signal, from zero for an activity; wide enough for what is on screen
        let low = 0, high = 1;
        for (const p of signal) { if (p.v < low) low = p.v; if (p.v > high) high = p.v; }
        const top = Math.max(high, -low, 50);
        return low < 0 ? [-top, top] : [0, Math.max(high, 5)];
      } }));
  }
  document.querySelectorAll(".settings input, [data-enable]").forEach((el) => el.addEventListener("input", sendSettings));
  sourceChanged();
  charts.forEach((chart) => chart.draw());
}

function sourceChanged() {
  const chosen = cfg.sources.find((s) => s.id === $("source").value);
  const notes = cfg.sources.filter((s) => s.note && (!s.available || s === chosen)).map((s) => `${s.label}: ${s.note}`);
  $("source-note").hidden = !notes.length;
  $("source-note").textContent = notes.join("  ·  ");
  $("speed-field").hidden = cfg.video.length > 0 || !chosen || chosen.group !== "recorded";
}

// ---------------------------------------------------------------- start, stop, the two loops
function note(text, corner = false) {
  const el = $("stage-note");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("corner", corner);
}

async function start() {
  const source = $("source").value;
  $("start").disabled = true;
  try {
    note(source === "camera" ? "Starting the camera…" : "Loading the recording…");
    showLevel("Starting…", "idle", "●", source === "camera" ? "Opening the camera." : "Loading the recording and the models.", null);
    if (source === "camera") {
      if (!navigator.mediaDevices || !window.isSecureContext) {
        throw new Error("The browser only gives a page the camera on https or on localhost. Open this page as http://localhost:PORT, or put the server behind https.");
      }
      const id = $("cameras").value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, deviceId: id ? { exact: id } : undefined }, audio: false,
      });
      video.srcObject = stream;
      await video.play();
      listCameras();
    }
    await openSession(source);
  } catch (error) {
    stopStream();
    const denied = error.name === "NotAllowedError" || error.name === "NotFoundError";
    const text = denied
      ? "The camera could not be opened. Allow the camera for this page (the icon in the address bar) and for the browser in System Settings > Privacy & Security > Camera, then press Start again."
      : String(error.message || error);
    note(text);
    showLevel("Not running", "idle", "●", text, null);
    $("start").disabled = false;
    return;
  }
  running = true;
  rows = []; signal = []; events = []; fps = 0;
  note("");
  video.hidden = replay;
  picture.hidden = !replay;
  $("recording").hidden = !replay || source === "sensor";
  $("mirror-field").hidden = replay;
  $("mirror").classList.toggle("flip", !replay && $("mirror-on").checked);
  $("latency-name").textContent = replay ? "time in the recording" : "round trip";
  $("link").textContent = replay ? (source === "sensor" ? "live sensor" : "pre-recorded") : "live camera";
  $("link").classList.add("live");
  for (const id of ["stop", "reset", "csv"]) $(id).disabled = false;
  $("recalibrate").disabled = !cfg.video.includes("ear");
  $("source").disabled = $("speed").disabled = true;
  if (replay) pollLoop(); else cameraLoop();
}

function stopStream() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.pause();
  video.srcObject = null;
}

function stop(message) {
  running = false;
  stopStream();
  if (session) navigator.sendBeacon(`/api/close?s=${session}`);
  session = null;
  note(typeof message === "string" ? message : "Stopped");
  $("link").textContent = "stopped";
  $("link").classList.remove("live");
  for (const id of ["stop", "reset", "recalibrate"]) $(id).disabled = true;
  $("start").disabled = $("source").disabled = $("speed").disabled = false;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  showLevel("Not running", "idle", "●", `${typeof message === "string" ? message : "Stopped."} The numbers of this run stay in the page: Download CSV saves them.`, null);
}

async function listCameras() {
  const select = $("cameras"), chosen = select.value;
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  select.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${escapeHtml(c.label || `Camera ${i + 1}`)}</option>`).join("");
  if (chosen) select.value = chosen;
  select.hidden = cams.length < 2;
}

function tick() {
  const now = performance.now(), instant = 1000 / Math.max(now - lastDone, 1);
  fps = fps ? 0.9 * fps + 0.1 * instant : instant;
  lastDone = now;
}

async function cameraLoop() {
  while (running) {
    const began = performance.now();
    if (video.readyState < 2 || !video.videoWidth) { await sleep(50); continue; }
    const scale = Math.min(1, SEND_WIDTH / video.videoWidth);
    work.width = Math.round(video.videoWidth * scale);
    work.height = Math.round(video.videoHeight * scale);
    work.getContext("2d").drawImage(video, 0, 0, work.width, work.height);
    const jpeg = await new Promise((done) => work.toBlob(done, "image/jpeg", 0.8));
    let answer;
    try {
      answer = await post("/api/frame", jpeg, "image/jpeg");
    } catch (error) {
      note("The server does not answer. Is it still running?", true);
      await sleep(1000);
      continue;
    }
    if (!running) return;
    if (answer.status === 410) { // the server forgot this session (idle too long, or it restarted)
      try { await openSession("camera"); } catch (error) { note(String(error.message), true); await sleep(2000); }
      continue;
    }
    if (answer.status !== 200) { await sleep(200); continue; }
    tick();
    render(answer.data, performance.now() - began);
    const wait = MIN_GAP_MS - (performance.now() - began);
    if (wait > 0) await sleep(wait);
  }
}

async function pollLoop() {
  let seq = 0;
  while (running) {
    let reply;
    try {
      const response = await fetch(`/api/replay/next?s=${session}&after=${seq}`);
      if (response.status === 410) return stop("The server ended this run.");
      reply = await response.json();
    } catch (error) {
      note("The server does not answer. Is it still running?", true);
      await sleep(1000);
      continue;
    }
    if (!running) return;
    if (reply.waiting) continue;
    if (reply.ended) return stop(reply.error ? `The run stopped: ${reply.error}` : "The recording has ended.");
    seq = reply.seq;
    tick();
    render(reply, null);
  }
}

// ---------------------------------------------------------------- drawing one result
function showLevel(level, kind, icon, reason, score) {
  $("banner").className = `banner ${kind}`;
  $("level").textContent = level;
  $("level-icon").textContent = icon;
  $("reason").textContent = reason;
  $("fused").textContent = fmt(score);
}

function render(reply, roundTripMs) {
  lastReply = reply;
  const [kind, icon] = STATUS[reply.level] || ["none", "?"];
  showLevel(reply.level, kind, icon, `${reply.reason}  ·  action: ${reply.action}`, reply.smoothed);
  $("fps").textContent = fmt(fps, 0);
  $("latency").textContent = roundTripMs === null ? clock(reply.t) : `${fmt(roundTripMs, 0)} ms`;
  $("ms").textContent = `${fmt(reply.ms, 0)} ms`;
  $("conf").textContent = fmt(reply.confidence);
  if (reply.duration) {
    $("progress").style.width = `${Math.min(100, 100 * reply.t / reply.duration)}%`;
    $("clock").textContent = `${clock(reply.t)} / ${clock(reply.duration)}`;
    $("label").textContent = reply.label ? `label (only shown, never used by the detector): ${reply.label}` : "this recording has no label";
  }
  if (reply.jpeg) picture.src = `data:image/jpeg;base64,${reply.jpeg}`;
  if (reply.frame) drawOverlay(reply);

  const row = { t: reply.t, level: reply.level, fused: reply.fused, smoothed: reply.smoothed, confidence: reply.confidence };
  for (const name of names) {
    const b = reply.branches[name] || {};
    branchCard(name, b);
    row[`${name}_score`] = b.score ?? null;
    row[`${name}_conf`] = b.confidence ?? null;
    if (name === "ear") { row.ear = b.frame ? b.frame.ear : null; row.ear_thr = b.frame ? b.frame.threshold : null; }
    if (name === "cnn") row.p_closed = b.frame ? b.frame.p_closed : null;
    if (name !== "eog") { row[`${name}_perclos`] = b.raw ? b.raw.perclos ?? null : null; row[`${name}_closure_s`] = b.raw ? b.raw.current_closure_s ?? null : null; }
    if (name === "eog") {
      row.eog_events_per_min = b.raw ? b.raw.events_per_min ?? null : null;
      if (b.signal) {
        for (const [t, v] of b.signal) signal.push({ t, v });
        const from = signal.findIndex((p) => p.t >= reply.t - SIGNAL_S);
        if (from > 0) signal.splice(0, from);
        events = b.events || [];
        $("signal-unit").textContent = b.unit === "µV" ? "filtered signal, µV" : "eye activity";
      }
      $("signal-empty").hidden = !!b.connected;
    }
  }
  rows.push(row);
  if (rows.length > KEEP_ROWS) rows.splice(0, rows.length - KEEP_ROWS);
  charts.forEach((chart) => chart.draw());
  drawStrip();
  alarm(reply.level);
}

function drawOverlay(reply) {
  const box = overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  overlay.width = box.width * dpr;
  overlay.height = box.height * dpr;
  const ctx = overlay.getContext("2d"), [fw, fh] = reply.frame;
  const scale = Math.min(overlay.width / fw, overlay.height / fh); // the picture is letter-boxed (object-fit: contain)
  ctx.setTransform(scale, 0, 0, scale, (overlay.width - fw * scale) / 2, (overlay.height - fh * scale) / 2);
  ctx.lineWidth = 2.5 / scale * dpr;
  const colour = { OPEN: css("--good"), CLOSED: css("--critical"), UNKNOWN: css("--warning") };
  const ear = reply.branches.ear || {}, cnn = reply.branches.cnn || {};
  const face = (ear.frame && ear.frame.face_box) || (cnn.frame && cnn.frame.face_box);
  const state = (cnn.frame && cnn.frame.state !== "UNKNOWN" ? cnn.frame.state : ear.frame && ear.frame.state) || "UNKNOWN";
  if (face) { ctx.strokeStyle = colour[state]; ctx.strokeRect(...face); }
  if (ear.frame && ear.frame.eyes) { // the 12 landmark points the EAR is measured from
    ctx.fillStyle = css("--s1");
    for (const [x, y] of ear.frame.eyes) { ctx.beginPath(); ctx.arc(x, y, 2.2 / scale * dpr, 0, 6.3); ctx.fill(); }
  }
  if (cnn.frame && cnn.frame.eyes && cnn.frame.side) { // the two squares the network looks at
    ctx.strokeStyle = css("--s2");
    const half = cnn.frame.side / 2;
    for (const [x, y] of cnn.frame.eyes) ctx.strokeRect(x - half, y - half, 2 * half, 2 * half);
  }
}

function meter(label, value) {
  const width = value === null || value === undefined ? 0 : Math.round(100 * Math.min(Math.max(value, 0), 1));
  return `<div class="meter"><span class="muted">${label}</span><div class="track"><div class="fill" style="width:${width}%"></div></div><span class="val">${fmt(value)}</span></div>`;
}

function branchCard(name, b) {
  const card = document.querySelector(`[data-branch="${name}"]`), body = card.querySelector(".branch-body");
  card.classList.toggle("waiting", !b.connected);
  if (!b.connected) {
    if (name === "eog") body.innerHTML = '<p class="muted">Not connected. This place is kept for the EOG device: once it is connected its score, confidence and eye events appear here and it joins the vote with its own weight.</p>';
    return;
  }
  if (!b.on) { body.innerHTML = '<p class="muted">Switched off: it takes no part in the vote.</p>'; return; }
  const f = b.frame || {}, raw = b.raw || {};
  const have = (v) => v !== undefined && v !== null;
  let state, facts, notice = "";
  if (name === "eog") {
    state = raw.eye_closing_now ? "EVENT" : (raw.status === "ok" ? "OK" : "–");
    facts = [
      ["eye events / min", fmt(raw.events_per_min, 0)], ["mean event", have(raw.mean_event_s) ? `${fmt(raw.mean_event_s)} s` : "–"],
      ["share of time in events", have(raw.closure_share) ? `${fmt(100 * raw.closure_share, 0)} %` : "–"],
      ["irregularity", fmt(raw.irregularity)], ["signal quality", fmt(raw.signal_quality)], ["events so far", have(raw.n_events) ? raw.n_events : "–"],
      ["share of the vote", `${fmt(100 * (b.share || 0), 0)} %`],
    ];
    notice = raw.status && raw.status !== "ok" ? raw.status : "";
  } else {
    state = f.state || "–";
    facts = [
      ...(name === "ear" && b.frame ? [["EAR", fmt(f.ear)], ["threshold", fmt(f.threshold)], ["left / right", `${fmt(f.left)} / ${fmt(f.right)}`]] : []),
      ...(name === "cnn" && b.frame ? [["p(closed)", fmt(f.p_closed)], ["left / right", `${fmt(f.left)} / ${fmt(f.right)}`], ["eye crop", `${fmt(f.side, 0)} px`]] : []),
      ["PERCLOS", have(raw.perclos) ? `${fmt(100 * raw.perclos, 0)} %` : "–"],
      ["blinks", have(raw.blink_count) ? `${raw.blink_count} (${fmt(raw.blink_rate_per_min, 0)}/min)` : "–"],
      ["closed now", have(raw.current_closure_s) ? `${fmt(raw.current_closure_s, 1)} s` : "–"],
      ["longest closure", have(raw.longest_closure_s) ? `${fmt(raw.longest_closure_s, 1)} s` : "–"],
      ["share of the vote", `${fmt(100 * (b.share || 0), 0)} %`],
      ["frame confidence", fmt(f.confidence)],
    ];
    const factors = Object.entries(f.factors || {});
    let weakest = "";
    if (factors.length) {
      const [key, value] = factors.reduce((a, c) => (c[1] < a[1] ? c : a));
      if (value < 0.8) weakest = `weakest input check: ${key} ${fmt(value)}`;
    }
    notice = [f.notice, weakest].filter(Boolean).join("  ·  ");
  }
  body.innerHTML = `<span class="state">${state}</span><span class="muted">${b.status === "ok" ? "" : b.status || ""}</span>`
    + meter("score", b.score) + meter("confidence", b.confidence)
    + `<div class="facts">${facts.map(([k, v]) => `<span><span class="muted">${k}</span><br><b>${v}</b></span>`).join("")}</div>`
    + `<div class="notice">${escapeHtml(notice)}</div>`;
}

// ---------------------------------------------------------------- charts (canvas, one y axis each, hover shows the values)
class LineChart {
  constructor(canvas, options) {
    Object.assign(this, { references: () => [], bands: () => [], digits: 2 }, options);
    this.canvas = canvas;
    this.tip = canvas.parentElement.querySelector(".tip"); this.hover = null;
    this.height = Number(canvas.getAttribute("height")); // drawing height in CSS pixels (the attribute itself is rescaled below)
    canvas.addEventListener("mousemove", (e) => { this.hover = e.offsetX; this.draw(); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = this.height;
    if (!w) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const data = this.rows(), [low, high] = this.range();
    const left = 42, right = 8, top = 8, bottom = 22, pw = w - left - right, ph = h - top - bottom;
    const end = data.length ? data[data.length - 1].t : 0, begin = end - this.windowS;
    const X = (t) => left + pw * (t - begin) / this.windowS, Y = (v) => top + ph * (1 - (v - low) / (high - low));
    for (const [from, to] of this.bands()) { // detected events: shaded behind the line
      const x0 = Math.max(X(from), left), x1 = Math.min(X(to), w - right);
      if (x1 <= left || x0 >= w - right) continue;
      ctx.fillStyle = css("--warning"); ctx.globalAlpha = 0.28; ctx.fillRect(x0, top, Math.max(x1 - x0, 1.5), ph); ctx.globalAlpha = 1;
    }
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) { // recessive grid, value labels on the left
      const v = low + (high - low) * i / 4;
      ctx.strokeStyle = v === 0 || i === 0 ? css("--axis") : css("--grid"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - right, Y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v.toFixed(this.digits), left - 6, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let i = 0; i <= 4; i++) {
      const back = this.windowS * (4 - i) / 4;
      ctx.fillText(back ? `-${back} s` : "now", Math.min(X(end - back), w - right - 12), h - bottom + 6);
    }
    for (const { y, label } of this.references()) { // thresholds: dashed, labelled at the left end (the newest data is on the right)
      if (y === null || y === undefined) continue;
      ctx.strokeStyle = css("--text2"); ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(left, Y(y)); ctx.lineTo(w - right, Y(y)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--text2"); ctx.fillText(label, left + 4, Y(y) - 2);
    }
    const first = data.findIndex((r) => r.t >= begin), shown = first < 0 ? [] : data.slice(first);
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();
    for (const s of this.series) {
      ctx.strokeStyle = css(s.colour); ctx.lineWidth = s.width || 2; ctx.lineJoin = "round"; ctx.beginPath();
      let pen = false;
      for (const r of shown) {
        const v = r[s.key];
        if (v === null || v === undefined) { pen = false; continue; } // a gap stays a gap (no line through missing data)
        const x = X(r.t), y = Y(Math.min(Math.max(v, low), high));
        if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    }
    ctx.restore();
    if (this.hover !== null && shown.length) this.showTip(ctx, shown, X, top, ph, w, end);
  }

  showTip(ctx, shown, X, top, ph, w, end) {
    const row = shown.reduce((a, r) => (Math.abs(X(r.t) - this.hover) < Math.abs(X(a.t) - this.hover) ? r : a));
    const x = X(row.t), back = end - row.t;
    ctx.strokeStyle = css("--text2"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
    this.tip.innerHTML = `<div class="muted">${back < 0.05 ? "now" : `${back.toFixed(1)} s ago`}${row.level ? ` · ${row.level}` : ""}</div>`
      + this.series.map((s) => `<div><i style="background:${css(s.colour)}"></i>${s.label} <b>${fmt(row[s.key], this.digits)}</b></div>`).join("");
    this.tip.hidden = false;
    this.tip.style.left = `${x > w / 2 ? x - this.tip.offsetWidth - 10 : x + 10}px`;
  }
}

function drawStrip() {
  const c = $("strip"), dpr = window.devicePixelRatio || 1, w = c.clientWidth;
  c.width = Math.round(w * dpr); c.height = 14 * dpr;
  const ctx = c.getContext("2d"), end = rows[rows.length - 1].t, begin = end - WINDOW_S;
  const left = 42 * dpr, pw = c.width - left - 8 * dpr; // same left and right edges as the chart above
  let previous = null;
  for (const r of rows) {
    if (r.t < begin) continue;
    const x = left + pw * (r.t - begin) / WINDOW_S;
    if (previous !== null) { ctx.fillStyle = css(`--${(STATUS[r.level] || ["none"])[0]}`); ctx.fillRect(previous, 0, Math.max(x - previous, 1), c.height); }
    previous = x;
  }
}

// ---------------------------------------------------------------- alarm sound, CSV, settings
function alarm(level) {
  const gap = ALARM_GAP_MS[level];
  if (!gap || !$("sound").checked || performance.now() - lastBeep < gap) return;
  lastBeep = performance.now();
  audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  const tone = audio.createOscillator(), volume = audio.createGain();
  tone.frequency.value = level === "DROWSY" ? 880 : 1320;
  tone.type = level === "DROWSY" ? "sine" : "square";
  volume.gain.setValueAtTime(0.15, audio.currentTime);
  volume.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.15);
  tone.connect(volume).connect(audio.destination);
  tone.start(); tone.stop(audio.currentTime + 0.16);
}

function downloadCsv() {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const cell = (v) => (v === null || v === undefined ? "" : typeof v === "number" ? +v.toFixed(4) : v);
  const text = [keys.join(","), ...rows.map((r) => keys.map((k) => cell(r[k])).join(","))].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = `drowsiness_${cfg.mode}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

const SLIDERS = { "trust-ear": ["trust", "ear"], "trust-cnn": ["trust", "cnn"], "trust-eog": ["trust", "eog"], "drowsy-on": ["drowsy_on"], "persist-s": ["persist_s"], "smooth-s": ["smooth_s"] };

function showSettings(data) {
  info = { ...info, ...data };
  for (const [id, path] of Object.entries(SLIDERS)) {
    const value = path.length === 2 ? data[path[0]][path[1]] : data[path[0]];
    if (value === undefined) continue;
    $(id).value = value;
    $(`o-${id}`).textContent = Number(value).toFixed(2);
  }
  $("warn-supp").checked = data.warn_while_suppressed;
  if (cfg.eog) $("trust-eog").disabled = !(data.signal && data.signal.eog);
  document.querySelectorAll("[data-enable]").forEach((box) => { box.checked = data.enabled[box.dataset.enable] !== false; box.disabled = !(box.dataset.enable in data.enabled); });
}

function sendSettings() {
  for (const id of Object.keys(SLIDERS)) $(`o-${id}`).textContent = Number($(id).value).toFixed(2);
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    if (!session) return;
    const enabled = {}, trust = {};
    document.querySelectorAll("[data-enable]").forEach((box) => { if (!box.disabled) enabled[box.dataset.enable] = box.checked; });
    for (const name of names) if (names.length > 1 && !$(`trust-${name}`).disabled) trust[name] = +$(`trust-${name}`).value;
    const body = { enabled, trust, drowsy_on: +$("drowsy-on").value, persist_s: +$("persist-s").value, smooth_s: +$("smooth-s").value, warn_while_suppressed: $("warn-supp").checked };
    const { status, data } = await post("/api/settings", JSON.stringify(body), "application/json");
    if (status === 200) info = { ...info, ...data };
  }, 150);
}

// ---------------------------------------------------------------- wiring
$("start").onclick = start;
$("stop").onclick = stop;
$("csv").onclick = downloadCsv;
$("recalibrate").onclick = () => post("/api/recalibrate");
$("reset").onclick = () => post("/api/reset");
$("source").onchange = sourceChanged;
$("cameras").onchange = async () => { if (running && stream) { stop(); await start(); } };
$("mirror-on").onchange = () => $("mirror").classList.toggle("flip", $("mirror-on").checked && !!stream);
window.addEventListener("pagehide", () => { if (session) navigator.sendBeacon(`/api/close?s=${session}`); });
document.addEventListener("visibilitychange", () => { // a hidden tab is slowed down by the browser to about one frame per second
  if (running && !replay) note(document.hidden ? "This tab is hidden: the browser slows it down, so detection is unreliable until it is visible again." : "", true);
});
window.addEventListener("resize", () => charts.forEach((chart) => chart.draw()));
fetch("/api/config").then((r) => r.json()).then((config) => { cfg = config; build(); })
  .catch(() => showLevel("No connection", "idle", "●", "The dashboard server does not answer.", null));
