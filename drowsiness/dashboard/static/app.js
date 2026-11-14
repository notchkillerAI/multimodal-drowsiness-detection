// Dashboard page: opens the camera, sends one JPEG frame at a time to the server, draws what comes back.
// All detection runs in Python on the server (the same code as the rest of the project); this file is display only.
"use strict";

const $ = (id) => document.getElementById(id);
const SEND_WIDTH = 640; // frames are shrunk to this width before sending
const MIN_GAP_MS = 33; // at most about 30 frames per second
const WINDOW_S = 60; // seconds shown in the charts
const KEEP_ROWS = 60000; // rows kept for the CSV (over 30 minutes)
const STATUS = {
  "ALERT": ["good", "✓"], "EARLY WARNING": ["warning", "⚠"], "DROWSY": ["serious", "⚠"],
  "MICROSLEEP": ["critical", "⛔"], "SLEEP": ["critical", "⛔"], "UNRESPONSIVE": ["critical", "⛔"],
  "SENSORS UNRELIABLE": ["none", "?"],
};
const ALARM_GAP_MS = { "DROWSY": 900, "MICROSLEEP": 450, "SLEEP": 300, "UNRESPONSIVE": 200 };

const video = $("video"), overlay = $("overlay"), work = document.createElement("canvas");
let session = null, running = false, stream = null, info = null;
let rows = [], lastReply = null, fps = 0, lastDone = 0, audio = null, lastBeep = 0, settingsTimer = null;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, digits = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(digits));

// ---------------------------------------------------------------- server calls
async function post(path, body, type) {
  const response = await fetch(`${path}${session ? `?s=${session}` : ""}`, {
    method: "POST", body, headers: type ? { "Content-Type": type } : undefined,
  });
  const data = await response.json().catch(() => ({}));
  return { status: response.status, data };
}

async function openSession() {
  session = null;
  const { status, data } = await post("/api/session");
  if (status !== 200) throw new Error(data.error || "The server did not start a session.");
  session = data.session;
  info = data;
  showSettings(data);
  for (const [name, why] of Object.entries(data.unavailable || {})) {
    const card = document.querySelector(`[data-branch="${name}"] .branch-body`);
    if (card) card.innerHTML = `<p class="muted">This branch could not start: ${why}</p>`;
  }
}

// ---------------------------------------------------------------- camera and the frame loop
function note(text, corner = false) {
  const el = $("stage-note");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("corner", corner);
}

async function start(source) {
  try {
    note("Starting…");
    if (source === "camera") {
      if (!navigator.mediaDevices || !window.isSecureContext) {
        throw new Error("The browser only gives a page the camera on https or on localhost. Open this page as http://localhost:PORT, or put the server behind https.");
      }
      const id = $("cameras").value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, deviceId: id ? { exact: id } : undefined }, audio: false,
      });
      video.srcObject = stream;
      listCameras();
    } else {
      video.srcObject = null;
      video.src = "/demo-video";
      video.loop = true;
    }
    await video.play();
    await openSession();
  } catch (error) {
    stopStream();
    const denied = error.name === "NotAllowedError" || error.name === "NotFoundError";
    note(denied
      ? "The camera could not be opened. Allow the camera for this page (the icon in the address bar) and for the browser in System Settings > Privacy & Security > Camera, then press Start again."
      : String(error.message || error));
    return;
  }
  running = true;
  rows = [];
  note("");
  $("link").textContent = source === "camera" ? "live camera" : "demo recording";
  $("link").classList.add("live");
  $("mirror").classList.toggle("flip", $("mirror-on").checked && source === "camera");
  for (const id of ["stop", "recalibrate", "reset", "csv"]) $(id).disabled = false;
  $("start").disabled = $("demo").disabled = true;
  loop();
}

function stopStream() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
}

function stop() {
  running = false;
  stopStream();
  if (session) navigator.sendBeacon(`/api/close?s=${session}`);
  session = null;
  note("Camera is off");
  $("link").textContent = "stopped";
  $("link").classList.remove("live");
  for (const id of ["stop", "recalibrate", "reset"]) $(id).disabled = true;
  $("start").disabled = $("demo").disabled = false;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  showLevel("Not running", "idle", "●", "Stopped. The numbers of this run stay in the page: Download CSV saves them.", null);
}

async function listCameras() {
  const select = $("cameras"), chosen = select.value;
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  select.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || `Camera ${i + 1}`}</option>`).join("");
  if (chosen) select.value = chosen;
  select.hidden = cams.length < 2;
}

async function loop() {
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
    if (answer.status === 410) { // the server forgot this session (it was idle too long, or it restarted)
      try { await openSession(); } catch (error) { note(String(error.message), true); await sleep(2000); }
      continue;
    }
    if (answer.status !== 200) { await sleep(200); continue; }
    const now = performance.now();
    const instant = 1000 / Math.max(now - lastDone, 1);
    fps = fps ? 0.9 * fps + 0.1 * instant : instant;
    lastDone = now;
    render(answer.data, now - began);
    const wait = MIN_GAP_MS - (performance.now() - began);
    if (wait > 0) await sleep(wait);
  }
}
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

// ---------------------------------------------------------------- drawing one reply
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
  $("latency").textContent = `${fmt(roundTripMs, 0)} ms`;
  $("ms").textContent = `${fmt(reply.ms, 0)} ms`;
  $("conf").textContent = fmt(reply.confidence);

  const { ear, cnn } = reply.branches;
  drawOverlay(reply);
  branchCard("ear", ear, (f) => [["EAR", fmt(f.ear)], ["threshold", fmt(f.threshold)], ["left / right", `${fmt(f.left)} / ${fmt(f.right)}`]]);
  branchCard("cnn", cnn, (f) => [["p(closed)", fmt(f.p_closed)], ["left / right", `${fmt(f.left)} / ${fmt(f.right)}`], ["eye crop", `${fmt(f.side, 0)} px`]]);

  rows.push({
    t: reply.t, level: reply.level, fused: reply.fused, smoothed: reply.smoothed, confidence: reply.confidence,
    ear_score: ear.score ?? null, ear_conf: ear.confidence ?? null, ear: ear.frame ? ear.frame.ear : null, ear_thr: ear.frame ? ear.frame.threshold : null,
    cnn_score: cnn.score ?? null, cnn_conf: cnn.confidence ?? null, p_closed: cnn.frame ? cnn.frame.p_closed : null,
    perclos_ear: ear.raw ? ear.raw.perclos : null, perclos_cnn: cnn.raw ? cnn.raw.perclos : null,
    closure_s: Math.max(ear.raw ? ear.raw.current_closure_s : 0, cnn.raw ? cnn.raw.current_closure_s : 0),
  });
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
  const scale = Math.min(overlay.width / fw, overlay.height / fh); // the video is letter-boxed (object-fit: contain)
  ctx.setTransform(scale, 0, 0, scale, (overlay.width - fw * scale) / 2, (overlay.height - fh * scale) / 2);
  ctx.lineWidth = 2.5 / scale * dpr;
  const colour = { OPEN: css("--good"), CLOSED: css("--critical"), UNKNOWN: css("--warning") };
  const { ear, cnn } = reply.branches;
  const face = (ear.frame && ear.frame.face_box) || (cnn.frame && cnn.frame.face_box);
  const state = (cnn.frame && cnn.frame.state !== "UNKNOWN" ? cnn.frame.state : ear.frame && ear.frame.state) || "UNKNOWN";
  if (face) { ctx.strokeStyle = colour[state]; ctx.strokeRect(...face); }
  if (ear.frame && ear.frame.eyes) { // the 12 landmark points the EAR is measured from
    ctx.fillStyle = css("--s1");
    for (const [x, y] of ear.frame.eyes) { ctx.beginPath(); ctx.arc(x, y, 2.2 / scale * dpr, 0, 6.3); ctx.fill(); }
  }
  if (cnn.frame && cnn.frame.eyes && cnn.frame.side) { // the two squares the CNN looks at
    ctx.strokeStyle = css("--s2");
    const half = cnn.frame.side / 2;
    for (const [x, y] of cnn.frame.eyes) ctx.strokeRect(x - half, y - half, 2 * half, 2 * half);
  }
}

function meter(label, value) {
  const width = value === null || value === undefined ? 0 : Math.round(100 * Math.min(Math.max(value, 0), 1));
  return `<div class="meter"><span class="muted">${label}</span><div class="track"><div class="fill" style="width:${width}%"></div></div><span class="val">${fmt(value)}</span></div>`;
}

function branchCard(name, b, frameFacts) {
  const body = document.querySelector(`[data-branch="${name}"] .branch-body`);
  if (!b.connected) return;
  if (!b.on) { body.innerHTML = '<p class="muted">Switched off: it takes no part in the vote.</p>'; return; }
  const f = b.frame || {}, raw = b.raw || {};
  const facts = [
    ...(b.frame ? frameFacts(f) : []),
    ["PERCLOS", raw.perclos === undefined ? "–" : `${fmt(100 * raw.perclos, 0)} %`],
    ["blinks", raw.blink_count === undefined ? "–" : `${raw.blink_count} (${fmt(raw.blink_rate_per_min, 0)}/min)`],
    ["closed now", raw.current_closure_s === undefined ? "–" : `${fmt(raw.current_closure_s, 1)} s`],
    ["longest closure", raw.longest_closure_s === undefined ? "–" : `${fmt(raw.longest_closure_s, 1)} s`],
    ["share of the vote", `${fmt(100 * (b.share || 0), 0)} %`],
    ["frame confidence", fmt(f.confidence)],
  ];
  let weakest = "";
  const factors = Object.entries(f.factors || {});
  if (factors.length) {
    const [key, value] = factors.reduce((a, c) => (c[1] < a[1] ? c : a));
    if (value < 0.8) weakest = `weakest input check: ${key} ${fmt(value)}`;
  }
  body.innerHTML = `<span class="state">${f.state || "–"}</span><span class="muted">${b.status === "ok" ? "" : b.status || ""}</span>`
    + meter("score", b.score) + meter("confidence", b.confidence)
    + `<div class="facts">${facts.map(([k, v]) => `<span><span class="muted">${k}</span><br><b>${v}</b></span>`).join("")}</div>`
    + `<div class="notice">${[f.notice, weakest].filter(Boolean).join("  ·  ")}</div>`;
}

// ---------------------------------------------------------------- charts (canvas, one y axis each, hover shows the values)
class LineChart {
  constructor(canvas, yMax, series, references) {
    this.canvas = canvas; this.yMax = yMax; this.series = series; this.references = references || (() => []);
    this.tip = canvas.parentElement.querySelector(".tip"); this.hover = null;
    this.height = Number(canvas.getAttribute("height")); // the drawing height in CSS pixels (the attribute itself is rescaled below)
    canvas.addEventListener("mousemove", (e) => { this.hover = e.offsetX; this.draw(); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = this.height;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const left = 34, right = 8, top = 8, bottom = 22, pw = w - left - right, ph = h - top - bottom;
    const end = rows.length ? rows[rows.length - 1].t : 0, begin = end - WINDOW_S;
    const X = (t) => left + pw * (t - begin) / WINDOW_S, Y = (v) => top + ph * (1 - v / this.yMax);
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) { // recessive grid, value labels on the left
      const v = this.yMax * i / 4;
      ctx.strokeStyle = i ? css("--grid") : css("--axis"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - right, Y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v.toFixed(2), left - 6, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const back of [60, 45, 30, 15, 0]) ctx.fillText(back ? `-${back} s` : "now", Math.min(X(end - back), w - right - 12), h - bottom + 6);
    for (const { y, label } of this.references()) { // thresholds: dashed, labelled at the right end
      if (y === null || y === undefined) continue;
      ctx.strokeStyle = css("--text2"); ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(left, Y(y)); ctx.lineTo(w - right, Y(y)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--text2"); ctx.fillText(label, w - right - 2, Y(y) - 2);
    }
    const first = rows.findIndex((r) => r.t >= begin), shown = first < 0 ? [] : rows.slice(first);
    for (const s of this.series) {
      ctx.strokeStyle = css(s.colour); ctx.lineWidth = s.width || 2; ctx.lineJoin = "round"; ctx.beginPath();
      let pen = false;
      for (const r of shown) {
        const v = r[s.key];
        if (v === null || v === undefined) { pen = false; continue; } // a gap stays a gap (no line through missing data)
        const x = X(r.t), y = Y(Math.min(v, this.yMax));
        if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    }
    if (this.hover !== null && shown.length) this.showTip(ctx, shown, X, top, ph, w);
  }

  showTip(ctx, shown, X, top, ph, w) {
    const row = shown.reduce((a, r) => (Math.abs(X(r.t) - this.hover) < Math.abs(X(a.t) - this.hover) ? r : a));
    const x = X(row.t);
    ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
    const back = rows[rows.length - 1].t - row.t;
    this.tip.innerHTML = `<div class="muted">${back < 0.05 ? "now" : `${back.toFixed(1)} s ago`} · ${row.level}</div>`
      + this.series.map((s) => `<div><i style="background:${css(s.colour)}"></i>${s.label} <b>${fmt(row[s.key])}</b></div>`).join("");
    this.tip.hidden = false;
    this.tip.style.left = `${x > w / 2 ? x - this.tip.offsetWidth - 10 : x + 10}px`;
  }
}

const charts = [
  new LineChart($("chart-scores"), 1, [
    { key: "ear_score", label: "EAR", colour: "--s1" }, { key: "cnn_score", label: "CNN", colour: "--s2" },
    { key: "smoothed", label: "fused (smoothed)", colour: "--text", width: 2.5 },
  ], () => (info ? [{ y: info.drowsy_on, label: "DROWSY above" }, { y: info.warn_on, label: "EARLY WARNING above" }] : [])),
  new LineChart($("chart-ear"), 0.5, [{ key: "ear", label: "EAR", colour: "--s1" }],
    () => (lastReply && lastReply.branches.ear.frame ? [{ y: lastReply.branches.ear.frame.threshold, label: "closed below" }] : [])),
  new LineChart($("chart-cnn"), 1, [{ key: "p_closed", label: "p(closed)", colour: "--s2" }], () => [{ y: 0.5, label: "closed above" }]),
];

function drawStrip() {
  const c = $("strip"), dpr = window.devicePixelRatio || 1, w = c.clientWidth;
  c.width = Math.round(w * dpr); c.height = 14 * dpr;
  const ctx = c.getContext("2d"), end = rows[rows.length - 1].t, begin = end - WINDOW_S;
  const left = 34 * dpr, pw = c.width - left - 8 * dpr; // same left and right edges as the chart above
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
  const text = [keys.join(","), ...rows.map((r) => keys.map((k) => (r[k] === null ? "" : typeof r[k] === "number" ? +r[k].toFixed(4) : r[k])).join(","))].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = `drowsiness_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

const SLIDERS = { "trust-ear": ["trust", "ear"], "trust-cnn": ["trust", "cnn"], "trust-eog": ["trust", "eog"], "drowsy-on": ["drowsy_on"], "persist-s": ["persist_s"], "smooth-s": ["smooth_s"] };

function showSettings(data) {
  info = { ...info, ...data };
  for (const [id, path] of Object.entries(SLIDERS)) {
    const value = path.length === 2 ? data[path[0]][path[1]] : data[path[0]];
    $(id).value = value;
    $(`o-${id}`).textContent = Number(value).toFixed(2);
  }
  $("warn-supp").checked = data.warn_while_suppressed;
  document.querySelectorAll("[data-enable]").forEach((box) => { box.checked = data.enabled[box.dataset.enable] !== false; });
}

function sendSettings() {
  for (const id of Object.keys(SLIDERS)) $(`o-${id}`).textContent = Number($(id).value).toFixed(2);
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    if (!session) return;
    const enabled = {};
    document.querySelectorAll("[data-enable]").forEach((box) => { enabled[box.dataset.enable] = box.checked; });
    const body = {
      enabled, trust: { ear: +$("trust-ear").value, cnn: +$("trust-cnn").value },
      drowsy_on: +$("drowsy-on").value, persist_s: +$("persist-s").value, smooth_s: +$("smooth-s").value,
      warn_while_suppressed: $("warn-supp").checked,
    };
    const { status, data } = await post("/api/settings", JSON.stringify(body), "application/json");
    if (status === 200) info = { ...info, ...data };
  }, 150);
}

// ---------------------------------------------------------------- wiring
$("start").onclick = () => start("camera");
$("demo").onclick = () => start("demo");
$("stop").onclick = stop;
$("csv").onclick = downloadCsv;
$("recalibrate").onclick = () => post("/api/recalibrate");
$("reset").onclick = () => post("/api/reset");
$("cameras").onchange = () => { if (running && stream) { stop(); start("camera"); } };
$("mirror-on").onchange = () => $("mirror").classList.toggle("flip", $("mirror-on").checked && !!stream);
document.querySelectorAll(".settings input, [data-enable]").forEach((el) => el.addEventListener("input", sendSettings));
window.addEventListener("pagehide", () => { if (session) navigator.sendBeacon(`/api/close?s=${session}`); });
document.addEventListener("visibilitychange", () => { // a hidden tab is slowed down by the browser to about one frame per second
  if (running) note(document.hidden ? "This tab is hidden: the browser slows it down, so detection is unreliable until it is visible again." : "", true);
});
window.addEventListener("resize", () => charts.forEach((chart) => chart.draw()));
fetch("/api/health").then((r) => r.json()).then((health) => { $("demo").hidden = !health.demo_video; }).catch(() => {});
charts.forEach((chart) => chart.draw());
