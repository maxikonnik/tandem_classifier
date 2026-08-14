const video = document.getElementById("video");
const canvas = document.getElementById("scrub");
const ctx = canvas.getContext("2d");
const tlabel = document.getElementById("tlabel");
const warnings = document.getElementById("warnings");
const videomsg = document.getElementById("videomsg");

let data = null;      // {t, ax, ay, az, amag, warnings}
let mode = "mag";
let curPath = null;

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function duration() {
  if (video.duration && isFinite(video.duration)) return video.duration;
  return data && data.t.length ? data.t[data.t.length - 1] : 1;
}

function draw() {
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!data || !data.t.length) return;
  const D = duration();
  const series = mode === "mag"
    ? [[data.amag, "#26215c", 2.4]]
    : [[data.ax, "#378ADD", 1.4], [data.ay, "#1D9E75", 1.4], [data.az, "#D85A30", 1.4]];
  let maxv = 1;
  for (const [arr] of series) for (const v of arr) maxv = Math.max(maxv, Math.abs(v));
  const pad = 8;
  const X = (t) => t / D * w;
  const Y = (v) => h - pad - (v / maxv) * (h - 2 * pad);
  // min/max envelope per pixel column preserves spikes
  for (const [arr, color, lw] of series) {
    ctx.beginPath();
    let px = -1, lo = Infinity, hi = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      const x = Math.floor(X(data.t[i]));
      if (x !== px && px >= 0) {
        ctx.moveTo(px, Y(lo)); ctx.lineTo(px, Y(hi));
        lo = Infinity; hi = -Infinity;
      }
      lo = Math.min(lo, arr[i]); hi = Math.max(hi, arr[i]); px = x;
    }
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  }
  const cx = X(video.currentTime || 0);
  ctx.strokeStyle = "#E24B4A"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
}

function loop() {
  tlabel.textContent = `${fmt(video.currentTime)} / ${fmt(duration())}`;
  draw();
  requestAnimationFrame(loop);
}

canvas.addEventListener("click", (e) => {
  const r = canvas.getBoundingClientRect();
  video.currentTime = (e.clientX - r.left) / r.width * duration();
});

let scrubbing = false;
function seekFromEvent(e) {
  const r = canvas.getBoundingClientRect();
  video.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration();
}
canvas.addEventListener("mousedown", (e) => { scrubbing = true; seekFromEvent(e); });
window.addEventListener("mousemove", (e) => { if (scrubbing) seekFromEvent(e); });
window.addEventListener("mouseup", () => { scrubbing = false; });

document.querySelectorAll(".mbtn").forEach((b) => {
  b.addEventListener("click", () => {
    mode = b.dataset.m;
    document.querySelectorAll(".mbtn").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    draw();
  });
});

async function openFile(path) {
  curPath = path;
  videomsg.textContent = "";
  video.src = `/api/video?path=${encodeURIComponent(path)}`;
  warnings.textContent = "";
  try {
    const res = await fetch(`/api/accel?path=${encodeURIComponent(path)}`);
    const body = await res.json();
    if (body.error) { warnings.textContent = body.error; data = null; return; }
    data = body;
    warnings.textContent = (body.warnings || []).join(" · ");
  } catch (err) {
    warnings.textContent = String(err);
  }
}

video.addEventListener("error", async () => {
  if (!curPath || video.src.includes("proxy")) return;
  videomsg.textContent = "готовлю совместимую копию…";
  const res = await fetch(`/api/proxy?path=${encodeURIComponent(curPath)}`, { method: "POST" });
  const body = await res.json();
  if (body.error) { videomsg.textContent = body.error; return; }
  videomsg.textContent = "";
  video.src = `/api/video?path=${encodeURIComponent(body.path)}&proxy=1`;
});

// --- file picker ---
const modal = document.getElementById("modal");
const fslist = document.getElementById("fslist");
const crumb = document.getElementById("crumb");
let browseCwd = null;

function makeRow(icon, label, onclick) {
  const d = document.createElement("div");
  d.className = "row";
  d.textContent = `${icon}  ${label}`;
  d.addEventListener("click", onclick);
  return d;
}

async function browse(path) {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
  const body = await res.json();
  if (body.error) { crumb.textContent = body.error; return; }
  browseCwd = body.cwd;
  crumb.textContent = body.cwd;
  fslist.innerHTML = "";
  fslist.appendChild(makeRow("⬆", "..", () => browse(body.parent)));
  for (const e of body.entries) {
    if (e.is_dir) fslist.appendChild(makeRow("📁", e.name, () => browse(e.path)));
    else fslist.appendChild(makeRow("🎬", e.name, () => {
      modal.classList.add("hidden");
      addFileOption(e.path, e.name);
      openFile(e.path);
    }));
  }
}

function addFileOption(path, name) {
  const sel = document.getElementById("filesel");
  const opt = document.createElement("option");
  opt.value = path; opt.textContent = name; opt.selected = true;
  sel.insertBefore(opt, sel.firstChild);
}

async function populateList(root) {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(root)}`);
  const body = await res.json();
  const sel = document.getElementById("filesel");
  sel.innerHTML = "";
  for (const e of (body.entries || []).filter((x) => !x.is_dir)) {
    const opt = document.createElement("option");
    opt.value = e.path; opt.textContent = e.name;
    sel.appendChild(opt);
  }
  if (sel.value) openFile(sel.value);
}

document.getElementById("filesel").addEventListener("change", (e) => openFile(e.target.value));
document.getElementById("browse").addEventListener("click", () => {
  modal.classList.remove("hidden"); browse(browseCwd || rootDir);
});
document.getElementById("closem").addEventListener("click", () => modal.classList.add("hidden"));
document.getElementById("usefolder").addEventListener("click", () => {
  modal.classList.add("hidden");
  if (browseCwd) { rootDir = browseCwd; populateList(rootDir); }
});

let rootDir = "";
(async function init() {
  const res = await fetch("/api/roots");
  const body = await res.json();
  rootDir = body.roots[0];
  await populateList(rootDir);
  requestAnimationFrame(loop);
})();
