/*
 * Pose Trainer (browser edition)
 * ==============================
 * Port of originClaude.py (Raspberry Pi 5 + Hailo pose server) to run fully
 * in the browser. The Hailo YOLOv8-pose model is replaced by TensorFlow.js
 * MoveNet, which outputs the same 17 COCO keypoints, so feature extraction,
 * matching, coaching and the trained pose library are unchanged.
 */
'use strict';

// ---------- Configuration (same values as originClaude.py) ----------
const PERSON_CONF_THRESH   = 0.25;
const KP_CONF_THRESH       = 0.15;
const TRAIN_COUNTDOWN_SECS = 5;
const TRAIN_SNAPSHOTS      = 10;
const TRAIN_SNAP_INTERVAL  = 0.4;
const REC_COUNTDOWN_SECS   = 3;
const MATCH_THRESHOLD      = 120.0;
const MATCH_MARGIN         = 0.75;
const DISPLAY_CONF_THRESH  = 0.50;
const POSE_HISTORY_SIZE    = 10;
const POSE_MIN_VOTES       = 3;
const ANGLE_TOLERANCE      = 15.0;
const CAPTURE_W            = 640;
const CAPTURE_H            = 480;
const KP_SMOOTH_ALPHA      = 0.30;
const MIN_VISIBLE_KPS      = 3;
const COACH_TIP_INTERVAL   = 2.0;
const STORAGE_KEY          = 'poseTrainer.library.v1';

const COCO_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];
const COCO = Object.fromEntries(COCO_NAMES.map((n, i) => [n, i]));

const COCO_CONNECTIONS = [
  [0,1],[0,2],[1,3],[2,4],[5,6],
  [5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
];

const CONNECTION_ANGLE_MAP = {
  '5,7': 'L_shoulder', '7,9': 'L_elbow',
  '6,8': 'R_shoulder', '8,10': 'R_elbow',
  '5,11': 'L_hip',     '6,12': 'R_hip',
  '11,13': 'L_knee',   '13,15': 'L_knee',
  '12,14': 'R_knee',   '14,16': 'R_knee',
  '11,12': 'hip_tilt', '5,6': 'shldr_tilt',
};

const FEATURE_ANGLE_KEYS = [
  'L_elbow','R_elbow','L_shoulder','R_shoulder',
  'L_hip','R_hip','L_knee','R_knee',
  'L_ankle','R_ankle','hip_tilt','shldr_tilt',
];
const FEATURE_EXTRA_KEYS = [
  'L_arm_ext','R_arm_ext','L_leg_ext','R_leg_ext',
  'L_forearm_ext','R_forearm_ext',
  'torso_lean','lr_symmetry','shoulder_hip_ratio',
];
const ALL_FEATURE_KEYS = [...FEATURE_ANGLE_KEYS, ...FEATURE_EXTRA_KEYS]; // 21
const FEATURE_SCALES = Object.fromEntries([
  ...FEATURE_ANGLE_KEYS.map(k => [k, 45.0]),
  ...FEATURE_EXTRA_KEYS.map(k => [k, 0.25]),
]);

const JOINT_NAMES = {
  L_elbow: 'left elbow', R_elbow: 'right elbow',
  L_shoulder: 'left shoulder', R_shoulder: 'right shoulder',
  L_hip: 'left hip', R_hip: 'right hip',
  L_knee: 'left knee', R_knee: 'right knee',
  L_ankle: 'left ankle', R_ankle: 'right ankle',
  hip_tilt: 'hip level', shldr_tilt: 'shoulder level',
};

const COL_GREEN = '#00dc50';
const COL_RED   = '#dc3c00';
const COL_GREY  = '#a0a0a0';
const COL_NODE  = '#ffb400';

const $ = id => document.getElementById(id);

// ===================================================================
//  POSE LIBRARY (replaces training_data/ + trained_model.json)
// ===================================================================
// library = { name: { vectors: [[21 floats], ...], active: bool, preview: [[x,y,conf]x17] | null } }

let library = {};
let trainedModel = {};   // name -> mean vector (active poses only)

function saveLibrary() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(library));
    return true;
  } catch (e) {
    toast('Could not save to browser storage. Use Export to keep your poses.', 5000);
    return false;
  }
}

function normalizeLibrary(raw) {
  const out = {};
  for (const [name, d] of Object.entries(raw || {})) {
    if (!d || !Array.isArray(d.vectors)) continue;
    const vecs = d.vectors
      .filter(v => Array.isArray(v) && v.length <= ALL_FEATURE_KEYS.length)
      .map(v => { const c = v.map(Number); while (c.length < ALL_FEATURE_KEYS.length) c.push(0); return c; });
    if (!vecs.length) continue;
    out[name] = {
      vectors: vecs,
      active: d.active !== false,
      preview: Array.isArray(d.preview) && d.preview.length === 17 ? d.preview : null,
    };
  }
  return out;
}

async function loadDefaultLibrary() {
  try {
    const r = await fetch('poses.json', { cache: 'no-cache' });
    return normalizeLibrary(await r.json());
  } catch (e) {
    return {};
  }
}

async function loadLibrary() {
  let stored = null;
  try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { /* storage blocked */ }
  if (stored) {
    try { library = normalizeLibrary(JSON.parse(stored)); }
    catch (e) { library = await loadDefaultLibrary(); }
  } else {
    library = await loadDefaultLibrary();
  }
  buildModel();
}

function buildModel() {
  trainedModel = {};
  for (const [name, d] of Object.entries(library)) {
    if (!d.active || !d.vectors.length) continue;
    const n = d.vectors.length;
    trainedModel[name] = ALL_FEATURE_KEYS.map((_, i) =>
      d.vectors.reduce((s, v) => s + v[i], 0) / n);
  }
  $('nmodel').textContent = Object.keys(trainedModel).length;
}

function meanVector(name) {
  const d = library[name];
  if (!d || !d.vectors.length) return null;
  const n = d.vectors.length;
  return ALL_FEATURE_KEYS.map((_, i) => d.vectors.reduce((s, v) => s + v[i], 0) / n);
}

// ===================================================================
//  FEATURE EXTRACTION (ported 1:1 from originClaude.py)
// ===================================================================

let kpCache = {};   // FIX-12: last-known-good keypoint positions
let emaKps  = null; // FIX-13: EMA smoothing state

function smoothKeypoints(kps) {
  if (kps === null) { emaKps = null; return null; }
  if (emaKps === null || emaKps.length !== kps.length) {
    emaKps = kps.map(k => ({ x: k.x, y: k.y, conf: k.conf }));
  } else {
    const a = KP_SMOOTH_ALPHA;
    kps.forEach((k, i) => {
      emaKps[i].x    = a * k.x    + (1 - a) * emaKps[i].x;
      emaKps[i].y    = a * k.y    + (1 - a) * emaKps[i].y;
      emaKps[i].conf = a * k.conf + (1 - a) * emaKps[i].conf;
    });
  }
  return emaKps.map(k => ({ ...k }));
}

function calcAngle(a, b, c) {
  const ax = a.x - b.x, ay = a.y - b.y;
  const cx = c.x - b.x, cy = c.y - b.y;
  const mag = Math.hypot(ax, ay) * Math.hypot(cx, cy);
  if (mag < 1e-6) return null;
  const cos = Math.max(-1, Math.min(1, (ax * cx + ay * cy) / mag));
  return Math.acos(cos) * 180 / Math.PI;
}

function confOk(kps, ...names) {
  return names.every(n => kps[COCO[n]].conf >= KP_CONF_THRESH);
}

function kpWithFallback(kps, name) {
  const idx = COCO[name];
  const k = kps[idx];
  if (k.conf >= KP_CONF_THRESH) {
    kpCache[idx] = { x: k.x, y: k.y };
    return { x: k.x, y: k.y };
  }
  return kpCache[idx] || null;
}

const round = (v, d) => { const f = 10 ** d; return Math.round(v * f) / f; };

function extractAngles(kps, aspect) {
  if (!kps) return {};
  const kpAsp = name => {
    const k = kpWithFallback(kps, name);
    return k ? { x: k.x * aspect, y: k.y } : null;
  };
  const pairs = [
    ['left_shoulder',  'left_elbow',     'left_wrist',     'L_elbow'],
    ['right_shoulder', 'right_elbow',    'right_wrist',    'R_elbow'],
    ['left_elbow',     'left_shoulder',  'left_hip',       'L_shoulder'],
    ['right_elbow',    'right_shoulder', 'right_hip',      'R_shoulder'],
    ['left_shoulder',  'left_hip',       'left_knee',      'L_hip'],
    ['right_shoulder', 'right_hip',      'right_knee',     'R_hip'],
    ['left_hip',       'left_knee',      'left_ankle',     'L_knee'],
    ['right_hip',      'right_knee',     'right_ankle',    'R_knee'],
    ['left_knee',      'left_ankle',     'right_ankle',    'L_ankle'],
    ['right_knee',     'right_ankle',    'left_ankle',     'R_ankle'],
    ['left_hip',       'right_hip',      'right_shoulder', 'hip_tilt'],
    ['left_shoulder',  'right_shoulder', 'right_hip',      'shldr_tilt'],
  ];
  const angles = {};
  for (const [an, bn, cn, label] of pairs) {
    const a = kpAsp(an), b = kpAsp(bn), c = kpAsp(cn);
    if (!a || !b || !c) continue;
    const v = calcAngle(a, b, c);
    if (v !== null) angles[label] = round(v, 2);
  }
  return angles;
}

function limbExt(a) {
  if (a === undefined || a === null) return null;
  return round(Math.max(0, Math.min(1, (a - 30) / 150)), 4);
}

function extractExtras(kps, angles, aspect) {
  const extras = {};
  for (const [src, dst] of [
    ['L_shoulder', 'L_arm_ext'], ['R_shoulder', 'R_arm_ext'],
    ['L_knee', 'L_leg_ext'],     ['R_knee', 'R_leg_ext'],
    ['L_elbow', 'L_forearm_ext'], ['R_elbow', 'R_forearm_ext'],
  ]) {
    const v = limbExt(angles[src]);
    if (v !== null) extras[dst] = v;
  }
  const P = n => kps[COCO[n]];
  if (confOk(kps, 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')) {
    const msx = (P('left_shoulder').x * aspect + P('right_shoulder').x * aspect) / 2;
    const msy = (P('left_shoulder').y + P('right_shoulder').y) / 2;
    const mhx = (P('left_hip').x * aspect + P('right_hip').x * aspect) / 2;
    const mhy = (P('left_hip').y + P('right_hip').y) / 2;
    extras.torso_lean = round(Math.atan2(msx - mhx, -(msy - mhy)) * 180 / Math.PI, 2);

    const hs = 1 - Math.abs(P('left_hip').y - P('right_hip').y) * 4;
    const ss = 1 - Math.abs(P('left_shoulder').y - P('right_shoulder').y) * 4;
    extras.lr_symmetry = round(Math.max(0, Math.min(1, (hs + ss) / 2)), 4);

    const shW  = Math.abs(P('left_shoulder').x - P('right_shoulder').x);
    const hipW = Math.abs(P('left_hip').x - P('right_hip').x);
    if (hipW > 1e-4) extras.shoulder_hip_ratio = round(shW / hipW, 4);
  }
  return extras;
}

function buildFeatureVector(angles, extras) {
  return [...FEATURE_ANGLE_KEYS.map(k => angles[k] ?? 0),
          ...FEATURE_EXTRA_KEYS.map(k => extras[k] ?? 0)];
}

function normalizedVectorDistance(v1, v2) {
  let total = 0, count = 0;
  ALL_FEATURE_KEYS.forEach((key, i) => {
    if (i >= v1.length || i >= v2.length) return;
    const live = v1[i], ref = v2[i];
    if (live === 0) return;
    const s = FEATURE_SCALES[key] || 1;
    total += ((live - ref) / s) ** 2;
    count++;
  });
  return count < 3 ? Infinity : Math.sqrt(total / count);
}

function smoothPoseLabel(history, label) {
  history.push(label);
  if (history.length > POSE_HISTORY_SIZE) history.shift();
  const counts = {};
  for (const x of history) if (x !== null) counts[x] = (counts[x] || 0) + 1;
  let best = null;
  for (const k in counts) if (best === null || counts[k] > counts[best]) best = k;
  return best !== null && counts[best] >= POSE_MIN_VOTES ? best : null;
}

function computeCoachingTips(currentAngles, refAngles) {
  const THRESHOLD = 15.0;
  const priority = ['L_knee','R_knee','L_hip','R_hip','hip_tilt','shldr_tilt',
                    'L_shoulder','R_shoulder','L_elbow','R_elbow','L_ankle','R_ankle'];
  const tips = [];
  for (const joint of priority) {
    const ref = refAngles[joint], live = currentAngles[joint];
    if (ref === undefined || live === undefined) continue;
    const diff = live - ref;
    if (Math.abs(diff) < THRESHOLD) continue;
    const name = JOINT_NAMES[joint] || joint;
    if (joint === 'hip_tilt' || joint === 'shldr_tilt') {
      tips.push(`Level your ${name}. You are tilting to one side.`);
    } else if (diff > 0) {
      tips.push(`Reduce your ${name} angle by about ${Math.round(Math.abs(diff))}°.`);
    } else {
      tips.push(`Increase your ${name} angle by about ${Math.round(Math.abs(diff))}°.`);
    }
    if (tips.length >= 2) break;
  }
  return tips;
}

// ===================================================================
//  DRAWING
// ===================================================================

function drawSkeleton(ctx, kps, w, h, refAngles) {
  let green = 0, total = 0;
  const scored = new Set();
  ctx.lineWidth = Math.max(2, w / 320);
  for (const [a, b] of COCO_CONNECTIONS) {
    if (kps[a].conf < KP_CONF_THRESH || kps[b].conf < KP_CONF_THRESH) continue;
    let color = COL_GREY;
    const ak = CONNECTION_ANGLE_MAP[a + ',' + b];
    if (refAngles && ak) {
      const lv = refAngles['_live_' + ak], rv = refAngles[ak];
      if (lv !== undefined && rv !== undefined) {
        const ok = Math.abs(lv - rv) <= ANGLE_TOLERANCE;
        color = ok ? COL_GREEN : COL_RED;
        if (!scored.has(ak)) { scored.add(ak); total++; if (ok) green++; }
      }
    }
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(kps[a].x * w, kps[a].y * h);
    ctx.lineTo(kps[b].x * w, kps[b].y * h);
    ctx.stroke();
  }
  const r = Math.max(4, w / 128);
  for (const k of kps) {
    if (k.conf < KP_CONF_THRESH) continue;
    ctx.fillStyle = COL_NODE; ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(k.x * w, k.y * h, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  return [green, total];
}

function outlinedText(ctx, text, x, y, size, color, align = 'left') {
  ctx.font = `700 ${size}px system-ui, sans-serif`;
  ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
  ctx.lineWidth = Math.max(3, size / 5); ctx.strokeStyle = '#000';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

function drawCountdown(ctx, text, color, w, h) {
  const size = Math.round(w / 22);
  ctx.font = `700 ${size}px system-ui, sans-serif`;
  const tw = ctx.measureText(text).width;
  const x = (w - tw) / 2, y = h / 2;
  ctx.fillStyle = 'rgba(0,0,0,.75)';
  ctx.fillRect(x - 14, y - size - 8, tw + 28, size + 24);
  outlinedText(ctx, text, x, y, size, color);
}

function drawCoachingOverlay(ctx, tips, stance, accuracy, w, h) {
  const fs = Math.round(w / 34), pad = 12, lh = fs + 10;
  const lines = [[`Coaching: ${stance.replace(/_/g, ' ')}` +
                  (accuracy !== null ? `   accuracy ${accuracy}%` : ''), '#b48cff'],
                 ...tips.map(t => [t, '#ffdc64'])];
  const y0 = h - pad - lines.length * lh - 8;
  ctx.fillStyle = 'rgba(0,0,0,.65)';
  ctx.fillRect(0, y0, w, h - y0);
  lines.forEach(([t, c], i) => outlinedText(ctx, t, pad, y0 + 8 + (i + 1) * lh - 8, fs, c));
}

// Sidebar skeleton (from _SIDEBAR_SKELETON_JS), also used for library thumbnails.
function drawMiniSkeleton(canvas, kps, refAngles, liveAngles, emptyText) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0d0d14'; ctx.fillRect(0, 0, W, H);
  const pts = kps && kps.length >= 17 ? kps.filter(k => k.conf >= KP_CONF_THRESH) : [];
  if (!pts.length) {
    if (emptyText) {
      ctx.fillStyle = '#334155'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(emptyText, W / 2, H / 2);
    }
    return;
  }
  const xs = pts.map(k => k.x), ys = pts.map(k => k.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = Math.max(6, W * 0.06);
  const sx = xmax > xmin ? (W - pad * 2) / (xmax - xmin) : 1;
  const sy = ymax > ymin ? (H - pad * 2) / (ymax - ymin) : 1;
  const s = Math.min(sx, sy);
  const px = k => pad + (k.x - xmin) * s + ((W - pad * 2) - (xmax - xmin) * s) / 2;
  const py = k => pad + (k.y - ymin) * s + ((H - pad * 2) - (ymax - ymin) * s) / 2;
  ctx.lineWidth = 2;
  for (const [a, b] of COCO_CONNECTIONS) {
    if (kps[a].conf < KP_CONF_THRESH || kps[b].conf < KP_CONF_THRESH) continue;
    const ak = CONNECTION_ANGLE_MAP[a + ',' + b];
    let color = '#555566';
    if (refAngles && liveAngles && ak && refAngles[ak] !== undefined && liveAngles[ak] !== undefined) {
      color = Math.abs(liveAngles[ak] - refAngles[ak]) <= ANGLE_TOLERANCE ? '#00dc50' : '#dc3c14';
    } else if (!liveAngles) {
      color = '#8b8bb0';
    }
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.moveTo(px(kps[a]), py(kps[a])); ctx.lineTo(px(kps[b]), py(kps[b])); ctx.stroke();
  }
  for (const k of kps) {
    if (k.conf < KP_CONF_THRESH) continue;
    ctx.fillStyle = '#00b4ff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(px(k), py(k), W > 100 ? 3.5 : 2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
}

// ===================================================================
//  UI HELPERS
// ===================================================================

let toastTimer = null;
function toast(msg, ms = 4000) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

function openOv(id) { $(id).classList.add('show'); }
function closeOv(id) { $(id).classList.remove('show'); }
document.querySelectorAll('[data-close]').forEach(b =>
  b.addEventListener('click', () => closeOv(b.dataset.close)));
document.querySelectorAll('.ov').forEach(ov =>
  ov.addEventListener('click', e => { if (e.target === ov) ov.classList.remove('show'); }));
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.querySelectorAll('.ov.show').forEach(o => o.classList.remove('show'));
});

const prettyName = s => s.replace(/_/g, ' ');
const cleanName  = s => s.trim().replace(/\s+/g, '_').replace(/[^\w\-]/g, '');

function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// ===================================================================
//  CAMERA + DETECTOR
// ===================================================================

const video = document.createElement('video');
video.playsInline = true; video.muted = true;
const src = document.createElement('canvas');      // (mirrored) camera frame fed to the model
const srcCtx = src.getContext('2d');
const out = $('out');                               // what the user sees / what gets recorded
const outCtx = out.getContext('2d');
const skel = document.createElement('canvas');      // skeleton-only video
const skelCtx = skel.getContext('2d');

let detector = null;
let stream = null;
let running = false;

const CAMERA_KEY = 'poseTrainer.camera';

async function startCamera(deviceId) {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  const base = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' };
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { ...base, width: { ideal: CAPTURE_W }, height: { ideal: CAPTURE_H } } });
  } catch (e) {
    // Some webcams refuse the requested resolution; retry with the camera's own default.
    if (e.name !== 'NotReadableError' && e.name !== 'OverconstrainedError' && e.name !== 'AbortError') throw e;
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: base });
  }
  try { if (deviceId) localStorage.setItem(CAMERA_KEY, deviceId); } catch (e) { /* storage blocked */ }
  video.srcObject = stream;
  await video.play();
  const w = video.videoWidth || CAPTURE_W, h = video.videoHeight || CAPTURE_H;
  for (const c of [src, out, skel]) { c.width = w; c.height = h; }
  await listCameras(stream.getVideoTracks()[0].getSettings().deviceId);
}

async function listCameras(currentId) {
  const sel = $('camSel');
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    sel.innerHTML = '';
    devs.forEach((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
      if (d.deviceId === currentId) o.selected = true;
      sel.appendChild(o);
    });
    sel.classList.toggle('hidden', devs.length < 2);
  } catch (e) {
    sel.classList.add('hidden');
  }
}

async function loadDetector() {
  const pd = window.poseDetection;
  await tf.setBackend('webgl');
  await tf.ready();
  detector = await pd.createDetector(pd.SupportedModels.MoveNet, {
    modelType: pd.movenet.modelType.SINGLEPOSE_LIGHTNING,
    modelUrl: new URL('model/model.json', location.href).href,
    enableSmoothing: false,   // we use our own EMA (FIX-13) so training == live
  });
}

// ===================================================================
//  APP STATE
// ===================================================================

const S = {
  mode: 'idle',           // idle | rec_countdown | recording | train_countdown | train_capture
  modeStart: 0,
  trainName: '',
  trainFrames: [],
  lastSnapTs: 0,
  coachTarget: null,
  coachTips: [],
  coachTipsTs: 0,
  poseHistory: [],
  fps: 0,
  // recording
  recorders: [],
  recChunks: { main: [], skel: [] },
  dataLines: [],
  frameId: 0,
  lastRecording: null,    // {t, main: Blob, skel: Blob, data: Blob, ext}
};

let latest = { angles: {}, extras: {}, kps: null, refAngles: null };

// ===================================================================
//  MAIN LOOP (camera_loop)
// ===================================================================

let fpsCount = 0, fpsStart = performance.now();

async function tick() {
  if (!running) return;
  if (video.readyState < 2) { requestAnimationFrame(tick); return; }

  const w = src.width, h = src.height, aspect = w / h;
  const mirror = $('mirrorChk').checked;

  srcCtx.save();
  if (mirror) { srcCtx.translate(w, 0); srcCtx.scale(-1, 1); }
  srcCtx.drawImage(video, 0, 0, w, h);
  srcCtx.restore();

  let raw = null;
  try {
    const poses = await detector.estimatePoses(src, { maxPoses: 1, flipHorizontal: false });
    const p = poses && poses[0];
    if (p && (p.score === undefined || p.score >= PERSON_CONF_THRESH)) {
      raw = p.keypoints.map(k => ({ x: k.x / w, y: k.y / h, conf: k.score ?? 0 }));
      const inBounds = raw.filter(k => k.x >= 0 && k.x <= 1 && k.y >= 0 && k.y <= 1 &&
                                       k.conf >= KP_CONF_THRESH).length;
      if (inBounds < MIN_VISIBLE_KPS) raw = null;
    }
  } catch (e) {
    console.error(e);
  }

  const kps = smoothKeypoints(raw);
  const poseDetected = kps !== null;
  let angles = {}, extras = {}, vector = null;
  if (poseDetected) {
    angles = extractAngles(kps, aspect);
    extras = extractExtras(kps, angles, aspect);
    vector = buildFeatureVector(angles, extras);
  }

  outCtx.drawImage(src, 0, 0);
  const now = performance.now() / 1000;

  let refForDraw = null, refForSidebar = null, detLabel = null, detClass = 'nodetect';
  let accuracy = null, tips = [], infoText = null, infoColor = '#00b4ff';

  // ---- coaching ----
  const coach = S.coachTarget;
  if (coach && poseDetected && Object.keys(angles).length) {
    const refVec = meanVector(coach);
    if (refVec) {
      const refAng = Object.fromEntries(ALL_FEATURE_KEYS.map((k, i) => [k, refVec[i]]));
      refForSidebar = Object.fromEntries(FEATURE_ANGLE_KEYS.map(k => [k, refAng[k]]));
      refForDraw = { ...refForSidebar };
      FEATURE_ANGLE_KEYS.forEach(k => { refForDraw['_live_' + k] = angles[k] ?? 0; });
      detLabel = `Coaching: ${prettyName(coach)}`; detClass = 'detect';
      if (now - S.coachTipsTs >= COACH_TIP_INTERVAL) {
        S.coachTips = computeCoachingTips(angles, refAng);
        S.coachTipsTs = now;
      }
      tips = S.coachTips;
    }
  }

  const nAngles = Object.values(angles).filter(v => v !== 0).length;
  const names = Object.keys(trainedModel);

  // ---- detection ----
  if (names.length && poseDetected && vector && !coach) {
    const ranked = names.map(s => [normalizedVectorDistance(vector, trainedModel[s]), s])
                        .sort((a, b) => a[0] - b[0]);
    let [bestD, bestS] = ranked[0];
    const secondD = ranked.length > 1 ? ranked[1][0] : Infinity;
    const conf = bestD === Infinity ? 0 : Math.max(0, Math.min(1, 1 - bestD / MATCH_THRESHOLD));

    if (bestD === Infinity) {
      infoText = `Too few joints visible (${nAngles}/12). Step back.`; infoColor = '#ff5000';
    } else {
      infoText = `dist=${bestD.toFixed(2)} conf=${Math.round(conf * 100)}% ` +
                 `(need >${DISPLAY_CONF_THRESH * 100}%)  joints=${nAngles}/12`;
      infoColor = conf > DISPLAY_CONF_THRESH ? '#64dc00' : '#ffb400';
    }

    const marginOk = secondD === Infinity || bestD <= secondD * MATCH_MARGIN;
    if (bestD < MATCH_THRESHOLD && marginOk && nAngles >= 6 && conf > DISPLAY_CONF_THRESH) {
      const stable = smoothPoseLabel(S.poseHistory, bestS);
      if (stable) {
        bestS = stable;
        detLabel = `${prettyName(bestS)} (${Math.round(conf * 100)}%)`; detClass = 'detect';
        const refVec = trainedModel[bestS];
        refForSidebar = Object.fromEntries(FEATURE_ANGLE_KEYS.map(k => [k, refVec[ALL_FEATURE_KEYS.indexOf(k)]]));
        refForDraw = { ...refForSidebar };
        FEATURE_ANGLE_KEYS.forEach(k => { refForDraw['_live_' + k] = angles[k] ?? 0; });
      } else {
        detLabel = `Stabilising: ${prettyName(bestS)}…`; detClass = 'detect';
      }
    } else {
      smoothPoseLabel(S.poseHistory, null);
      detLabel = 'Unknown Pose';
    }
  } else if (!poseDetected) {
    S.poseHistory = [];
    kpCache = {};
  }

  if (poseDetected) {
    const [g, t] = drawSkeleton(outCtx, kps, w, h, refForDraw);
    if (refForDraw && t > 0) accuracy = Math.round(100 * g / t);
  }

  if (infoText && !coach) outlinedText(outCtx, infoText, 10, h - 12, Math.max(12, Math.round(w / 52)), infoColor);
  if (coach) {
    if (!poseDetected) drawCoachingOverlay(outCtx, ['Step into frame so I can see you.'], coach, null, w, h);
    else drawCoachingOverlay(outCtx, tips.length ? tips : ['Good form! Hold it.'], coach, accuracy, w, h);
  }

  latest = { angles, extras, kps, refAngles: refForSidebar, poseDetected, detLabel, detClass, accuracy };

  handleModes(now, w, h, poseDetected, kps, angles, extras, vector, refForDraw);

  // ---- fps ----
  fpsCount++;
  const pnow = performance.now();
  if (pnow - fpsStart >= 1000) {
    S.fps = fpsCount * 1000 / (pnow - fpsStart);
    fpsCount = 0; fpsStart = pnow;
  }

  requestAnimationFrame(tick);
}

function handleModes(now, w, h, poseDetected, kps, angles, extras, vector, refForDraw) {
  // ---- recording ----
  if (S.mode === 'rec_countdown') {
    const rem = Math.ceil(REC_COUNTDOWN_SECS - (now - S.modeStart));
    if (rem > 0) drawCountdown(outCtx, `Recording in ${rem}…`, '#00dcdc', w, h);
    else beginRecording();
  }
  if (S.mode === 'recording') {
    if (Math.floor(now * 2) % 2 === 0) {
      outCtx.fillStyle = '#dc0000';
      outCtx.beginPath(); outCtx.arc(22, 26, 8, 0, Math.PI * 2); outCtx.fill();
    }
    outlinedText(outCtx, 'REC', 36, 34, 22, '#ff3030');
    skelCtx.fillStyle = '#000'; skelCtx.fillRect(0, 0, w, h);
    if (poseDetected) {
      drawSkeleton(skelCtx, kps, w, h, refForDraw);
      if (Object.keys(angles).length) {
        S.dataLines.push(`${S.frameId}: ` + kps.map(k => `${k.x.toFixed(4)},${k.y.toFixed(4)}`).join(' '));
        S.frameId++;
      }
    }
  }

  // ---- training ----
  if (S.mode === 'train_countdown') {
    const rem = Math.ceil(TRAIN_COUNTDOWN_SECS - (now - S.modeStart));
    if (rem > 0) drawCountdown(outCtx, `Training '${prettyName(S.trainName)}' in ${rem}…`, '#ffc400', w, h);
    else { S.mode = 'train_capture'; S.modeStart = now; S.trainFrames = []; S.lastSnapTs = 0; }
  }
  if (S.mode === 'train_capture') {
    let taken = S.trainFrames.length;
    const bw = (taken / TRAIN_SNAPSHOTS) * (w - 40);
    outCtx.fillStyle = '#323232'; outCtx.fillRect(20, h - 30, w - 40, 15);
    outCtx.fillStyle = '#00c864'; outCtx.fillRect(20, h - 30, Math.min(bw, w - 40), 15);
    outlinedText(outCtx, `CAPTURING '${prettyName(S.trainName)}': photo ${taken}/${TRAIN_SNAPSHOTS}`,
                 20, h - 40, Math.max(13, Math.round(w / 40)), '#00dc78');
    if (now - S.lastSnapTs < 0.1) {
      outCtx.fillStyle = 'rgba(255,255,255,.25)'; outCtx.fillRect(0, 0, w, h);
    }
    if (poseDetected && vector && taken < TRAIN_SNAPSHOTS && now - S.lastSnapTs >= TRAIN_SNAP_INTERVAL) {
      S.trainFrames.push({ vector, kps: kps.map(k => [round(k.x, 4), round(k.y, 4), round(k.conf, 3)]) });
      S.lastSnapTs = now;
      taken++;
    }
    if (taken >= TRAIN_SNAPSHOTS) finishTraining();
    else if (now - S.modeStart > 30 && taken === 0) {
      S.mode = 'idle';
      toast(`No pose detected while training '${prettyName(S.trainName)}'. Nothing saved.`, 5000);
    }
  }
}

function finishTraining() {
  const name = S.trainName, frames = S.trainFrames;
  S.mode = 'idle'; S.trainFrames = [];
  if (!frames.length) return;
  const existing = library[name];
  const mid = frames[Math.floor(frames.length / 2)].kps;
  library[name] = {
    vectors: [...(existing ? existing.vectors : []), ...frames.map(f => f.vector)],
    active: existing ? existing.active : true,
    preview: mid.map(([x, y, c]) => [x, y, c]),
  };
  saveLibrary();
  buildModel();
  toast(`Saved ${frames.length} photos for '${prettyName(name)}'.\nModel rebuilt: ` +
        `${library[name].vectors.length} frames for this pose.`, 5000);
}

// ===================================================================
//  RECORDING (MediaRecorder replaces cv2.VideoWriter)
// ===================================================================

function pickMime() {
  const opts = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return opts.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
}

function beginRecording() {
  const mime = pickMime();
  if (!window.MediaRecorder) { S.mode = 'idle'; toast('Recording is not supported in this browser.'); return; }
  S.recChunks = { main: [], skel: [] };
  S.dataLines = []; S.frameId = 0;
  S.recorders = [['main', out], ['skel', skel]].map(([key, canvas]) => {
    const r = new MediaRecorder(canvas.captureStream(20), mime ? { mimeType: mime } : undefined);
    r.ondataavailable = e => { if (e.data.size) S.recChunks[key].push(e.data); };
    r.start(1000);
    return r;
  });
  S.recMime = mime || 'video/webm';
  S.recT = Math.floor(Date.now() / 1000);
  S.mode = 'recording';
  S.modeStart = performance.now() / 1000;
}

function stopRecording() {
  const recs = S.recorders; S.recorders = [];
  if (!recs.length) return;
  let pending = recs.length;
  const done = () => {
    if (--pending) return;
    const type = S.recMime.split(';')[0];
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    S.lastRecording = {
      t: S.recT, ext,
      main: new Blob(S.recChunks.main, { type }),
      skel: new Blob(S.recChunks.skel, { type }),
      data: new Blob([S.dataLines.join('\n') + '\n'], { type: 'text/plain' }),
    };
    $('dlBtn').disabled = false;
    toast('Recording saved. Press Download to get the files.', 4000);
  };
  recs.forEach(r => { r.onstop = done; try { r.stop(); } catch (e) { done(); } });
}

// ===================================================================
//  BUTTONS
// ===================================================================

$('recBtn').onclick = () => {
  if (S.mode === 'recording' || S.mode === 'rec_countdown') return toast('Already recording.', 3000);
  if (S.mode.startsWith('train')) return toast('Training active. Stop first.', 3000);
  S.mode = 'rec_countdown'; S.modeStart = performance.now() / 1000;
};

$('stopBtn').onclick = () => {
  const was = S.mode;
  if (was === 'recording') stopRecording();
  S.mode = 'idle'; S.trainFrames = [];
  toast(was === 'idle' ? 'Nothing running.' : 'Stopped.', 2000);
};

$('dlBtn').onclick = () => {
  const r = S.lastRecording;
  if (!r) return;
  downloadBlob(r.main, `video_${r.t}.${r.ext}`);
  setTimeout(() => downloadBlob(r.skel, `skeleton_${r.t}.${r.ext}`), 400);
  setTimeout(() => downloadBlob(r.data, `data_${r.t}.txt`), 800);
};

$('trainBtn').onclick = () => {
  if (S.mode === 'recording' || S.mode === 'rec_countdown') return toast('Recording active. Stop first.', 3000);
  if (S.mode.startsWith('train')) return toast('Already training.', 3000);
  $('trainName').value = '';
  openOv('trainOv');
  setTimeout(() => $('trainName').focus(), 50);
};

function goTrain() {
  const name = cleanName($('trainName').value);
  if (!name) { toast('Enter a name (letters, numbers, _ or -).', 3000); return; }
  closeOv('trainOv');
  S.trainName = name; S.trainFrames = [];
  S.mode = 'train_countdown'; S.modeStart = performance.now() / 1000;
  toast(`Capturing '${prettyName(name)}'. Get into position (~${TRAIN_COUNTDOWN_SECS + 5}s).`, 5000);
}
$('trainGo').onclick = goTrain;
$('trainName').addEventListener('keydown', e => { if (e.key === 'Enter') goTrain(); });

// ---- coaching ----
$('coachBtn').onclick = () => {
  const list = $('coachList');
  list.innerHTML = '';
  const names = Object.keys(library).sort();
  if (!names.length) list.innerHTML = '<div class="empty">No poses yet. Use Train Pose first.</div>';
  names.forEach(s => {
    const b = document.createElement('button');
    b.className = 'coach-pick' + (s === S.coachTarget ? ' on' : '');
    b.textContent = prettyName(s);
    b.onclick = () => selectCoach(s);
    list.appendChild(b);
  });
  $('coachDesc').textContent = S.coachTarget ? coachDescription(S.coachTarget) : '';
  openOv('coachOv');
};

function coachDescription(name) {
  const n = library[name] ? library[name].vectors.length : 0;
  return `Comparing your live pose against '${prettyName(name)}'. Trained on ${n} frame(s). ` +
         `Joint corrections will appear on screen.`;
}

function selectCoach(name) {
  S.coachTarget = name; S.coachTips = []; S.coachTipsTs = 0;
  $('coachBtn').classList.add('on');
  $('coachDesc').textContent = coachDescription(name);
  document.querySelectorAll('#coachList .coach-pick').forEach(b =>
    b.classList.toggle('on', b.textContent === prettyName(name)));
  toast(`Coaching: ${prettyName(name)}`, 2500);
}

$('coachStop').onclick = () => {
  S.coachTarget = null;
  $('coachBtn').classList.remove('on');
  closeOv('coachOv');
  toast('Coaching mode off.', 2000);
};

// ---- library ----
$('libBtn').onclick = () => { renderLibrary(); openOv('libOv'); };

function renderLibrary() {
  const list = $('libList');
  list.innerHTML = '';
  const names = Object.keys(library).sort();
  if (!names.length) {
    list.innerHTML = '<div class="empty">No poses yet. Press Train Pose to record one.</div>';
    return;
  }
  for (const name of names) {
    const d = library[name];
    const item = document.createElement('div');
    item.className = 'item';

    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = d.active; cb.title = 'Use for live detection';
    cb.onchange = () => { d.active = cb.checked; saveLibrary(); buildModel(); };

    const cv = document.createElement('canvas');
    cv.width = 44; cv.height = 52;
    if (d.preview) {
      drawMiniSkeleton(cv, d.preview.map(([x, y, c]) => ({ x, y, conf: c })), null, null, '');
    } else {
      drawMiniSkeleton(cv, null, null, null, '');
    }

    const meta = document.createElement('div');
    meta.className = 'meta';
    const nm = document.createElement('div'); nm.className = 'nm'; nm.textContent = prettyName(name);
    const ct = document.createElement('div'); ct.className = 'ct';
    ct.textContent = `${d.vectors.length} frame(s)` + (d.active ? '' : ' · disabled');
    meta.append(nm, ct);

    const acts = document.createElement('div');
    acts.className = 'acts';
    const ren = document.createElement('button');
    ren.className = 'b bk'; ren.textContent = 'Rename';
    ren.onclick = () => renamePose(name);
    const del = document.createElement('button');
    del.className = 'b br'; del.textContent = 'Delete';
    del.onclick = () => {
      if (!confirm(`Delete '${prettyName(name)}' and all ${d.vectors.length} frames?`)) return;
      delete library[name];
      if (S.coachTarget === name) { S.coachTarget = null; $('coachBtn').classList.remove('on'); }
      saveLibrary(); buildModel(); renderLibrary();
    };
    acts.append(ren, del);
    item.append(cb, cv, meta, acts);
    list.appendChild(item);
  }
}

function renamePose(name) {
  const input = prompt('New name for this pose:', name);
  if (input === null) return;
  const nn = cleanName(input);
  if (!nn || nn === name) return;
  if (library[nn]) {
    if (!confirm(`'${prettyName(nn)}' already exists. Merge '${prettyName(name)}' into it?`)) return;
    library[nn].vectors.push(...library[name].vectors);
    library[nn].preview = library[nn].preview || library[name].preview;
  } else {
    library[nn] = library[name];
  }
  delete library[name];
  if (S.coachTarget === name) S.coachTarget = nn;
  saveLibrary(); buildModel(); renderLibrary();
}

$('exportBtn').onclick = () => {
  const blob = new Blob([JSON.stringify(library)], { type: 'application/json' });
  downloadBlob(blob, `pose_library_${Math.floor(Date.now() / 1000)}.json`);
};

$('importFile').onchange = async e => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const incoming = normalizeLibrary(JSON.parse(await file.text()));
    const names = Object.keys(incoming);
    if (!names.length) return toast('That file has no poses in it.', 3000);
    for (const n of names) {
      if (library[n]) library[n].vectors.push(...incoming[n].vectors);
      else library[n] = incoming[n];
    }
    saveLibrary(); buildModel(); renderLibrary();
    toast(`Imported ${names.length} pose(s).`, 3000);
  } catch (err) {
    toast('Could not read that file. Is it an exported pose library?', 4000);
  }
};

$('resetBtn').onclick = async () => {
  if (!confirm('Replace your pose library with the default poses? Your trained poses will be lost unless exported.')) return;
  library = await loadDefaultLibrary();
  S.coachTarget = null; $('coachBtn').classList.remove('on');
  saveLibrary(); buildModel(); renderLibrary();
};

$('camSel').onchange = async e => {
  try { await startCamera(e.target.value); } catch (err) { toast(cameraErrorText(err), 7000); }
};

// ===================================================================
//  SIDEBAR + STATUS (replaces /status and /analysis polling)
// ===================================================================

const ANGLE_ROWS = FEATURE_ANGLE_KEYS;
const EXTRA_ROWS = [['L_arm_ext','L Arm Ext',false],['R_arm_ext','R Arm Ext',false],
  ['L_leg_ext','L Leg Ext',false],['R_leg_ext','R Leg Ext',false],
  ['L_forearm_ext','L Fore',false],['R_forearm_ext','R Fore',false],
  ['torso_lean','Torso Lean',true],['lr_symmetry','Symmetry',false],
  ['shoulder_hip_ratio','Sh/Hip',false]];

function buildRows(container, items) {
  container.innerHTML = '';
  const cells = {};
  for (const [key, label] of items) {
    const row = document.createElement('div'); row.className = 'row';
    const l = document.createElement('span'); l.className = 'label'; l.textContent = label;
    const v = document.createElement('span'); v.className = 'val'; v.textContent = '--';
    row.append(l, v); container.appendChild(row);
    cells[key] = v;
  }
  return cells;
}
const angleCells = buildRows($('angles'), ANGLE_ROWS.map(k => [k, k.replace(/_/g, ' ')]));
const extraCells = buildRows($('extras'), EXTRA_ROWS.map(([k, l]) => [k, l]));

function updateSidebar() {
  const a = latest;
  const det = $('detection');
  if (a.detLabel) { det.textContent = a.detLabel; det.className = a.detClass; }
  else {
    det.textContent = a.poseDetected ? (Object.keys(trainedModel).length ? 'Pose found' : 'Pose found (no poses trained)') : 'No pose detected';
    det.className = a.poseDetected ? 'detect' : 'nodetect';
  }
  $('accuracy').textContent = a.accuracy !== null && a.accuracy !== undefined ? `Accuracy: ${a.accuracy}%` : '';
  $('pose-status').textContent = a.poseDetected ? 'YES' : 'no';
  $('fps').textContent = S.fps.toFixed(1);
  for (const k of ANGLE_ROWS) {
    const v = a.angles[k];
    angleCells[k].textContent = v !== undefined ? Math.round(v) + '°' : '--';
  }
  for (const [k, , signed] of EXTRA_ROWS) {
    const v = a.extras[k];
    extraCells[k].textContent = v === undefined ? '--' : signed ? (v > 0 ? '+' : '') + v.toFixed(1) : v.toFixed(2);
  }
  drawMiniSkeleton($('skelCanvas'), a.kps, a.refAngles, a.refAngles ? a.angles : {}, 'No pose detected');

  const b = $('badge');
  b.className = 'badge';
  if (S.mode === 'recording') {
    const dur = Math.floor(performance.now() / 1000 - S.modeStart);
    b.textContent = `● REC ${dur}s`; b.classList.add('badge-rec', 'pulse');
  } else if (S.mode === 'train_capture') {
    b.textContent = 'TRAINING'; b.classList.add('badge-train', 'pulse');
  } else if (S.mode.endsWith('countdown')) {
    b.textContent = 'COUNTDOWN'; b.classList.add('badge-countdown', 'pulse');
  } else {
    b.textContent = `IDLE  ${S.fps.toFixed(1)} fps`; b.classList.add('badge-idle');
  }
}

// ===================================================================
//  STARTUP
// ===================================================================

function cameraErrorText(e) {
  switch (e && e.name) {
    case 'NotAllowedError':
      return 'Camera permission was denied. Click the camera icon in the address bar, allow access, then try again. ' +
             'On Windows also check Settings → Privacy & security → Camera → "Let desktop apps access your camera".';
    case 'NotReadableError':
    case 'AbortError':
      return 'That camera could not be opened. It is probably in use by another app (Teams, Zoom, Discord, the Camera app ' +
             'or another browser tab). Close those and try again, or pick a different camera above.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'That camera was not found. Check it is plugged in, or pick a different camera.';
    default:
      return `Could not open the camera (${(e && (e.name || e.message)) || 'unknown error'}).`;
  }
}

async function videoInputs() {
  return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
}

// Step 1: "Start camera" asks for permission (so camera names are visible) and lists the cameras.
$('startBtn').onclick = async () => {
  const btn = $('startBtn'), err = $('menuErr');
  err.textContent = '';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    err.textContent = 'This browser cannot access a camera. The page must be opened over https:// (or localhost).';
    return;
  }
  btn.disabled = true; btn.textContent = 'Finding cameras…';
  let probeErr = null;
  try {
    let devs = await videoInputs();
    if (!devs.some(d => d.label)) {
      // Camera names stay hidden until permission is granted, so ask once and release immediately.
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        s.getTracks().forEach(t => t.stop());
      } catch (e) {
        probeErr = e;
        if (e.name === 'NotAllowedError' || e.name === 'SecurityError') throw e;
      }
      devs = await videoInputs();
    }
    if (!devs.length) throw Object.assign(new Error('No camera found'), { name: 'NotFoundError' });

    let saved = null;
    try { saved = localStorage.getItem(CAMERA_KEY); } catch (e) { /* storage blocked */ }
    const sel = $('menuCamSel');
    sel.innerHTML = '';
    devs.forEach((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
      if (d.deviceId === saved) o.selected = true;
      sel.appendChild(o);
    });
    btn.classList.add('hidden');
    $('camPick').classList.remove('hidden');
    if (probeErr) err.textContent = 'Your default camera could not be opened. Pick a camera from the list.';
  } catch (e) {
    btn.disabled = false; btn.textContent = 'Start camera';
    err.textContent = cameraErrorText(e);
  }
};

// Step 2: open the chosen camera and start the app.
$('useCamBtn').onclick = async () => {
  const btn = $('useCamBtn'), err = $('menuErr');
  err.textContent = '';
  btn.disabled = true; btn.textContent = 'Starting camera…';
  try {
    await startCamera($('menuCamSel').value || undefined);
  } catch (e) {
    btn.disabled = false; btn.textContent = 'Use this camera';
    err.textContent = cameraErrorText(e);
    return;
  }
  $('menu').classList.add('hidden');
  $('app').classList.remove('hidden');
  try {
    await loadLibrary();
    await loadDetector();
  } catch (e) {
    console.error(e);
    $('loading').textContent = 'Could not load the pose model. Check your connection and reload.';
    return;
  }
  $('loading').classList.add('hidden');
  running = true;
  requestAnimationFrame(tick);
  setInterval(updateSidebar, 250);
};
