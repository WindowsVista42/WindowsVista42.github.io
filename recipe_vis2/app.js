import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DRACOLoader }   from 'three/addons/loaders/DRACOLoader.js';

// ── Constants ────────────────────────────────────────────────────────────────
const DATA       = 'data/';
const LEFT_W     = 280;
const RIGHT_W    = 270;
const TOPBAR_H   = 44;
const CAT_TEX_W  = 4096;   // texture width for category data buffer
const MAX_LABELS = 64;     // max labels per category family for uniform arrays
const DRAG_THRESH  = 5;
const DBL_CLICK_MS = 350;

// ── Vertex shader (GLSL3 — uses gl_VertexID + texelFetch) ───────────────────
const VERT = /* glsl */`
  out vec3  vColor;
  out float vAlpha;

  uniform float uPointSize;

  uniform highp usampler2D uCategoryTex;
  uniform int  uActiveFamilyIdx;
  uniform int  uN;

  uniform vec3 uPalette[${MAX_LABELS * 3}];
  uniform int  uPaletteN;
  uniform int  uCategoryModes[${MAX_LABELS}];

  void main() {
    int flatIdx = uActiveFamilyIdx * uN + gl_VertexID;
    int tx = flatIdx % ${CAT_TEX_W};
    int ty = flatIdx / ${CAT_TEX_W};
    int catId = int(texelFetch(uCategoryTex, ivec2(tx, ty), 0).r);

    int mode = uCategoryModes[catId];

    // mode 0 = normal, 1 = light, 2 = dark/desaturated — never hidden
    vAlpha = 1.0;
    int paletteIdx = mode * uPaletteN + catId % uPaletteN;
    vColor = uPalette[paletteIdx];

    vec4 mvPos    = modelViewMatrix * vec4(position, 1.0);
    gl_Position   = projectionMatrix * mvPos;
    float depth   = max(-mvPos.z, 0.001);
    gl_PointSize  = (uPointSize * 5.0) / depth;
  }
`;

// ── Fragment shader ──────────────────────────────────────────────────────────
const FRAG = /* glsl */`
  in  vec3  vColor;
  in  float vAlpha;
  out vec4  fragColor;

  uniform float uOutline;

  void main() {
    if (vAlpha < 0.5) discard;
    vec2  uv   = gl_PointCoord - vec2(0.5);
    float dist = length(uv);
    if (dist > 0.5) discard;
    vec3 col = vColor;
    if (uOutline > 0.5 && dist > 0.38) {
      float t = smoothstep(0.38, 0.48, dist);
      col = mix(col, vec3(0.0), t * 0.85);
    }
    fragColor = vec4(col, 1.0);
  }
`;

// ── State ────────────────────────────────────────────────────────────────────
let N = 0;
let meta = null;
let posArr = null;          // Float32Array(N*3) — positions for raycasting
let recipeIds = null;       // Uint32Array(N)
let chunkIds  = null;       // Uint16Array(N)
let categoryData = null;    // Uint32Array(N * N_families) — packed category IDs
let alphaCache = null;      // Float32Array(N) — 0=hidden, 1=visible (for raycasting)

const chunkCache   = new Map(); // chunkId -> {recipe_id_str: {...}}
const pendingChunks = new Map(); // chunkId -> Promise<data> (in-flight fetches)

let renderer, scene, camera, controls;
let geometry, pointsMesh;
let uniforms = {};
let catTexture = null;
let activeFamilyIdx = 0;
let categoryModes = new Int32Array(MAX_LABELS).fill(0);
let highlightedLabelIdx = -1; // -1 = none

let lockedIdx  = -1;
let hoverIdx   = -1;   // currently hovered point index
let pointerIsDown = false;
let mouseDownX = 0, mouseDownY = 0;
let lastClickTime = 0;
let camAnim = null;

// Computed from coord_bounds after meta loads — used by camera presets
let dataCenterVec = new THREE.Vector3(0, 0, 0);
let defaultCamPos = new THREE.Vector3(0.8, 0.6, 2.0);
let dataExtent    = 14;  // approximate, updated on boot

let appMode = 'story';  // 'story' | 'explore'
let storyData = null;
let currentStep = 0;

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();

// ── Color helpers ─────────────────────────────────────────────────────────────
function hexToVec3(hex) {
  return [
    parseInt(hex.slice(1,3), 16) / 255,
    parseInt(hex.slice(3,5), 16) / 255,
    parseInt(hex.slice(5,7), 16) / 255,
  ];
}

function lighten(rgb) {
  return [
    rgb[0] * 0.6 + 0.4,
    rgb[1] * 0.6 + 0.4,
    rgb[2] * 0.6 + 0.4,
  ];
}

function darken(rgb) {
  return [rgb[0] * 0.12 + 0.03, rgb[1] * 0.12 + 0.03, rgb[2] * 0.12 + 0.03];
}

function buildPalette(hexColors) {
  const n = hexColors.length;
  // flat array: [normal×n, light×n, dark×n]
  const flat = new Float32Array(n * 3 * 3);
  hexColors.forEach((hex, i) => {
    const base  = hexToVec3(hex);
    const light = lighten(base);
    const dark  = darken(base);
    flat.set(base,  i * 3);
    flat.set(light, (n + i) * 3);
    flat.set(dark,  (n * 2 + i) * 3);
  });
  return { flat, n };
}

// ── Category texture ──────────────────────────────────────────────────────────
function buildCategoryTexture(families) {
  const nFamilies = families.length;
  const totalEls  = N * nFamilies;
  const texH      = Math.ceil(totalEls / CAT_TEX_W);
  const texData   = new Uint32Array(CAT_TEX_W * texH); // zero-padded

  families.forEach((arr, fi) => {
    for (let i = 0; i < N; i++) {
      texData[fi * N + i] = arr[i];
    }
  });

  const tex = new THREE.DataTexture(texData, CAT_TEX_W, texH,
    THREE.RedIntegerFormat, THREE.UnsignedIntType);
  tex.internalFormat = 'R32UI';
  tex.needsUpdate    = true;
  return tex;
}

// ── Palette color lookup ──────────────────────────────────────────────────────
function getPaletteRgb(labelIdx) {
  const palN    = uniforms.uPaletteN?.value ?? 1;
  const palette = uniforms.uPalette?.value;
  if (!palette) return [78, 121, 167];
  const base = (labelIdx % palN) * 3;
  return [
    Math.round(palette[base]     * 255),
    Math.round(palette[base + 1] * 255),
    Math.round(palette[base + 2] * 255),
  ];
}

// ── Decompress helpers ────────────────────────────────────────────────────────
async function decompressBuffer(ab) {
  const stream = new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

async function decompressJson(ab) {
  const stream = new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}

// ── Chunk loading ─────────────────────────────────────────────────────────────
function pad(i) { return String(i).padStart(6, '0'); }

function loadChunk(chunkId) {
  if (chunkCache.has(chunkId))   return Promise.resolve(chunkCache.get(chunkId));
  if (pendingChunks.has(chunkId)) return pendingChunks.get(chunkId);

  const promise = (async () => {
    try {
      const resp = await fetch(`${DATA}chunks/chunk_${pad(chunkId)}.json.gz`);
      if (!resp.ok) return null;
      const data = await resp.arrayBuffer().then(decompressJson);
      chunkCache.set(chunkId, data);
      return data;
    } catch { return null; }
    finally   { pendingChunks.delete(chunkId); }
  })();

  pendingChunks.set(chunkId, promise);
  return promise;
}

async function getRecipeData(idx) {
  const chunk = await loadChunk(chunkIds[idx]);
  return chunk?.[String(recipeIds[idx])] ?? null;
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function initScene(palette) {
  const canvas = document.getElementById('three-canvas');
  const w = window.innerWidth - LEFT_W - RIGHT_W;
  const h = window.innerHeight - TOPBAR_H;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, w / h, 0.001, 1000);
  camera.position.copy(defaultCamPos);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance   = 0.05;
  controls.maxDistance   = dataExtent * 6;
  controls.target.copy(dataCenterVec);
  controls.update();

  geometry = new THREE.BufferGeometry();
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0,0,0), 1000);

  const posAttr = new THREE.BufferAttribute(posArr, 3);
  posAttr.usage = THREE.StaticDrawUsage;
  geometry.setAttribute('position', posAttr);
  geometry.setDrawRange(0, N);

  // Category texture
  const families = meta.categories.map(cat => {
    const raw = categoryData.subarray(
      meta.categories.indexOf(cat) * N,
      (meta.categories.indexOf(cat) + 1) * N
    );
    const u32 = new Uint32Array(N);
    for (let i = 0; i < N; i++) u32[i] = raw[i];
    return u32;
  });
  catTexture = buildCategoryTexture(families);

  uniforms = {
    uPointSize:      { value: 4.0 },
    uOutline:        { value: 1.0 },
    uCategoryTex:    { value: catTexture },
    uActiveFamilyIdx:{ value: 0 },
    uN:              { value: N },
    uPalette:        { value: palette.flat },
    uPaletteN:       { value: palette.n },
    uCategoryModes:  { value: Array.from(categoryModes) },
  };

  const mat = new THREE.ShaderMaterial({
    glslVersion:    THREE.GLSL3,
    vertexShader:   VERT,
    fragmentShader: FRAG,
    uniforms,
    depthTest:  true,
    depthWrite: true,
  });

  pointsMesh = new THREE.Points(geometry, mat);
  scene.add(pointsMesh);

  // Initial alpha cache (all visible)
  alphaCache = new Float32Array(N).fill(1.0);

  // UI wiring
  document.getElementById('fc-reset').addEventListener('click', () => setCameraPreset('reset'));
  document.getElementById('fc-top').addEventListener('click',   () => setCameraPreset('top'));
  document.getElementById('fc-front').addEventListener('click', () => setCameraPreset('front'));
  document.getElementById('fc-side').addEventListener('click',  () => setCameraPreset('side'));

  const sizeSlider = document.getElementById('fc-size');
  sizeSlider.addEventListener('input', () => {
    uniforms.uPointSize.value = parseFloat(sizeSlider.value);
    document.getElementById('fc-size-val').textContent = sizeSlider.value + '×';
  });

  const outlineBtn = document.getElementById('fc-outline');
  outlineBtn.addEventListener('click', () => {
    uniforms.uOutline.value = outlineBtn.classList.toggle('active') ? 1.0 : 0.0;
  });

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('pointerdown',  onPointerDown);
  renderer.domElement.addEventListener('pointermove',  onPointerMove);
  renderer.domElement.addEventListener('pointerup',    onPointerUp);
  renderer.domElement.addEventListener('pointerleave', onPointerLeave);

  animate();
}

// ── Category mode management ──────────────────────────────────────────────────
function setHighlightLabel(labelIdx) {
  // -1 = clear all (all normal)
  // else: selected label stays mode=0 (normal color), others go to mode=2 (dark/desaturated)
  // Points are never hidden — mode=2 is dark but still visible and raycasted
  highlightedLabelIdx = labelIdx;
  const nLabels = meta.categories[activeFamilyIdx].labels.length;
  for (let i = 0; i < MAX_LABELS; i++) {
    if (i >= nLabels) { categoryModes[i] = 0; continue; }
    categoryModes[i] = (labelIdx < 0 || i === labelIdx) ? 0 : 2;
  }
  uniforms.uCategoryModes.value = Array.from(categoryModes);
  alphaCache.fill(1.0); // all points always raycasted
  renderCategoryList();
}

function setActiveFamily(idx) {
  activeFamilyIdx = idx;
  highlightedLabelIdx = -1;
  categoryModes.fill(0);
  uniforms.uActiveFamilyIdx.value = idx;
  uniforms.uCategoryModes.value   = Array.from(categoryModes);
  alphaCache.fill(1.0);
  renderCategoryList();
}

function getCategoryFamilyData(familyIdx) {
  // Returns Uint8/Uint32 view into categoryData for a given family
  const start = familyIdx * N;
  return categoryData.subarray(start, start + N);
}

// ── Progress ──────────────────────────────────────────────────────────────────
function showProgress(pct, label) {
  document.getElementById('progress-wrap').style.display  = 'block';
  document.getElementById('progress-label').style.display = 'block';
  document.getElementById('progress-bar').style.width     = `${pct}%`;
  document.getElementById('progress-label').textContent   = label;
}
function hideProgress() {
  document.getElementById('progress-wrap').style.display  = 'none';
  document.getElementById('progress-label').style.display = 'none';
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function worldToScreen(i) {
  const i3 = i * 3;
  const v  = new THREE.Vector3(posArr[i3], posArr[i3+1], posArr[i3+2]).project(camera);
  const w  = window.innerWidth - LEFT_W - RIGHT_W;
  const h  = window.innerHeight - TOPBAR_H;
  return { x: (v.x + 1) / 2 * w + LEFT_W, y: (-v.y + 1) / 2 * h + TOPBAR_H };
}

function positionHoverTip(idx) {
  const tip = document.getElementById('hover-tip');
  if (tip.style.display === 'none') return;
  const sp = worldToScreen(idx);
  const tw = tip.offsetWidth || 180;
  const th = tip.offsetHeight || 32;
  const canvasRight = window.innerWidth - RIGHT_W;
  const margin = 14;

  let left, arrowSide;
  if (sp.x + margin + tw + 16 <= canvasRight) {
    left = sp.x + margin; arrowSide = 'left';
  } else {
    left = sp.x - margin - tw; arrowSide = 'right';
  }
  left = Math.max(LEFT_W + 4, Math.min(left, canvasRight - tw - 4));

  let top = sp.y - th / 2;
  top = Math.max(TOPBAR_H + 4, Math.min(top, window.innerHeight - th - 4));

  tip.style.left = `${Math.round(left)}px`;
  tip.style.top  = `${Math.round(top)}px`;

  const arrow  = document.getElementById('hover-tip-arrow');
  const arrowY = Math.max(8, Math.min(sp.y - top - 7, th - 16));
  arrow.style.top = `${Math.round(arrowY)}px`;
  arrow.className = `arrow-${arrowSide}`;
  arrow.style.left = arrowSide === 'left' ? '-7px' : 'auto';
  arrow.style.right = arrowSide === 'right' ? '-7px' : 'auto';
}

function populateHoverTip(idx, recipe) {
  const tip = document.getElementById('hover-tip');
  const nameEl = tip.querySelector('.ht-name');
  const tagsEl = tip.querySelector('.ht-tags');
  const metaEl = tip.querySelector('.ht-meta');
  const descEl = tip.querySelector('.ht-desc');

  if (!recipe) {
    nameEl.innerHTML = '<span class="ht-loading">Loading…</span>';
    tagsEl.innerHTML = '';
    metaEl.textContent = '';
    descEl.textContent = '';
    return;
  }

  nameEl.textContent = recipe.name || `Recipe #${recipeIds[idx]}`;

  // All category pills in a single row, active family gets outline ring
  tagsEl.innerHTML = '';
  meta.categories.forEach((cat, fi) => {
    const famData  = getCategoryFamilyData(fi);
    const labelIdx = famData[idx];
    const label    = cat.labels[labelIdx];
    if (!label) return;
    const [r, g, b] = getPaletteRgb(labelIdx);
    const tr = Math.round(r * 0.45 + 255 * 0.55);
    const tg = Math.round(g * 0.45 + 255 * 0.55);
    const tb = Math.round(b * 0.45 + 255 * 0.55);
    const span = document.createElement('span');
    span.className = 'ht-tag' + (fi === activeFamilyIdx ? ' active-family' : '');
    span.textContent = label;
    span.style.background  = `rgba(${r},${g},${b},0.35)`;
    span.style.borderColor = `rgba(${r},${g},${b},0.80)`;
    span.style.color       = `rgb(${tr},${tg},${tb})`;
    tagsEl.appendChild(span);
  });

  // Meta line: date + cook time + steps
  const parts = [];
  if (recipe.submitted) parts.push(recipe.submitted.slice(0, 7)); // YYYY-MM
  if (recipe.minutes)   parts.push(`${recipe.minutes} min`);
  if (recipe.n_steps)   parts.push(`${recipe.n_steps} steps`);
  if (recipe.avg_rating != null) parts.push(`★ ${recipe.avg_rating.toFixed(1)}`);
  metaEl.textContent = parts.join(' · ');

  descEl.textContent = (recipe.description || '').trim().slice(0, 200);
}

function showHoverTip(idx) {
  hoverIdx = idx;
  const tip     = document.getElementById('hover-tip');
  const chunkId = chunkIds[idx];
  const chunk   = chunkCache.get(chunkId);

  populateHoverTip(idx, chunk?.[String(recipeIds[idx])] ?? null);
  tip.style.display = 'block';
  positionHoverTip(idx);

  if (!chunk) {
    loadChunk(chunkId).then(loaded => {
      if (hoverIdx !== idx || !loaded) return;
      populateHoverTip(idx, loaded[String(recipeIds[idx])] ?? null);
      positionHoverTip(idx);
    });
  }
}

function hideHoverTip() {
  hoverIdx = -1;
  document.getElementById('hover-tip').style.display = 'none';
}

// ── Raycasting ────────────────────────────────────────────────────────────────
const _rayPt = new THREE.Vector3();

function raycastBest(e) {
  const rc = renderer.domElement.getBoundingClientRect();
  mouse.x  =  ((e.clientX - rc.left) / rc.width)  * 2 - 1;
  mouse.y  = -((e.clientY - rc.top)  / rc.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const dist = camera.position.distanceTo(controls.target);
  raycaster.params.Points = { threshold: Math.max(0.01, dist * 0.02) };
  const hits = raycaster.intersectObject(pointsMesh);
  if (!hits.length) return -1;

  let best = -1, bestD = Infinity;
  for (const hit of hits) {
    const i = hit.index;
    if (i >= N || alphaCache[i] < 0.5) continue;
    _rayPt.set(posArr[i * 3], posArr[i * 3 + 1], posArr[i * 3 + 2]);
    const d = raycaster.ray.distanceToPoint(_rayPt);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ── Pointer events ────────────────────────────────────────────────────────────
function onPointerDown(e) {
  pointerIsDown = true;
  mouseDownX = e.clientX; mouseDownY = e.clientY;
}

function onPointerMove(e) {
  if (pointerIsDown) {
    const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
    if (d > DRAG_THRESH) { hideHoverTip(); return; }
  }
  const idx = raycastBest(e);
  if (idx >= 0) {
    showHoverTip(idx);
  } else {
    if (lockedIdx < 0) hideHoverTip();
  }
}

function onPointerUp(e) {
  pointerIsDown = false;
  const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
  if (d > DRAG_THRESH) return;

  const idx = raycastBest(e);
  const now = Date.now();

  if (now - lastClickTime < DBL_CLICK_MS && lastClickTime > 0) {
    lastClickTime = 0;
    hideHoverTip();
    lockedIdx = -1;
    if (idx >= 0) {
      // Double-click: isolate the category of this point
      const famData = getCategoryFamilyData(activeFamilyIdx);
      const catId   = famData[idx];
      if (highlightedLabelIdx === catId) {
        setHighlightLabel(-1);
        showExploreDefault();
      } else {
        setHighlightLabel(catId);
        showClusterInfo(catId);
      }
    } else {
      setHighlightLabel(-1);
      showExploreDefault();
    }
    return;
  }

  lastClickTime = now;
  if (appMode === 'explore') {
    if (idx >= 0) {
      lockedIdx = idx;
      showRecipeInfo(idx);
    } else {
      lockedIdx = -1;
      hideHoverTip();
      showExploreDefault();
    }
  }
}

function onPointerLeave() {
  pointerIsDown = false;
  if (lockedIdx < 0) hideHoverTip();
}

// ── Camera ────────────────────────────────────────────────────────────────────
const CAM_BACK = new THREE.Vector3(0, 0, 1); // camera looks down local -Z in Three.js
const MS_PER_RAD = 350;
const MS_MIN     = 200;
const MS_MAX     = 900;

function quatFromLookAt(from, to) {
  const lookDir = to.clone().sub(from).normalize();
  const m = new THREE.Matrix4().lookAt(new THREE.Vector3(0,0,0), lookDir, new THREE.Vector3(0,1,0));
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

function quatFromLookDir(lookDir, up = new THREE.Vector3(0,1,0)) {
  const m = new THREE.Matrix4().lookAt(new THREE.Vector3(0,0,0), lookDir.clone().normalize(), up);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

/**
 * Animate camera to a new orientation using quaternion slerp.
 * @param {THREE.Quaternion} toQ     - target camera quaternion
 * @param {number}           toDist  - distance from target
 * @param {THREE.Vector3}    toTarget - orbit target point
 */
function animateCameraTo(toQ, toDist, toTarget) {
  const fromQ      = camera.quaternion.clone();
  const fromDist   = camera.position.distanceTo(controls.target);
  const fromTarget = controls.target.clone();
  const t0         = performance.now();

  // Duration proportional to rotation angle — fast small moves, longer large ones
  const cosHalf = Math.abs(fromQ.dot(toQ));
  const angle   = 2 * Math.acos(Math.min(1, cosHalf));
  const ms      = Math.max(MS_MIN, Math.min(MS_MAX, angle * MS_PER_RAD));

  camAnim = (now) => {
    const t = Math.min((now - t0) / ms, 1);
    const e = t * t * (3 - 2 * t); // smoothstep

    // Slerp orientation — single op, no gimbal issues
    const q = fromQ.clone().slerp(toQ, e);

    // Lerp distance and target separately
    const dist   = fromDist + (toDist - fromDist) * e;
    const target = fromTarget.clone().lerp(toTarget, e);

    // Reconstruct position from orientation and distance
    const back = CAM_BACK.clone().applyQuaternion(q);
    camera.position.copy(target).addScaledVector(back, dist);
    camera.quaternion.copy(q);

    // Sync orbit target but DON'T call controls.update() — it would fight the slerp
    controls.target.copy(target);

    if (t >= 1) {
      // Final frame: snap exactly and let OrbitControls resync
      const finalBack = CAM_BACK.clone().applyQuaternion(toQ);
      camera.position.copy(toTarget).addScaledVector(finalBack, toDist);
      camera.quaternion.copy(toQ);
      camera.up.set(0, 1, 0);
      controls.target.copy(toTarget);
      controls.update();
    }

    return t < 1;
  };
}

/** Animate from a story-step camera object {position, target}. */
function animateCameraToPosition(pos, target) {
  const posVec    = new THREE.Vector3(...pos);
  const targetVec = new THREE.Vector3(...target);
  const q         = quatFromLookAt(posVec, targetVec);
  const dist      = posVec.distanceTo(targetVec);
  animateCameraTo(q, dist, targetVec);
}

function setCameraPreset(preset) {
  const c = dataCenterVec;
  const d = dataExtent * 1.5;
  let q, dist;
  switch (preset) {
    case 'top':
      q    = quatFromLookDir(new THREE.Vector3(0, -1, -0.00001), new THREE.Vector3(0, 0, -1));
      dist = d;
      break;
    case 'front':
      q    = quatFromLookDir(new THREE.Vector3(0, 0, -1));
      dist = d;
      break;
    case 'side':
      q    = quatFromLookDir(new THREE.Vector3(-1, 0, 0));
      dist = d;
      break;
    default: // reset
      q    = quatFromLookAt(defaultCamPos, c);
      dist = defaultCamPos.distanceTo(c);
      break;
  }
  animateCameraTo(q, dist, c.clone());
}

// ── Resize ────────────────────────────────────────────────────────────────────
function onResize() {
  const w = window.innerWidth - LEFT_W - RIGHT_W;
  const h = window.innerHeight - TOPBAR_H;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (lockedIdx >= 0) positionHoverTip(lockedIdx);
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  if (camAnim) {
    if (!camAnim(performance.now())) camAnim = null;
  } else {
    controls.update(); // only when not animating — update() fights quaternion slerp
  }
  renderer.render(scene, camera);
  if (lockedIdx >= 0) positionHoverTip(lockedIdx);
}

// ── Right panel: category tabs + label list ───────────────────────────────────
function initRightPanel() {
  const tabsEl = document.getElementById('category-tabs');
  tabsEl.innerHTML = '';
  meta.categories.forEach((cat, i) => {
    const btn = document.createElement('button');
    btn.className = 'cat-tab' + (i === 0 ? ' active' : '');
    btn.textContent = cat.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setActiveFamily(i);
    });
    tabsEl.appendChild(btn);
  });
  renderCategoryList();
}

function renderCategoryList() {
  const listEl  = document.getElementById('category-list');
  const family  = meta.categories[activeFamilyIdx];
  const famData = getCategoryFamilyData(activeFamilyIdx);

  // Count per label
  const counts = new Array(family.labels.length).fill(0);
  for (let i = 0; i < N; i++) {
    const id = famData[i];
    if (id < counts.length) counts[id]++;
  }

  // Determine palette hex for this label
  const palette    = uniforms.uPalette.value;
  const paletteN   = uniforms.uPaletteN.value;

  listEl.innerHTML = '';
  family.labels.forEach((label, labelIdx) => {
    const row = document.createElement('div');
    row.className = 'cat-row' + (highlightedLabelIdx === labelIdx ? ' active' : '');

    const palBase = labelIdx % paletteN;
    const r = Math.round(palette[palBase * 3]     * 255);
    const g = Math.round(palette[palBase * 3 + 1] * 255);
    const b = Math.round(palette[palBase * 3 + 2] * 255);
    const hex = `rgb(${r},${g},${b})`;

    const dot = document.createElement('span');
    dot.className        = 'cat-dot';
    dot.style.background = hex;

    const nameEl = document.createElement('span');
    nameEl.className = 'cat-label' + (highlightedLabelIdx >= 0 && highlightedLabelIdx !== labelIdx ? ' dimmed' : '');
    nameEl.textContent = label;

    const countEl = document.createElement('span');
    countEl.className   = 'cat-count';
    countEl.textContent = counts[labelIdx].toLocaleString();

    row.appendChild(dot);
    row.appendChild(nameEl);
    row.appendChild(countEl);

    row.addEventListener('click', () => {
      if (highlightedLabelIdx === labelIdx) {
        setHighlightLabel(-1);
      } else {
        setHighlightLabel(labelIdx);
        if (appMode === 'explore') showClusterInfo(labelIdx);
      }
    });

    listEl.appendChild(row);
  });
}

// ── Left panel: explore mode ──────────────────────────────────────────────────
function showExploreDefault() {
  document.getElementById('explore-default').style.display = 'block';
  document.getElementById('explore-recipe').style.display  = 'none';
  document.getElementById('explore-cluster').style.display = 'none';
}

function showRecipeInfo(idx) {
  document.getElementById('explore-default').style.display = 'none';
  document.getElementById('explore-recipe').style.display  = 'block';
  document.getElementById('explore-cluster').style.display = 'none';

  // Show placeholder immediately, fill in async
  document.getElementById('recipe-name').textContent = `Recipe #${recipeIds[idx]}`;
  document.getElementById('recipe-tags').innerHTML   = '';
  document.getElementById('recipe-stats').textContent = 'Loading…';
  document.getElementById('recipe-ingredients').textContent = '';
  document.getElementById('recipe-description').textContent = '';

  getRecipeData(idx).then(r => {
    if (!r) return;
    document.getElementById('recipe-name').textContent = r.name || `Recipe #${recipeIds[idx]}`;

    // Tags from active categories
    const tagsEl = document.getElementById('recipe-tags');
    tagsEl.innerHTML = '';
    meta.categories.forEach((cat, fi) => {
      const famData = getCategoryFamilyData(fi);
      const catId   = famData[idx];
      const label   = cat.labels[catId] ?? '';
      if (label) {
        const tag = document.createElement('span');
        tag.className   = 'recipe-tag';
        tag.textContent = label;
        tagsEl.appendChild(tag);
      }
    });

    const stats = [];
    if (r.avg_rating != null) stats.push(`★ ${r.avg_rating.toFixed(1)} (${r.n_ratings} ratings)`);
    if (r.minutes)      stats.push(`${r.minutes} min`);
    if (r.n_steps)      stats.push(`${r.n_steps} steps`);
    if (r.n_ingredients) stats.push(`${r.n_ingredients} ingredients`);
    document.getElementById('recipe-stats').textContent = stats.join(' · ');

    if (r.ingredients?.length) {
      const shown = r.ingredients.slice(0, 8);
      document.getElementById('recipe-ingredients').textContent =
        shown.join(', ') + (r.ingredients.length > 8 ? ` +${r.ingredients.length - 8} more` : '');
    }

    if (r.description) {
      const desc = r.description.trim();
      document.getElementById('recipe-description').textContent =
        desc.length > 220 ? desc.slice(0, 220) + '…' : desc;
    }
  });
}

function showClusterInfo(labelIdx) {
  if (appMode !== 'explore') return;
  const family = meta.categories[activeFamilyIdx];
  const label  = family.labels[labelIdx] ?? `Category ${labelIdx}`;
  const famData = getCategoryFamilyData(activeFamilyIdx);
  let count = 0;
  for (let i = 0; i < N; i++) if (famData[i] === labelIdx) count++;

  document.getElementById('explore-default').style.display = 'none';
  document.getElementById('explore-recipe').style.display  = 'none';
  document.getElementById('explore-cluster').style.display = 'block';
  document.getElementById('cluster-name').textContent  = label;
  document.getElementById('cluster-count').textContent = `${count.toLocaleString()} recipes`;
}

// ── Left panel: story mode ────────────────────────────────────────────────────
function applyStep(step) {
  // Switch category family
  if (step.colorBy) {
    const idx = meta.categories.findIndex(c => c.name === step.colorBy);
    if (idx >= 0) {
      setActiveFamily(idx);
      document.querySelectorAll('.cat-tab').forEach((btn, i) => {
        btn.classList.toggle('active', i === idx);
      });
    }
  }
  // Apply highlight
  if (step.highlight) {
    const famIdx = meta.categories.findIndex(c => c.name === step.highlight.family);
    if (famIdx >= 0) {
      if (famIdx !== activeFamilyIdx) {
        setActiveFamily(famIdx);
        document.querySelectorAll('.cat-tab').forEach((btn, i) => {
          btn.classList.toggle('active', i === famIdx);
        });
      }
      const labelIdx = meta.categories[famIdx].labels.indexOf(step.highlight.label);
      if (labelIdx >= 0) setHighlightLabel(labelIdx);
    }
  } else {
    setHighlightLabel(-1);
  }
  // Animate camera
  if (step.camera) {
    animateCameraToPosition(step.camera.position, step.camera.target);
  }
  // Update panel text
  document.getElementById('story-title').textContent    = step.title    || '';
  document.getElementById('story-subtitle').textContent = step.subtitle || '';
  document.getElementById('story-counter').textContent  =
    `${currentStep + 1} / ${storyData.steps.length}`;

  document.getElementById('btn-prev').disabled = currentStep === 0;
  document.getElementById('btn-next').disabled = currentStep === storyData.steps.length - 1;
}

function initStoryPanel() {
  document.getElementById('btn-prev').addEventListener('click', () => {
    if (currentStep > 0) { currentStep--; applyStep(storyData.steps[currentStep]); }
  });
  document.getElementById('btn-next').addEventListener('click', () => {
    if (currentStep < storyData.steps.length - 1) {
      currentStep++;
      applyStep(storyData.steps[currentStep]);
    }
  });
  applyStep(storyData.steps[0]);
}

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(mode) {
  appMode = mode;
  document.getElementById('btn-story').classList.toggle('active',   mode === 'story');
  document.getElementById('btn-explore').classList.toggle('active', mode === 'explore');
  document.getElementById('story-panel').style.display   = mode === 'story'   ? 'flex' : 'none';
  document.getElementById('explore-panel').style.display = mode === 'explore' ? 'flex' : 'none';

  if (mode === 'explore') {
    setHighlightLabel(-1);
    showExploreDefault();
    lockedIdx = -1;
    hideHoverTip();
  } else {
    applyStep(storyData.steps[currentStep]);
  }
}

// ── Debug: copy state ─────────────────────────────────────────────────────────
function copyState() {
  const pos    = camera.position;
  const target = controls.target;
  const family = meta.categories[activeFamilyIdx];
  const hl     = highlightedLabelIdx >= 0
    ? { family: family.name, label: family.labels[highlightedLabelIdx] }
    : null;

  const snippet = {
    title:     '',
    subtitle:  '',
    colorBy:   family.name,
    // position is reconstructed from quaternion + distance — store the visual position
    camera: {
      position: [+pos.x.toFixed(3), +pos.y.toFixed(3), +pos.z.toFixed(3)],
      target:   [+target.x.toFixed(3), +target.y.toFixed(3), +target.z.toFixed(3)],
    },
    highlight: hl,
  };

  navigator.clipboard.writeText(JSON.stringify(snippet, null, 2))
    .then(() => alert('Story step snippet copied to clipboard.'))
    .catch(() => console.log(JSON.stringify(snippet, null, 2)));
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  showProgress(0, 'Loading metadata…');

  // Load meta, palette, story in parallel
  const [metaRes, paletteRes, storyRes] = await Promise.all([
    fetch(`${DATA}meta.json`).then(r => r.json()),
    fetch('palette.json').then(r => r.json()),
    fetch('story.json').then(r => r.json()),
  ]);

  meta      = metaRes;
  storyData = storyRes;
  N         = meta.total;

  const palette = buildPalette(paletteRes);

  // Compute data center and extent for camera positioning
  const b  = meta.coord_bounds;
  dataCenterVec.set(
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  );
  dataExtent = Math.max(b.max[0]-b.min[0], b.max[1]-b.min[1], b.max[2]-b.min[2]);
  defaultCamPos = new THREE.Vector3(
    dataCenterVec.x,
    dataCenterVec.y + dataExtent * 0.2,
    dataCenterVec.z + dataExtent * 1.4,
  );

  showProgress(10, 'Loading geometry…');

  // Load Draco geometry
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/libs/draco/');
  const drcGeom = await new Promise((resolve, reject) => {
    dracoLoader.load(`${DATA}geometry.drc`, resolve, null, reject);
  });

  posArr = new Float32Array(drcGeom.attributes.position.array);
  showProgress(40, 'Loading attributes…');

  // Load and parse attributes.bin.gz
  const attrRaw    = await fetch(`${DATA}attributes.bin.gz`).then(r => r.arrayBuffer());
  const attrBuffer = await decompressBuffer(attrRaw);

  const dtypeSize = { uint32: 4, uint16: 2, uint8: 1, float32: 4 };
  const dtypeArray = {
    uint32: Uint32Array, uint16: Uint16Array, uint8: Uint8Array, float32: Float32Array,
  };
  const attributes = {};
  let offset = 0;
  for (const attr of meta.attribute_layout) {
    const sz  = dtypeSize[attr.dtype];
    const Arr = dtypeArray[attr.dtype];
    // Always slice to guarantee alignment — avoids issues when offset is not
    // a multiple of the element size (e.g. float32 after several uint8 blocks).
    attributes[attr.name] = new Arr(attrBuffer.slice(offset, offset + N * sz));
    offset += N * sz;
  }

  recipeIds = attributes['recipe_id'];
  chunkIds  = attributes['chunk_id'];

  // Build packed category data: Uint32Array(N * N_families)
  const nFamilies = meta.categories.length;
  categoryData    = new Uint32Array(N * nFamilies);
  meta.categories.forEach((cat, fi) => {
    const src = attributes[cat.attribute];
    for (let i = 0; i < N; i++) {
      categoryData[fi * N + i] = src[i];
    }
  });

  showProgress(70, 'Building scene…');
  initScene(palette);

  showProgress(90, 'Initialising UI…');
  initRightPanel();

  // Mode buttons
  document.getElementById('btn-story').addEventListener('click',   () => setMode('story'));
  document.getElementById('btn-explore').addEventListener('click', () => setMode('explore'));

  // About modal
  document.getElementById('btn-about').addEventListener('click', () => {
    document.getElementById('about-overlay').classList.add('open');
  });
  document.getElementById('about-close').addEventListener('click', () => {
    document.getElementById('about-overlay').classList.remove('open');
  });
  document.getElementById('about-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('about-overlay'))
      document.getElementById('about-overlay').classList.remove('open');
  });

  // Debug copy state
  document.getElementById('btn-copy-state').addEventListener('click', copyState);

  // Story mode default
  initStoryPanel();
  setMode('story');

  hideProgress();
}

boot().catch(e => {
  document.getElementById('progress-label').textContent = `Error: ${e.message}`;
  document.getElementById('progress-label').style.display = 'block';
  document.getElementById('progress-wrap').style.display  = 'none';
  console.error(e);
});
