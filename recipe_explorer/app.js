import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

// ── Constants ────────────────────────────────────────────────────────────────
const DATA = 'data/';
const LEFT_W = 0;
const RIGHT_W = 0;
const TOPBAR_H = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--topbar-h')) || 44;
const CAT_TEX_W = 4096;   // texture width for category data buffer
const MAX_LABELS = 64;     // max labels per category family for uniform arrays
const DRAG_THRESH = 5;
const DBL_CLICK_MS = 350;

// ── Vertex shader (GLSL3 — uses gl_VertexID + texelFetch) ───────────────────
const VERT = /* glsl */`
  out vec3  vColor;
  out float vAlpha;
  out float vOutline;

  uniform float uPointSize;

  uniform highp usampler2D uCategoryTex;
  uniform int  uActiveFamilyIdx;
  uniform int  uN;

  uniform vec3 uPalette[${MAX_LABELS * 3}];
  uniform int  uPaletteN;
  uniform int  uCategoryModes[${MAX_LABELS}];

  uniform int   uSecFamilyIdx;     // -1 = inactive
  uniform int   uSecLabelIdx;
  uniform float uSecDimFactor;     // dim for non-matching L1 points (default 1.0)
  uniform float uSecOutlineFactor; // outline brightness for matching L2 points: 0=none, 0.4=hover, 1=locked
  uniform int   uLockedVertexIdx;  // -1 = none; locked point gets mode 1 (highlight)

  void main() {
    int flatIdx = uActiveFamilyIdx * uN + gl_VertexID;
    int tx = flatIdx % ${CAT_TEX_W};
    int ty = flatIdx / ${CAT_TEX_W};
    int catId = int(texelFetch(uCategoryTex, ivec2(tx, ty), 0).r);

    int   mode      = uCategoryModes[catId];
    if (uLockedVertexIdx == gl_VertexID) mode = 1;
    float dimFactor = 1.0;
    bool  inL2      = false;

    // Secondary filter: dim non-matching points within L1 selection (mode 0 = normal, 1 = light)
    if (uSecFamilyIdx >= 0 && mode <= 1) {
      int secFlat  = uSecFamilyIdx * uN + gl_VertexID;
      int secCatId = int(texelFetch(uCategoryTex, ivec2(secFlat % ${CAT_TEX_W}, secFlat / ${CAT_TEX_W}), 0).r);
      if (secCatId != uSecLabelIdx) dimFactor = uSecDimFactor;
      else inL2 = true;
    }

    // mode 0 = normal, 1 = light, 2 = dark/desaturated — never hidden
    vAlpha = 1.0;
    int paletteIdx = mode * uPaletteN + catId % uPaletteN;
    vColor = uPalette[paletteIdx] * dimFactor;
    vOutline = inL2 ? uSecOutlineFactor : 0.0;

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
  in float vOutline;
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
      col = mix(col, vec3(vOutline), t * 0.85);
    }
    fragColor = vec4(col, 1.0);
  }
`;

// ── State ────────────────────────────────────────────────────────────────────
let N = 0;
let meta = null;
let posArr = null;          // Float32Array(N*3) — positions for raycasting
let recipeIds = null;       // Uint32Array(N)
let chunkIds = null;       // Uint16Array(N)
let categoryData = null;    // Uint32Array(N * N_families) — packed category IDs
let alphaCache = null;      // Float32Array(N) — 0=hidden, 1=visible (for raycasting)

const chunkCache = new Map(); // chunkId -> {recipe_id_str: {...}}
const pendingChunks = new Map(); // chunkId -> Promise<data> (in-flight fetches)

const recipeMetricsCache = new Map(); // shard -> {rid_str: metrics | {}}
const categoryMetricsCache = new Map(); // filename -> data
let categoryMetricsIndex = null;      // {family: {label: filename}}

// Multi-label highlight — when set, overrides highlightedLabelIdx.
// null = use single-label system; Set<number> = active set of label indices.
let highlightLabelSet = null;

let savedIntersectionState = null; // {activeFam, modes[]} saved before intersection hover

const REVIEWS_YEAR_MIN = 1999;
const REVIEWS_YEAR_MAX = 2018;


let renderer, scene, camera, controls;
let geometry, pointsMesh;
let uniforms = {};
let catTexture = null;
let activeFamilyIdx = 0;
let categoryModes = new Int32Array(MAX_LABELS).fill(0);
let highlightedLabelIdx = -1; // -1 = none

// ── Filter state ─────────────────────────────────────────────────────────────
let filterLevel = 0;      // 0=root, 1=L1 selected, 2=both selected
let level1LabelIdx = -1;  // label index within activeFamilyIdx
let level2FamilyIdx = -1; // family index for Level 2 chart
let level2LabelIdx = -1;  // label index within level2 family

let lockedIdx = -1;
function setLockedIdx(idx) {
  lockedIdx = idx;
  if (uniforms.uLockedVertexIdx !== undefined) uniforms.uLockedVertexIdx.value = idx;
}

let hoverIdx = -1;   // currently hovered point index
let pointerIsDown = false;
let mouseDownX = 0, mouseDownY = 0;
let lastMouseX = -1, lastMouseY = -1;
let hoveredChartFamilyIdx = -1, hoveredChartLi = -1;
let lastClickTime = 0;
let camAnim = null;

// Computed from coord_bounds after meta loads — used by camera presets
let dataCenterVec = new THREE.Vector3(0, 0, 0);
let defaultCamPos = new THREE.Vector3(0.8, 0.6, 2.0);
let dataExtent = 14;  // approximate, updated on boot

let appMode = 'story';  // 'story' | 'explore'
let storyData = null;
let currentStep = 0;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// ── Color helpers ─────────────────────────────────────────────────────────────
function hexToVec3(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
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
  return [rgb[0] * 0.20 + 0.03, rgb[1] * 0.20 + 0.03, rgb[2] * 0.20 + 0.03];
}

function buildPalette(hexColors) {
  const n = hexColors.length;
  // flat array: [normal×n, light×n, dark×n]
  const flat = new Float32Array(n * 3 * 3);
  hexColors.forEach((hex, i) => {
    const base = hexToVec3(hex);
    const light = lighten(base);
    const dark = darken(base);
    flat.set(base, i * 3);
    flat.set(light, (n + i) * 3);
    flat.set(dark, (n * 2 + i) * 3);
  });
  return { flat, n };
}

// ── Category texture ──────────────────────────────────────────────────────────
function buildCategoryTexture(families) {
  const nFamilies = families.length;
  const totalEls = N * nFamilies;
  const texH = Math.ceil(totalEls / CAT_TEX_W);
  const texData = new Uint32Array(CAT_TEX_W * texH); // zero-padded

  families.forEach((arr, fi) => {
    for (let i = 0; i < N; i++) {
      texData[fi * N + i] = arr[i];
    }
  });

  const tex = new THREE.DataTexture(texData, CAT_TEX_W, texH,
    THREE.RedIntegerFormat, THREE.UnsignedIntType);
  tex.internalFormat = 'R32UI';
  tex.needsUpdate = true;
  return tex;
}

// ── Label display names ───────────────────────────────────────────────────────
const LABEL_OVERRIDES = {
  'northeastern-united-states': 'northeastern-us',
  'southern-united-states':     'southern-us',
  'southwestern-united-states': 'southwestern-us',
};
function shortenLabel(label) {
  return LABEL_OVERRIDES[label] ?? label;
}

// ── Palette color lookup ──────────────────────────────────────────────────────
function getPaletteRgb(labelIdx) {
  const palN = uniforms.uPaletteN?.value ?? 1;
  const palette = uniforms.uPalette?.value;
  if (!palette) return [78, 121, 167];
  const base = (labelIdx % palN) * 3;
  return [
    Math.round(palette[base] * 255),
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
  if (chunkCache.has(chunkId)) return Promise.resolve(chunkCache.get(chunkId));
  if (pendingChunks.has(chunkId)) return pendingChunks.get(chunkId);

  const promise = (async () => {
    try {
      const resp = await fetch(`${DATA}chunks/chunk_${pad(chunkId)}.json.gz`);
      if (!resp.ok) return null;
      const data = await resp.arrayBuffer().then(decompressJson);
      chunkCache.set(chunkId, data);
      return data;
    } catch { return null; }
    finally { pendingChunks.delete(chunkId); }
  })();

  pendingChunks.set(chunkId, promise);
  return promise;
}

async function getRecipeData(idx) {
  const chunk = await loadChunk(chunkIds[idx]);
  return chunk?.[String(recipeIds[idx])] ?? null;
}

// ── Star field ────────────────────────────────────────────────────────────────
function createStarField() {
  const COUNT = 3000;
  const RADIUS = 120;
  const pos = new Float32Array(COUNT * 3);
  for (let i = 0; i < COUNT; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    pos[i * 3]     = RADIUS * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = RADIUS * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = RADIUS * Math.cos(phi);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.12,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.35,
  });
  return new THREE.Points(geo, mat);
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function initScene(palette) {
  const canvas = document.getElementById('three-canvas');
  const w = window.innerWidth - LEFT_W - RIGHT_W;
  const h = window.innerHeight;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);

  scene = new THREE.Scene();
  scene.add(createStarField());
  camera = new THREE.PerspectiveCamera(60, w / h, 0.001, 1000);
  camera.position.copy(defaultCamPos);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.05;
  controls.maxDistance = dataExtent * 6;
  controls.target.copy(dataCenterVec);
  controls.update();
  controls.addEventListener('start', () => stopIdleAutoRotate());
  controls.addEventListener('end', () => {
    if (appMode === 'story' && currentStep === 0) startIdleAutoRotate();
  });

  geometry = new THREE.BufferGeometry();
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1000);

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
    uPointSize: { value: 4.0 },
    uOutline: { value: 1.0 },
    uCategoryTex: { value: catTexture },
    uActiveFamilyIdx: { value: 0 },
    uN: { value: N },
    uPalette: { value: palette.flat },
    uPaletteN: { value: palette.n },
    uCategoryModes: { value: Array.from(categoryModes) },
    uSecFamilyIdx:     { value: -1 },
    uSecLabelIdx:      { value: 0 },
    uSecDimFactor:     { value: 1.0 },
    uSecOutlineFactor: { value: 0.0 },
    uLockedVertexIdx:  { value: -1 },
  };

  const mat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms,
    depthTest: true,
    depthWrite: true,
  });

  pointsMesh = new THREE.Points(geometry, mat);
  scene.add(pointsMesh);

  // Initial alpha cache (all visible)
  alphaCache = new Float32Array(N).fill(1.0);

  // UI wiring — elements may be absent if their panel is commented out in HTML
  document.getElementById('fc-reset')?.addEventListener('click', () => setCameraPreset('reset'));
  document.getElementById('fc-top')?.addEventListener('click', () => setCameraPreset('top'));
  document.getElementById('fc-front')?.addEventListener('click', () => setCameraPreset('front'));
  document.getElementById('fc-side')?.addEventListener('click', () => setCameraPreset('side'));


  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerup', onPointerUp);
  renderer.domElement.addEventListener('pointerleave', onPointerLeave);

  animate();
}

// ── Category mode management ──────────────────────────────────────────────────
function setHighlightLabel(labelIdx) {
  highlightLabelSet = null;
  highlightedLabelIdx = labelIdx;
  const nLabels = meta.categories[activeFamilyIdx].labels.length;
  for (let i = 0; i < MAX_LABELS; i++) {
    if (i >= nLabels) { categoryModes[i] = 0; continue; }
    categoryModes[i] = (labelIdx < 0 || i === labelIdx) ? 0 : 2;
  }
  uniforms.uCategoryModes.value = Array.from(categoryModes);
  if (labelIdx < 0) {
    alphaCache.fill(1.0);
  } else {
    const famData = getCategoryFamilyData(activeFamilyIdx);
    for (let i = 0; i < N; i++) {
      alphaCache[i] = famData[i] === labelIdx ? 1.0 : 0.0;
    }
  }
}

function applyHighlightLabels(labelIdxs) {
  highlightLabelSet = new Set(labelIdxs);
  const nLabels = meta.categories[activeFamilyIdx].labels.length;
  for (let i = 0; i < MAX_LABELS; i++) {
    if (i >= nLabels) { categoryModes[i] = 0; continue; }
    categoryModes[i] = (highlightLabelSet.size === 0 || highlightLabelSet.has(i)) ? 0 : 2;
  }
  uniforms.uCategoryModes.value = Array.from(categoryModes);
  if (highlightLabelSet.size === 0) {
    alphaCache.fill(1.0);
  } else {
    const famData = getCategoryFamilyData(activeFamilyIdx);
    for (let i = 0; i < N; i++) {
      alphaCache[i] = highlightLabelSet.has(famData[i]) ? 1.0 : 0.0;
    }
  }
}

function applyIntersectionHighlight(secondaryFamName, secondaryLabel) {
  const secFamIdx = meta.categories.findIndex(c => c.name === secondaryFamName);
  if (secFamIdx < 0) return;
  const secLabelIdx = meta.categories[secFamIdx].labels.indexOf(secondaryLabel);
  if (secLabelIdx < 0) return;

  if (!savedIntersectionState) {
    savedIntersectionState = { alpha: alphaCache.slice() };
  }

  uniforms.uSecFamilyIdx.value = secFamIdx;
  uniforms.uSecLabelIdx.value = secLabelIdx;
  uniforms.uSecDimFactor.value = 0.7;

  // Restrict raycasting to the intersection subset
  const secData = getCategoryFamilyData(secFamIdx);
  for (let i = 0; i < N; i++) {
    alphaCache[i] = savedIntersectionState.alpha[i] > 0 && secData[i] === secLabelIdx ? 1.0 : 0.0;
  }
}

function restoreIntersectionHighlight() {
  if (!savedIntersectionState) return;
  uniforms.uSecFamilyIdx.value     = -1;
  uniforms.uSecDimFactor.value     = 1.0;
  uniforms.uSecOutlineFactor.value = 0.0;
  alphaCache.set(savedIntersectionState.alpha);
  savedIntersectionState = null;
}

function setActiveFamily(idx) {
  if (filterLevel === 0) {
    activeFamilyIdx = idx;
    highlightedLabelIdx = -1;
    highlightLabelSet = null;
    categoryModes.fill(0);
    uniforms.uActiveFamilyIdx.value = idx;
    uniforms.uCategoryModes.value = Array.from(categoryModes);
    alphaCache.fill(1.0);
    renderRightPanelChart(idx);
  } else if (filterLevel === 1) {
    if (idx === activeFamilyIdx) {
      level2FamilyIdx = -1;
      renderRightPanelChart(idx);
      updateTabVisualState();
    } else {
      level2FamilyIdx = idx;
      const filteredCounts = computeFilteredCounts(idx, activeFamilyIdx, level1LabelIdx);
      renderRightPanelChart(idx, filteredCounts);
      updateTabVisualState();
    }
  } else if (filterLevel === 2) {
    if (idx === activeFamilyIdx) {
      // Clicking L1 family tab — stay at L2, show full L1 chart
      renderRightPanelChart(activeFamilyIdx);
      updateTabVisualState();
    } else if (idx === level2FamilyIdx) {
      // Clicking current L2 family tab — re-show the L2 filtered chart
      const filteredCounts = computeFilteredCounts(idx, activeFamilyIdx, level1LabelIdx);
      renderRightPanelChart(idx, filteredCounts);
      updateTabVisualState();
    } else {
      // Switching to a different L2 family — clear L2 selection, drop to L1
      level2FamilyIdx = idx;
      level2LabelIdx = -1;
      filterLevel = 1;
      restoreIntersectionHighlight();
      const filteredCounts = computeFilteredCounts(idx, activeFamilyIdx, level1LabelIdx);
      renderRightPanelChart(idx, filteredCounts);
      renderFilterChips();
      updateTabVisualState();
    }
  }
}

function getCategoryFamilyData(familyIdx) {
  // Returns Uint8/Uint32 view into categoryData for a given family
  const start = familyIdx * N;
  return categoryData.subarray(start, start + N);
}

// ── Progress ──────────────────────────────────────────────────────────────────
function showProgress(pct, label) {
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('progress-label').style.display = 'block';
  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('progress-label').textContent = label;
}
function hideProgress() {
  document.getElementById('progress-wrap').style.display = 'none';
  document.getElementById('progress-label').style.display = 'none';
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function worldToScreen(i) {
  const i3 = i * 3;
  const v = new THREE.Vector3(posArr[i3], posArr[i3 + 1], posArr[i3 + 2]).project(camera);
  const w = window.innerWidth - LEFT_W - RIGHT_W;
  const h = window.innerHeight;
  return { x: (v.x + 1) / 2 * w + LEFT_W, y: (-v.y + 1) / 2 * h };
}

function positionHoverTip(idx) {
  const tip = document.getElementById('hover-tip');
  if (tip.style.display === 'none') return;
  const sp = worldToScreen(idx);
  const tw = tip.offsetWidth || 180;
  const th = tip.offsetHeight || 32;
  const margin = 14;

  // Horizontal zone: between right edge of left panel and left edge of right column
  const lpEl = document.getElementById('left-panel');
  const edEl = document.getElementById('explore-default');
  const rcEl = document.getElementById('right-column');
  const leftBound = (lpEl && lpEl.style.display !== 'none')
    ? lpEl.getBoundingClientRect().right + 4
    : (edEl && edEl.style.display !== 'none')
    ? edEl.getBoundingClientRect().right + 4
    : 4;
  const rightBound = rcEl ? rcEl.getBoundingClientRect().left - 4 : window.innerWidth - 4;

  let left, arrowSide;
  if (sp.x + margin + tw + 16 <= rightBound) {
    left = sp.x + margin; arrowSide = 'left';
  } else {
    left = sp.x - margin - tw; arrowSide = 'right';
  }
  left = Math.max(leftBound, Math.min(left, rightBound - tw));

  let top = sp.y - th / 2;
  top = Math.max(TOPBAR_H + 4, Math.min(top, window.innerHeight - th - 4));

  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;

  const arrow = document.getElementById('hover-tip-arrow');
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
    const famData = getCategoryFamilyData(fi);
    const labelIdx = famData[idx];
    const label = cat.labels[labelIdx];
    if (!label) return;
    const [r, g, b] = getPaletteRgb(labelIdx);
    const tr = Math.round(r * 0.45 + 255 * 0.55);
    const tg = Math.round(g * 0.45 + 255 * 0.55);
    const tb = Math.round(b * 0.45 + 255 * 0.55);
    const span = document.createElement('span');
    span.className = 'ht-tag' + (fi === activeFamilyIdx ? ' active-family' : '');
    span.textContent = label;
    span.style.background = `rgba(${r},${g},${b},0.35)`;
    span.style.borderColor = `rgba(${r},${g},${b},0.80)`;
    span.style.color = `rgb(${tr},${tg},${tb})`;
    tagsEl.appendChild(span);
  });

  // Meta line
  const parts = [];
  if (recipe.avg_rating != null) {
    const ratingStr = `★ ${recipe.avg_rating.toFixed(1)}`;
    const countStr = recipe.n_ratings != null ? ` (${recipe.n_ratings} ratings)` : '';
    parts.push(ratingStr + countStr);
  }
  if (recipe.minutes) parts.push(`${recipe.minutes} min`);
  if (recipe.n_steps) parts.push(`${recipe.n_steps} steps`);
  if (recipe.n_ingredients) parts.push(`${recipe.n_ingredients} ingredients`);
  if (recipe.submitted) parts.push(recipe.submitted.slice(0, 7));
  metaEl.textContent = parts.join(' · ');

  descEl.textContent = (recipe.description || '').trim().slice(0, 220);
}

function showHoverTip(idx) {
  hoverIdx = idx;
  const tip = document.getElementById('hover-tip');
  const chunkId = chunkIds[idx];
  const chunk = chunkCache.get(chunkId);

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
  mouse.x = ((e.clientX - rc.left) / rc.width) * 2 - 1;
  mouse.y = -((e.clientY - rc.top) / rc.height) * 2 + 1;
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
  lastMouseX = e.clientX; lastMouseY = e.clientY;
  if (pointerIsDown) {
    const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
    if (d > DRAG_THRESH) { if (lockedIdx < 0) hideHoverTip(); return; }
  }
  if (lockedIdx >= 0) return;
  const idx = raycastBest(e);
  if (idx >= 0) {
    showHoverTip(idx);
  } else {
    hideHoverTip();
  }
}

function onPointerUp(e) {
  pointerIsDown = false;
  const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
  if (d > DRAG_THRESH) return;

  const idx = raycastBest(e);
  const now = Date.now();

  // Check double-click FIRST — cancel single-click effect, then filter category
  if (now - lastClickTime < DBL_CLICK_MS && lastClickTime > 0) {
    lastClickTime = 0;
    if (appMode === 'story') return;
    // Undo the first-click recipe selection
    setLockedIdx(-1);
    hideHoverTip();
    showExploreDefault();
    if (filterLevel >= 1) {
      resetToLevel0();
    } else if (idx >= 0) {
      const famData = getCategoryFamilyData(activeFamilyIdx);
      transitionToLevel1(activeFamilyIdx, famData[idx]);
    }
    return;
  }

  lastClickTime = now;
  if (appMode === 'explore') {
    if (lockedIdx >= 0) {
      setLockedIdx(-1);
      showExploreDefault();
      if (idx >= 0) showHoverTip(idx); else hideHoverTip();
    } else if (idx >= 0) {
      setLockedIdx(idx);
      showRecipeInfo(idx);
    }
  }
}

function onPointerLeave() {
  pointerIsDown = false;
  if (lockedIdx < 0) hideHoverTip();
}

// ── Camera ────────────────────────────────────────────────────────────────────
const CAM_BACK = new THREE.Vector3(0, 0, 1); // camera looks down local -Z in Three.js
const MS_PER_RAD = 210;
const MS_PER_UNIT_TARGET = 36;
const MS_PER_UNIT_DIST = 24;
const MS_MIN = 240;
const MS_MAX = 840;

function quatFromLookAt(from, to) {
  const lookDir = to.clone().sub(from).normalize();
  const m = new THREE.Matrix4().lookAt(new THREE.Vector3(0, 0, 0), lookDir, new THREE.Vector3(0, 1, 0));
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

function quatFromLookDir(lookDir, up = new THREE.Vector3(0, 1, 0)) {
  const m = new THREE.Matrix4().lookAt(new THREE.Vector3(0, 0, 0), lookDir.clone().normalize(), up);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

/**
 * Animate camera to a new orientation using quaternion slerp.
 * @param {THREE.Quaternion} toQ     - target camera quaternion
 * @param {number}           toDist  - distance from target
 * @param {THREE.Vector3}    toTarget - orbit target point
 */
function animateCameraTo(toQ, toDist, toTarget) {
  const fromQ = camera.quaternion.clone();
  const fromDist = camera.position.distanceTo(controls.target);
  const fromTarget = controls.target.clone();
  const t0 = performance.now();

  // Duration based on rotation angle + target displacement + distance change
  const cosHalf = Math.abs(fromQ.dot(toQ));
  const angle = 2 * Math.acos(Math.min(1, cosHalf));
  const targetMove = fromTarget.distanceTo(toTarget);
  const distMove = Math.abs(fromDist - toDist);
  const ms = Math.max(MS_MIN, Math.min(MS_MAX,
    angle * MS_PER_RAD + targetMove * MS_PER_UNIT_TARGET + distMove * MS_PER_UNIT_DIST
  ));

  camAnim = (now) => {
    const t = Math.min((now - t0) / ms, 1);
    const e = t * t * (3 - 2 * t); // smoothstep

    // Slerp orientation — single op, no gimbal issues
    const q = fromQ.clone().slerp(toQ, e);

    // Lerp distance and target separately
    const dist = fromDist + (toDist - fromDist) * e;
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

// ── Idle auto-rotate (step 0 only) ───────────────────────────────────────────
let _autoRotateTimer = null;

let _autoRotateFadeStart = -1;
const AUTO_ROTATE_TARGET = 0.25;
const AUTO_ROTATE_FADE_MS = 3000;

function startIdleAutoRotate() {
  clearTimeout(_autoRotateTimer);
  _autoRotateTimer = setTimeout(() => {
    if (appMode === 'story' && currentStep === 0) {
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0;
      _autoRotateFadeStart = performance.now();
    }
  }, 3000);
}

function stopIdleAutoRotate() {
  clearTimeout(_autoRotateTimer);
  _autoRotateFadeStart = -1;
  if (controls) { controls.autoRotate = false; controls.autoRotateSpeed = 0; }
}

/** Animate from a story-step camera object {position, target}. */
function animateCameraToPosition(pos, target) {
  const posVec = new THREE.Vector3(...pos);
  const targetVec = new THREE.Vector3(...target);
  const q = quatFromLookAt(posVec, targetVec);
  const dist = posVec.distanceTo(targetVec);
  animateCameraTo(q, dist, targetVec);
}

function setCameraPreset(preset) {
  const c = dataCenterVec;
  const d = dataExtent * 1.5;
  let q, dist;
  switch (preset) {
    case 'top':
      q = quatFromLookDir(new THREE.Vector3(0, -1, -0.00001), new THREE.Vector3(0, 0, -1));
      dist = d;
      break;
    case 'front':
      q = quatFromLookDir(new THREE.Vector3(0, 0, -1));
      dist = d;
      break;
    case 'side':
      q = quatFromLookDir(new THREE.Vector3(-1, 0, 0));
      dist = d;
      break;
    default: // reset
      q = quatFromLookAt(defaultCamPos, c);
      dist = defaultCamPos.distanceTo(c);
      break;
  }
  animateCameraTo(q, dist, c.clone());
}

// ── Resize ────────────────────────────────────────────────────────────────────
function onResize() {
  const w = window.innerWidth - LEFT_W - RIGHT_W;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (lockedIdx >= 0) positionHoverTip(lockedIdx);
  const btn = document.getElementById('btn-randomize');
  if (btn && btn.style.display !== 'none') {
    const anchor = document.getElementById('left-panel')?.style.display !== 'none'
      ? document.getElementById('left-panel')
      : document.getElementById('explore-default');
    if (anchor) positionRandomize(anchor);
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  if (camAnim) {
    if (!camAnim(performance.now())) camAnim = null;
  } else {
    if (_autoRotateFadeStart >= 0) {
      const t = Math.min((performance.now() - _autoRotateFadeStart) / AUTO_ROTATE_FADE_MS, 1);
      controls.autoRotateSpeed = t * t * AUTO_ROTATE_TARGET;
      if (t >= 1) _autoRotateFadeStart = -1;
    }
    controls.update(); // only when not animating — update() fights quaternion slerp
  }
  renderer.render(scene, camera);
  if (lockedIdx >= 0) positionHoverTip(lockedIdx);
}

// ── Right panel: category tabs + horizontal bar chart ─────────────────────────
function initRightPanel() {
  const tabsEl = document.getElementById('category-tabs');
  tabsEl.innerHTML = '';
  meta.categories.forEach((cat, i) => {
    const btn = document.createElement('button');
    btn.className = 'cat-tab' + (i === 0 ? ' active' : '');
    const nameSpan = document.createElement('span');
    nameSpan.textContent = cat.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const sym = document.createElement('span');
    sym.className = 'tab-symbol';
    sym.textContent = '○';
    btn.appendChild(nameSpan);
    btn.appendChild(sym);
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setActiveFamily(i);
    });
    tabsEl.appendChild(btn);
  });
  renderFilterChips();
  renderRightPanelChart(activeFamilyIdx);

  const chartContainer = document.getElementById('right-panel-chart');
  if (chartContainer && typeof ResizeObserver !== 'undefined') {
    let _chartResizePending = false;
    new ResizeObserver(() => {
      if (_chartResizePending) return;
      _chartResizePending = true;
      requestAnimationFrame(() => {
        _chartResizePending = false;
        const famIdx = appMode === 'explore' && filterLevel >= 2 ? level2FamilyIdx : activeFamilyIdx;
        const counts = appMode === 'explore' && filterLevel >= 2
          ? computeFilteredCounts(level2FamilyIdx, activeFamilyIdx, level1LabelIdx)
          : undefined;
        renderRightPanelChart(famIdx, counts);
      });
    }).observe(chartContainer);
  }
}

function updateTabVisualState() {
  document.querySelectorAll('.cat-tab').forEach((btn, i) => {
    btn.classList.remove('tab-locked', 'tab-available');
    const sym = btn.querySelector('.tab-symbol');
    if (!sym) return;
    const isActive = (i === activeFamilyIdx && filterLevel >= 1) ||
                     (i === level2FamilyIdx && filterLevel >= 2);
    sym.textContent = isActive ? '●' : '○';
  });
}

function computeFullCounts(familyIdx) {
  const family = meta.categories[familyIdx];
  const famData = getCategoryFamilyData(familyIdx);
  const counts = new Array(family.labels.length).fill(0);
  for (let i = 0; i < N; i++) {
    const id = famData[i];
    if (id < counts.length) counts[id]++;
  }
  return counts;
}

function computeFilteredCounts(familyIdx, l1FamilyIdx, l1LabelIdx) {
  const family = meta.categories[familyIdx];
  const famData = getCategoryFamilyData(familyIdx);
  const l1Data = getCategoryFamilyData(l1FamilyIdx);
  const counts = new Array(family.labels.length).fill(0);
  for (let i = 0; i < N; i++) {
    if (l1Data[i] !== l1LabelIdx) continue;
    const id = famData[i];
    if (id < counts.length) counts[id]++;
  }
  return counts;
}


function restoreSecondaryAfterHover() {
  if (filterLevel === 2) {
    uniforms.uSecFamilyIdx.value     = level2FamilyIdx;
    uniforms.uSecLabelIdx.value      = level2LabelIdx;
    uniforms.uSecDimFactor.value     = 0.80;  // noticeable but not harsh
    uniforms.uSecOutlineFactor.value = 1.0;   // full white outline when locked
  } else {
    uniforms.uSecFamilyIdx.value     = -1;
    uniforms.uSecDimFactor.value     = 1.0;
    uniforms.uSecOutlineFactor.value = 0.0;
  }
}

function renderRightPanelChart(familyIdx, scopedCounts) {
  const container = document.getElementById('right-panel-chart');
  if (!container || !meta) return;

  const family = meta.categories[familyIdx];
  const counts = scopedCounts ?? computeFullCounts(familyIdx);
  const labels = family.labels;
  const maxCount = Math.max(...counts, 1);

  // For unordered categorical families, sort bars by displayed count descending
  const SORT_BY_COUNT = new Set(['cuisines', 'meal_types']);
  // For ordered families that read better high→low, reverse natural order
  const REVERSE_ORDER = new Set(['avg_rating']);
  const renderOrder = SORT_BY_COUNT.has(family.name)
    ? Array.from({length: labels.length}, (_, i) => i).sort((a, b) => counts[b] - counts[a])
    : REVERSE_ORDER.has(family.name)
    ? Array.from({length: labels.length}, (_, i) => labels.length - 1 - i)
    : Array.from({length: labels.length}, (_, i) => i);

  const W = container.offsetWidth || 300;
  const MIN_ROW_H = 18;
  const TOP_H = 18;
  const containerH = Math.max(0, (container.clientHeight || 0) - 12 - TOP_H);
  const rowH = containerH > 0
    ? Math.max(MIN_ROW_H, containerH / labels.length)
    : MIN_ROW_H;

  // Measure longest label to align all bar starts consistently
  const _ctx = document.createElement('canvas').getContext('2d');
  _ctx.font = '11px "Source Serif 4", serif';
  const maxTextW = Math.max(...labels.map(l => _ctx.measureText(shortenLabel(l)).width));
  const labelW = Math.max(60, Math.ceil(maxTextW) + 20);
  const barGap = 6;
  const countPad = 40;
  const barMaxW = W - labelW - barGap - countPad;
  const H = rowH * labels.length;

  container.innerHTML = '';
  const svg = d3.select(container).append('svg')
    .attr('width', W).attr('height', TOP_H + H - 1)
    .style('display', 'block').style('overflow', 'visible');

  const FONT = '"Source Serif 4", serif';
  const TEXT_DIM = 'rgba(255,255,255,0.72)';
  const TEXT_BODY = 'rgba(255,255,255,0.88)';

  // Clip path for label column so long names don't overflow
  svg.append('defs').append('clipPath').attr('id', 'rpc-label-clip')
    .append('rect').attr('x', 0).attr('y', 0).attr('width', labelW - 2).attr('height', TOP_H + H);

  // Bar rects and label texts keyed by label index for hover manipulation
  const barRects = new Map();
  const labelTexts = new Map();

  const isL1Chart = familyIdx === activeFamilyIdx;
  const isSelected = li => {
    if (appMode === 'story') {
      if (highlightLabelSet) return highlightLabelSet.has(li);
      return highlightedLabelIdx >= 0 && li === highlightedLabelIdx;
    }
    return (isL1Chart && filterLevel >= 1 && li === level1LabelIdx) ||
           (!isL1Chart && filterLevel >= 2 && familyIdx === level2FamilyIdx && li === level2LabelIdx);
  };

  const hasSelection = appMode === 'story'
    ? (highlightedLabelIdx >= 0 || (highlightLabelSet?.size ?? 0) > 0)
    : ((isL1Chart && filterLevel >= 1) || (!isL1Chart && filterLevel >= 2));
  const baseOpacity = li => isSelected(li) ? 1.0 : (hasSelection ? 0.28 : 0.88);

  renderOrder.forEach((li, renderIdx) => {
    const label = labels[li];
    const count = counts[li];
    const barW = count > 0 ? Math.max(2, (count / maxCount) * barMaxW) : 0;
    const y = TOP_H + renderIdx * rowH;
    const [r, g, b] = getPaletteRgb(li);
    const barColor = `rgba(${r},${g},${b},0.75)`;
    const sel = isSelected(li);

    const g_row = svg.append('g').attr('transform', `translate(0,${y})`);

    // Label (clipped)
    g_row.append('text')
      .attr('clip-path', 'url(#rpc-label-clip)')
      .attr('x', labelW - 6).attr('y', rowH / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('font-size', '11px').attr('font-family', FONT)
      .attr('fill', sel ? TEXT_BODY : TEXT_DIM)
      .attr('font-weight', sel ? '600' : '400')
      .text(shortenLabel(label));
    labelTexts.set(li, g_row.select('text'));

    // Bar
    if (barW > 0) {
      const bar = g_row.append('rect')
        .attr('class', 'rpc-bar')
        .attr('x', labelW + barGap).attr('y', 2)
        .attr('width', barW).attr('height', rowH - 4)
        .attr('rx', 2)
        .attr('fill', barColor)
        .attr('opacity', baseOpacity(li))
        .attr('stroke', sel ? 'rgba(255,255,255,0.65)' : 'none')
        .attr('stroke-width', 1.5);
      barRects.set(li, bar);
    }

    // Count
    if (count > 0) {
      g_row.append('text')
        .attr('x', labelW + barGap + barW + 4).attr('y', rowH / 2)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px').attr('font-family', FONT)
        .attr('fill', TEXT_DIM)
        .text(count >= 1000 ? d3.format('.2s')(count) : count);
    }

    // Hit rect
    if (count > 0) {
      g_row.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', W).attr('height', rowH)
        .attr('fill', 'transparent')
        .style('cursor', (isSelected(li) || !hasSelection) ? 'pointer' : 'default')
        .on('mouseenter', () => {
          if (appMode === 'story') {
            // 1 selection: disable hover entirely
            if (hasSelection && (highlightLabelSet == null || highlightLabelSet.size < 2)) return;
            const hoverModes = Array.from(categoryModes);
            for (let i = 0; i < MAX_LABELS; i++) {
              if (hoverModes[i] === 0 && i !== li) hoverModes[i] = 2;
            }
            uniforms.uCategoryModes.value = hoverModes;
            const hasMultiSelection = highlightLabelSet != null && highlightLabelSet.size >= 2;
            const liHighlighted = !hasMultiSelection || isSelected(li);
            barRects.forEach((rect, idx) => {
              if (!hasMultiSelection) {
                rect.attr('opacity', idx === li ? 1.0 : 0.28);
              } else {
                if (idx === li && isSelected(li)) rect.attr('opacity', 1.0);
                else if (isSelected(idx))         rect.attr('opacity', 0.28);
                else                              rect.attr('opacity', 0.10);
              }
            });
            labelTexts.forEach((text, idx) => {
              const bold = idx === li && liHighlighted;
              text.attr('fill', bold ? TEXT_BODY : TEXT_DIM);
              text.attr('font-weight', bold ? '600' : '400');
            });
            return;
          }
          if (appMode !== 'explore') return;
          // L2 locked → no hover anywhere
          if (filterLevel >= 2) return;
          // L1 selected and viewing L1 chart → bars are locked, no hover
          if (filterLevel === 1 && isL1Chart) return;

          hoveredChartFamilyIdx = familyIdx;
          hoveredChartLi = li;

          // Bar + label visuals
          barRects.forEach((rect, idx) => {
            if (idx === li) rect.attr('opacity', 1.0).attr('stroke', 'none');
            else if (isSelected(idx)) rect.attr('opacity', 1.0);
            else rect.attr('opacity', 0.55);
          });
          labelTexts.forEach((text, idx) => {
            text.attr('fill', idx === li ? TEXT_BODY : TEXT_DIM);
          });

          // 3D highlight
          if (filterLevel === 0) {
            // L0: dim non-hovered categories
            uniforms.uSecFamilyIdx.value     = activeFamilyIdx;
            uniforms.uSecLabelIdx.value      = li;
            uniforms.uSecDimFactor.value     = 0.45;
            uniforms.uSecOutlineFactor.value = 0.0;
          } else {
            // L1 on non-L1 chart: preview L2 intersection within L1 selection
            uniforms.uSecFamilyIdx.value     = familyIdx;
            uniforms.uSecLabelIdx.value      = li;
            uniforms.uSecDimFactor.value     = 0.80;
            uniforms.uSecOutlineFactor.value = 0.80;
          }

          showPhantomChip(familyIdx, li);
        })
        .on('mouseleave', () => {
          if (appMode === 'story') {
            uniforms.uCategoryModes.value = Array.from(categoryModes);
            barRects.forEach((rect, idx) => {
              rect.attr('opacity', baseOpacity(idx));
            });
            labelTexts.forEach((text, idx) => {
              text.attr('fill', (isSelected(idx) || !hasSelection) ? TEXT_BODY : TEXT_DIM);
              text.attr('font-weight', isSelected(idx) ? '600' : '400');
            });
            return;
          }
          if (appMode !== 'explore') return;
          if (filterLevel >= 2) return;
          if (filterLevel === 1 && isL1Chart) return;

          barRects.forEach((rect, idx) => {
            rect.attr('opacity', baseOpacity(idx))
                .attr('stroke', isSelected(idx) ? 'rgba(255,255,255,0.65)' : 'none');
          });
          labelTexts.forEach((text, idx) => {
            text.attr('fill', (isSelected(idx) || !hasSelection) ? TEXT_BODY : TEXT_DIM);
          });

          // Clear secondary uniforms — locked L2 state is restored by transitionToLevel2 on click
          hoveredChartFamilyIdx = -1;
          hoveredChartLi = -1;
          uniforms.uSecFamilyIdx.value     = -1;
          uniforms.uSecDimFactor.value     = 1.0;
          uniforms.uSecOutlineFactor.value = 0.0;
          clearPhantomChip();
        })
        .on('click', () => {
          if (appMode !== 'explore') return;
          if (isL1Chart) {
            if (filterLevel >= 1) {
              // Deselecting — reset first, then immediately re-apply hover dim so
              // the next render never sees a fully-bright frame
              resetToLevel0();
              if (hoveredChartFamilyIdx >= 0 && hoveredChartLi >= 0) {
                uniforms.uSecFamilyIdx.value     = hoveredChartFamilyIdx;
                uniforms.uSecLabelIdx.value      = hoveredChartLi;
                uniforms.uSecDimFactor.value     = 0.45;
                uniforms.uSecOutlineFactor.value = 0.0;
              }
            } else {
              // Selecting — clear hover preview before locking in
              uniforms.uSecFamilyIdx.value     = -1;
              uniforms.uSecDimFactor.value     = 1.0;
              uniforms.uSecOutlineFactor.value = 0.0;
              transitionToLevel1(familyIdx, li);
            }
          } else {
            uniforms.uSecFamilyIdx.value     = -1;
            uniforms.uSecDimFactor.value     = 1.0;
            uniforms.uSecOutlineFactor.value = 0.0;
            if (filterLevel <= 1) transitionToLevel2(familyIdx, li);
            else resetToLevel1();
          }
        });
    }
  });

  // Column headers
  const barStart = labelW + barGap;
  svg.append('text')
    .attr('x', labelW - 6).attr('y', TOP_H - 6)
    .attr('text-anchor', 'end')
    .attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.40)')
    .text('↓ Cluster');
  svg.append('text')
    .attr('x', labelW + (barGap - 6) / 2).attr('y', TOP_H - 6)
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.25)')
    .text('|');
  svg.append('text')
    .attr('x', barStart).attr('y', TOP_H - 6)
    .attr('text-anchor', 'start')
    .attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.40)')
    .text('Recipe Count →');

  // Re-apply hover at the start of the next frame so it lands after the fresh render paints
  if (hoveredChartFamilyIdx === familyIdx && hoveredChartLi >= 0) {
    const capturedRect = barRects.get(hoveredChartLi);
    requestAnimationFrame(() => {
      if (capturedRect && container.contains(capturedRect.node())) {
        capturedRect.node().dispatchEvent(new MouseEvent('mouseenter', { bubbles: false }));
      }
    });
  }
}

// ── Filter state machine ──────────────────────────────────────────────────────
function transitionToLevel1(familyIdx, labelIdx) {
  filterLevel = 1;
  level1LabelIdx = labelIdx;
  level2FamilyIdx = -1; // no L2 family selected yet
  setHighlightLabel(labelIdx);
  updateTabVisualState();
  // Pulse non-L1 tabs to hint at L2 interaction
  document.querySelectorAll('.cat-tab').forEach((btn, i) => {
    if (i === familyIdx) return;
    btn.classList.add('tab-pulse');
    btn.addEventListener('animationend', () => btn.classList.remove('tab-pulse'), { once: true });
  });
  // Stay on the L1 family chart — user can switch tabs manually for L2
  renderRightPanelChart(familyIdx);
  renderFilterChips();
}

function transitionToLevel2(familyIdx, labelIdx) {
  filterLevel = 2;
  level2FamilyIdx = familyIdx;
  level2LabelIdx = labelIdx;
  // Apply secondary filter via shader uniforms (locked at 0.35 dim)
  if (!savedIntersectionState) savedIntersectionState = { alpha: alphaCache.slice() };
  uniforms.uSecFamilyIdx.value     = familyIdx;
  uniforms.uSecLabelIdx.value      = labelIdx;
  uniforms.uSecDimFactor.value     = 0.80;
  uniforms.uSecOutlineFactor.value = 1.0;
  // Restrict raycasting to intersection
  const secData = getCategoryFamilyData(familyIdx);
  for (let i = 0; i < N; i++) {
    alphaCache[i] = savedIntersectionState.alpha[i] > 0 && secData[i] === labelIdx ? 1.0 : 0.0;
  }
  updateTabVisualState();
  renderRightPanelChart(familyIdx, computeFilteredCounts(familyIdx, activeFamilyIdx, level1LabelIdx));
  renderFilterChips();
}

function resetToLevel0() {
  filterLevel = 0;
  level1LabelIdx = -1;
  level2FamilyIdx = -1;
  level2LabelIdx = -1;
  restoreIntersectionHighlight();
  uniforms.uSecFamilyIdx.value     = -1;
  uniforms.uSecDimFactor.value     = 1.0;
  uniforms.uSecOutlineFactor.value = 0.0;
  setHighlightLabel(-1);
  updateTabVisualState();
  document.querySelectorAll('.cat-tab').forEach((btn, i) => {
    btn.classList.toggle('active', i === activeFamilyIdx);
  });
  renderRightPanelChart(activeFamilyIdx);
  renderFilterChips();
}

function resetToLevel1() {
  filterLevel = 1;
  level2LabelIdx = -1;
  uniforms.uSecFamilyIdx.value     = -1;
  uniforms.uSecDimFactor.value     = 1.0;
  uniforms.uSecOutlineFactor.value = 0.0;
  if (savedIntersectionState) {
    alphaCache.set(savedIntersectionState.alpha);
    savedIntersectionState = null;
  }
  updateTabVisualState();
  // Render L2 family chart if one was chosen, otherwise fall back to L1 family chart
  if (level2FamilyIdx >= 0 && level2FamilyIdx !== activeFamilyIdx) {
    const filteredCounts = computeFilteredCounts(level2FamilyIdx, activeFamilyIdx, level1LabelIdx);
    renderRightPanelChart(level2FamilyIdx, filteredCounts);
  } else {
    level2FamilyIdx = -1;
    renderRightPanelChart(activeFamilyIdx);
  }
  renderFilterChips();
}

// ── Filter chip rendering ─────────────────────────────────────────────────────
function makeChip(familyIdx, labelIdx, isPhantom, onRemove) {
  const [r, g, b] = getPaletteRgb(labelIdx);
  const label = shortenLabel(meta.categories[familyIdx].labels[labelIdx]);
  const chip = document.createElement('span');
  chip.className = 'filter-chip' + (isPhantom ? ' filter-chip-phantom' : '');
  chip.style.background = `rgba(${r},${g},${b},0.35)`;
  chip.style.borderColor = `rgba(${r},${g},${b},0.80)`;
  chip.textContent = label;
  if (onRemove && !isPhantom) {
    chip.style.cursor = 'pointer';
    chip.addEventListener('click', onRemove);
    const x = document.createElement('button');
    x.className = 'filter-chip-remove';
    x.textContent = '×';
    x.addEventListener('click', e => { e.stopPropagation(); onRemove(); });
    chip.appendChild(x);
  }
  return chip;
}


function renderFilterChips() {
  const bar = document.getElementById('breadcrumb-bar');
  bar.innerHTML = '';
  if (filterLevel === 0) {
    const txt = document.createElement('span');
    txt.className = 'flavor-text';
    txt.textContent = 'Hover to preview · Click to filter · Up to two categories';
    bar.appendChild(txt);
    return;
  }
  const txt = s => { const sp = document.createElement('span'); sp.className = 'flavor-text'; sp.textContent = s; return sp; };
  if (filterLevel === 1) {
    bar.appendChild(txt('Showing'));
    bar.appendChild(makeChip(activeFamilyIdx, level1LabelIdx, false, resetToLevel0));
    bar.appendChild(txt('with something else?'));
  } else {
    bar.appendChild(txt('Showing'));
    bar.appendChild(makeChip(activeFamilyIdx, level1LabelIdx, false, resetToLevel0));
    bar.appendChild(txt('with'));
    bar.appendChild(makeChip(level2FamilyIdx, level2LabelIdx, false, resetToLevel1));
  }
}

function showPhantomChip(familyIdx, labelIdx) {
  const bar = document.getElementById('breadcrumb-bar');
  const txt = s => { const sp = document.createElement('span'); sp.className = 'flavor-text phantom-sep'; sp.textContent = s; return sp; };

  if (filterLevel === 0) {
    bar.innerHTML = '';
    bar.appendChild(txt('Showing'));
    bar.appendChild(makeChip(familyIdx, labelIdx, true, null));
  } else if (filterLevel === 1) {
    // Rebuild as: Showing [pill1] and [phantom pill2]
    bar.innerHTML = '';
    bar.appendChild(txt('Showing'));
    bar.appendChild(makeChip(activeFamilyIdx, level1LabelIdx, false, resetToLevel0));
    bar.appendChild(txt('with'));
    bar.appendChild(makeChip(familyIdx, labelIdx, true, null));
  } else {
    // L2 locked — append phantom after existing chips
    bar.querySelectorAll('.filter-chip-phantom, .phantom-sep').forEach(e => e.remove());
    bar.appendChild(txt('with'));
    bar.appendChild(makeChip(familyIdx, labelIdx, true, null));
  }
}

function clearPhantomChip() {
  if (filterLevel >= 1) {
    // showPhantomChip cleared the bar entirely — restore full state
    renderFilterChips();
    return;
  }
  const bar = document.getElementById('breadcrumb-bar');
  bar.querySelectorAll('.filter-chip-phantom, .phantom-sep').forEach(e => e.remove());
  if (!bar.querySelector('.filter-chip')) {
    const txt = document.createElement('span');
    txt.className = 'flavor-text';
    txt.textContent = 'Hover to preview · Click to filter · Up to two categories';
    bar.appendChild(txt);
  }
}

// ── Left panel: explore mode ──────────────────────────────────────────────────
function randomizeRecipe() {
  if (!N || !posArr) return;
  const idx = Math.floor(Math.random() * N);
  setLockedIdx(idx);

  const i3   = idx * 3;
  const pt   = new THREE.Vector3(posArr[i3], posArr[i3 + 1], posArr[i3 + 2]);
  const dist = dataExtent * 0.25;

  // Camera sits outside the dataset looking inward toward the point
  let outward = pt.clone().sub(dataCenterVec);
  if (outward.length() < 0.001) outward.set(0, 0, 1);
  outward.normalize();

  animateCameraToPosition(
    [pt.x + outward.x * dist, pt.y + outward.y * dist, pt.z + outward.z * dist],
    [pt.x, pt.y, pt.z]
  );

  showHoverTip(idx);
  showRecipeInfo(idx);
}

function positionRandomize(anchorEl) {
  const btn = document.getElementById('btn-randomize');
  if (!btn || btn.style.display === 'none') return;
  requestAnimationFrame(() => {
    const rect = anchorEl.getBoundingClientRect();
    btn.style.top = `${rect.bottom + 6}px`;
    btn.style.width = `${rect.width}px`;
  });
}

function showExploreDefault() {
  document.getElementById('left-panel').style.display = 'none';
  document.getElementById('explore-default').style.display = 'flex';
  document.getElementById('recipe-chart-btns').style.display = 'none';
  hideLeftPanelChart();
  positionRandomize(document.getElementById('explore-default'));
}

function showRecipeInfo(idx) {
  document.getElementById('explore-default').style.display = 'none';
  document.getElementById('left-panel').style.display = 'flex';
  positionRandomize(document.getElementById('left-panel'));
  document.getElementById('explore-content').style.display = 'block';
  document.getElementById('explore-recipe').style.display = 'block';
  hideLeftPanelChart();

  // Render "Show X" buttons after metrics load
  const btnsFooter = document.getElementById('recipe-chart-btns');
  btnsFooter.innerHTML = '';
  btnsFooter.style.display = 'flex';

  // Show placeholder immediately, fill in async
  document.getElementById('recipe-name').textContent = `Recipe #${recipeIds[idx]}`;
  document.getElementById('recipe-tags').innerHTML = '';
  document.getElementById('recipe-stats').textContent = 'Loading…';
  document.getElementById('recipe-ingredients').textContent = '';
  document.getElementById('recipe-description').textContent = '';
  document.getElementById('recipe-link').href = '';

  getRecipeData(idx).then(r => {
    if (!r) return;
    document.getElementById('recipe-name').textContent = r.name || `Recipe #${recipeIds[idx]}`;

    // Tags from active categories
    const tagsEl = document.getElementById('recipe-tags');
    tagsEl.innerHTML = '';
    meta.categories.forEach((cat, fi) => {
      const famData = getCategoryFamilyData(fi);
      const labelIdx = famData[idx];
      const label = cat.labels[labelIdx] ?? '';
      if (!label) return;
      const [r, g, b] = getPaletteRgb(labelIdx);
      const tr = Math.round(r * 0.45 + 255 * 0.55);
      const tg = Math.round(g * 0.45 + 255 * 0.55);
      const tb = Math.round(b * 0.45 + 255 * 0.55);
      const tag = document.createElement('span');
      tag.className = 'recipe-tag';
      tag.textContent = label;
      tag.style.background = `rgba(${r},${g},${b},0.35)`;
      tag.style.borderColor = `rgba(${r},${g},${b},0.80)`;
      tag.style.color = `rgb(${tr},${tg},${tb})`;
      tagsEl.appendChild(tag);
    });

    const stats = [];
    if (r.avg_rating != null) stats.push(`★ ${r.avg_rating.toFixed(1)} (${r.n_ratings} ratings)`);
    if (r.minutes) stats.push(`${r.minutes} min`);
    if (r.n_steps) stats.push(`${r.n_steps} steps`);
    if (r.n_ingredients) stats.push(`${r.n_ingredients} ingredients`);
    if (r.submitted) stats.push(r.submitted.slice(0, 7));
    document.getElementById('recipe-stats').textContent = stats.join(' · ');

    if (r.ingredients?.length) {
      const shown = r.ingredients.slice(0, 12);
      document.getElementById('recipe-ingredients').textContent =
        shown.join(', ') + (r.ingredients.length > 12 ? ` +${r.ingredients.length - 12} more` : '');
    }

    if (r.description) {
      const desc = r.description.trim();
      document.getElementById('recipe-description').textContent =
        desc.length > 300 ? desc.slice(0, 300) + '…' : desc;
    }

    const recipeName = r.name || `Recipe #${recipeIds[idx]}`;
    document.getElementById('recipe-link').href =
      `https://www.google.com/search?q=${encodeURIComponent('food.com ' + recipeName)}`;

    // "Show X" chart buttons — loaded after metrics resolve
    renderRecipeChartButtons(recipeIds[idx]);
  });
}

async function renderRecipeChartButtons(recipeId) {
  const btnsEl = document.getElementById('recipe-chart-btns');
  btnsEl.innerHTML = '';
  const shard = String(recipeId).slice(-2).padStart(2, '0');
  if (!recipeMetricsCache.has(shard)) {
    try {
      const ab = await fetch(`${DATA}recipe_metrics/${shard}.json.gz`).then(r => r.ok ? r.arrayBuffer() : null);
      recipeMetricsCache.set(shard, ab ? await decompressJson(ab) : {});
    } catch { recipeMetricsCache.set(shard, {}); }
  }
  const metrics = recipeMetricsCache.get(shard)?.[String(recipeId)];
  const hasRatings = !!(metrics?.n_ratings);
  const hasReviews = !!(metrics?.n_reviews);
  [
    { label: 'Show Ratings', tabId: 'ratings', enabled: hasRatings },
    { label: 'Show Reviews / Year', tabId: 'reviews', enabled: hasReviews },
  ].forEach(({ label, tabId, enabled }) => {
    const btn = document.createElement('button');
    btn.className = 'recipe-chart-btn' + (enabled ? '' : ' disabled');
    btn.textContent = label;
    if (enabled) {
      btn.addEventListener('click', () => showRecipeChartInPanel(recipeId, tabId, label));
    }
    btnsEl.appendChild(btn);
  });
}


// ── Left panel chart view ─────────────────────────────────────────────────────
function showLeftPanelChart(title, renderFn) {
  document.getElementById('recipe-chart-btns').style.display = 'none';
  document.getElementById('explore-content').style.display = 'none';
  const view = document.getElementById('left-panel-chart-view');
  view.style.display = 'flex';
  document.getElementById('left-panel-chart-title').textContent = title;
  const body = document.getElementById('left-panel-chart-body');
  body.innerHTML = '';
  requestAnimationFrame(() => renderFn(body));
}

function hideLeftPanelChart() {
  document.getElementById('left-panel-chart-view').style.display = 'none';
  document.getElementById('left-panel-chart-body').innerHTML = '';
  document.getElementById('recipe-chart-btns').style.display = 'none';
}

async function showRecipeChartInPanel(recipeId, tabId, title) {
  const shard = String(recipeId).slice(-2).padStart(2, '0');
  const metrics = recipeMetricsCache.get(shard)?.[String(recipeId)];
  if (!metrics) return;

  showLeftPanelChart(title, body => {
    if (tabId === 'ratings') {
      const stat = document.createElement('div');
      stat.className = 'chart-stat-header';
      stat.innerHTML =
        `<span class="chart-stat-avg">★ ${metrics.avg_rating?.toFixed(1) ?? '—'}</span>` +
        `<span class="chart-stat-meta"> · ${(metrics.n_ratings ?? 0).toLocaleString()} ratings · ${(metrics.n_reviews ?? 0).toLocaleString()} reviews</span>`;
      body.appendChild(stat);

      const chartEl = document.createElement('div');
      chartEl.style.cssText = 'flex:1;min-height:0;overflow:hidden;';
      body.appendChild(chartEl);

      const noRating = Math.max(0, (metrics.n_reviews ?? 0) - (metrics.n_ratings ?? 0));
      const labels = ['5', '4', '3', '2', '1', 'No Rating'];
      const counts = [metrics.count_5, metrics.count_4, metrics.count_3, metrics.count_2, metrics.count_1, noRating];
      renderRecipeHorizontalBars(chartEl, labels, counts, 'rgba(212,175,55,0.80)', 'Rating', 0);

    } else if (tabId === 'reviews') {
      const labels = [], counts = [];
      for (let y = REVIEWS_YEAR_MIN; y <= REVIEWS_YEAR_MAX; y++) {
        labels.push(String(y));
        counts.push(metrics.n_per_year?.[y] ?? 0);
      }
      const colors = resolveDistColors('submitted', labels);
      renderRecipeHorizontalBars(body, labels, counts, colors, 'Year', 20);
    }
  });
}

function renderRecipeHorizontalBars(container, labels, counts, barColor, colLabel, vertPad = 0) {
  const W = container.offsetWidth || 280;
  const TOP_H = 18;
  const containerH = Math.max(0, (container.clientHeight || 0) - vertPad - TOP_H);
  const MIN_ROW_H = 18;
  const rowH = containerH > 0 ? Math.max(MIN_ROW_H, containerH / labels.length) : MIN_ROW_H;

  const _ctx = document.createElement('canvas').getContext('2d');
  _ctx.font = '11px "Source Serif 4", serif';
  const maxTextW = Math.max(...labels.map(l => _ctx.measureText(l).width));
  const labelW = Math.max(20, Math.ceil(maxTextW) + 12);
  const barGap = 6;
  const countPad = 40;
  const barMaxW = W - labelW - barGap - countPad;
  const maxCount = Math.max(...counts, 1);
  const H = rowH * labels.length;

  container.innerHTML = '';
  const svg = d3.select(container).append('svg')
    .attr('width', W).attr('height', TOP_H + H)
    .style('display', 'block').style('overflow', 'visible');

  const FONT = '"Source Serif 4", serif';
  const TEXT_DIM = 'rgba(255,255,255,0.72)';

  svg.append('defs').append('clipPath').attr('id', 'rhb-label-clip')
    .append('rect').attr('x', 0).attr('y', 0).attr('width', labelW - 2).attr('height', TOP_H + H);

  // Column headers
  svg.append('text')
    .attr('x', labelW - 6).attr('y', TOP_H - 6)
    .attr('text-anchor', 'end').attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.40)').text(`↓ ${colLabel}`);
  svg.append('text')
    .attr('x', labelW + (barGap - 6) / 2).attr('y', TOP_H - 6)
    .attr('text-anchor', 'middle').attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.25)').text('|');
  svg.append('text')
    .attr('x', labelW + barGap).attr('y', TOP_H - 6)
    .attr('text-anchor', 'start').attr('font-size', '10px').attr('font-family', FONT)
    .attr('fill', 'rgba(255,255,255,0.40)').text('Count →');

  labels.forEach((label, i) => {
    const count = counts[i];
    const barW = count > 0 ? Math.max(2, (count / maxCount) * barMaxW) : 0;
    const y = TOP_H + i * rowH;
    const g = svg.append('g').attr('transform', `translate(0,${y})`);

    g.append('text')
      .attr('clip-path', 'url(#rhb-label-clip)')
      .attr('x', labelW - 6).attr('y', rowH / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('font-size', '11px').attr('font-family', FONT)
      .attr('fill', TEXT_DIM).text(label);

    if (barW > 0) {
      const fill = Array.isArray(barColor) ? barColor[i] : barColor;
      g.append('rect')
        .attr('x', labelW + barGap).attr('y', 2)
        .attr('width', barW).attr('height', rowH - 4)
        .attr('rx', 2).attr('fill', fill).attr('opacity', 0.88);
    }

    if (count > 0) {
      g.append('text')
        .attr('x', labelW + barGap + barW + 4).attr('y', rowH / 2)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px').attr('font-family', FONT)
        .attr('fill', 'rgba(255,255,255,0.52)')
        .text(count >= 1000 ? d3.format('.2s')(count) : count);
    }
  });
}

async function loadAndRenderChart(config, container) {
  if (container === null) return; // panel charts removed
  try {
    const resp = await fetch(config.dataFile);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const fileData = await resp.json();
    // story.json block fields override data file defaults
    const merged = { ...fileData, ...config };
    // Resolve palette colors for pre-binned histograms
    if (merged.labels && merged.counts && config.categoryFamily && meta) {
      const fam = meta.categories.find(c => c.name === config.categoryFamily);
      if (fam) {
        merged.colors = merged.labels.map(lbl => {
          const li = fam.labels.indexOf(lbl);
          if (li < 0) return '#4e79a7';
          const [r, g, b] = getPaletteRgb(li);
          return `rgb(${r},${g},${b})`;
        });
      }
    }
    renderChart(target, merged, fileData.data);
  } catch (e) {
    console.warn('Chart load failed:', e);
  }
}

function renderChart(container, config, data) {
  const type = config.chartType;
  const isBinned = !!(config.labels && config.counts);
  if (!type || (!data?.length && !isBinned)) return;

  const W = container.clientWidth > 10 ? container.clientWidth : 266;
  const isInline = config.placement === 'inline';
  const H = isInline ? 130 : (container.clientHeight > 60 ? container.clientHeight : 190);
  const margin = type === 'beeswarm'
    ? { top: 8, right: 10, bottom: 26, left: 10 }
    : isBinned
      ? { top: 26, right: 14, bottom: 52, left: 24 }
      : { top: 8, right: 14, bottom: 26, left: 34 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const AXIS_COLOR = 'rgba(255,255,255,0.25)';
  const BAR_COLOR = '#4e79a7';
  const TEXT_COLOR = 'rgba(255,255,255,0.72)';

  container.innerHTML = '';
  const svg = d3.select(container)
    .append('svg')
    .attr('width', W).attr('height', H)
    .style('display', 'block');

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const FONT = '"Source Serif 4", serif';
  const styleAxis = ax => ax
    .call(a => a.select('.domain').attr('stroke', AXIS_COLOR))
    .call(a => a.selectAll('.tick line').attr('stroke', AXIS_COLOR))
    .call(a => a.selectAll('text').attr('fill', TEXT_COLOR).attr('font-size', '11px').attr('font-family', FONT));

  if (type === 'histogram') {
    if (isBinned) {
      const labels = config.labels;
      const counts = config.counts;
      const x = d3.scaleBand().domain(labels).range([0, innerW]).paddingInner(0.12).paddingOuter(0.5);
      const yMax = config.yMax ?? d3.max(counts);
      const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).tickSizeOuter(0))
        .call(ax => {
          ax.select('.domain').attr('stroke', AXIS_COLOR);
          ax.selectAll('.tick line').attr('stroke', AXIS_COLOR);
          ax.selectAll('text')
            .attr('fill', TEXT_COLOR).attr('font-size', '11px')
            .attr('transform', 'rotate(-40)')
            .attr('text-anchor', 'end')
            .attr('dx', '-0.4em').attr('dy', '0.15em');
        });
      styleAxis(g.append('g').call(d3.axisLeft(y).tickValues([]))
        .call(a => a.selectAll('.tick').remove()));

      const bars = g.selectAll('.bar').data(counts).join('rect')
        .attr('class', 'bar')
        .attr('x', (_, i) => x(labels[i]))
        .attr('width', x.bandwidth())
        .attr('y', d => y(d))
        .attr('height', d => innerH - y(d))
        .attr('fill', (_, i) => config.colors?.[i] ?? BAR_COLOR)
        .attr('opacity', 0.88)
        .attr('stroke', 'none')
        .attr('stroke-width', 1.5);

      const fmtFull = d => d.toLocaleString();
      const fmtAbbr = d => d >= 1000 ? d3.format('.1s')(d) : String(d);
      const labelFits = (d, bw) => fmtFull(d).length * 5.4 <= bw;
      const abbrFits = (d, bw) => fmtAbbr(d).length * 5.4 <= bw;
      const LABEL_Y = 4;
      g.selectAll('.bar-count').data(counts).join('text')
        .attr('class', 'bar-count')
        .attr('pointer-events', 'none')
        .attr('font-family', FONT)
        .attr('font-size', '9px')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'hanging')
        .attr('fill', TEXT_COLOR)
        .attr('x', (_, i) => x(labels[i]) + x.bandwidth() / 2)
        .attr('y', LABEL_Y)
        .attr('opacity', d => {
          if (!d) return 0;
          const bw = x.bandwidth();
          return abbrFits(d, bw) ? 1 : 0;
        })
        .text(d => {
          if (!d) return '';
          const bw = x.bandwidth();
          return labelFits(d, bw) ? fmtFull(d) : abbrFits(d, bw) ? fmtAbbr(d) : '';
        });

      if (config.onBarEnter) {
        let lockedLabel = null;

        const setLocked = lbl => {
          lockedLabel = lbl;
          bars.attr('opacity', (_, i) => labels[i] === lbl ? 1.0 : 0.15)
            .attr('stroke', (_, i) => labels[i] === lbl ? 'rgba(255,255,255,0.72)' : 'none');
          config.onBarEnter(lbl);
        };

        const clearLocked = () => {
          lockedLabel = null;
          bars.attr('opacity', 0.88).attr('stroke', 'none');
          config.onBarLeave?.();
        };

        const halfGap = (x.step() - x.bandwidth()) / 2;
        const activeLabels = labels.filter((_, i) => counts[i] > 0);
        g.selectAll('.bar-hit').data(activeLabels).join('rect')
          .attr('class', 'bar-hit')
          .attr('x', lbl => x(lbl) - halfGap)
          .attr('width', x.step())
          .attr('y', 0)
          .attr('height', innerH)
          .attr('fill', 'transparent')
          .style('cursor', 'pointer')
          .on('mouseover', (_, lbl) => {
            if (lockedLabel) return;
            bars.attr('opacity', (_, i) => labels[i] === lbl ? 1.0 : 0.28);
            config.onBarEnter(lbl);
          })
          .on('mouseleave', () => {
            if (lockedLabel) return;
            bars.attr('opacity', 0.88);
            config.onBarLeave?.();
          })
          .on('click', (_, lbl) => {
            if (lockedLabel === lbl) {
              // Same bar clicked: deselect, resume hover on this bar
              lockedLabel = null;
              bars.attr('opacity', (_, i) => labels[i] === lbl ? 1.0 : 0.28)
                .attr('stroke', 'none');
              config.onBarEnter(lbl);
            } else if (lockedLabel) {
              // Different bar clicked while locked: just deselect
              clearLocked();
            } else {
              // Nothing locked: lock this bar
              setLocked(lbl);
            }
          });
      }

    } else {
      const xMin = config.xMin ?? d3.min(data);
      const xMax = config.xMax ?? d3.max(data);
      const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
      const bins = d3.bin().domain([xMin, xMax]).thresholds(config.bins ?? 20)(data);
      const yMax = config.yMax ?? d3.max(bins, b => b.length);
      const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      styleAxis(g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)));
      styleAxis(g.append('g').call(d3.axisLeft(y).tickValues(y.ticks(4).filter(Number.isInteger)).tickFormat(d => d >= 1000 ? d3.format('.0s')(d) : d)));

      g.selectAll('rect').data(bins).join('rect')
        .attr('x', b => x(b.x0) + 1)
        .attr('width', b => Math.max(0, x(b.x1) - x(b.x0) - 1))
        .attr('y', b => y(b.length))
        .attr('height', b => innerH - y(b.length))
        .attr('fill', BAR_COLOR).attr('opacity', 0.75);
    }

    if (config.yLabel) {
      svg.append('text')
        .attr('x', margin.left)
        .attr('y', margin.top - 7)
        .attr('fill', TEXT_COLOR)
        .attr('font-size', '11px')
        .attr('font-family', FONT)
        .attr('text-anchor', 'start')
        .text('↑ ' + config.yLabel);
    }

  } else if (type === 'beeswarm') {
    const xMin = config.xMin ?? d3.min(data);
    const xMax = config.xMax ?? d3.max(data);
    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
    const r = config.radius ?? 3;
    const midY = innerH / 2;

    styleAxis(g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)));

    const sample = data.length > 600
      ? Array.from({ length: 600 }, (_, i) => data[Math.floor(i * data.length / 600)])
      : data;
    const nodes = sample.map(v => ({ tx: x(v), x: x(v), y: midY }));
    const sim = d3.forceSimulation(nodes)
      .force('x', d3.forceX(d => d.tx).strength(0.9))
      .force('y', d3.forceY(midY).strength(0.05))
      .force('collide', d3.forceCollide(r + 0.8))
      .stop();
    for (let i = 0; i < 150; i++) sim.tick();

    g.selectAll('circle').data(nodes).join('circle')
      .attr('cx', d => Math.max(r, Math.min(innerW - r, d.x)))
      .attr('cy', d => Math.max(r, Math.min(innerH - r, d.y)))
      .attr('r', r)
      .attr('fill', BAR_COLOR).attr('opacity', 0.60);

  } else if (type === 'line') {
    const xVals = data.map(d => d[0]);
    const yVals = data.map(d => d[1]);
    const x = d3.scaleLinear()
      .domain([config.xMin ?? d3.min(xVals), config.xMax ?? d3.max(xVals)])
      .range([0, innerW]);
    const y = d3.scaleLinear()
      .domain([config.yMin ?? d3.min(yVals), config.yMax ?? d3.max(yVals)])
      .range([innerH, 0]);

    styleAxis(g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)));
    styleAxis(g.append('g').call(d3.axisLeft(y).tickValues(y.ticks(4).filter(Number.isInteger)).tickFormat(d => d >= 1000 ? d3.format('.0s')(d) : d)));

    g.append('path').datum(data)
      .attr('fill', 'none').attr('stroke', BAR_COLOR).attr('stroke-width', 1.5)
      .attr('d', d3.line().x(d => x(d[0])).y(d => y(d[1])));

    if (config.yLabel) {
      svg.append('text')
        .attr('x', margin.left)
        .attr('y', margin.top - 7)
        .attr('fill', TEXT_COLOR)
        .attr('font-size', '11px')
        .attr('font-family', FONT)
        .attr('text-anchor', 'start')
        .text('↑ ' + config.yLabel);
    }

  } else if (type === 'scatter') {
    const xVals = data.map(d => d[0]);
    const yVals = data.map(d => d[1]);
    const x = d3.scaleLinear()
      .domain([config.xMin ?? d3.min(xVals), config.xMax ?? d3.max(xVals)])
      .range([0, innerW]);
    const y = d3.scaleLinear()
      .domain([config.yMin ?? d3.min(yVals), config.yMax ?? d3.max(yVals)])
      .range([innerH, 0]);

    styleAxis(g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)));
    styleAxis(g.append('g').call(d3.axisLeft(y).tickValues(y.ticks(4).filter(Number.isInteger)).tickFormat(d => d >= 1000 ? d3.format('.0s')(d) : d)));

    g.selectAll('circle').data(data).join('circle')
      .attr('cx', d => x(d[0])).attr('cy', d => y(d[1]))
      .attr('r', 2.5).attr('fill', BAR_COLOR).attr('opacity', 0.55);

    if (config.yLabel) {
      svg.append('text')
        .attr('x', margin.left)
        .attr('y', margin.top - 7)
        .attr('fill', TEXT_COLOR)
        .attr('font-size', '11px')
        .attr('font-family', FONT)
        .attr('text-anchor', 'start')
        .text('↑ ' + config.yLabel);
    }
  }

  if (config.xLabel) {
    svg.append('text')
      .attr('x', margin.left + innerW / 2).attr('y', H - 2)
      .attr('text-anchor', 'middle').attr('fill', TEXT_COLOR).attr('font-size', '11px').attr('font-family', FONT)
      .text(config.xLabel);
  }
}


function resolveDistColors(familyName, labels) {
  const fam = meta?.categories.find(c => c.name === familyName);
  if (!fam) return labels.map(() => '#4e79a7');
  return labels.map(lbl => {
    const li = fam.labels.indexOf(lbl);
    if (li < 0) return '#4e79a7';
    const [r, g, b] = getPaletteRgb(li);
    return `rgb(${r},${g},${b})`;
  });
}


// ── Left panel: story mode ────────────────────────────────────────────────────
function applyStep(step) {
  // Clear any secondary-dim state left over from hover or explore mode
  uniforms.uSecFamilyIdx.value     = -1;
  uniforms.uSecDimFactor.value     = 1.0;
  uniforms.uSecOutlineFactor.value = 0.0;

  // Clear locked recipe from previous step
  setLockedIdx(-1);
  hideHoverTip();

  // Switch category family
  if (step.colorBy) {
    const idx = meta.categories.findIndex(c => c.name === step.colorBy);
    if (idx >= 0) {
      setActiveFamily(idx);
      document.querySelectorAll('.cat-tab').forEach((btn, i) => {
        btn.classList.toggle('active', i === idx);
      });
      const familyName = step.colorBy.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      document.getElementById('right-panel-title').textContent = familyName;
      document.getElementById('right-panel-subtitle').textContent = step.categorySubtitle || '';
    }
  }
  // Apply highlight — string, array of strings, or null. Family is always colorBy.
  if (step.highlight) {
    const labels = Array.isArray(step.highlight) ? step.highlight : [step.highlight];
    const labelIdxs = labels
      .map(lbl => meta.categories[activeFamilyIdx].labels.indexOf(lbl))
      .filter(i => i >= 0);
    if (labelIdxs.length === 1) {
      setHighlightLabel(labelIdxs[0]);
    } else if (labelIdxs.length > 1) {
      applyHighlightLabels(labelIdxs);
    }
  } else {
    setHighlightLabel(-1);
  }
  // Animate camera
  if (step.camera) {
    animateCameraToPosition(step.camera.position, step.camera.target);
  }
  // Render content blocks
  const contentEl = document.getElementById('story-content');
  contentEl.innerHTML = '';
  for (const block of step.content ?? []) {
    if (block.type === 'text') {
      const el = document.createElement('p');
      el.className = `story-${block.style || 'body'}`;
      el.textContent = block.value || '';
      contentEl.appendChild(el);
    } else if (block.type === 'description') {
      const el = document.createElement('p');
      el.className = 'story-description';
      el.textContent = block.value || '';
      contentEl.appendChild(el);
    } else if (block.type === 'link') {
      const el = document.createElement('a');
      el.className = 'story-link';
      el.textContent = block.value || 'View';
      el.href = block.href || '#';
      el.target = '_blank';
      el.rel = 'noopener noreferrer';
      contentEl.appendChild(el);
    } else if (block.type === 'chart' && block.placement === 'inline') {
      const wrap = document.createElement('div');
      wrap.className = 'story-chart-inline';
      contentEl.appendChild(wrap);
      loadAndRenderChart(block, wrap);
    }
  }

  document.getElementById('story-counter').textContent =
    `${currentStep + 1} / ${storyData.steps.length}`;

  document.getElementById('btn-prev').disabled = currentStep === 0;
  document.getElementById('btn-next').disabled = currentStep === storyData.steps.length - 1;

  const pct = ((currentStep + 1) / storyData.steps.length) * 100;
  document.getElementById('story-progress-fill').style.width = `${pct}%`;

  renderRightPanelChart(activeFamilyIdx);

  // Lock a specific recipe point if the step requests it
  if (step.lockedRecipe != null) {
    const idx = step.lockedRecipe;
    if (idx >= 0 && idx < N) {
      setLockedIdx(idx);
      showHoverTip(idx);
    }
  }

  // Auto-rotate on step 0 only — stops permanently on first user interaction
  stopIdleAutoRotate();
  if (currentStep === 0) startIdleAutoRotate();
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
  document.addEventListener('keydown', e => {
    if (appMode !== 'story') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft' && currentStep > 0) {
      currentStep--; applyStep(storyData.steps[currentStep]);
    } else if (e.key === 'ArrowRight' && currentStep < storyData.steps.length - 1) {
      currentStep++; applyStep(storyData.steps[currentStep]);
    }
  });
  applyStep(storyData.steps[0]);
}

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(mode) {
  appMode = mode;
  document.getElementById('btn-story').classList.toggle('active', mode === 'story');
  document.getElementById('btn-explore').classList.toggle('active', mode === 'explore');
  document.getElementById('story-panel').style.display = mode === 'story' ? 'flex' : 'none';
  document.getElementById('explore-panel').style.display = mode === 'explore' ? 'flex' : 'none';
  document.getElementById('btn-randomize').style.display = mode === 'explore' ? 'block' : 'none';
  document.getElementById('right-column').classList.toggle('story-mode', mode === 'story');

  if (mode === 'explore') {
    stopIdleAutoRotate();
    highlightLabelSet = null;
    setHighlightLabel(-1);
    setLockedIdx(-1);
    hideHoverTip();
    hideLeftPanelChart();
    showExploreDefault();
    document.getElementById('btn-randomize').textContent = 'Surprise Me';
    document.getElementById('right-panel-title').textContent = 'Categories';
    renderFilterChips();
    requestAnimationFrame(() => renderRightPanelChart(activeFamilyIdx));
  } else {
    setLockedIdx(-1);
    hideHoverTip();
    document.getElementById('left-panel').style.display = 'flex';
    document.getElementById('explore-default').style.display = 'none';
    resetToLevel0();
    restoreIntersectionHighlight();
    applyStep(storyData.steps[currentStep]);
  }
}

// ── Share ─────────────────────────────────────────────────────────────────────
function buildShareUrl() {
  const params = new URLSearchParams();
  params.set('mode', appMode);
  if (appMode === 'story') {
    params.set('step', currentStep);
  } else {
    const pos = camera.position;
    const target = controls.target;
    params.set('cam', [pos.x, pos.y, pos.z, target.x, target.y, target.z]
      .map(v => v.toFixed(3)).join(','));
    params.set('family', activeFamilyIdx);
    if (filterLevel >= 1) {
      params.set('fl', filterLevel);
      params.set('l1', level1LabelIdx);
    }
    if (filterLevel >= 2) {
      params.set('l2fam', level2FamilyIdx);
      params.set('l2', level2LabelIdx);
    }
    if (lockedIdx >= 0) params.set('locked', lockedIdx);
  }
  return `${location.origin}${location.pathname}#${params.toString()}`;
}

function applyShareState() {
  const hash = location.hash.slice(1);
  if (!hash) return;
  const params = new URLSearchParams(hash);
  const mode = params.get('mode') ?? 'story';

  if (mode === 'story') {
    const step = Math.max(0, Math.min(
      parseInt(params.get('step') ?? '0'), storyData.steps.length - 1
    ));
    currentStep = step;
    setMode('story');
  } else {
    setMode('explore');

    const family = params.get('family');
    const familyIdx = family !== null ? parseInt(family) : 0;
    setActiveFamily(familyIdx);
    document.querySelectorAll('.cat-tab').forEach((btn, i) => {
      btn.classList.toggle('active', i === familyIdx);
    });

    const fl = params.get('fl');
    if (fl !== null && parseInt(fl) >= 1) {
      const l1 = parseInt(params.get('l1') ?? '-1');
      if (l1 >= 0) transitionToLevel1(familyIdx, l1);

      if (parseInt(fl) >= 2) {
        const l2fam = parseInt(params.get('l2fam') ?? '-1');
        const l2    = parseInt(params.get('l2')    ?? '-1');
        if (l2fam >= 0 && l2 >= 0) transitionToLevel2(l2fam, l2);
      }
    }

    const camStr = params.get('cam');
    if (camStr) {
      const [px, py, pz, tx, ty, tz] = camStr.split(',').map(Number);
      animateCameraToPosition([px, py, pz], [tx, ty, tz]);
    }

    const locked = params.get('locked');
    if (locked !== null) {
      const idx = parseInt(locked);
      if (idx >= 0 && idx < N) {
        setLockedIdx(idx);
        showRecipeInfo(idx);
        showHoverTip(idx);
      }
    }
  }

  history.replaceState(null, '', location.pathname);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  showProgress(0, 'Loading metadata…');

  // Load meta, palette, story, and category metrics index in parallel
  const [metaRes, paletteRes, storyRes, catIdxRes] = await Promise.all([
    fetch(`${DATA}meta.json`).then(r => r.json()),
    fetch('palette.json').then(r => r.json()),
    fetch('story.json').then(r => r.json()),
    fetch(`${DATA}category_metrics/index.json`).then(r => r.json()).catch(() => null),
  ]);
  categoryMetricsIndex = catIdxRes;

  meta = metaRes;
  storyData = storyRes;
  N = meta.total;

  const palette = buildPalette(paletteRes);

  // Compute data center and extent for camera positioning
  const b = meta.coord_bounds;
  dataCenterVec.set(
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  );
  dataExtent = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
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
  const attrRaw = await fetch(`${DATA}attributes.bin.gz`).then(r => r.arrayBuffer());
  const attrBuffer = await decompressBuffer(attrRaw);

  const dtypeSize = { uint32: 4, uint16: 2, uint8: 1, float32: 4 };
  const dtypeArray = {
    uint32: Uint32Array, uint16: Uint16Array, uint8: Uint8Array, float32: Float32Array,
  };
  const attributes = {};
  let offset = 0;
  for (const attr of meta.attribute_layout) {
    const sz = dtypeSize[attr.dtype];
    const Arr = dtypeArray[attr.dtype];
    // Always slice to guarantee alignment — avoids issues when offset is not
    // a multiple of the element size (e.g. float32 after several uint8 blocks).
    attributes[attr.name] = new Arr(attrBuffer.slice(offset, offset + N * sz));
    offset += N * sz;
  }

  recipeIds = attributes['recipe_id'];
  chunkIds = attributes['chunk_id'];

  // Build packed category data: Uint32Array(N * N_families)
  const nFamilies = meta.categories.length;
  categoryData = new Uint32Array(N * nFamilies);
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
  document.getElementById('btn-story').addEventListener('click',    () => setMode('story'));
  const RANDOMIZE_LABELS = [
    'Roll the Dice', 'Spin the Wheel', 'Take a Chance',
    'Why Not?', 'Just Wing It', 'Mystery Pick', 'Go for It',
    'Hit Me', 'Keep Going', 'I Trust You', 'Dealer\'s Choice',
    'Wild Card', 'Leap of Faith', 'Fortune Favors...', 'Pull the Lever',
    'Close Your Eyes', 'No Peeking', 'Let Fate Decide', 'Next!',
    'Show Me Something', 'Feeling Adventurous?',
    // food puns
    'Chef\'s Choice', 'Recipe Roulette', 'Pot Luck', 'Stir the Pot',
    'What\'s Cooking?', 'Today\'s Special', 'House Special',
    'Take a Whisk', 'Feeling Saucy', 'Secret Ingredient',
    'Catch of the Day', 'Serve It Up', 'On the Menu',
    'Roll the Dough', 'Just Whisk It', 'Spice It Up',
    'Heat Things Up', 'Season to Taste', 'Simmer and See',
    'Bake It Till You Make It', 'Something Smells Good...',
    'Bring the Heat', 'Toss It Up', 'A Dash of Luck',
    'Mix It Up', 'Shake It Up', 'Feeling Hungry?', 'From the Vault',
    'Fork Around and Find Out', 'Lettuce Find One', 'Thyme to Explore',
    'Dough You Feel Lucky?', 'Roux-lette', 'Leek of Faith',
    'Braise Yourself', 'Crust Me', 'Turnip the Beet',
    'Miso Ready', 'Udon Know What You\'ll Get', 'Pasta la Vista',
    'Shell We?', 'Wanna Taco Bout It?', 'Kale Yeah!',
    'The Yolk\'s on You', 'Holy Guacamole', 'Nacho Average Pick',
    'Pour Decisions', 'Oil Be Surprised', 'No Whey!',
    'Sear-iously?', 'Pho-nomenal Find', 'Something\'s Brewing',
    'Worth Its Salt', 'Can\'t Beet the Suspense', 'Knead Something New',
    'Let\'s Get This Bread', 'Going With Your Gut',
  ];
  const btnRandomize = document.getElementById('btn-randomize');
  btnRandomize.addEventListener('click', () => {
    randomizeRecipe();
    const available = RANDOMIZE_LABELS.filter(l => l !== btnRandomize.textContent);
    btnRandomize.textContent = available[Math.floor(Math.random() * available.length)];
  });
  document.getElementById('btn-explore').addEventListener('click', () => setMode('explore'));

  // Left panel chart back button
  document.getElementById('btn-chart-back').addEventListener('click', () => {
    hideLeftPanelChart();
    document.getElementById('explore-content').style.display = 'block';
    document.getElementById('recipe-chart-btns').style.display = 'flex';
  });

  // About modal
  const btnAbout = document.getElementById('btn-about');
  const aboutOverlay = document.getElementById('about-overlay');
  const closeAbout = () => { aboutOverlay.classList.remove('open'); btnAbout.classList.remove('active'); };
  btnAbout.addEventListener('click', () => {
    aboutOverlay.classList.add('open');
    btnAbout.classList.add('active');
  });
  document.getElementById('about-close').addEventListener('click', closeAbout);
  aboutOverlay.addEventListener('click', e => { if (e.target === aboutOverlay) closeAbout(); });

  // Share popup
  const sharePopup = document.getElementById('share-popup');
  const shareUrlInput = document.getElementById('share-url');
  const shareInclude = document.getElementById('share-include-view');
  const plainUrl = `${location.origin}${location.pathname}`;

  const refreshShareUrl = () => {
    shareUrlInput.value = shareInclude.checked ? buildShareUrl() : plainUrl;
  };

  const btnShare = document.getElementById('btn-share');
  btnShare.addEventListener('click', () => {
    refreshShareUrl();
    sharePopup.classList.toggle('open');
    btnShare.classList.toggle('active', sharePopup.classList.contains('open'));
  });
  shareInclude.addEventListener('change', refreshShareUrl);
  document.getElementById('share-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(shareUrlInput.value).then(() => {
      const btn = document.getElementById('share-copy');
      btn.textContent = '✓ Copied';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  });
  document.getElementById('share-close').addEventListener('click', () => {
    sharePopup.classList.remove('open');
    btnShare.classList.remove('active');
  });
  document.addEventListener('click', e => {
    if (!sharePopup.contains(e.target) && e.target !== document.getElementById('btn-share')) {
      sharePopup.classList.remove('open');
      btnShare.classList.remove('active');
    }
  });


  // Story mode default
  initStoryPanel();
  setMode('story');
  applyShareState();

  hideProgress();
}

boot().catch(e => {
  document.getElementById('progress-label').textContent = `Error: ${e.message}`;
  document.getElementById('progress-label').style.display = 'block';
  document.getElementById('progress-wrap').style.display = 'none';
  console.error(e);
});
