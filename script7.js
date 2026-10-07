// ══════════════════════════════════════════════════════════════════════════
//  AirPen v7  — lecture mode, stable tracking, fixed sliders + bridge
//  + AUTO WORD RECOGNITION (recognizes when you pause writing)
//  + LOCAL CNN (TensorFlow.js, EMNIST-balanced 47-class) — no API key,
//    no network call per word. Requires train_emnist_cnn.py's output
//    (tfjs_model/) to be hosted alongside this app and the TF.js script
//    tag added to your HTML:
//      <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
//  + FIX: transpose correction for TFDS EMNIST orientation (predictions
//    were scrambled because tensorflow_datasets serves EMNIST images
//    transposed relative to natural reading orientation — the old `emnist`
//    pip package auto-corrected this internally, tfds does not).
//  + FIX: single-frame landmark spike rejection (stray MediaPipe
//    misdetections no longer cause visible jump/scramble in strokes).
// ══════════════════════════════════════════════════════════════════════════

// ── TUNING (let not const — sliders reassign these) ───────────────────────
let SMOOTH        = 0.15;
let MIN_DIST      = 3.0;
const PT_HIST       = 4;
const TENSION       = 0.5;
const PRESSURE_MIN  = 0.7;
const PRESSURE_MAX  = 1.05;
const PRESSURE_SPDS = 45;

const SWIPE_MIN_DIST   = 65;
const SWIPE_MAX_TIME   = 800;
const SWIPE_REARM_TIME = 350;
const FIST_COOLDOWN    = 1800;
const CONFIRM_FRAMES   = 3;
const EXT_THRESHOLD    = 1.60;

// How long (ms) to ignore new drawing strokes right after a slide swipe
// fires. Without this, the hand's residual motion while relaxing out of
// the HORNS gesture back toward a pointing pose can register as a stray
// stroke the instant the gesture reclassifies as "point".
const WRITE_LOCK_MS = 600;
let writeLockUntil = 0;

// How many consecutive empty frames we tolerate before we actually
// treat the hand as "gone". MediaPipe's internal tracker already re-runs
// detection on a lost hand, but if a single frame whiffs (motion blur,
// hand briefly at frame edge, or fast movement) we don't want the cursor
// to vanish or the stroke/UI to reset.
const LOST_GRACE_FRAMES = 12;

// Max plausible per-frame fingertip movement, in canvas pixels. Raised
// generously so genuinely fast hand motion isn't mistaken for a bad
// detection — a too-low value here was causing the cursor ring to freeze
// mid-move (looking like it "dropped") whenever you wrote or gestured
// quickly. Only truly implausible teleports get rejected now.
const MAX_JUMP_PX = 160;

// Set true only while tuning recognition — shows a live preview of exactly
// what image is fed to the CNN. Left on, it costs an extra canvas render +
// async readback per glyph every recognition, which is the main thing that
// made recognition feel slow. Off by default for normal use.
const DEBUG_GLYPHS = true;

// ── ONE EURO FILTER ─────────────────────────────────────────────────────
// Replaces simple lerp-based smoothing. A plain lerp forces a tradeoff:
// smooth enough to kill jitter also means laggy/heavy during fast strokes.
// One Euro adapts automatically — heavy smoothing when the fingertip is
// nearly still (kills the "vibrating" look), and lighter smoothing as
// speed increases (stays responsive, doesn't feel slow to write with).
class OneEuroFilter {
  constructor(freq = 60, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.freq = freq;
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
  _alpha(cutoff) {
    const te  = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }
  filter(x, timestampMs) {
    if (this.tPrev === null) {
      this.tPrev = timestampMs;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dt = (timestampMs - this.tPrev) / 1000;
    if (dt > 0) this.freq = 1 / dt;
    const dx    = (x - this.xPrev) * this.freq;
    const aD    = this._alpha(this.dcutoff);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.mincutoff + this.beta * Math.abs(dxHat);
    const a     = this._alpha(cutoff);
    const xHat  = a * x + (1 - a) * this.xPrev;
    this.xPrev  = xHat;
    this.dxPrev = dxHat;
    this.tPrev  = timestampMs;
    return xHat;
  }
  reset() {
    this.xPrev = null; this.dxPrev = 0; this.tPrev = null;
  }
}
const oneEuroX = new OneEuroFilter(60, 1.0, 0.007, 1.0);
const oneEuroY = new OneEuroFilter(60, 1.0, 0.007, 1.0);

// ── WORD RECOGNITION (NEW) ─────────────────────────────────────────────────
// How long (ms) the pen has to be idle — i.e. no new ink added — before we
// treat the current word as "finished" and send it off to be recognized.
// Kept above SWIPE_MAX_TIME/FIST_COOLDOWN scale so a brief gesture-switch
// (e.g. point -> peace -> point) mid-word doesn't fire it prematurely.
const WORD_PAUSE_MS     = 1100;
// How often (ms) we poll for the pause condition.
const WORD_POLL_MS      = 250;
// Ignore microscopic ink blobs (accidental taps/jitter) below this bbox size.
const MIN_INK_SPAN_PX   = 14;
// Padding (px) added around the ink bounding box before cropping, so
// ascenders/descenders and stroke edges aren't clipped.
const INK_CROP_PAD      = 24;

// ── LOCAL CNN (NEW) ──────────────────────────────────────────────────────
// Where the converted TF.js model lives (output of train_emnist_cnn.py,
// copied into your static assets). Adjust to match your deployment.
const CNN_MODEL_URL = 'models/emnist/model.json';

// Official EMNIST-balanced label order — index i of the model's softmax
// output corresponds to EMNIST_CLASSES[i]. MUST match the Python script's
// EMNIST_CLASSES list exactly, or recognition will silently scramble.
const EMNIST_CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'.split('');

// Per-character segmentation tuning (tune against your own handwriting).
const CHAR_INK_THRESHOLD = 40;   // 0-255 grayscale cutoff for "this pixel is ink"
const CHAR_GAP_PX        = 10;   // empty-column run length that splits two characters
const CHAR_MIN_WIDTH_PX  = 4;    // ignore stray segments narrower than this

// ── CORE STATE ────────────────────────────────────────────────────────────
let mpHands = null;
let video, dCanvas, dCtx, oCanvas, oCtx;

let brushSize = 6;
let penColor  = '#ffffff';
let isEraser  = false;

let isDrawing   = false;
let strokeCount = 0;

let smoothX = null, smoothY = null;
let ptHist    = [];
let pressureW = 1.0;

let gestHist         = [];
let confirmedGesture = 'none';

let swipeBuf     = [];
let swipeLocked  = false;
let swipeRearmAt = 0;
let hornsCtr     = 0;
let fistLastTime = 0;

let isPinching    = false;
let pinchStartX   = null, pinchStartY   = null;
let pinchOffsetX  = 0,    pinchOffsetY  = 0;
let pinchSnapshot = null;

let slideWS = null, slideWSStatus = 'disconnected';

// tracks consecutive frames with no detected hand, so a single dropped
// frame doesn't reset drawing state / gesture history
let lostFrames = 0;

// video's native (intrinsic) resolution — needed to correctly map
// normalized landmarks through the same "cover" crop the <video> element
// uses visually, since the stage canvas aspect ratio (16:10) doesn't match
// the camera's captured resolution (320x240, ~4:3).
let videoNativeW = 0, videoNativeH = 0;

// Smoothed copy of all 21 hand landmarks (not just the fingertip cursor).
// Raw per-frame landmarks are noisy enough to cause visible skeleton
// jitter and occasional gesture misclassification (e.g. a finger briefly
// reads as "not extended" due to landmark noise); smoothing all of them
// makes both the skeleton drawing and gesture detection more stable.
let smoothedLandmarks = null;
const LANDMARK_SMOOTH = 0.5; // 0 = no smoothing, 1 = frozen; weight toward new sample

// ── WORD RECOGNITION STATE (NEW) ───────────────────────────────────────────
let lastInkTime   = 0;     // Date.now() of the most recently drawn point
let inkBounds     = null;  // {minX,minY,maxX,maxY} of ink since last recognition
let wordPending   = false; // true while recognition is running
let transcript    = [];    // array of recognized word strings, in order

// ── LOCAL CNN STATE (NEW) ───────────────────────────────────────────────
let cnnModel      = null;  // tf.LayersModel once loaded
let cnnLoading    = false;
let cnnLoadError  = null;

// ── WEBSOCKET ─────────────────────────────────────────────────────────────
function connectSlideWS() {
  try {
    slideWS = new WebSocket('ws://localhost:8765');
    slideWS.onopen  = () => { slideWSStatus = 'connected';    updateSlideStatus(); };
    slideWS.onclose = () => {
      slideWSStatus = 'disconnected'; updateSlideStatus();
      setTimeout(connectSlideWS, 3000);
    };
    slideWS.onerror = () => { slideWSStatus = 'error'; updateSlideStatus(); };
  } catch(e) { slideWSStatus = 'error'; updateSlideStatus(); }
}
function updateSlideStatus() {
  const el = document.getElementById('slide-status');
  if (el) el.textContent = 'bridge: ' + slideWSStatus;
}
function sendSlideCommand(cmd) {
  if (slideWS && slideWS.readyState === WebSocket.OPEN) slideWS.send(cmd);
  flashSlide(cmd === 'next' ? '▶▶ NEXT' : '◀◀ PREV');
}
function flashSlide(msg) {
  const el = document.getElementById('slide-status');
  if (!el) return;
  el.textContent = msg;
  el.style.color = '#ffdd00';
  setTimeout(() => { updateSlideStatus(); el.style.color = ''; }, 900);
}

// ── MATH ──────────────────────────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * t; }
function dist3(a, b) {
  const dx = a.x-b.x, dy = a.y-b.y, dz=(a.z||0)-(b.z||0);
  return Math.sqrt(dx*dx+dy*dy+dz*dz);
}

// ── CATMULL-ROM ────────────────────────────────────────────────────────────
function drawCR(ctx, p0, p1, p2, p3, w) {
  const t    = TENSION;
  const cp1x = p1.x + (p2.x - p0.x) / (6*t);
  const cp1y = p1.y + (p2.y - p0.y) / (6*t);
  const cp2x = p2.x - (p3.x - p1.x) / (6*t);
  const cp2y = p2.y - (p3.y - p1.y) / (6*t);
  ctx.lineWidth = w;
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  ctx.stroke();
}

// ── GESTURE ────────────────────────────────────────────────────────────────
function confirmGesture(raw) {
  gestHist.push(raw);
  if (gestHist.length > CONFIRM_FRAMES) gestHist.shift();
  if (gestHist.length === CONFIRM_FRAMES && gestHist.every(g => g === raw))
    confirmedGesture = raw;
  return confirmedGesture;
}

function isExtended(lm, tip, mcp) {
  const w = lm[0];
  return dist3(lm[mcp],w) > 0.001 && (dist3(lm[tip],w)/dist3(lm[mcp],w)) > EXT_THRESHOLD;
}

function classify(lm) {
  const palmSize  = dist3(lm[0], lm[9]);
  const pinchDist = dist3(lm[4], lm[8]);
  if (palmSize > 0.001 && (pinchDist/palmSize) < 0.30) return 'pinch';
  const idx   = isExtended(lm,8,5);
  const mid   = isExtended(lm,12,9);
  const ring  = isExtended(lm,16,13);
  const pinky = isExtended(lm,20,17);
  const n = [idx,mid,ring,pinky].filter(Boolean).length;
  if (n===0)                       return 'fist';
  if (n>=4)                        return 'open';
  if (idx&&pinky&&!mid&&!ring)     return 'horns';
  if (idx&&mid&&!ring&&!pinky)     return 'peace';
  if (idx)                         return 'point';
  return 'other';
}

// ── SAVE ──────────────────────────────────────────────────────────────────
function savePageAsImage() {
  if (!dCanvas) return;
  const tmp = document.createElement('canvas');
  tmp.width  = dCanvas.width;
  tmp.height = dCanvas.height;
  const tc = tmp.getContext('2d');
  tc.fillStyle = '#ffffff';
  tc.fillRect(0, 0, tmp.width, tmp.height);
  tc.drawImage(dCanvas, 0, 0);
  const a = document.createElement('a');
  a.href     = tmp.toDataURL('image/png');
  a.download = 'airpen-notes.png';
  a.click();
}

function saveAllPages() {
  savePageAsImage();
}

// ── BOOTSTRAP ─────────────────────────────────────────────────────────────
async function boot() {
  const btn  = document.getElementById('start-btn');
  const prog = document.getElementById('load-progress');
  btn.disabled = true; btn.textContent = '⏳ LOADING...';

  try {
    prog.textContent = 'Starting webcam...';

    // Remove any leftover glyph-debug-panel from earlier testing sessions —
    // it's a plain DOM element that persists across recognitions once
    // created, so simply flipping DEBUG_GLYPHS off doesn't remove one that
    // already exists on the page.
    document.getElementById('glyph-debug-panel')?.remove();

    video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    await new Promise(r => { video.onloadedmetadata = r; });
    video.play();

    videoNativeW = video.videoWidth  || 640;
    videoNativeH = video.videoHeight || 480;

    prog.textContent = 'Loading MediaPipe Hands...';
    mpHands = new Hands({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    mpHands.setOptions({
      maxNumHands            : 1,
      modelComplexity        : 1,
      minDetectionConfidence : 0.6,
      minTrackingConfidence  : 0.6
    });

    prog.textContent = 'Warming up model...';
    await mpHands.initialize();

    dCanvas = document.getElementById('draw-canvas');
    oCanvas = document.getElementById('overlay-canvas');

    function resize() {
      const s = document.getElementById('stage');
      const w = s.clientWidth, h = s.clientHeight;
      const tmp = document.createElement('canvas');
      tmp.width = w; tmp.height = h;
      if (dCtx) tmp.getContext('2d').drawImage(dCanvas, 0, 0, w, h);
      dCanvas.width = oCanvas.width  = w;
      dCanvas.height = oCanvas.height = h;
      if (dCtx) {
        dCtx.drawImage(tmp, 0, 0);
        _setDrawCtx();
      }
    }
    resize();
    window.addEventListener('resize', resize);

    dCtx = dCanvas.getContext('2d');
    oCtx  = oCanvas.getContext('2d');
    _setDrawCtx();

    document.getElementById('start-overlay').style.display = 'none';
    document.getElementById('pulse').classList.add('on');
    document.getElementById('status-txt').textContent = 'tracking';

    // ── GATED FRAME LOOP ─────────────────────────────────────────────────
    let frameReady = true;
    mpHands.onResults(result => {
      frameReady = true;
      processHands(result);
    });
    function frameLoop() {
      if (frameReady) {
        frameReady = false;
        mpHands.send({ image: video }).catch(() => { frameReady = true; });
      }
      requestAnimationFrame(frameLoop);
    }
    requestAnimationFrame(frameLoop);
    connectSlideWS();

    // NEW: word-recognition UI + idle-watcher + local CNN model
    ensureTranscriptUI();
    loadCNNModel();
    setInterval(checkWordPause, WORD_POLL_MS);

  } catch(e) {
    prog.textContent = 'ERROR: ' + e.message;
    btn.disabled = false; btn.textContent = '▶ RETRY';
  }
}

function _setDrawCtx() {
  dCtx.lineCap  = 'round';
  dCtx.lineJoin = 'round';
}

// ── SKELETON ──────────────────────────────────────────────────────────────
const CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

// ── STOP DRAWING ──────────────────────────────────────────────────────────
function _stopDraw() {
  isDrawing = false;
  ptHist    = [];
  pressureW = 1.0;
  document.getElementById('pulse').classList.remove('draw');
}

function mapLandmark(lm, W, H) {
  const vw = videoNativeW || W, vh = videoNativeH || H;
  const videoAspect  = vw / vh;
  const canvasAspect = W / H;

  let scale, offsetX = 0, offsetY = 0;
  if (videoAspect > canvasAspect) {
    scale = H / vh;
    offsetX = (vw * scale - W) / 2;
  } else {
    scale = W / vw;
    offsetY = (vh * scale - H) / 2;
  }

  const px = lm.x * vw * scale - offsetX;
  const py = lm.y * vh * scale - offsetY;
  return { x: W - px, y: py };
}

// ── WORD RECOGNITION HELPERS (NEW) ─────────────────────────────────────────

// Crops just the ink-bounding-box region (plus padding) from the draw
// canvas onto a small BLACK-backed temp canvas (matches EMNIST convention:
// bright strokes on dark background). Using black instead of white also
// sidesteps a real bug the old white-crop had: a white pen stroke flattened
// onto a white background became invisible.
function _cropInkRegionBlack() {
  const pad = INK_CROP_PAD;
  const sx = Math.max(0, Math.floor(inkBounds.minX - pad));
  const sy = Math.max(0, Math.floor(inkBounds.minY - pad));
  const ex = Math.min(dCanvas.width,  Math.ceil(inkBounds.maxX + pad));
  const ey = Math.min(dCanvas.height, Math.ceil(inkBounds.maxY + pad));
  const sw = Math.max(1, ex - sx), sh = Math.max(1, ey - sy);

  const tmp = document.createElement('canvas');
  tmp.width = sw; tmp.height = sh;
  const tc = tmp.getContext('2d');
  tc.fillStyle = '#000000';
  tc.fillRect(0, 0, sw, sh);
  tc.drawImage(dCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
  return tmp;
}

// Loads the EMNIST CNN once at boot. Recognition silently no-ops until this
// resolves — checkWordPause() guards on cnnModel being non-null.
async function loadCNNModel() {
  cnnLoading = true;
  updateTranscriptUI('loading model…');
  try {
    cnnModel = await tf.loadLayersModel(CNN_MODEL_URL);
    // warm up so the first real recognition isn't slowed by lazy kernel compilation
    tf.tidy(() => cnnModel.predict(tf.zeros([1, 28, 28, 1])));
    updateTranscriptUI('idle');
  } catch (e) {
    cnnLoadError = e.message || String(e);
    updateTranscriptUI('model load failed');
    console.error('CNN model failed to load from', CNN_MODEL_URL, e);
  }
  cnnLoading = false;
}

// Builds (and caches) a small floating panel showing the live transcript,
// a pending/idle indicator, and copy/clear controls. Created at runtime so
// no HTML changes are required to use this module.
function ensureTranscriptUI() {
  if (document.getElementById('transcript-panel')) return;
  const panel = document.createElement('div');
  panel.id = 'transcript-panel';
  panel.style.cssText = `
    position:fixed; right:16px; top:16px; width:280px; max-height:240px;
    background:rgba(15,15,25,0.92); border:1px solid rgba(102,85,255,0.4);
    border-radius:10px; padding:10px 12px; z-index:9999;
    font-family:monospace; font-size:13px; color:#e8e8f5;
    box-shadow:0 4px 20px rgba(0,0,0,0.4); backdrop-filter:blur(4px);
  `;
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <strong style="color:#6655ff;">✎ AUTO-RECOGNIZE (CNN)</strong>
      <span id="word-status" style="font-size:11px;color:#888;">idle</span>
    </div>
    <div id="transcript-text" style="min-height:48px;max-height:140px;overflow-y:auto;
      line-height:1.5;word-wrap:break-word;color:#fff;"></div>
    <div style="display:flex;gap:6px;margin-top:8px;">
      <button id="transcript-copy" style="flex:1;font-size:11px;padding:4px;cursor:pointer;
        background:#222;color:#fff;border:1px solid #444;border-radius:5px;">COPY</button>
      <button id="transcript-clear" style="flex:1;font-size:11px;padding:4px;cursor:pointer;
        background:#222;color:#fff;border:1px solid #444;border-radius:5px;">CLEAR</button>
    </div>
  `;
  document.body.appendChild(panel);
  document.getElementById('transcript-copy').onclick = () => {
    navigator.clipboard?.writeText(transcript.join(' '));
  };
  document.getElementById('transcript-clear').onclick = () => {
    transcript = [];
    updateTranscriptUI();
  };
}

function updateTranscriptUI(statusText) {
  const t = document.getElementById('transcript-text');
  const s = document.getElementById('word-status');
  // Show only the current (most recent) word — full history is still kept
  // in `transcript` for the COPY button, just not displayed all at once.
  if (t) t.textContent = transcript.length ? transcript[transcript.length - 1] : '';
  if (s && statusText !== undefined) s.textContent = statusText;
}

// Expands the running ink bounding box to include a freshly-drawn point,
// and stamps lastInkTime so the idle-watcher knows the pen is still "live".
function _markInk(x, y) {
  lastInkTime = Date.now();
  if (!inkBounds) {
    inkBounds = { minX: x, minY: y, maxX: x, maxY: y };
  } else {
    if (x < inkBounds.minX) inkBounds.minX = x;
    if (y < inkBounds.minY) inkBounds.minY = y;
    if (x > inkBounds.maxX) inkBounds.maxX = x;
    if (y > inkBounds.maxY) inkBounds.maxY = y;
  }
}

// ── CHARACTER SEGMENTATION ─────────────────────────────────────────────────
// Splits a word-region canvas into individual glyph canvases using a simple
// vertical-projection profile: sum ink per column, treat runs of >=
// CHAR_GAP_PX empty columns as the boundary between two characters. Good
// enough for printed/block air-writing; cursive that connects letters will
// under-segment (treated as one blob) — acceptable for a first pass.
function _segmentCharacters(cropCanvas) {
  const w = cropCanvas.width, h = cropCanvas.height;
  const ctx = cropCanvas.getContext('2d');
  const { data } = ctx.getImageData(0, 0, w, h);

  // grayscale luminance per pixel (max channel — robust to whatever pen color)
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
    gray[i] = Math.max(r, g, b);
  }

  const colHasInk = new Array(w).fill(false);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (gray[y*w + x] > CHAR_INK_THRESHOLD) { colHasInk[x] = true; break; }
    }
  }

  // Close small internal gaps (e.g. a "t" crossbar not quite touching its
  // stem, or a shaky stroke that briefly lifts mid-letter due to air-writing
  // imprecision) so a single letter doesn't get mis-segmented as two.
  // Only closes gaps that are bounded by ink on BOTH sides and shorter than
  // GAP_CLOSE_PX — genuine spacing between separate letters is bounded by
  // CHAR_GAP_PX below and is normally much wider than this.
  const GAP_CLOSE_PX = 4;
  const colHasInkClosed = colHasInk.slice();
  {
    let i = 0;
    while (i < colHasInkClosed.length) {
      if (!colHasInkClosed[i]) {
        let j = i;
        while (j < colHasInkClosed.length && !colHasInkClosed[j]) j++;
        const gapLen   = j - i;
        const hasBefore = i > 0 && colHasInkClosed[i - 1];
        const hasAfter  = j < colHasInkClosed.length && colHasInkClosed[j];
        if (gapLen <= GAP_CLOSE_PX && hasBefore && hasAfter) {
          for (let k = i; k < j; k++) colHasInkClosed[k] = true;
        }
        i = j;
      } else {
        i++;
      }
    }
  }

  // group into [startX, endX] runs, splitting on gaps >= CHAR_GAP_PX
  const segments = [];
  let runStart = -1, gapLen = 0;
  for (let x = 0; x < w; x++) {
    if (colHasInkClosed[x]) {
      if (runStart === -1) runStart = x;
      gapLen = 0;
    } else if (runStart !== -1) {
      gapLen++;
      if (gapLen >= CHAR_GAP_PX) {
        segments.push([runStart, x - gapLen]);
        runStart = -1; gapLen = 0;
      }
    }
  }
  if (runStart !== -1) segments.push([runStart, w - 1]);

  // for each column-run, tighten the vertical bounds to the actual ink
  const glyphs = [];
  for (const [sx, ex] of segments) {
    if (ex - sx < CHAR_MIN_WIDTH_PX) continue;
    let minY = h, maxY = 0;
    for (let x = sx; x <= ex; x++) {
      for (let y = 0; y < h; y++) {
        if (gray[y*w + x] > CHAR_INK_THRESHOLD) { if (y < minY) minY = y; if (y > maxY) maxY = y; }
      }
    }
    if (minY > maxY) continue; // shouldn't happen, safety
    glyphs.push({ sx, sy: minY, ex, ey: maxY });
  }
  return glyphs.map(g => ({ x: g.sx, canvas: _toGlyphCanvas(cropCanvas, g) }));
}

// Thickens a canvas's ink by stamping it repeatedly at small offsets within
// a disk of the given radius, using additive ('lighter') blending so
// overlapping bright pixels just stay bright (clamped at 255) rather than
// overflowing. This is a cheap dilation. It matters here because EMNIST's
// training strokes are proportionally thick (real pen, scanned), while our
// thin air-written lines nearly vanish once squeezed down to 28x28 —
// dilating first ensures enough stroke survives the downscale to actually
// resemble the training distribution.
function _dilateCanvas(srcCanvas, radiusPx) {
  const w = srcCanvas.width, h = srcCanvas.height;
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  octx.fillStyle = '#000000';
  octx.fillRect(0, 0, w, h);
  octx.globalCompositeOperation = 'lighter';
  for (let dx = -radiusPx; dx <= radiusPx; dx++) {
    for (let dy = -radiusPx; dy <= radiusPx; dy++) {
      if (dx * dx + dy * dy <= radiusPx * radiusPx) {
        octx.drawImage(srcCanvas, dx, dy);
      }
    }
  }
  octx.globalCompositeOperation = 'source-over';
  return out;
}

// Crops one glyph's tight bbox, pads it to square, dilates the strokes so
// they survive the downscale, and resizes into the classic MNIST/EMNIST
// 28x28-with-4px-margin layout the model expects.
function _toGlyphCanvas(srcCanvas, g) {
  const bw = g.ex - g.sx + 1, bh = g.ey - g.sy + 1;
  const side = Math.max(bw, bh);
  const squarePad = Math.ceil(side * 0.2); // breathing room before the 4px model margin
  const squareSide = side + squarePad * 2;

  const square = document.createElement('canvas');
  square.width = squareSide; square.height = squareSide;
  const sc = square.getContext('2d');
  sc.fillStyle = '#000000';
  sc.fillRect(0, 0, squareSide, squareSide);
  sc.drawImage(
    srcCanvas,
    g.sx, g.sy, bw, bh,
    squarePad + (side - bw) / 2, squarePad + (side - bh) / 2, bw, bh
  );

  // Dilate proportional to the glyph's own size so thin strokes end up
  // thick enough, relative to the letter, to survive shrinking to 28x28.
  const dilateRadius = Math.max(2, Math.round(squareSide * 0.05));
  const dilated = _dilateCanvas(square, dilateRadius);

  const out = document.createElement('canvas');
  out.width = 28; out.height = 28;
  const oc = out.getContext('2d');
  oc.fillStyle = '#000000';
  oc.fillRect(0, 0, 28, 28);
  oc.imageSmoothingEnabled = true;
  // draw into a 20x20 inner box (4px margin all round), matching EMNIST norm
  oc.drawImage(dilated, 0, 0, squareSide, squareSide, 4, 4, 20, 20);
  return out;
}

// Converts a 28x28 glyph canvas into the [1,28,28,1] tensor the model
// expects, applying the orientation correction needed because the model
// was trained on tensorflow_datasets EMNIST images (which are transposed
// relative to natural reading orientation, unlike the old `emnist` pip
// package's output). Without this, predictions come out scrambled even
// though the model itself trained/evaluated fine on internally-consistent
// (also-transposed) data.
function _glyphToTensor(glyphCanvas) {
  return tf.browser.fromPixels(glyphCanvas, 1)   // grayscale, [28,28,1]
    .toFloat().div(255.0)
    .transpose([1, 0, 2])                        // <- orientation fix
    .expandDims(0);                              // [1,28,28,1]
}

// DEBUG ONLY: renders the exact 28x28 image the model receives (after
// orientation transform) into a small visible canvas on the page, so we
// can see with our own eyes whether the letter looks upright/correct or
// flipped/rotated/mirrored, instead of guessing blindly. Also shows the
// model's top-3 predictions with confidence — this tells us whether a
// wrong answer is a near-miss (model roughly right, correct letter is
// 2nd/3rd guess -> accuracy/preprocessing issue) or completely unrelated
// (points to a label-mapping bug instead).
function _debugShowGlyph(glyphCanvas, top3Text) {
  let dbg = document.getElementById('glyph-debug-panel');
  if (!dbg) {
    dbg = document.createElement('div');
    dbg.id = 'glyph-debug-panel';
    dbg.style.cssText = `
      position:fixed; left:16px; bottom:16px; z-index:9999;
      background:rgba(15,15,25,0.92); border:1px solid rgba(102,85,255,0.4);
      border-radius:10px; padding:10px; font-family:monospace; font-size:11px;
      color:#e8e8f5; display:flex; gap:10px; align-items:flex-start;
      max-width:90vw; overflow-x:auto;
    `;
    document.body.appendChild(dbg);
  }

  const rawPreview = document.createElement('canvas');
  rawPreview.width = 28; rawPreview.height = 28;
  rawPreview.style.cssText = 'width:84px;height:84px;image-rendering:pixelated;border:1px solid #555;';
  rawPreview.getContext('2d').drawImage(glyphCanvas, 0, 0);

  const wrap1 = document.createElement('div');
  wrap1.appendChild(document.createTextNode('RAW'));
  wrap1.appendChild(document.createElement('br'));
  wrap1.appendChild(rawPreview);

  // render the actual tensor fed to the model back to a canvas, so we see
  // it exactly as the network sees it (post transpose/reverse/etc.)
  const tensor = tf.tidy(() => _glyphToTensor(glyphCanvas).squeeze([0]));
  const transformedPreview = document.createElement('canvas');
  transformedPreview.width = 28; transformedPreview.height = 28;
  transformedPreview.style.cssText = 'width:84px;height:84px;image-rendering:pixelated;border:1px solid #6655ff;';
  tf.browser.toPixels(tensor, transformedPreview).then(() => tensor.dispose());

  const wrap2 = document.createElement('div');
  wrap2.appendChild(document.createTextNode('FED TO MODEL'));
  wrap2.appendChild(document.createElement('br'));
  wrap2.appendChild(transformedPreview);

  const wrap3 = document.createElement('div');
  wrap3.style.cssText = 'white-space:pre-line;min-width:120px;';
  wrap3.textContent = 'TOP-3:\n' + (top3Text || '');

  const group = document.createElement('div');
  group.style.cssText = 'display:flex;gap:10px;align-items:flex-start;border-right:1px solid #444;padding-right:10px;margin-right:4px;';
  group.appendChild(wrap1);
  group.appendChild(wrap2);
  group.appendChild(wrap3);

  dbg.appendChild(group);
}

// Idle-watcher: called every WORD_POLL_MS. Fires recognition once the pen
// has been lifted (not mid-stroke) and idle for WORD_PAUSE_MS.
function checkWordPause() {
  if (!cnnModel || !inkBounds || isDrawing || wordPending) return;
  const span = Math.max(
    inkBounds.maxX - inkBounds.minX,
    inkBounds.maxY - inkBounds.minY
  );
  if (span < MIN_INK_SPAN_PX) { inkBounds = null; return; } // ignore noise
  if (Date.now() - lastInkTime < WORD_PAUSE_MS) return;
  recognizeWordCNN();
}

// Runs the full local pipeline: crop ink region -> segment into glyphs ->
// classify each glyph with the CNN -> concatenate into a word string.
// Entirely client-side, no network call.
async function recognizeWordCNN() {
  if (!inkBounds || !cnnModel) return;
  wordPending = true;
  updateTranscriptUI('recognizing…');

  const cropCanvas = _cropInkRegionBlack();
  // reset tracking immediately so new ink drawn during inference starts a
  // fresh word instead of merging into this one
  inkBounds = null;

  try {
    const glyphs = _segmentCharacters(cropCanvas);
    if (!glyphs.length) {
      updateTranscriptUI('unclear');
      wordPending = false;
      return;
    }

    let word = '';
    if (DEBUG_GLYPHS) {
      const dbg = document.getElementById('glyph-debug-panel');
      if (dbg) dbg.innerHTML = '';
    }
    for (const g of glyphs) {
      let top3Text = '';
      const ch = tf.tidy(() => {
        const input = _glyphToTensor(g.canvas);
        const pred = cnnModel.predict(input);
        const probs = pred.dataSync();
        if (DEBUG_GLYPHS) {
          const ranked = Array.from(probs)
            .map((p, i) => ({ p, i }))
            .sort((a, b) => b.p - a.p)
            .slice(0, 3);
          top3Text = ranked
            .map(r => `${EMNIST_CLASSES[r.i]}: ${(r.p * 100).toFixed(1)}%`)
            .join('\n');
        }
        const idx = pred.argMax(-1).dataSync()[0];
        return EMNIST_CLASSES[idx] || '?';
      });
      if (DEBUG_GLYPHS) _debugShowGlyph(g.canvas, top3Text);
      word += ch;
    }
    if (DEBUG_GLYPHS) console.log(`[AirPen] segmented ${glyphs.length} glyph(s) -> "${word}"`);

    transcript.push(word);
    updateTranscriptUI('ok');
  } catch (e) {
    console.error('recognizeWordCNN failed', e);
    updateTranscriptUI('error');
  }
  wordPending = false;
}

// Draws the gesture-colored cursor ring (and point-mode center dot) at the
// given position. Extracted so it can be called both during normal
// per-frame hand tracking AND during brief tracking-loss grace frames, so
// the ring doesn't visibly disappear/"drop" every time MediaPipe misses a
// frame (motion blur, fast movement) — it just holds at its last position.
function _drawCursorRing(gesture, curX, curY) {
  const GCOL = { point:'#6655ff',open:'#ff4d8f',fist:'#ffaa30',peace:'#39ffb0',horns:'#ffdd00',pinch:'#00e5ff',other:'#404080',none:'#404080' };
  oCtx.strokeStyle = GCOL[gesture]||'#404080'; oCtx.lineWidth = 2;
  const cR = gesture==='open' ? 38 : gesture==='pinch' ? 18 : brushSize+8;
  oCtx.beginPath(); oCtx.arc(curX, curY, cR, 0, Math.PI*2); oCtx.stroke();
  if (gesture==='point') {
    oCtx.fillStyle = penColor;
    oCtx.beginPath(); oCtx.arc(curX, curY, 3, 0, Math.PI*2); oCtx.fill();
  }
}

// ── MAIN PROCESSING ───────────────────────────────────────────────────────
function processHands(results) {
  if (!oCtx || !dCtx) return;
  const W = oCanvas.width, H = oCanvas.height;
  oCtx.clearRect(0, 0, W, H);

  if (!results.multiHandLandmarks?.length) {
    lostFrames++;
    if (lostFrames <= LOST_GRACE_FRAMES) {
      // Hold the cursor ring at its last known position instead of letting
      // it vanish — a single missed detection frame shouldn't make the
      // indicator flicker/drop while you're actively writing.
      if (smoothX !== null) {
        _drawCursorRing(confirmedGesture, smoothX, smoothY);
      }
      return;
    }
    _stopDraw();
    smoothX = null; smoothY = null;
    oneEuroX.reset(); oneEuroY.reset();
    smoothedLandmarks = null;
    gestHist.length = 0; confirmedGesture = 'none';
    swipeBuf.length = 0; swipeLocked = false; hornsCtr = 0;
    if (isPinching) { isPinching = false; pinchSnapshot = null; }
    document.getElementById('c-gest').textContent  = '—';
    document.getElementById('c-score').textContent = '—';
    return;
  }
  lostFrames = 0;

  const rawLm = results.multiHandLandmarks[0];

  if (!smoothedLandmarks) {
    smoothedLandmarks = rawLm.map(p => ({ x: p.x, y: p.y, z: p.z || 0 }));
  } else {
    smoothedLandmarks = rawLm.map((p, i) => ({
      x: lerp(smoothedLandmarks[i].x, p.x, LANDMARK_SMOOTH),
      y: lerp(smoothedLandmarks[i].y, p.y, LANDMARK_SMOOTH),
      z: lerp(smoothedLandmarks[i].z, p.z || 0, LANDMARK_SMOOTH),
    }));
  }
  const lm = smoothedLandmarks;

  const mapped = lm.map(p => mapLandmark(p, W, H));
  const cx = i => mapped[i].x;
  const cy = i => mapped[i].y;

  const rawG    = classify(lm);
  const gesture = confirmGesture(rawG);
  document.getElementById('c-gest').textContent  = rawG.toUpperCase();
  document.getElementById('c-score').textContent = '✓';

  const rawX = rawG === 'pinch' ? (cx(4)+cx(8))/2 : cx(8);
  const rawY = rawG === 'pinch' ? (cy(4)+cy(8))/2 : cy(8);
  const frameTime = performance.now();

  // Spike rejection: if the raw fingertip position jumped further than is
  // physically plausible in a single frame, treat it as a misdetection and
  // keep the previous smoothed position instead of feeding bad data into
  // the filter (which would otherwise pollute its velocity estimate).
  if (smoothX === null) {
    smoothX = oneEuroX.filter(rawX, frameTime);
    smoothY = oneEuroY.filter(rawY, frameTime);
  } else {
    const jump = Math.hypot(rawX - smoothX, rawY - smoothY);
    if (jump <= MAX_JUMP_PX) {
      smoothX = oneEuroX.filter(rawX, frameTime);
      smoothY = oneEuroY.filter(rawY, frameTime);
    }
    // else: skip this frame's raw point entirely, curX/curY stay put below
  }
  const curX = smoothX, curY = smoothY;

  // ── Skeleton ─────────────────────────────────────────────────────────
  oCtx.strokeStyle = 'rgba(102,85,255,0.25)';
  oCtx.lineWidth   = 1.5;
  for (const [a,b] of CONN) {
    oCtx.beginPath(); oCtx.moveTo(cx(a),cy(a)); oCtx.lineTo(cx(b),cy(b)); oCtx.stroke();
  }
  for (let i = 0; i < 21; i++) {
    oCtx.fillStyle = i===8 ? '#6655ff' : i===4 ? '#00e5ff' : '#ff4d8f';
    oCtx.beginPath();
    oCtx.arc(cx(i), cy(i), (i===8||i===4) ? 6 : 3, 0, Math.PI*2);
    oCtx.fill();
  }

  // ── Cursor ring ───────────────────────────────────────────────────────
  _drawCursorRing(gesture, curX, curY);

  // Pinch connector
  if (gesture==='pinch') {
    oCtx.strokeStyle='#00e5ff'; oCtx.lineWidth=1.5; oCtx.setLineDash([4,3]);
    oCtx.beginPath(); oCtx.moveTo(cx(4),cy(4)); oCtx.lineTo(cx(8),cy(8)); oCtx.stroke();
    oCtx.setLineDash([]);
    [[cx(4),cy(4)],[cx(8),cy(8)]].forEach(([px,py])=>{
      oCtx.fillStyle='#00e5ff'; oCtx.beginPath(); oCtx.arc(px,py,6,0,Math.PI*2); oCtx.fill();
    });
  }

  // Horns trail
  if (gesture==='horns' && swipeBuf.length>1) {
    const sx=swipeBuf[0].x, ex=swipeBuf[swipeBuf.length-1].x, dir=ex-sx;
    const pct=Math.min(Math.abs(dir)/SWIPE_MIN_DIST,1);
    oCtx.strokeStyle=`rgba(255,221,0,${0.4+pct*0.5})`; oCtx.lineWidth=3;
    oCtx.beginPath(); oCtx.moveTo(sx,curY); oCtx.lineTo(ex,curY); oCtx.stroke();
    const d=dir>0?1:-1; oCtx.fillStyle='#ffdd00';
    oCtx.beginPath(); oCtx.moveTo(ex+d*12,curY); oCtx.lineTo(ex-d*7,curY-8); oCtx.lineTo(ex-d*7,curY+8); oCtx.fill();
  }

  const now = Date.now();

  // ══════════════════════════════════════════════════════════════════════
  //  GESTURE ACTIONS
  // ══════════════════════════════════════════════════════════════════════

  if (gesture === 'pinch') {
    _stopDraw();
    if (!isPinching) {
      isPinching=true; pinchStartX=curX; pinchStartY=curY;
      pinchOffsetX=0; pinchOffsetY=0;
      pinchSnapshot=dCtx.getImageData(0,0,dCanvas.width,dCanvas.height);
    } else {
      const dx=curX-pinchStartX, dy=curY-pinchStartY;
      if (Math.abs(dx-pinchOffsetX)>0.5||Math.abs(dy-pinchOffsetY)>0.5) {
        pinchOffsetX=dx; pinchOffsetY=dy;
        const t=document.createElement('canvas');
        t.width=dCanvas.width; t.height=dCanvas.height;
        t.getContext('2d').putImageData(pinchSnapshot,0,0);
        dCtx.clearRect(0,0,dCanvas.width,dCanvas.height);
        dCtx.drawImage(t,Math.round(dx),Math.round(dy));
      }
    }

  } else {
    if (isPinching) {
      isPinching=false; pinchSnapshot=null;
      pinchStartX=null; pinchStartY=null; pinchOffsetX=0; pinchOffsetY=0;
    }

    if (gesture === 'point') {
      hornsCtr=0; swipeBuf.length=0; swipeLocked=false;

      if (now < writeLockUntil) {
        // Still inside the brief post-swipe lock window — ignore this as
        // a drawing stroke so leftover hand motion from the slide gesture
        // doesn't leave stray ink on the canvas.
        _stopDraw();
      } else {
      document.getElementById('pulse').classList.add('draw');
      const cur = { x: curX, y: curY };

      if (!isDrawing) {
        strokeCount++;
        document.getElementById('c-strk').textContent = strokeCount;
        isDrawing=true; ptHist=[cur]; pressureW=PRESSURE_MIN;
        dCtx.globalCompositeOperation = isEraser ? 'destination-out' : 'source-over';
        dCtx.fillStyle = isEraser ? 'rgba(0,0,0,1)' : penColor;
        dCtx.beginPath();
        dCtx.arc(curX,curY,(isEraser?brushSize*5:brushSize)*PRESSURE_MIN/2,0,Math.PI*2);
        dCtx.fill();
        // NEW: count this as ink (unless erasing — erasing shouldn't trigger recognition)
        if (!isEraser) { updateTranscriptUI('writing…'); _markInk(curX, curY); }
      } else {
        const prev=ptHist[ptHist.length-1];
        const ddx=curX-prev.x, ddy=curY-prev.y;
        const d=Math.sqrt(ddx*ddx+ddy*ddy);
        if (d >= MIN_DIST) {
          ptHist.push(cur);
          if (ptHist.length>PT_HIST) ptHist.shift();
          const speed=Math.min(d/PRESSURE_SPDS,1);
          pressureW=lerp(pressureW, lerp(PRESSURE_MIN,PRESSURE_MAX,speed), 0.2);
          const segW=(isEraser?brushSize*5:brushSize)*pressureW;
          dCtx.globalCompositeOperation = isEraser ? 'destination-out' : 'source-over';
          dCtx.strokeStyle = isEraser ? 'rgba(0,0,0,1)' : penColor;
          dCtx.lineCap='round'; dCtx.lineJoin='round'; dCtx.shadowBlur=0;
          if (ptHist.length===PT_HIST) {
            drawCR(dCtx,ptHist[0],ptHist[1],ptHist[2],ptHist[3],segW);
          } else {
            dCtx.lineWidth=segW;
            dCtx.beginPath(); dCtx.moveTo(prev.x,prev.y); dCtx.lineTo(curX,curY); dCtx.stroke();
          }
          // NEW: extend ink bbox with every drawn segment
          if (!isEraser) _markInk(curX, curY);
        }
      }
      }

    } else if (gesture === 'horns') {
      _stopDraw();
      hornsCtr=Math.min(hornsCtr+1,CONFIRM_FRAMES+10);
      if (swipeLocked&&now>=swipeRearmAt) { swipeLocked=false; swipeBuf.length=0; }
      if (!swipeLocked&&hornsCtr>=CONFIRM_FRAMES) {
        swipeBuf.push({x:curX,t:now});
        while (swipeBuf.length&&now-swipeBuf[0].t>SWIPE_MAX_TIME) swipeBuf.shift();
        if (swipeBuf.length>=3) {
          const dx=swipeBuf[swipeBuf.length-1].x-swipeBuf[0].x;
          if (Math.abs(dx)>=SWIPE_MIN_DIST) {
            sendSlideCommand(dx<0?'prev':'next');
            swipeLocked=true; swipeRearmAt=now+SWIPE_REARM_TIME; swipeBuf.length=0;
            writeLockUntil = now + WRITE_LOCK_MS;
          }
        }
      }

    } else if (gesture === 'fist') {
      _stopDraw();
      hornsCtr=0; swipeBuf.length=0; swipeLocked=false;
      if (now-fistLastTime>FIST_COOLDOWN) { clearCanvas(); fistLastTime=now; }

    } else if (gesture === 'peace') {
      _stopDraw();
      hornsCtr=0; swipeBuf.length=0; swipeLocked=false;

    } else if (gesture === 'open') {
      _stopDraw(); hornsCtr=0;
      dCtx.globalCompositeOperation='destination-out';
      dCtx.beginPath(); dCtx.arc(curX,curY,38,0,Math.PI*2); dCtx.fill();
      dCtx.globalCompositeOperation='source-over';

    } else {
      _stopDraw(); hornsCtr=0;
    }
  }
}

// ── HELPERS ───────────────────────────────────────────────────────────────
function clearCanvas() {
  if (!dCtx) return;
  dCtx.clearRect(0, 0, dCanvas.width, dCanvas.height);
  strokeCount=0;
  if (isPinching) pinchSnapshot=dCtx.getImageData(0,0,dCanvas.width,dCanvas.height);
  document.getElementById('c-strk').textContent  = '0';
  document.getElementById('c-recog').textContent = '—';
  document.getElementById('c-conf').textContent  = '';
  // NEW: a board clear means any in-progress (unrecognized) word is gone too
  inkBounds = null;
  updateTranscriptUI('idle');
}

function toggleEraser() {
  isEraser=!isEraser;
  const b=document.getElementById('erase-btn');
  b.classList.toggle('on',isEraser);
  b.textContent=isEraser?'✓ ERASER':'◻ ERASER';
}

function setColor(c, el) {
  penColor=c; isEraser=false;
  document.getElementById('erase-btn').classList.remove('on');
  document.getElementById('erase-btn').textContent='◻ ERASER';
  document.querySelectorAll('.dot').forEach(d=>d.classList.remove('sel'));
  el.classList.add('sel');
}

// Manual "AI RECOGNIZE" button — runs the same local CNN segmentation
// pipeline as the auto word-recognizer, but on the whole canvas on demand
// (handy if you don't want to wait for the pause-trigger, or want to
// re-run after the auto pass guessed wrong).
async function recognizeLetter() {
  if (!dCanvas) return;
  const btn = document.getElementById('recog-btn');
  if (!cnnModel) {
    document.getElementById('c-recog').textContent = 'ERR';
    document.getElementById('c-conf').textContent = cnnLoadError
      ? 'model failed to load: ' + cnnLoadError.slice(0, 50)
      : 'model still loading…';
    return;
  }
  btn.textContent = '◌ ANALYSING...'; btn.style.color = '#ffaa30';

  // black-backed full-canvas copy, same convention the CNN expects
  const tmp = document.createElement('canvas');
  tmp.width = dCanvas.width; tmp.height = dCanvas.height;
  const tc = tmp.getContext('2d');
  tc.fillStyle = '#000000';
  tc.fillRect(0, 0, tmp.width, tmp.height);
  tc.drawImage(dCanvas, 0, 0);

  try {
    const glyphs = _segmentCharacters(tmp);
    if (DEBUG_GLYPHS) {
      const dbg = document.getElementById('glyph-debug-panel');
      if (dbg) dbg.innerHTML = '';
    }
    if (!glyphs.length) {
      document.getElementById('c-recog').textContent = '—';
      document.getElementById('c-conf').textContent = 'nothing detected';
    } else {
      let word = '';
      for (const g of glyphs) {
        let top3Text = '';
        const ch = tf.tidy(() => {
          const input = _glyphToTensor(g.canvas);
          const pred = cnnModel.predict(input);
          const probs = pred.dataSync();
          if (DEBUG_GLYPHS) {
            const ranked = Array.from(probs)
              .map((p, i) => ({ p, i }))
              .sort((a, b) => b.p - a.p)
              .slice(0, 3);
            top3Text = ranked
              .map(r => `${EMNIST_CLASSES[r.i]}: ${(r.p * 100).toFixed(1)}%`)
              .join('\n');
          }
          const idx = pred.argMax(-1).dataSync()[0];
          return EMNIST_CLASSES[idx] || '?';
        });
        if (DEBUG_GLYPHS) _debugShowGlyph(g.canvas, top3Text);
        word += ch;
      }
      document.getElementById('c-recog').textContent = word;
      document.getElementById('c-conf').textContent = glyphs.length + ' char(s), local CNN';
    }
  } catch (e) {
    document.getElementById('c-recog').textContent = 'ERR';
    document.getElementById('c-conf').textContent = e.message.slice(0, 60);
  }
  btn.textContent = '◈ AI RECOGNIZE'; btn.style.color = '';
}