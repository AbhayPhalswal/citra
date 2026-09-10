import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// =============================================================================
// CONSTANTS
// =============================================================================
// wss:// when the page itself is https:, ws:// when it's plain http: --
// derived from the page's own protocol rather than hardcoded, since
// citra_ui_server.py serves either depending on whether a TLS cert is
// present (see generate_tls_cert.py). An https: page can't open a plain
// ws:// socket at all in any modern browser (blocked as mixed content),
// so this has to track the page's real scheme, not assume one.
const WS_SCHEME = location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_SCHEME}//${location.hostname}:8765/ws`;
const RECONNECT_DELAY_MS = 2000;
const CAPTION_HOLD_MS = 4500;

// Amber/gold, matching --accent in style.css. Warm rather than the earlier
// cyan for two reasons that happen to agree: it's the JARVIS holo-globe
// look this is modelled on, and a warm hue emits substantially less blue
// light than cyan at the same perceived brightness — which is the actual
// mechanism behind "easier on the eyes at night", so the reference look
// and the eye-comfort goal point the same direction here.
const GOLD = new THREE.Color(0xffa832);        // deep amber — the base mass
const GOLD_BRIGHT = new THREE.Color(0xffe3a0);  // pale gold — the hot highlights

// Visual "energy" per assistant state, driving globe brightness, rotation
// and bloom together so it genuinely reads as a status indicator rather
// than generic ambient motion. IDLE is deliberately low: at rest this
// should be calm enough to leave on screen indefinitely.
const STATE_ENERGY = {
  IDLE: 0.12,
  LISTENING: 0.85,
  PROCESSING: 0.5,
  SPEAKING: 1.0,
  COOLDOWN: 0.2,
};

const PREFERS_REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// =============================================================================
// THREE.JS SCENE — the JARVIS holographic globe
// =============================================================================
// A dense sphere of glowing golden LINE SEGMENTS, not a shaded surface.
//
// This is a genuinely different construction from a fresnel-shaded orb,
// and the difference is the whole look: the reference is a hologram you
// can see THROUGH, built from thousands of individual bright strokes
// (radial spikes, tangential dashes, latitude rings, sweeping arcs) whose
// density and overlap is what makes it read as an object. A shaded sphere
// can't produce that no matter how it's lit, because its silhouette is
// solid — you never see the far side's linework through the near side.
//
// Performance: every layer below is ONE THREE.LineSegments with all its
// segments packed into a single BufferGeometry, so ~4600 line segments
// cost 4 draw calls total, not 4600. The geometry is built once at load
// and never touched again — animation is entirely group rotation plus
// material-level brightness, so the per-frame CPU cost is a handful of
// matrix and uniform updates regardless of how dense the globe looks.
const canvas = document.getElementById("core-canvas");
const renderer = new THREE.WebGLRenderer({
  canvas,
  // antialias OFF on purpose. Every line here is 1px and additively
  // blended into bloom, so MSAA has essentially nothing to smooth — it
  // would be a real per-frame cost buying an invisible difference.
  antialias: false,
  alpha: false,
  powerPreference: "high-performance",
});
// Capped below the panel's true devicePixelRatio (2+ on a 16" high-DPI
// OLED). A soft, glow-based scene reads identically at 1.5x while
// rendering ~44% fewer pixels than at 2x.
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);

const scene = new THREE.Scene();

// BASE_FOV/CAMERA_DISTANCE are tuned for a landscape/desktop aspect ratio.
// A FIXED vertical FOV crops the globe on a phone's portrait screen -- real
// bug, caught from an actual screenshot, not hypothetical: Three.js's `fov`
// is always the VERTICAL field of view; the effective HORIZONTAL fov is
// narrower whenever aspect (width/height) drops below 1, which is every
// phone held upright. At BASE_FOV=45 and this distance, the globe's outer
// content (arcs reaching ~1.12x GLOBE_RADIUS) needs roughly a 20 degree
// half-angle to stay fully on screen -- comfortably inside a desktop's wide
// horizontal frustum, but WIDER than what a narrow portrait frustum has
// available at the same vertical FOV, so the sides get cut off.
//
// computeFramingFov() below fixes this properly instead of picking one
// more fixed number that will just crop differently on some OTHER screen:
// on every resize (and once at startup), it computes the MINIMUM vertical
// FOV that guarantees CONTENT_RADIUS stays within frame on the CURRENT
// aspect ratio, for both the horizontal and vertical directions, and only
// grows past BASE_FOV when the aspect ratio actually requires it. Wide
// screens keep the original, deliberately-tuned 45 degrees; narrow ones
// automatically get however much more they need -- this is the actual
// "auto fix aspect ratio" fix, not a value guessed for one specific phone.
const BASE_FOV = 45;
const CAMERA_DISTANCE = 7.5;
const CONTENT_RADIUS = 2.7; // outer sweeping arcs reach ~1.12x GLOBE_RADIUS
                             // (2.05), plus a safety margin -- see GLOBE_
                             // RADIUS and arcRadius below for the source
                             // numbers this was computed from.

function computeFramingFov(aspect) {
  const theta = Math.atan(CONTENT_RADIUS / CAMERA_DISTANCE); // required half-angle, radians
  // Half-vertical-FOV needed so the VERTICAL extent fits:
  const verticalNeed = theta;
  // Half-vertical-FOV needed so the HORIZONTAL extent fits, given
  // tan(horizontalHalf) = tan(verticalHalf) * aspect -- solved for
  // verticalHalf so that horizontalHalf >= theta:
  const horizontalNeed = Math.atan(Math.tan(theta) / aspect);
  const requiredHalfFovDeg = Math.max(verticalNeed, horizontalNeed) * (180 / Math.PI);
  return Math.max(BASE_FOV, requiredHalfFovDeg * 2);
}

const camera = new THREE.PerspectiveCamera(
  computeFramingFov(window.innerWidth / window.innerHeight),
  window.innerWidth / window.innerHeight,
  0.1, 100,
);
camera.position.set(0, 0, CAMERA_DISTANCE);

// --- The globe --------------------------------------------------------------
const GLOBE_RADIUS = 2.05;

// Everything lives under one group so the whole hologram rotates as a
// single rigid object — the layers must stay locked together or they'd
// visibly shear apart, which instantly breaks the illusion that they're
// one structure.
const globe = new THREE.Group();
scene.add(globe);

// Shared material factory. `vertexColors` is the important part: it lets
// every individual segment carry its own brightness in the geometry
// buffer, so the globe has bright hot-spots and dim filler WITHOUT
// needing a separate material (and therefore a separate draw call) per
// brightness level. The reference's texture comes almost entirely from
// that per-stroke variation.
function lineMaterial(opacity) {
  return new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity,
    // Additive against true black: overlapping strokes ACCUMULATE, so
    // dense regions naturally burn brighter than sparse ones exactly the
    // way the reference's packed clusters do. Alpha blending would just
    // average them toward a flat wash instead.
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
}

// Pushes one segment (with a bright->dim gradient along its length) into
// the position/color arrays. The gradient is what makes strokes read as
// motion-streaked light rather than as uniform sticks.
function pushSegment(pos, col, a, b, color, brightness, tailFade = 0.25) {
  pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
  col.push(
    color.r * brightness, color.g * brightness, color.b * brightness,
    color.r * brightness * tailFade, color.g * brightness * tailFade, color.b * brightness * tailFade
  );
}

function buildLines(positions, colors, opacity) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const lines = new THREE.LineSegments(geo, lineMaterial(opacity));
  globe.add(lines);
  return lines;
}

// A random unit vector perpendicular to `n` — used to lay tangential
// strokes flat against the sphere surface in an arbitrary direction.
const _tmpAxis = new THREE.Vector3();
function randomTangent(n) {
  _tmpAxis.set(0, 0, 1);
  if (Math.abs(n.z) > 0.9) _tmpAxis.set(1, 0, 0);
  return new THREE.Vector3()
    .crossVectors(n, _tmpAxis)
    .normalize()
    .applyAxisAngle(n, Math.random() * Math.PI * 2);
}

function randomOnSphere() {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  return new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    Math.cos(phi)
  );
}

// ---- Layer 1: the dense "data city" surface --------------------------------
// Generated in CLUSTERS rather than uniformly at random. This is the
// single most important detail for matching the reference: uniform
// scatter produces an even grey-gold haze, whereas the reference has
// obvious dense districts separated by sparser gaps. Clustering is what
// creates that structure — pick a district center, then scatter strokes
// tightly around it.
const surfacePos = [];
const surfaceCol = [];
const CLUSTER_COUNT = 340;
const PER_CLUSTER = 30;

for (let c = 0; c < CLUSTER_COUNT; c++) {
  const center = randomOnSphere();
  const spread = 0.05 + Math.random() * 0.15;
  // Districts vary in overall brightness across a WIDE range, deliberately
  // skewed dark: squaring a 0-1 random pushes most districts toward the
  // low end, so a few hubs blaze and the majority sit faint. An even
  // spread here was what made the first attempt read as a uniform haze —
  // the reference's texture comes from most of it being dim.
  const districtBrightness = 0.2 + Math.pow(Math.random(), 1.5) * 0.8;

  for (let i = 0; i < PER_CLUSTER; i++) {
    const n = center
      .clone()
      .add(new THREE.Vector3(
        (Math.random() - 0.5) * spread * 2,
        (Math.random() - 0.5) * spread * 2,
        (Math.random() - 0.5) * spread * 2
      ))
      .normalize();

    // Slight radial jitter so strokes sit at marginally different depths.
    // Without this every stroke lands on one perfect shell and the globe
    // looks like a decal; with it, it gains real thickness.
    const r = GLOBE_RADIUS * (0.975 + Math.random() * 0.05);
    const base = n.clone().multiplyScalar(r);
    const shade = GOLD.clone().lerp(GOLD_BRIGHT, Math.random() * 0.8);
    const brightness = districtBrightness * (0.35 + Math.random() * 0.65);

    if (Math.random() < 0.28) {
      // Radial spike — the fine bristling quills around the reference's
      // rim. Kept SHORT: long spikes turned the silhouette into a sea
      // urchin, where the reference reads as a crisp sphere with a
      // fine fuzz on it.
      const len = GLOBE_RADIUS * (0.012 + Math.pow(Math.random(), 2) * 0.055);
      pushSegment(surfacePos, surfaceCol, base,
        n.clone().multiplyScalar(r + len), shade, brightness, 0.1);
    } else {
      // Tangential dash lying flat on the surface — the panelling and
      // circuitry between the spikes, and the bulk of the texture.
      const len = GLOBE_RADIUS * (0.015 + Math.random() * 0.075);
      pushSegment(surfacePos, surfaceCol, base,
        base.clone().addScaledVector(randomTangent(n), len), shade, brightness, 0.55);
    }
  }
}
const surfaceLines = buildLines(surfacePos, surfaceCol, 0.95);

// ---- Layer 2: latitude rings -----------------------------------------------
// These give the mass an unmistakable SPHERE reading. Without them the
// clustered strokes alone could be any blob; a few circles of latitude
// immediately establish curvature and a rotation axis.
const latPos = [];
const latCol = [];
const LAT_RINGS = 13;
const LAT_SEGMENTS = 190;

for (let i = 1; i < LAT_RINGS; i++) {
  const phi = (i / LAT_RINGS) * Math.PI;
  const ringRadius = Math.sin(phi) * GLOBE_RADIUS;
  const y = Math.cos(phi) * GLOBE_RADIUS;
  // Rings nearer the equator are brighter, which reinforces the sense of
  // a lit sphere rather than a flat wireframe.
  const ringBrightness = 0.07 + Math.sin(phi) * 0.13;

  for (let s = 0; s < LAT_SEGMENTS; s++) {
    // Broken, not continuous: skipping segments turns a solid circle into
    // a dashed data-track, which is far closer to the reference and also
    // keeps the rings from overpowering the surface detail.
    if (Math.random() < 0.32) continue;
    const t0 = (s / LAT_SEGMENTS) * Math.PI * 2;
    const t1 = ((s + 1) / LAT_SEGMENTS) * Math.PI * 2;
    pushSegment(latPos, latCol,
      new THREE.Vector3(Math.cos(t0) * ringRadius, y, Math.sin(t0) * ringRadius),
      new THREE.Vector3(Math.cos(t1) * ringRadius, y, Math.sin(t1) * ringRadius),
      GOLD, ringBrightness * (0.6 + Math.random() * 0.6), 1.0);
  }
}
const latitudeLines = buildLines(latPos, latCol, 0.9);

// ---- Layer 3: sweeping great-circle arcs -----------------------------------
// The few bold curves that swoop across the reference's face. Only four,
// each tilted differently — they're the strongest single visual element,
// so more than a handful would fight the surface detail for attention.
const arcPos = [];
const arcCol = [];
const ARC_COUNT = 4;
const ARC_SEGMENTS = 150;

for (let a = 0; a < ARC_COUNT; a++) {
  const tilt = new THREE.Euler(
    Math.random() * Math.PI,
    Math.random() * Math.PI,
    Math.random() * Math.PI
  );
  const arcRadius = GLOBE_RADIUS * (0.82 + Math.random() * 0.3);
  const sweep = Math.PI * (0.8 + Math.random() * 0.9); // partial, not closed
  const startAngle = Math.random() * Math.PI * 2;

  for (let s = 0; s < ARC_SEGMENTS; s++) {
    const t0 = startAngle + (s / ARC_SEGMENTS) * sweep;
    const t1 = startAngle + ((s + 1) / ARC_SEGMENTS) * sweep;
    // Fade in and out at the arc's ends so it dissolves into the globe
    // instead of stopping at a hard, obviously-artificial edge.
    const fade = Math.sin((s / ARC_SEGMENTS) * Math.PI);
    pushSegment(arcPos, arcCol,
      new THREE.Vector3(Math.cos(t0) * arcRadius, Math.sin(t0) * arcRadius, 0).applyEuler(tilt),
      new THREE.Vector3(Math.cos(t1) * arcRadius, Math.sin(t1) * arcRadius, 0).applyEuler(tilt),
      GOLD_BRIGHT, 0.32 * fade, 1.0);
  }
}
const arcLines = buildLines(arcPos, arcCol, 0.95);

// ---- Layer 4: the hot core -------------------------------------------------
// Small and bright, sitting inside the shell. The reference has a
// concentrated glow at the center that reads through the surrounding
// structure; this plus bloom reproduces it.
// Small and tight. An earlier attempt used a 0.2-radius sphere at 0.85
// opacity and it rendered as a flat pale DISC pasted over the middle —
// the reference's center is a concentrated knot of light, so this is
// kept small enough that bloom does the work of making it glow rather
// than the geometry being visibly a ball.
const core = new THREE.Mesh(
  new THREE.SphereGeometry(0.075, 16, 16),
  new THREE.MeshBasicMaterial({
    color: GOLD_BRIGHT,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
);
globe.add(core);

// Concentric inner rings around the core, replacing what used to be a
// translucent haze sphere. That haze was a real mistake: filling the
// interior with an even wash destroyed the single most important quality
// of the reference — that it's a HOLOGRAM you see straight through, with
// black between the strokes. These rings add the same sense of interior
// depth using more linework instead of a fill, so the inside stays dark.
const innerRingPos = [];
const innerRingCol = [];
for (let i = 0; i < 3; i++) {
  const rr = GLOBE_RADIUS * (0.16 + i * 0.13);
  const tilt = new THREE.Euler(Math.random() * Math.PI, Math.random() * Math.PI, 0);
  const segs = 120;
  for (let s = 0; s < segs; s++) {
    if (Math.random() < 0.25) continue;
    const t0 = (s / segs) * Math.PI * 2;
    const t1 = ((s + 1) / segs) * Math.PI * 2;
    pushSegment(innerRingPos, innerRingCol,
      new THREE.Vector3(Math.cos(t0) * rr, Math.sin(t0) * rr, 0).applyEuler(tilt),
      new THREE.Vector3(Math.cos(t1) * rr, Math.sin(t1) * rr, 0).applyEuler(tilt),
      GOLD_BRIGHT, 0.3 + Math.random() * 0.4, 1.0);
  }
}
const innerRings = buildLines(innerRingPos, innerRingCol, 0.8);

// Slight axial tilt — a globe spinning on a perfectly vertical screen axis
// looks like a machine part; a tilted one looks like an object in space.
globe.rotation.z = 0.19;

// ---- Sparse outer particle field -------------------------------------------
const PARTICLE_COUNT = 180;
const particlePositions = new Float32Array(PARTICLE_COUNT * 3);
for (let i = 0; i < PARTICLE_COUNT; i++) {
  const r = 4 + Math.random() * 6;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  particlePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
  particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
  particlePositions[i * 3 + 2] = r * Math.cos(phi) * 0.35;
}
const particleGeo = new THREE.BufferGeometry();
particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
const particles = new THREE.Points(
  particleGeo,
  new THREE.PointsMaterial({ color: GOLD, size: 0.016, transparent: true, opacity: 0.3 })
);
scene.add(particles);

// --- Post-processing --------------------------------------------------------
// Bloom is rendered at half resolution: it's an inherently blurred effect,
// so full-res input buys no visible sharpness while quadrupling the pass's
// pixel cost. Strength is far lower than the previous version's — heavy
// bloom on a true-black panel is precisely the halation that makes a dark
// UI tiring to look at, so this is tuned to suggest glow rather than
// flood the screen with it.
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2),
  0.6,  // strength
  0.8,  // radius — wide and soft, which diffuses the glow across the
        // linework instead of concentrating hot spots. This is what
        // fuses thousands of individually thin 1px lines into something
        // that reads as one luminous MASS, which is most of why the
        // reference looks like a hologram rather than a wireframe.
  0.18  // threshold
);
composer.addPass(bloomPass);

// --- Frame loop -------------------------------------------------------------
let currentEnergy = STATE_ENERGY.IDLE;
let targetEnergy = STATE_ENERGY.IDLE;
const clock = new THREE.Clock();

function animate() {
  // Clamped delta. clock.getDelta() returns the real wall-clock gap, which
  // can be seconds long after the tab was backgrounded or the machine
  // slept — feeding that straight into the integrators below would make
  // everything lurch forward on the first frame back. Clamping to ~3
  // frames' worth turns that lurch into a normal frame.
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.getElapsedTime();

  // Frame-rate-independent easing. The naive `x += (target - x) * k * dt`
  // form changes its effective speed with frame rate; this exponential
  // form converges at the same real-time rate at 60Hz, 120Hz or 30Hz,
  // which matters on a high-refresh laptop panel.
  currentEnergy += (targetEnergy - currentEnergy) * (1 - Math.exp(-3.5 * dt));

  const motion = PREFERS_REDUCED_MOTION ? 0.25 : 1;

  // The whole hologram spins as one rigid body — a single matrix update
  // for ~4600 line segments, which is why this stays cheap no matter how
  // dense the globe looks.
  globe.rotation.y += dt * (0.06 + currentEnergy * 0.3) * motion;

  // Layers counter-rotate slightly around a second axis. This is what
  // makes the structure feel alive and mechanical rather than like a
  // single texture spinning: parallax between layers at different depths
  // is the cue that tells the eye it's looking at a volume.
  latitudeLines.rotation.x = Math.sin(elapsed * 0.13 * motion) * 0.06;
  arcLines.rotation.y -= dt * (0.05 + currentEnergy * 0.22) * motion;
  arcLines.rotation.x += dt * 0.03 * motion;
  particles.rotation.y += dt * 0.012 * motion;

  // Brightness, not geometry, carries the state response. Rebuilding the
  // line buffers per frame would be enormously more expensive and look no
  // different — the globe already has all its detail, so "more energy"
  // just means the same structure lit harder.
  const breath = 1 + Math.sin(elapsed * 1.6 * motion) * 0.05 * (0.3 + currentEnergy);
  surfaceLines.material.opacity = 0.5 + currentEnergy * 0.5;
  arcLines.material.opacity = 0.45 + currentEnergy * 0.55;
  latitudeLines.material.opacity = 0.4 + currentEnergy * 0.5;
  core.material.opacity = (0.45 + currentEnergy * 0.5) * breath;
  core.scale.setScalar(breath);
  innerRings.material.opacity = 0.35 + currentEnergy * 0.45;
  innerRings.rotation.z += dt * (0.08 + currentEnergy * 0.35) * motion;

  bloomPass.strength = 0.4 + currentEnergy * 0.5;

  composer.render();
}
// setAnimationLoop, not a manual requestAnimationFrame chain: it's the
// renderer's own scheduler and it stops cleanly when the tab is hidden,
// rather than a raw rAF that keeps re-queuing itself.
renderer.setAnimationLoop(animate);

function handleResize() {
  const aspect = window.innerWidth / window.innerHeight;
  camera.aspect = aspect;
  camera.fov = computeFramingFov(aspect); // re-derived every resize, not
                                            // just aspect -- see
                                            // computeFramingFov's comment;
                                            // this is what keeps the globe
                                            // fully framed across ANY
                                            // orientation/device, not one
                                            // fixed number tuned for
                                            // whichever screen was tested.
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  bloomPass.setSize(window.innerWidth / 2, window.innerHeight / 2);
}
window.addEventListener("resize", handleResize);
// iOS Safari's own toolbar showing/hiding changes window.innerHeight
// WITHOUT necessarily firing a plain "resize" event reliably in every iOS
// version -- orientationchange is the more dependable signal specifically
// for the phone-rotation case, and costs nothing to also listen for here
// since handleResize() is idempotent (safe to call repeatedly with the
// same values).
window.addEventListener("orientationchange", handleResize);

// =============================================================================
// WEBSOCKET CLIENT
// =============================================================================
const statusDot = document.getElementById("status-dot");
const stateLabel = document.getElementById("state-label");
const captionLine = document.getElementById("caption-line");
const transcriptLine = document.getElementById("transcript-line");

let ws = null;
let captionHideTimer = null;
let transcriptHideTimer = null;

// Declared HERE, above connect(), not further down the file.
//
// ws.onopen reads queuedChat. `let` bindings sit in a temporal dead
// zone until their declaration runs, and touching one there throws a
// ReferenceError - even through `typeof`. With the declarations below
// connect(), any socket that opened before the file finished executing
// would kill the whole onopen handler.
const CHAT_TIMEOUT_MS = 45000;
let chatWatchdog = null;
let queuedChat = null;


function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusDot.classList.remove("offline");
    statusDot.classList.add("online");
    // Send anything typed while the socket was down. Without this the
    // message is simply lost and the only way to find out is that no
    // answer ever arrives - which is how "it's always thinking" started.
    if (typeof queuedChat === "string" && queuedChat) {
      const text = queuedChat;
      queuedChat = null;
      pendingBubble = addBubble("her", "thinking...", "pending");
      if (sendAction("chat", { text })) armChatWatchdog(text);
    }
  };

  ws.onclose = () => {
    statusDot.classList.remove("online");
    statusDot.classList.add("offline");
    setTimeout(connect, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleServerMessage(msg);
  };
}
connect();

// Mobile browsers (confirmed on iOS Safari via a real session — see
// app.js's git history for the log evidence) aggressively suspend a
// backgrounded tab's JavaScript AND kill its open WebSocket within
// roughly 30 seconds of losing focus or the screen locking. The passive
// setTimeout(connect, RECONNECT_DELAY_MS) in ws.onclose above can't help
// here: JS execution itself is frozen while backgrounded, so that timer
// doesn't even fire until the tab is foregrounded again — meaning the
// dashboard could sit disconnected for an arbitrary length of time after
// you unlock your phone, not just RECONNECT_DELAY_MS. Reconnecting
// explicitly the INSTANT the tab becomes visible again closes that gap
// at the moment JS actually resumes running, rather than waiting on a
// timer that was frozen the whole time anyway.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && (!ws || ws.readyState !== WebSocket.OPEN)) {
    connect();
  }
});

function sendAction(action, extra = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    // Previously a silent no-op — a button tap while disconnected did
    // NOTHING with zero feedback, the only indicator being a 6px status
    // dot easy to miss, especially on a phone. Confirmed via a real
    // session: connection dropped in the background, and any tap after
    // that point looked like the dashboard just wasn't responding, with
    // no way to tell why. A visible toast plus an immediate reconnect
    // attempt turns "silently does nothing" into "tells you what's
    // wrong and starts fixing it".
    showToast("Not connected — reconnecting...");
    connect();
    return false;
  }
  ws.send(JSON.stringify({ action, ...extra }));
  return true;
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "chat_reply":
      handleChatReply(msg);
      break;
    case "toast":
      showToast(msg.message);
      break;
    case "caption":
      showCaption(msg.text);
      break;
    case "transcript":
      showTranscript(msg.text);
      break;
    case "assistant_state":
      applyAssistantState(msg.state);
      break;
    case "hw_state":
      applyHardwareState(msg.relays, msg.ac);
      break;
    case "error":
      showToast(msg.message);
      break;
  }
}

// A failed action (unreachable board, an endpoint the board's current
// firmware doesn't have yet) used to just disappear and the button looked
// like it did nothing. This surfaces it.
let toastTimer = null;
function showToast(message) {
  let toast = document.getElementById("error-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "error-toast";
    toast.className = "error-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 5000);
}

function showCaption(text) {
  captionLine.textContent = text;
  captionLine.classList.add("show");
  clearTimeout(captionHideTimer);
  // Hold longer for longer lines — a caption should stay up roughly as
  // long as it takes to read, not a fixed duration regardless of length.
  captionHideTimer = setTimeout(
    () => captionLine.classList.remove("show"),
    CAPTION_HOLD_MS + text.length * 40
  );
}

function showTranscript(text) {
  transcriptLine.textContent = text;
  transcriptLine.classList.add("show");
  clearTimeout(transcriptHideTimer);
  transcriptHideTimer = setTimeout(
    () => transcriptLine.classList.remove("show"),
    CAPTION_HOLD_MS
  );
}

function applyAssistantState(state) {
  stateLabel.textContent = state;
  stateLabel.classList.toggle("active", state !== "IDLE" && state !== "COOLDOWN");
  targetEnergy = STATE_ENERGY[state] ?? STATE_ENERGY.IDLE;
}

// =============================================================================
// ROOM CONTROL DRAWER
// =============================================================================
const controlFab = document.getElementById("control-fab");
const drawer = document.getElementById("control-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawerClose = document.getElementById("drawer-close");

function openDrawer() {
  drawer.classList.add("open");
  drawerBackdrop.classList.add("show");
  controlFab.classList.add("drawer-open");
}
function closeDrawer() {
  drawer.classList.remove("open");
  drawerBackdrop.classList.remove("show");
  controlFab.classList.remove("drawer-open");
}

controlFab.addEventListener("click", openDrawer);
drawerClose.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer); // "click outside"
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDrawer();
});

// =============================================================================
// TALK TO CITRA — records via THIS device's microphone, sends to /voice
// =============================================================================
// This is genuinely a different thing from the room's own wake-word
// pipeline, not a remote copy of it: there's no "hey citra" detection
// here at all. Tapping this button IS the wake signal — the recording
// starts immediately and runs until you tap again (or the safety cap
// below fires), then citra_ui_server.py forwards the raw bytes to
// jarvis_voice_assistant.py's ingest endpoint, which decodes and routes
// it through the exact same transcription/routing/speaking pipeline the
// physical microphone uses. The spoken reply comes out of THIS MACHINE's
// speakers (the room's), matching this being a smart-ROOM system meant
// to be heard by whoever's in it — not routed back to the phone that
// sent the clip.
const micFab = document.getElementById("mic-fab");

const MIC_MAX_RECORDING_MS = 10000; // matches the local mic's own
                                      // MAX_RECORDING_SECONDS convention
                                      // in jarvis_voice_assistant.py

// getUserMedia/MediaRecorder only work in a "secure context" — https:// or
// localhost specifically. A plain http:// LAN address (exactly how this
// page is reached from another device unless generate_tls_cert.py has
// been run — see that file's docstring) does NOT qualify, and browsers
// enforce this at the platform level; no amount of app code works around
// it. Checked once at load, not only discovered when tapped, so the
// button visibly communicates "this won't work here" up front rather
// than failing silently on the first tap.
const micSupported = !!(navigator.mediaDevices && window.MediaRecorder);
if (!micSupported) {
  micFab.classList.add("unavailable");
  micFab.title = window.isSecureContext
    ? "Microphone not supported in this browser"
    : "Microphone needs a secure (https) connection to this page";
}

let mediaRecorder = null;
let recordedChunks = [];
let micState = "idle"; // idle | recording | processing
let micMaxTimer = null;

// Preference order matters: opus-in-webm is what Chrome/Firefox/Android
// produce and is the best available quality; mp4/aac is what iOS Safari
// actually produces — Safari doesn't support audio/webm AT ALL. Asking
// the browser what it actually supports, rather than hardcoding one
// assumption, is what makes this work on both without a platform branch;
// the server side (_decode_audio_to_pcm16_16k_mono) is already verified
// against both.
function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/mp4;codecs=mp4a.40.2",
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return ""; // let the browser fall back to its own default
}

async function startRecording() {
  if (!micSupported) {
    showToast("Microphone isn't available on this connection.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = pickMimeType();
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      // Release the microphone the instant recording stops. Holding it
      // open any longer than needed is both a real battery/privacy
      // concern on a phone and unnecessary — the whole clip is already
      // buffered in recordedChunks by this point.
      stream.getTracks().forEach((t) => t.stop());
      sendRecording();
    };

    mediaRecorder.start();
    micState = "recording";
    micFab.classList.add("recording");
    micMaxTimer = setTimeout(() => {
      if (micState === "recording") stopRecording();
    }, MIC_MAX_RECORDING_MS);
  } catch (err) {
    showToast(
      err.name === "NotAllowedError"
        ? "Microphone permission was denied."
        : `Couldn't access the microphone: ${err.message}`
    );
  }
}

function stopRecording() {
  clearTimeout(micMaxTimer);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop(); // onstop (above) does the actual send
  }
  micFab.classList.remove("recording");
  micFab.classList.add("processing");
  micState = "processing";
}

async function sendRecording() {
  const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
  recordedChunks = [];

  try {
    const response = await fetch("/voice", {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      showToast(body.message || `Couldn't process that (${response.status}).`);
    }
    // On success, the actual reply (transcript, caption, spoken response)
    // arrives through the SAME WebSocket broadcast every other command
    // already uses — nothing further to do here.
  } catch (err) {
    showToast("Couldn't reach the server to send that recording.");
  } finally {
    micFab.classList.remove("processing");
    micState = "idle";
  }
}

micFab.addEventListener("click", () => {
  if (micState === "idle") startRecording();
  else if (micState === "recording") stopRecording();
  // "processing": ignore taps, a request is already in flight.
});

// =============================================================================
// HARDWARE PANELS
// =============================================================================
const relayList = document.getElementById("relay-list");
const RELAY_COUNT = 4;
const relayToggles = [];

for (let i = 1; i <= RELAY_COUNT; i++) {
  const row = document.createElement("div");
  row.className = "relay-row";

  const label = document.createElement("div");
  label.className = "relay-label";
  label.textContent = `Light ${i}`;

  const toggle = document.createElement("label");
  toggle.className = "toggle-switch";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.dataset.relay = String(i);

  const track = document.createElement("span");
  track.className = "toggle-track";

  checkbox.addEventListener("change", () => {
    sendAction("relay_set", { relay: i, state: checkbox.checked });
  });

  toggle.appendChild(checkbox);
  toggle.appendChild(track);
  row.appendChild(label);
  row.appendChild(toggle);
  relayList.appendChild(row);
  relayToggles.push(checkbox);
}

document.querySelectorAll(".lights-panel [data-action]").forEach((btn) => {
  btn.addEventListener("click", () => sendAction(btn.dataset.action));
});

const acPowerBtn = document.getElementById("ac-power-btn");
const acTempValue = document.getElementById("ac-temp-value");
const fanDisplay = document.getElementById("fan-display");
const modeButtons = document.querySelectorAll(".mode-btn");

let acIsOn = false;

acPowerBtn.addEventListener("click", () => {
  acIsOn = !acIsOn;
  acPowerBtn.classList.toggle("on", acIsOn);
  sendAction("ac_power", { state: acIsOn });
});

document.querySelectorAll(".ac-panel [data-action]").forEach((btn) => {
  btn.addEventListener("click", () => sendAction(btn.dataset.action));
});

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sendAction("ac_mode", { value: btn.dataset.mode });
  });
});

function applyHardwareState(relays, ac) {
  if (relays) {
    relayToggles.forEach((checkbox, idx) => {
      const key = `relay${idx + 1}`;
      if (relays[key] === undefined) return;
      checkbox.checked = relays[key] === "on";
    });
  }
  if (ac) {
    if (ac.power !== undefined) {
      acIsOn = ac.power === "on";
      acPowerBtn.classList.toggle("on", acIsOn);
    }
    if (ac.temp !== undefined) acTempValue.textContent = ac.temp;
    if (ac.fan !== undefined) fanDisplay.textContent = ac.fan;
    if (ac.mode !== undefined) {
      modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === ac.mode));
    }
  }
}

/* ============================================================================
   TYPED CHAT + CONSOLE
   ============================================================================
   Routes through the SAME JarvisRouter the voice path uses (see
   citra_ui_server._answer), so a typed "turn on the bedroom lights" does
   exactly what the spoken one does. This is a second INPUT, not a second
   brain.

   READ-ALOUD USES THE BROWSER'S OWN VOICE, not Piper, and that is
   deliberate. If you are typing on your phone you want the answer read
   out of the phone in your hand, not out of the speaker in a room you
   may not be standing in. It also means read-aloud works when the voice
   assistant process is not running at all, and it never contends with
   Citra's own audio device. The trade is that it is not her voice.
   ============================================================================ */

const chatFab = document.getElementById("chat-fab");
const chatSheet = document.getElementById("chat-sheet");
const chatBackdrop = document.getElementById("chat-backdrop");
const chatClose = document.getElementById("chat-close");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatLog = document.getElementById("chat-log");
const chatSpeak = document.getElementById("chat-speak-toggle");
const consoleBtn = document.getElementById("console-btn");

let pendingBubble = null;

// Remembered per device. Wrapped because storage throws outright in some
// privacy modes rather than just returning null.
try {
  chatSpeak.checked = localStorage.getItem("citra.chat.speak") === "1";
} catch { /* default off */ }
chatSpeak.addEventListener("change", () => {
  try { localStorage.setItem("citra.chat.speak", chatSpeak.checked ? "1" : "0"); } catch {}
  if (!chatSpeak.checked && window.speechSynthesis) window.speechSynthesis.cancel();
});

function openChat() {
  chatSheet.classList.add("open");
  chatBackdrop.classList.add("show");
  chatFab.classList.add("sheet-open");
  // Focus only on a pointer-capable screen: focusing on a phone yanks the
  // keyboard up before the sheet has finished sliding, which lands the
  // animation in the wrong place.
  if (window.matchMedia("(hover: hover)").matches) chatInput.focus();
}
function closeChat() {
  chatSheet.classList.remove("open");
  chatBackdrop.classList.remove("show");
  chatFab.classList.remove("sheet-open");
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

chatFab.addEventListener("click", openChat);
chatClose.addEventListener("click", closeChat);
chatBackdrop.addEventListener("click", closeChat);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && chatSheet.classList.contains("open")) closeChat();
});

function addBubble(who, text, extraClass = "") {
  const empty = chatLog.querySelector(".chat-empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "chat-msg " + who + (extraClass ? " " + extraClass : "");
  el.textContent = text;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  addBubble("me", text);
  chatInput.value = "";
  // Echo and placeholder go up IMMEDIATELY. The round trip can be over a
  // second on the Smart Path, and a text box that swallows your message
  // and shows nothing reads as broken - the same dead-air problem
  // jarvis_presence solves for the spoken path.
  pendingBubble = addBubble("her", "thinking...", "pending");

  // If the socket is down, sendAction reconnects and returns false -
  // WITHOUT sending. The bubble used to sit on "thinking..." forever at
  // that point, which is the whole reason the chat looked permanently
  // stuck: a dropped connection is invisible, and the only clue was a
  // toast that is easy to miss. Queue the message and say so.
  if (!sendAction("chat", { text })) {
    failPending("Reconnecting… I'll send that as soon as I'm back.");
    queuedChat = text;
    return;
  }
  armChatWatchdog(text);
});

function armChatWatchdog(text) {
  clearTimeout(chatWatchdog);
  chatWatchdog = setTimeout(() => {
    if (!pendingBubble) return;
    failPending("No answer came back. Tap send again to retry.");
  }, CHAT_TIMEOUT_MS);
}

function failPending(message) {
  if (!pendingBubble) return;
  pendingBubble.textContent = message;
  pendingBubble.classList.remove("pending");
  pendingBubble.classList.add("failed");
  pendingBubble = null;
}

consoleBtn.addEventListener("click", () => {
  sendAction("open_console", { log: "assistant" });
});

function handleChatReply(msg) {
  clearTimeout(chatWatchdog);
  const text = msg.reply || "(no answer)";
  if (pendingBubble) {
    pendingBubble.textContent = text;
    pendingBubble.classList.remove("pending");
    if (!msg.ok) pendingBubble.classList.add("failed");
    pendingBubble = null;
  } else {
    addBubble("her", text, msg.ok ? "" : "failed");
  }
  if (msg.path) {
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = msg.path + " \u00b7 " + msg.ms + "ms";
    chatLog.appendChild(meta);
    chatLog.scrollTop = chatLog.scrollHeight;
  }
  if (chatSpeak.checked && msg.ok && window.speechSynthesis) {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
}
