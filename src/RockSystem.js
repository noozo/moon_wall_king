/**
 * RockSystem — scatter low-poly moon rocks near the player using InstancedMesh.
 *
 * Strategy
 * ─────────
 * • The moon surface is divided into a global spherical grid of cells
 *   (CELL_SIZE_RAD radians per cell ≈ 3 m arc).  Each cell's rocks are
 *   deterministically seeded by (latIdx, lonIdx) so they are stable as the
 *   player moves — no popping caused by player-relative coordinate drift.
 *
 * • Every frame we iterate the ~30×30 patch of cells within RENDER_RADIUS
 *   of the camera, build the instance matrix for each visible rock, and
 *   upload them to the GPU in a single draw call per shape type.
 *
 * • NUM_SHAPES distinct rock meshes are pre-generated (different seeds →
 *   different anisotropic scaling + gaussian bump deformations).  Rocks are
 *   assigned to shapes by hash so the same rock always has the same shape.
 *
 * • IcosahedronGeometry (detail=1) is made indexed via mergeVertices so that
 *   computeVertexNormals() produces smooth interpolated normals — giving the
 *   "low-poly but smooth-shaded" appearance requested.
 *
 * • Rocks use a ShaderMaterial with analytical-gradient FBM bump normals
 *   (same technique as the terrain).  This avoids triplanar seam artifacts
 *   and adds micro-roughness without any UV texture dependency.
 *
 * Rock sizes range from 0.05 to 0.75 game units (≈ 5 cm – 75 cm visual scale).
 * Only rocks within RENDER_RADIUS ≈ 40 game units are activated, matching the
 * highest-detail terrain LOD zone (LOD split distance level 5 ≈ 46 units).
 */

import * as THREE from 'three';
import { sunDirection, earthDirection } from './SunDirection.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Number of distinct rock silhouette types. */
const NUM_SHAPES = 8;

/**
 * Maximum instances per shape in the GPU buffer.
 * At average density (~0.8 rocks / cell, ~700 cells in range, 8 shapes)
 * the expected count per shape is ≈ 70.  200 gives comfortable headroom.
 */
const MAX_INSTANCES = 2000;

/**
 * Spherical grid cell size in radians.
 * 0.003 rad × moonRadius(1000) ≈ 3 game-unit arc per cell side.
 */
const CELL_SIZE_RAD = 0.003;

/**
 * Total number of longitude cells around the full 2π circle.
 * Used for modular wrapping so the rock grid is seamless at any meridian.
 * Without this, Math.atan2 returns [-π, +π] — the lonIdx jumps from +1047
 * to -1048 at the 180° meridian, giving completely different hash seeds
 * on either side and causing all rocks to regenerate when crossing that line.
 */
const TWO_PI = Math.PI * 2;
const TOTAL_LON_CELLS = Math.ceil(TWO_PI / CELL_SIZE_RAD); // ≈ 2095

/**
 * Render radius in game units.  Kept just below the LOD-5 split distance
 * (46 units) so rocks only appear on the highest-resolution terrain tiles.
 */
const RENDER_RADIUS    = 160;
const RENDER_RAD_SQ    = RENDER_RADIUS * RENDER_RADIUS;
/** Loose pre-cull: reject cells whose centre is more than this far away. */
const LOOSE_RAD_SQ     = RENDER_RAD_SQ * 4;

// ---------------------------------------------------------------------------
// Shaders — inlined to keep RockSystem self-contained
// ---------------------------------------------------------------------------

/**
 * Vertex shader for instanced rocks.
 *
 * instanceMatrix — Three.js automatically prepends "attribute mat4 instanceMatrix;"
 *   for any InstancedMesh; do NOT declare it again (duplicate = compile error).
 *
 * Transform chain:
 *   worldPos = modelMatrix * instanceMatrix * position
 *
 *   modelMatrix   = moonGroup world transform (rotation in Phase 2, translation
 *                   in Phase 3).  Provided automatically by Three.js.
 *   instanceMatrix = per-rock scale + align-to-surface + yaw.  Stored in
 *                   Moon-local space.
 *
 * Normal:
 *   instance normal matrix: mat3(instanceMatrix) with columns normalised
 *   (correct for uniform scale per rock, avoids full inverse/transpose).
 *   Then rotated into world space by mat3(modelMatrix) (rotation-only, safe).
 */
const ROCK_VERT = /* glsl */`
varying vec3 vNormal;
varying vec3 vWorldPos;

void main() {
  // Full world position: moonGroup rotation × instance placement × vertex
  vec4 instancePos = instanceMatrix * vec4(position, 1.0);
  vec4 worldPos4   = modelMatrix * instancePos;
  vWorldPos = worldPos4.xyz;

  // Instance normal matrix: normalize each column (uniform-scale shortcut)
  mat3 instNorm = mat3(instanceMatrix);
  instNorm[0] = normalize(instNorm[0]);
  instNorm[1] = normalize(instNorm[1]);
  instNorm[2] = normalize(instNorm[2]);
  // Apply moonGroup rotation on top (mat3(modelMatrix) = rotation part only)
  vNormal = mat3(modelMatrix) * (instNorm * normal);

  gl_Position = projectionMatrix * modelViewMatrix * instancePos;
}
`;

/**
 * Fragment shader for rocks.
 *
 * Uses the same analytical-gradient FBM bump-normal technique as the terrain
 * shader.  No texture sampling at all — avoids every possible seam artifact:
 *   • No triplanar UV projection seams
 *   • No mipmap/anisotropy edge cases
 *
 * Two noise scales perturb the interpolated geometry normal:
 *   • Coarse (vWorldPos × 3)  — ~30 cm undulations  (rock face character)
 *   • Fine   (vWorldPos × 18) — ~5 cm micro-roughness (surface grain)
 *
 * Albedo variation is derived from the coarse FBM value itself (free,
 * already computed) so each rock surface has subtle brightness variation
 * without any additional texture lookup.
 */
const ROCK_FRAG = /* glsl */`
#define PI 3.14159265358979

varying vec3 vNormal;
varying vec3 vWorldPos;

uniform vec3 uSunDirection;
uniform vec3 uEarthDirection;
uniform sampler2D uShadowMap0;
uniform sampler2D uShadowMap1;
uniform sampler2D uShadowMap2;
uniform mat4 uShadowMatrix0;
uniform mat4 uShadowMatrix1;
uniform mat4 uShadowMatrix2;
uniform vec3 uCascadeSplits;
uniform float uCamNear;
uniform float uCamFar;
uniform float uWireframe;

// Player spotlight uniforms
uniform float     uSpotlightOn;
uniform vec3      uSpotlightPos;
uniform vec3      uSpotlightDir;
uniform float     uSpotlightAngle;
uniform float     uSpotlightRange;
uniform sampler2D uSpotlightShadowMap;
uniform mat4      uSpotlightMatrix;

// 16-tap Poisson disk (used by both CSM and spotlight shadows)
const vec2 POISSON_DISK[16] = vec2[16](
  vec2(-0.94201624, -0.39906216),
  vec2( 0.94558609, -0.76890725),
  vec2(-0.09418410, -0.92938870),
  vec2( 0.34495938,  0.29387760),
  vec2(-0.91588581,  0.45771432),
  vec2(-0.81544232, -0.87912464),
  vec2(-0.38277543,  0.27676845),
  vec2( 0.97484398,  0.75648379),
  vec2( 0.44323325, -0.97511554),
  vec2( 0.53742981, -0.47373420),
  vec2(-0.26496911, -0.41893023),
  vec2( 0.79197514,  0.19090188),
  vec2(-0.24188840,  0.99706507),
  vec2(-0.81409955,  0.91437590),
  vec2( 0.19984126,  0.78641367),
  vec2( 0.14383161, -0.14100790)
);

// --- Analytical-gradient value noise (identical to terrain.frag) -----------

float hash(vec2 p) {
  p  = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

// Returns vec3(dv/dp.x, dv/dp.y, value)
vec3 noiseGV(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash(i),                  b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
  vec2 u  = f * f * (3.0 - 2.0 * f);
  vec2 du = 6.0 * f * (1.0 - f);
  float val = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  float gx  = du.x * ((b - a) * (1.0 - u.y) + (d - c) * u.y);
  float gy  = du.y * ((c - a) * (1.0 - u.x) + (d - b) * u.x);
  return vec3(gx, gy, val);
}

const mat2 ROT = mat2(0.86602540378, 0.5, -0.5, 0.86602540378); // 30° per octave

vec3 fbmGV4(vec2 p) {
  float val = 0.0; vec2 grad = vec2(0.0); float amp = 0.5;
  vec3 n;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy;
  return vec3(grad, val);
}

// ---------------------------------------------------------------------------

void main() {
  vec3 N   = normalize(vNormal);
  vec3 sun = normalize(uSunDirection);

  // --- Analytical bump normals -------------------------------------------
  // Tangent frame anchored to the interpolated geometry normal N.
  vec3 tX;
  if (abs(N.y) < 0.999) {
    tX = normalize(cross(vec3(0.0, 1.0, 0.0), N));
  } else {
    tX = normalize(cross(vec3(1.0, 0.0, 0.0), N));
  }
  vec3 tZ = normalize(cross(N, tX));

  vec3 b1 = fbmGV4(vWorldPos.xz * 3.0);
  vec3 b2 = fbmGV4(vWorldPos.xz * 18.0);

  // Bump normals for SUN lighting: masked by sun-facing factor so that FBM
  // gradients don't tilt shadow-facing normals toward the sun (bright seams).
  float baseDiff  = max(dot(N, sun), 0.0);
  vec2 bumpGradSun = (b1.xy * 0.16 + b2.xy * 0.08) * baseDiff;
  vec3 bN_sun      = normalize(N - tX * bumpGradSun.x - tZ * bumpGradSun.y);

  // Bump normals for SPOTLIGHT: no masking — spotlight direction varies per
  // fragment so there's no "always-bright-seam" risk.  Giving bumps at night
  // makes rocks look properly textured under the flashlight.
  vec2 bumpGrad = b1.xy * 0.16 + b2.xy * 0.08;
  vec3 bN       = normalize(N - tX * bumpGrad.x - tZ * bumpGrad.y);

  // --- Albedo ------------------------------------------------------------
  // Base value matches the terrain's desaturated LROC texture (~0.45-0.55).
  // Lunar rocks are the same regolith material — they only look darker in
  // photos because of self-shadowing and viewing angle, not intrinsic albedo.
  float fbmVal = b1.z;
  vec3 albedo = vec3(clamp(0.46 + (fbmVal - 0.5) * 0.10, 0.28, 0.64));

  // --- Sun lighting (physically-motivated hemisphere ambient) ---------------
  //
  // sunElev = dot(sunDir, outward moon normal at this rock).
  // When sunElev <= 0 the moon body is between this rock and the sun —
  // no direct illumination can reach it (moon is opaque).
  // This gates BOTH the direct Lambert term AND the sky ambient so rocks
  // on the night side receive zero sun contribution regardless of face normal.
  vec3  moonNorm = normalize(vWorldPos);
  float sunElev  = max(dot(sun, moonNorm), 0.0);

  // Soft terminator: ramp direct sun from 0→1 over a small horizon band
  // so the day/night boundary isn't a hard step on rock surfaces.
  float moonOcclusion = smoothstep(0.0, 0.08, sunElev);

  float nDotSun  = dot(bN_sun, sun);
  float directSun = max(nDotSun, 0.0) * moonOcclusion;
  float diff     = min(directSun + sunElev * 0.9, 1.0);
  
  // --- Shadows -----------------------------------------------------------
  float z_ndc     = gl_FragCoord.z * 2.0 - 1.0;
  float viewDepth = (2.0 * uCamNear * uCamFar)
                  / (uCamFar + uCamNear - z_ndc * (uCamFar - uCamNear));

  // Select cascade and shadow matrix
  mat4 shadowMatrix;
  int  cascade;
  if (viewDepth < uCascadeSplits.x) {
    cascade      = 0;
    shadowMatrix = uShadowMatrix0;
  } else if (viewDepth < uCascadeSplits.y) {
    cascade      = 1;
    shadowMatrix = uShadowMatrix1;
  } else {
    cascade      = 2;
    shadowMatrix = uShadowMatrix2;
  }

  vec4 shadowUVW = shadowMatrix * vec4(vWorldPos, 1.0);
  vec2 shadowUV  = shadowUVW.xy;

  float shadow = 1.0;
  if (shadowUV.x > 0.001 && shadowUV.x < 0.999 &&
      shadowUV.y > 0.001 && shadowUV.y < 0.999) {

    float currentDepth = shadowUVW.z;
    float bias = max(0.0008 * (1.0 - diff), 0.0002);

    // 16-tap Poisson disk

    // Per-fragment random rotation — breaks repeating banding pattern
    float phi    = fract(sin(dot(shadowUV, vec2(127.1, 311.7))) * 43758.5453) * 6.28318;
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float spread = 5.0 / 1024.0;

    float sum = 0.0;
    for (int i = 0; i < 16; i++) {
      vec2 rotated  = vec2(
        POISSON_DISK[i].x * cosPhi - POISSON_DISK[i].y * sinPhi,
        POISSON_DISK[i].x * sinPhi + POISSON_DISK[i].y * cosPhi
      );
      vec2 sampleUV = clamp(shadowUV + rotated * spread, 0.001, 0.999);

      float d;
      if      (cascade == 0) d = texture2D(uShadowMap0, sampleUV).r;
      else if (cascade == 1) d = texture2D(uShadowMap1, sampleUV).r;
      else                   d = texture2D(uShadowMap2, sampleUV).r;

      sum += (currentDepth - bias > d) ? 0.0 : 1.0;
    }
    shadow = mix(0.15, 1.0, sum / 16.0);
  }
  
  vec3  eDir       = normalize(uEarthDirection);
  float earthFace  = max(dot(bN_sun, eDir), 0.0);
  vec3  earthshine = vec3(0.45, 0.65, 1.0) * earthFace * 0.018;
  float ambient    = 0.04;

  // --- Player spotlight ---------------------------------------------------
  // Uses unmasked bN (bumps active at night) and wrapped diffuse so that
  // tilted convex-hull faces still receive meaningful illumination from the
  // flashlight rather than appearing nearly black.
  float spotLight = 0.0;
  if (uSpotlightOn > 0.5) {
    vec3  toFrag   = vWorldPos - uSpotlightPos;
    float dist     = length(toFrag);
    vec3  fragDir  = toFrag / dist;
    vec3  L        = -fragDir;

    float cosAngle = dot(fragDir, uSpotlightDir);
    float spotCos  = cos(uSpotlightAngle);

    if (cosAngle > spotCos && dist < uSpotlightRange) {
      float spotAtten = smoothstep(spotCos, spotCos + 0.05, cosAngle);
      float distAtten = 1.0 - smoothstep(0.0, uSpotlightRange, dist);

      // Wrapped diffuse: small wrap (0.12) so only slightly-past-perpendicular
      // faces get some light — avoids the "every face is white" over-exposure
      // that a large wrap caused while still softening the hard Lambert cutoff.
      float nDotL       = dot(bN, L);
      float wrappedDiff = max(nDotL * 0.88 + 0.12, 0.0);
      spotLight = wrappedDiff * spotAtten * distAtten * 2.0;
    }
  }

  vec3 totalLight = vec3(diff * shadow + ambient) + vec3(spotLight);
  gl_FragColor = vec4(albedo * totalLight + albedo * earthshine, 1.0);

  if (uWireframe > 0.5) gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------------------
// RockSystem
// ---------------------------------------------------------------------------

export class RockSystem {
  /**
   * @param {THREE.Scene}   scene
   * @param {number}        moonRadius   Game-space sphere radius (e.g. 1000)
   * @param {{ getHeightAt: (nx: number, ny: number, nz: number) => number }} terrainSystem
   * @param {THREE.Vector3} sunDir       Normalised world-space sun direction (unused, kept for API compat)
   * @param {THREE.Texture} _detailTexture  Unused (bump normals are procedural)
   * @param {THREE.Group}   moonGroup    Parent group — rocks must live here so they
   *                                     move with the Moon body in Phase 3.
   */
  constructor(scene, moonRadius, terrainSystem, _sunDir, _detailTexture, moonGroup) {
    this._scene      = scene;
    this._moonRadius = moonRadius;
    this._terrain    = terrainSystem;
    this._parent     = moonGroup ?? scene;  // fall back to scene if not provided

    // Shared material.  FrontSide is correct: ConvexGeometry always produces
    // outward-pointing normals with no flipped triangles.
this._material = new THREE.ShaderMaterial({
  vertexShader:   ROCK_VERT,
  fragmentShader: ROCK_FRAG,
  side: THREE.FrontSide,
  uniforms: {
    uSunDirection:   { value: sunDirection },
    uEarthDirection: { value: earthDirection },
    uShadowMap0:     { value: null },
    uShadowMap1:     { value: null },
    uShadowMap2:     { value: null },
    uShadowMatrix0:  { value: new THREE.Matrix4() },
    uShadowMatrix1:  { value: new THREE.Matrix4() },
    uShadowMatrix2:  { value: new THREE.Matrix4() },
    uCascadeSplits:  { value: new THREE.Vector3(20, 200, 2000) },
    uCamNear:        { value: 0.5 },
    uCamFar:         { value: 350000.0 },
    uWireframe:      { value: 0 },
    // Player spotlight
    uSpotlightOn:    { value: 0 },
    uSpotlightPos:   { value: new THREE.Vector3() },
    uSpotlightDir:   { value: new THREE.Vector3() },
    uSpotlightAngle: { value: 0.5 },
    uSpotlightRange: { value: 30.0 },
    uSpotlightShadowMap: { value: null },
    uSpotlightMatrix:    { value: new THREE.Matrix4() },
  },
});

    // One InstancedMesh per rock shape type
    this._meshes = [];
    for (let s = 0; s < NUM_SHAPES; s++) {
      const geo  = this._makeRockGeo(s * 73_856 + 12_479);
      const mesh = new THREE.InstancedMesh(geo, this._material, MAX_INSTANCES);
      mesh.count = 0;
      mesh.layers.enable(1);  // visible to shadow camera (layer 1) and main camera (layer 0)
      mesh.frustumCulled = false;
      this._parent.add(mesh);
      this._meshes.push(mesh);
    }

    // Pre-allocated temporaries — avoids GC pressure in the update loop
    this._mtx    = new THREE.Matrix4();
    this._pos    = new THREE.Vector3();
    this._scl    = new THREE.Vector3();
    this._alignQ = new THREE.Quaternion();
    this._yawQ   = new THREE.Quaternion();
    this._surfN  = new THREE.Vector3();
    this._upRef  = new THREE.Vector3(0, 1, 0);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Per-frame update.  Iterates the global spherical grid cells within
   * RENDER_RADIUS of the camera, computes each rock's world transform, and
   * uploads the instance matrix buffer for every shape type.
   *
   * @param {THREE.Vector3} cameraPos  World-space camera position
   */
  update(cameraPos) {
    const counts = new Array(NUM_SHAPES).fill(0);

    // Camera → sphere direction → lat/lon
    const camR = cameraPos.length();
    const lat0 = Math.asin(Math.max(-1, Math.min(1, cameraPos.y / camR)));

    // Normalize longitude to [0, 2π) to avoid the atan2 sign flip at ±180°.
    // atan2 returns [-π, +π]; crossing 180°E flips the sign and causes lonIdx
    // to jump by ~2094 cells → completely different hash seeds → rock pop-in.
    // With [0, 2π), the wrap point is at 0°/360° and we handle it with
    // modular arithmetic on lonIdx (see inner loop below).
    const lon0 = ((Math.atan2(cameraPos.z, cameraPos.x) % TWO_PI) + TWO_PI) % TWO_PI;

    // Iterate a square patch of lat/lon cells around the camera position
    const HALF    = Math.ceil(RENDER_RADIUS / (CELL_SIZE_RAD * this._moonRadius)) + 1;
    const latIdx0 = Math.floor(lat0 / CELL_SIZE_RAD);
    const lonIdx0 = Math.floor(lon0 / CELL_SIZE_RAD);

    for (let dlat = -HALF; dlat <= HALF; dlat++) {
      for (let dlon = -HALF; dlon <= HALF; dlon++) {
        const latIdx = latIdx0 + dlat;
        // Wrap longitude index into [0, TOTAL_LON_CELLS) so the grid is
        // seamless everywhere, including the 0°/360° prime-meridian wrap.
        const lonIdx = ((lonIdx0 + dlon) % TOTAL_LON_CELLS + TOTAL_LON_CELLS) % TOTAL_LON_CELLS;

        // Cell centre in world space (on the reference sphere, ignoring elevation)
        const cLat = (latIdx + 0.5) * CELL_SIZE_RAD;
        const cLon = (lonIdx + 0.5) * CELL_SIZE_RAD;
        const cCos = Math.cos(cLat);
        const cWX  = cCos * Math.cos(cLon) * this._moonRadius;
        const cWY  = Math.sin(cLat)        * this._moonRadius;
        const cWZ  = cCos * Math.sin(cLon) * this._moonRadius;

        // Loose cull: skip cells whose centre is far outside RENDER_RADIUS
        const ddx = cWX - cameraPos.x;
        const ddy = cWY - cameraPos.y;
        const ddz = cWZ - cameraPos.z;
        if (ddx * ddx + ddy * ddy + ddz * ddz > LOOSE_RAD_SQ) continue;

        // Deterministic rock count for this cell (stable, hash-seeded)
        const nRocks = this._cellCount(latIdx, lonIdx);

        for (let r = 0; r < nRocks; r++) {
          // Rock lat/lon offset within the cell (deterministic)
          const rLat = cLat + (this._h(latIdx, lonIdx, r * 7    ) - 0.5) * CELL_SIZE_RAD;
          const rLon = cLon + (this._h(latIdx, lonIdx, r * 7 + 1) - 0.5) * CELL_SIZE_RAD;

          // Rock sphere direction (clamped away from exact poles)
          const rl   = Math.max(-Math.PI / 2 + 1e-4, Math.min(Math.PI / 2 - 1e-4, rLat));
          const rCos = Math.cos(rl);
          const rdx  = rCos * Math.cos(rLon);
          const rdy  = Math.sin(rl);
          const rdz  = rCos * Math.sin(rLon);

          // Query real LOLA elevation to place rock on the terrain surface
          const hgt = this._terrain.getHeightAt(rdx, rdy, rdz);
          const rR  = this._moonRadius + hgt;
          this._pos.set(rdx * rR, rdy * rR, rdz * rR);

          // Exact per-rock distance cull
          const fx = this._pos.x - cameraPos.x;
          const fy = this._pos.y - cameraPos.y;
          const fz = this._pos.z - cameraPos.z;
          if (fx * fx + fy * fy + fz * fz > RENDER_RAD_SQ) continue;

          // Shape and instance-count guard
          const shape = (this._h(latIdx, lonIdx, r * 7 + 3) * NUM_SHAPES | 0) % NUM_SHAPES;
          if (counts[shape] >= MAX_INSTANCES) continue;

          // Rock size in game units: 0.05 → 0.75
          const sGU = 0.05 + this._h(latIdx, lonIdx, r * 7 + 2) * 0.70;
          this._scl.setScalar(sGU);

          // Orientation:
          //   1. Align rock's local +Y to the sphere surface normal (≈ rdx/rdy/rdz).
          //   2. Apply random yaw rotation around that normal.
          this._surfN.set(rdx, rdy, rdz); // already unit-length
          this._alignQ.setFromUnitVectors(this._upRef, this._surfN);
          const yaw = this._h(latIdx, lonIdx, r * 7 + 4) * Math.PI * 2;
          this._yawQ.setFromAxisAngle(this._surfN, yaw);
          // Combined: yaw THEN align  →  yawQ × alignQ
          this._yawQ.multiply(this._alignQ);

          // Lift rock so its base sits on the terrain surface rather than
          // being centred at the surface point (which embeds the lower half
          // underground).  0.40 × size matches the average half-height of the
          // rock geometry (sy is biased toward ~0.45 in _makeRockGeo).
          this._pos.addScaledVector(this._surfN, 0.40 * sGU);

          this._mtx.compose(this._pos, this._yawQ, this._scl);
          this._meshes[shape].setMatrixAt(counts[shape], this._mtx);
          counts[shape]++;
        }
      }
    }

    // Upload counts + instance matrices to GPU
    for (let i = 0; i < NUM_SHAPES; i++) {
      this._meshes[i].count = counts[i];
      this._meshes[i].instanceMatrix.needsUpdate = true;
    }
  }

  /** Toggle wireframe on all rock meshes (X key diagnostic). */
  setWireframe(enabled) {
    this._material.wireframe = enabled;
    this._material.uniforms.uWireframe.value = enabled ? 1 : 0;
  }

  setSpotlightUniforms(on, pos, dir, angle, range, shadowTex, shadowMatrix) {
    const u = this._material.uniforms;
    u.uSpotlightOn.value    = on ? 1 : 0;
    u.uSpotlightPos.value.copy(pos);
    u.uSpotlightDir.value.copy(dir);
    u.uSpotlightAngle.value = angle;
    u.uSpotlightRange.value = range;
    u.uSpotlightShadowMap.value = shadowTex;
    u.uSpotlightMatrix.value.copy(shadowMatrix);
  }

  dispose() {
    for (const m of this._meshes) {
      this._parent.remove(m);
      m.geometry.dispose();
    }
    this._material.dispose();
  }

  // ---------------------------------------------------------------------------
  // Internal — deterministic hash
  // ---------------------------------------------------------------------------

  /**
   * Fast 32-bit integer hash → [0, 1).
   * Three independent integer inputs ensure no cross-parameter correlation.
   */
  _h(a, b, c = 0) {
    let h = (Math.imul(a | 0, 0x4b511117) ^
             Math.imul(b | 0, 0x8e3a1789) ^
             Math.imul(c | 0, 0x12345679)) | 0;
    h ^= h >>> 14;
    h  = Math.imul(h, 0x9e3779b9) | 0;
    h ^= h >>> 16;
    return ((h >>> 0) & 0xFFFF) / 65536.0;
  }

  /**
   * How many rocks live in the cell at (latIdx, lonIdx).
   *
   * Distribution chosen to give lunar highland rock density:
   *   35 % → 0 rocks   (open regolith)
   *   40 % → 1 rock
   *   18 % → 2 rocks
   *    7 % → 3 rocks
   */
  _cellCount(latIdx, lonIdx) {
    const v = this._h(latIdx, lonIdx, 0xBEEF);
    if (v < 0.35) return 0;
    if (v < 0.75) return 1;
    if (v < 0.93) return 2;
    return 3;
  }

  // ---------------------------------------------------------------------------
  // Internal — procedural rock geometry
  // ---------------------------------------------------------------------------

  /**
   * Generates one low-poly smooth-shaded rock mesh.
   *
   * Process:
   *   1. IcosahedronGeometry(1, 1)  — 80 faces, non-indexed by default.
   *   2. mergeVertices()            — converts to ~42 shared-vertex indexed
   *                                   geometry so computeVertexNormals() can
   *                                   average across adjacent faces (smooth).
   *   3. Wide anisotropic scaling   — sx/sz 0.30–1.60, sy 0.15–0.75.
   *                                   Produces slabs, chunks, and columns.
   *   4. High-sharpness gaussian bumps — 5-9 bumps per rock, amplitude ±0.42,
   *                                   sharpness 6–22.  HIGH sharpness creates
   *                                   tight protrusions/depressions separated
   *                                   by flat regions → angular, faceted look.
   *   5. Per-vertex roughness jitter — small independent random offset per
   *                                   vertex (±0.10) breaks remaining symmetry.
   *   6. computeVertexNormals()     — smooth Gouraud shading across shared verts.
   *
   * @param {number} seed  Integer seed — each of the NUM_SHAPES uses a distinct value.
   * @returns {THREE.BufferGeometry}
   */
  /**
   * Generates one rock mesh as a convex hull of random points.
   *
   * Why ConvexGeometry instead of displaced icosahedron:
   *   Any per-vertex displacement applied to a subdivided sphere can flip
   *   triangle winding (creating back-facing triangles and visible holes
   *   under FrontSide culling).  There is no displacement magnitude that is
   *   simultaneously "enough to look varied" and "guaranteed never to flip".
   *
   *   A convex hull is mathematically guaranteed:
   *     • Closed manifold — no missing faces, no T-junctions
   *     • All normals point outward — FrontSide culling always correct
   *     • No cracks, holes, or winding inversions
   *
   * Shape variety comes from:
   *   1. Anisotropic scaling (sx, sy, sz) — slabs, chunks, ovals
   *   2. Random point count (14–22) — more points → rounder, fewer → angrier
   *   3. Per-point radius jitter (±25%) — irregular protrusions
   *   4. Random point distribution on the sphere
   */
  /**
   * Generate one rock mesh as a displaced IcosahedronGeometry.
   *
   * Why displaced icosahedron instead of ConvexGeometry:
   *   Convex hulls of 14–22 points give only 20–40 faces — too few for natural
   *   rock detail.  An IcosahedronGeometry at detail=2 gives 320 faces.
   *   Per-vertex 3-octave 3D FBM displacement creates irregular protrusions,
   *   concavities and surface roughness that reads convincingly as real rock.
   *
   * Safety: displacement amplitude is kept well below the minimum face edge
   * length so no face can flip winding.  The geometry remains a valid manifold
   * after displacement and computeVertexNormals() gives smooth Gouraud shading.
   */
  _makeRockGeo(seed) {
    // Anisotropic scale — flatter in Y so rocks sit on the ground naturally.
    const sx = 0.55 + this._h(seed, 10) * 0.70;   // 0.55 – 1.25
    const sy = 0.25 + this._h(seed, 11) * 0.45;   // 0.25 – 0.70
    const sz = 0.55 + this._h(seed, 12) * 0.70;   // 0.55 – 1.25

    // Subdivision level 2 → 320 faces, 162 vertices.  Enough complexity for
    // convincing rock silhouettes at all viewing distances.
    const geo = new THREE.IcosahedronGeometry(1.0, 2);
    const pos = geo.attributes.position.array;
    const numVerts = pos.length / 3;

    for (let i = 0; i < numVerts; i++) {
      const ox = pos[i * 3], oy = pos[i * 3 + 1], oz = pos[i * 3 + 2];

      // Normalised sphere direction for this vertex.
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
      const nx = ox / len, ny = oy / len, nz = oz / len;

      // 3-octave 3D FBM — each octave doubles frequency and halves amplitude.
      // Frequencies tuned so that coarse bumps (oct0) give the overall boulder
      // silhouette and fine bumps (oct2) add surface roughness / facets.
      const d0 = this._noise3D(seed,       nx * 2.2, ny * 2.2, nz * 2.2) * 0.22;
      const d1 = this._noise3D(seed + 137, nx * 4.8, ny * 4.8, nz * 4.8) * 0.11;
      const d2 = this._noise3D(seed + 271, nx * 10.5, ny * 10.5, nz * 10.5) * 0.055;

      const r = 1.0 + d0 + d1 + d2;  // r always > 0 (max disp ±0.385, so r > 0.615)

      pos[i * 3]     = nx * r * sx;
      pos[i * 3 + 1] = ny * r * sy;
      pos[i * 3 + 2] = nz * r * sz;
    }

    geo.deleteAttribute('normal');
    geo.attributes.position.needsUpdate = true;

    // IcosahedronGeometry is non-indexed (each triangle owns 3 duplicated
    // vertices). Displacement is position-based (same sphere direction →
    // same displacement) so duplicates remain co-located after deformation.
    // mergeVertices fuses them back into a shared-vertex indexed mesh so that
    // computeVertexNormals() can average across all incident faces → smooth
    // Gouraud shading instead of the flat-shaded faceted look.
    const merged = mergeVertices(geo);
    geo.dispose();
    merged.computeVertexNormals();
    return merged;
  }

  /**
   * Smooth 3D value noise via trilinear interpolation of hashed lattice corners.
   * Returns a value in [-1, 1].
   */
  _noise3D(seed, x, y, z) {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    const fx = x - ix, fy = y - iy, fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const uz = fz * fz * (3 - 2 * fz);

    const c = (dx, dy, dz) => this._hashInt3(seed, ix + dx, iy + dy, iz + dz);

    const v = c(0,0,0)*(1-ux)*(1-uy)*(1-uz) + c(1,0,0)*ux*(1-uy)*(1-uz)
            + c(0,1,0)*(1-ux)*uy*(1-uz)     + c(1,1,0)*ux*uy*(1-uz)
            + c(0,0,1)*(1-ux)*(1-uy)*uz     + c(1,0,1)*ux*(1-uy)*uz
            + c(0,1,1)*(1-ux)*uy*uz         + c(1,1,1)*ux*uy*uz;

    return (v - 0.5) * 2.0;
  }

  /** Hash three integers + seed to [0, 1). */
  _hashInt3(seed, ix, iy, iz) {
    let h = (Math.imul(seed | 0, 0x4b511117) ^
             Math.imul(ix   | 0, 0x8e3a1789) ^
             Math.imul(iy   | 0, 0x12345679) ^
             Math.imul(iz   | 0, 0x87654321)) | 0;
    h ^= h >>> 14;
    h  = Math.imul(h, 0x9e3779b9) | 0;
    h ^= h >>> 16;
    return ((h >>> 0) & 0xFFFF) / 65536.0;
  }
}
