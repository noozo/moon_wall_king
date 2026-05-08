/**
 * DustSystem — footstep and landing dust particles for the Moon surface.
 *
 * Each particle follows a ballistic trajectory under inverse-square lunar
 * gravity (no air drag — the Moon has no atmosphere).  Physics matches
 * PhysicsBody.integrate() exactly: semi-implicit Euler, same gravity constant.
 *
 * All positions are in Moon-local space; the Points object is parented to
 * moonGroup so Moon spin is inherited automatically.
 *
 * Two spawn triggers (called from main.js):
 *   spawnStep(pos, N)              — footstep puff while walking
 *   spawnLand(pos, N, impactSpeed) — landing burst scaled by impact speed
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_PARTICLES  = 128;
const MOON_GRAV      = 1.62;   // game-units / s²  (matches PlayerController)
const LIFETIME       = 2.5;    // max seconds per particle
const STEP_PARTICLES = 6;
const LAND_PARTICLES = 14;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const DUST_VERT = /* glsl */`
attribute float aAlpha;
varying  float vAlpha;
void main() {
  vAlpha = aAlpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // Start at 3.5 px, shrink to 0.5 px as the particle ages.
  gl_PointSize = max(0.5, (1.0 - aAlpha) * 3.5);
  gl_Position  = projectionMatrix * mv;
}
`;

const DUST_FRAG = /* glsl */`
varying float vAlpha;
void main() {
  if (vAlpha >= 1.0) discard;
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  float a = (1.0 - vAlpha) * (1.0 - smoothstep(0.3, 0.5, d)) * 0.55;
  gl_FragColor = vec4(0.78, 0.74, 0.70, a);
}
`;

// ---------------------------------------------------------------------------

export class DustSystem {
  /**
   * @param {number}       moonRadius
   * @param {THREE.Group}  moonGroup   Parent; positions are Moon-local.
   */
  constructor(moonRadius, moonGroup) {
    this._moonRadius = moonRadius;

    // Per-particle state — flat arrays for cache-friendly update loop.
    this._px  = new Float32Array(MAX_PARTICLES);
    this._py  = new Float32Array(MAX_PARTICLES);
    this._pz  = new Float32Array(MAX_PARTICLES);
    this._vx  = new Float32Array(MAX_PARTICLES);
    this._vy  = new Float32Array(MAX_PARTICLES);
    this._vz  = new Float32Array(MAX_PARTICLES);
    this._lt  = new Float32Array(MAX_PARTICLES);          // lifetime (s)
    this._age = new Float32Array(MAX_PARTICLES).fill(1);  // 0-1; 1 = dead

    // GPU-upload buffers.
    this._posArr   = new Float32Array(MAX_PARTICLES * 3);
    this._alphaArr = new Float32Array(MAX_PARTICLES).fill(1);

    const geo = new THREE.BufferGeometry();
    this._posBuf   = new THREE.BufferAttribute(this._posArr,   3).setUsage(THREE.DynamicDrawUsage);
    this._alphaBuf = new THREE.BufferAttribute(this._alphaArr, 1).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this._posBuf);
    geo.setAttribute('aAlpha',   this._alphaBuf);

    this._points               = new THREE.Points(geo, new THREE.ShaderMaterial({
      vertexShader:   DUST_VERT,
      fragmentShader: DUST_FRAG,
      transparent:    true,
      depthWrite:     false,
    }));
    this._points.frustumCulled = false;
    moonGroup.add(this._points);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Footstep puff — called when the player takes a step on the surface.
   * @param {THREE.Vector3} pos  Moon-local body position
   * @param {THREE.Vector3} N    Moon-local surface normal (unit vector)
   */
  spawnStep(pos, N) {
    this._spawn(pos, N, STEP_PARTICLES, 0.45 + Math.random() * 0.25);
  }

  /**
   * Landing burst — called when the player transitions from airborne to grounded.
   * Particle count and speed scale with impact velocity.
   * @param {THREE.Vector3} pos
   * @param {THREE.Vector3} N
   * @param {number}        impactSpeed  Absolute vertical speed at touchdown (u/s)
   */
  spawnLand(pos, N, impactSpeed) {
    const count = Math.round(Math.min(LAND_PARTICLES, 4 + impactSpeed * 3));
    const speed = 0.3 + impactSpeed * 0.22;
    this._spawn(pos, N, count, speed);
  }

  /** Advance all live particles by one time step. */
  update(dt) {
    const R = this._moonRadius;

    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (this._age[i] >= 1.0) {
        this._alphaArr[i] = 1.0;
        continue;
      }

      // Inverse-square gravity toward Moon centre.
      const px = this._px[i], py = this._py[i], pz = this._pz[i];
      const dist = Math.sqrt(px * px + py * py + pz * pz) || R;
      const nx = px / dist, ny = py / dist, nz = pz / dist;
      const g  = MOON_GRAV * (R / dist) * (R / dist);

      // Semi-implicit Euler — velocity first.
      this._vx[i] -= nx * g * dt;
      this._vy[i] -= ny * g * dt;
      this._vz[i] -= nz * g * dt;
      this._px[i] += this._vx[i] * dt;
      this._py[i] += this._vy[i] * dt;
      this._pz[i] += this._vz[i] * dt;

      // Cheap surface clamp against base sphere — kill on contact.
      const nd = Math.sqrt(this._px[i] ** 2 + this._py[i] ** 2 + this._pz[i] ** 2);
      if (nd < R) {
        this._age[i] = 1.0;
        this._alphaArr[i] = 1.0;
        continue;
      }

      this._age[i] = Math.min(1.0, this._age[i] + dt / this._lt[i]);
      this._alphaArr[i] = this._age[i];

      this._posArr[i * 3]     = this._px[i];
      this._posArr[i * 3 + 1] = this._py[i];
      this._posArr[i * 3 + 2] = this._pz[i];
    }

    this._posBuf.needsUpdate   = true;
    this._alphaBuf.needsUpdate = true;
  }

  dispose() {
    this._points.geometry.dispose();
    this._points.material.dispose();
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _spawn(pos, N, count, speed) {
    // Build an orthonormal tangent frame from the surface normal.
    const ax = N.x, ay = N.y, az = N.z;
    let tx = az, ty = 0, tz = -ax;                 // cross(N, Y) approx
    if (Math.abs(ay) > 0.9) { tx = 1; tz = 0; }   // fallback near poles
    const tl = Math.sqrt(tx * tx + tz * tz) || 1;
    tx /= tl; tz /= tl;
    // Bitangent = N × T
    const bx = ay * tz - az * ty;
    const by = az * tx - ax * tz;
    const bz = ax * ty - ay * tx;

    let spawned = 0;
    for (let i = 0; i < MAX_PARTICLES && spawned < count; i++) {
      if (this._age[i] < 1.0) continue;

      // Offset slightly off the surface so gravity doesn't immediately reclaim them.
      this._px[i] = pos.x + ax * 0.1;
      this._py[i] = pos.y + ay * 0.1;
      this._pz[i] = pos.z + az * 0.1;

      // Hemispherical velocity spray biased toward the surface normal.
      const theta  = Math.random() * Math.PI * 2;
      const spread = Math.random() * 0.65;
      const upComp = Math.sqrt(1 - spread * spread);
      const sp     = speed * (0.55 + Math.random() * 0.45);
      const cs = Math.cos(theta), sn = Math.sin(theta);

      this._vx[i] = (ax * upComp + (cs * tx + sn * bx) * spread) * sp;
      this._vy[i] = (ay * upComp + (cs * ty + sn * by) * spread) * sp;
      this._vz[i] = (az * upComp + (cs * tz + sn * bz) * spread) * sp;

      this._lt[i]  = LIFETIME * (0.55 + Math.random() * 0.45);
      this._age[i] = 0;
      spawned++;
    }
  }
}
