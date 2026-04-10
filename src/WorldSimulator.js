/**
 * WorldSimulator — heliocentric double-precision orbital simulation.
 *
 * Coordinate system
 * ─────────────────
 *   World space (float64, JS numbers)
 *     Inertial heliocentric frame.  The Sun sits at (0,0,0).
 *     Earth orbits the Sun.  The Moon orbits the Earth.
 *     All positions are computed here at full float64 precision.
 *
 *   Render space (float32, Three.js / WebGL)
 *     Floating-origin frame:  renderPos = worldPos − floatingOrigin
 *     floatingOrigin is always set to the Moon's current world position,
 *     so the Moon terrain (at render origin) stays at (0,0,0) and the
 *     GPU always operates on small, float32-safe coordinates.
 *
 * Scale (compressed, not real SI)
 *   Moon   radius :  1 000 units
 *   Earth–Moon    : 25 000 units   (real ≈ 221 × Moon radius)
 *   Sun–Earth     :200 000 units   (real ≈ 86 M × Moon radius)
 *
 * Time scale: TIME_SCALE = 1/3000
 *   1 real second = 3 000 simulated seconds  (50 simulated minutes)
 *   Moon orbit period    ≈ 787 s real   (~13 min)
 *   Earth orbit period   ≈ 10 522 s real (~2.9 h)
 *   Moon day (synodic)   ≈ 849 s real   (~14 min)
 */

import * as THREE from 'three';
import { SIM_TIME_SCALE } from './SimConfig.js';

// ============================================================================
// Float64 Keplerian solver
// ============================================================================

/**
 * Solve Kepler's equation  M = E − e·sin(E)  via Newton–Raphson (float64).
 */
function solveKepler64(M, e) {
  let E = M;
  for (let i = 0; i < 12; i++) {
    E -= (E - e * Math.sin(E) - M) / (1.0 - e * Math.cos(E));
  }
  return E;
}

/**
 * Compute Cartesian position in the orbital reference plane
 * from the Keplerian elements and elapsed time.
 *
 * @param {number} t        Elapsed simulation time (seconds)
 * @param {object} elems    { a, e, i, omega, w, M0, P }
 * @returns {{ x: number, y: number, z: number }}  Float64 world position
 */
function keplerPos64(t, { a, e, i, omega, w, M0, P }) {
  const M = M0 + (2 * Math.PI * t) / P;
  const E = solveKepler64(M, e);

  // Position in orbital plane (x = along periapsis, z = 90° ahead)
  const px =  a * (Math.cos(E) - e);
  const pz =  a * Math.sqrt(1.0 - e * e) * Math.sin(E);

  // Rotate by argument of periapsis ω
  const cosW = Math.cos(w), sinW = Math.sin(w);
  const q1x  =  px * cosW - pz * sinW;
  const q1z  =  px * sinW + pz * cosW;

  // Tilt by inclination i around the line of nodes
  const cosI = Math.cos(i), sinI = Math.sin(i);
  const q2x  =  q1x;
  const q2y  = -q1z * sinI;
  const q2z  =  q1z * cosI;

  // Rotate by longitude of ascending node Ω around world-Y
  const cosO = Math.cos(omega), sinO = Math.sin(omega);
  return {
    x:  q2x * cosO - q2z * sinO,
    y:  q2y,
    z:  q2x * sinO + q2z * cosO,
  };
}

// ============================================================================
// WorldSimulator
// ============================================================================

export class WorldSimulator {
  /**
   * @param {number} moonRadius – must match MOON_RADIUS in main.js (default 1000)
   */
  constructor(moonRadius = 1000) {
    this._moonRadius = moonRadius;
    this._time       = 0;

    // ── Time scale ────────────────────────────────────────────────────────
    const DAY_S      = 86400;
    const TIME_SCALE = SIM_TIME_SCALE;

    // ── Orbital elements ──────────────────────────────────────────────────

    // Earth around Sun (heliocentric).
    // Distances and period use the same compressed scale as CelestialBody.js.
    this._earthOrbit = {
      a:     moonRadius * 200,          // 200 000 — Sun–Earth distance
      e:     0.0167,                    // real Earth-orbit eccentricity
      i:     0.0,                       // ecliptic = reference plane
      omega: 0,
      w:     0,
      M0:    0,                         // start near perihelion
      P:     365.25 * DAY_S * TIME_SCALE,
    };

    // Moon around Earth (geocentric).
    // M0 = π places the Moon at apoapsis from Earth at t=0, so Earth appears
    // on the +X side of the Moon — matching the player's initial look direction.
    this._moonOrbit = {
      a:     moonRadius * 25,           // 25 000 — Earth–Moon distance
      e:     0.0549,                    // real Moon-orbit eccentricity
      i:     0.0898,                    // 5.14° real Moon-orbit inclination
      omega: 0,
      w:     0,
      M0:    Math.PI,                   // Moon at apoapsis → Earth at +X from Moon
      P:     27.32 * DAY_S * TIME_SCALE,
    };

    // ── World-space body positions (float64) ──────────────────────────────
    // Sun is always at origin.
    this.sunWorldPos  = { x: 0, y: 0, z: 0 };

    // Earth and Moon initialised by calling advance(0).
    this.earthWorldPos = { x: 0, y: 0, z: 0 };
    this.moonWorldPos  = { x: 0, y: 0, z: 0 };

    // ── Floating origin ───────────────────────────────────────────────────
    // Always equals moonWorldPos.  Kept as a separate field so Phase 4+
    // (player leaves Moon) can switch it to playerWorldPos without changing
    // any caller.
    this.floatingOrigin = { x: 0, y: 0, z: 0 };

    // Compute initial positions and seed floatingOrigin to Moon.
    this.advance(0);
    this.floatingOrigin.x = this.moonWorldPos.x;
    this.floatingOrigin.y = this.moonWorldPos.y;
    this.floatingOrigin.z = this.moonWorldPos.z;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Advance the simulation by dt real seconds.
   * Updates earthWorldPos, moonWorldPos, floatingOrigin.
   */
  advance(dt) {
    this._time += dt;

    // Earth heliocentric position
    const ep = keplerPos64(this._time, this._earthOrbit);
    this.earthWorldPos.x = ep.x;
    this.earthWorldPos.y = ep.y;
    this.earthWorldPos.z = ep.z;

    // Moon geocentric offset → add Earth world position
    const mp = keplerPos64(this._time, this._moonOrbit);
    this.moonWorldPos.x = ep.x + mp.x;
    this.moonWorldPos.y = ep.y + mp.y;
    this.moonWorldPos.z = ep.z + mp.z;

    // NOTE: floatingOrigin is NOT updated here.
    // main.js sets it every frame based on the active camera mode:
    //   player mode → floatingOrigin = moonWorldPos   (Moon at render origin)
    //   fly mode    → floatingOrigin = camWorldPos     (camera at render origin,
    //                                                    Moon appears to orbit)
  }

  // ---------------------------------------------------------------------------
  // Render-space accessors  (float64 → float32 safe)
  // ---------------------------------------------------------------------------

  /**
   * Convert world-space position to render space.
   * renderPos = worldPos − floatingOrigin
   *
   * @param {{ x,y,z }} worldPos
   * @param {THREE.Vector3} [out]
   */
  toRenderPos(worldPos, out = new THREE.Vector3()) {
    out.set(
      worldPos.x - this.floatingOrigin.x,
      worldPos.y - this.floatingOrigin.y,
      worldPos.z - this.floatingOrigin.z,
    );
    return out;
  }

  /**
   * Moon render position — always (0,0,0) because floatingOrigin = moonWorldPos.
   * The call is kept explicit so Phase 4 (player leaves Moon) can change this.
   */
  getMoonRenderPos(out = new THREE.Vector3()) {
    return this.toRenderPos(this.moonWorldPos, out);
  }

  /** Earth render position — Earth relative to floating origin. */
  getEarthRenderPos(out = new THREE.Vector3()) {
    return this.toRenderPos(this.earthWorldPos, out);
  }

  /** Sun render position — Sun (world origin) relative to floating origin. */
  getSunRenderPos(out = new THREE.Vector3()) {
    return this.toRenderPos(this.sunWorldPos, out);
  }

  /**
   * Convert a render-space position to Moon-local coordinates.
   * Moon-local = renderPos − moonRenderPos = renderPos − (0,0,0) = renderPos.
   * Kept explicit for Phase 4 when moonGroup may have a non-zero render pos.
   *
   * @param {THREE.Vector3} renderPos
   * @param {THREE.Vector3} [out]
   */
  toMoonLocal(renderPos, out = new THREE.Vector3()) {
    const mr = this.getMoonRenderPos();
    out.set(
      renderPos.x - mr.x,
      renderPos.y - mr.y,
      renderPos.z - mr.z,
    );
    return out;
  }
}
