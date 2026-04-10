/**
 * PhysicsBody — 3-D rigid-body state for a player-scale object on a
 * spherical Moon surface.
 *
 * Replaces the old scalar "playerDir + heightAboveTerrain + verticalVelocity"
 * model in PlayerController with a proper force-integration approach.
 *
 * Integration method: semi-implicit (symplectic) Euler
 * ───────────────────────────────────────────────────
 * The key insight from Glenn Fiedler's "Integration Basics" (gafferongames.com):
 *
 *   // WRONG — explicit Euler, energy grows, unstable springs
 *   position += velocity * dt;
 *   velocity += acceleration * dt;
 *
 *   // CORRECT — semi-implicit Euler, symplectic, energy-preserving on average
 *   velocity += acceleration * dt;   ← velocity first
 *   position += velocity * dt;        ← then position from NEW velocity
 *
 * This one-line swap makes the integrator symplectic, which means:
 *   • Constant gravity / ballistic arcs are integrated exactly.
 *   • Energy is conserved on average — objects don't spontaneously accelerate.
 *   • Oscillatory systems (spring, pendulum) remain stable.
 *
 * Surface contact
 * ───────────────
 * The body's position is a free 3-D Moon-local vector — it is NOT clamped
 * to a sphere radius.  constrainToSurface() is called every frame and:
 *   1. Reads the actual terrain height at the current direction.
 *   2. If the body is inside the surface, pushes it out.
 *   3. Cancels the inward (radial) velocity component — inelastic contact,
 *      no bounce.
 *
 * This replaces the old positional snap with a proper contact constraint,
 * eliminating the "correction jitter" when LOD tiles reload.
 *
 * All positions are in Moon-local space.  main.js converts to world/render
 * space by applying moonGroup's transform.
 */

import * as THREE from 'three';

export class PhysicsBody {
  /**
   * @param {number} moonRadius  Moon sphere radius in game units (default 1000)
   * @param {number} eyeHeight   Camera offset above terrain contact point (m)
   */
  constructor(moonRadius = 1000, eyeHeight = 1.8) {
    this._moonRadius = moonRadius;
    this._eyeHeight  = eyeHeight;

    // Starting surface direction — on the sun-facing side of the Moon.
    // At t=0 the Sun is in the -X direction (Moon starts at ~170 000, 0, 0
    // in world space, Sun at origin), so a negative-X start direction gives
    // NdotL ≈ 0.93 — bright midday lighting from the first frame.
    const startDir = new THREE.Vector3(-1, 0.3, 0.3).normalize();

    /**
     * Moon-local world position (game units).
     * Magnitude ≈ moonRadius + terrainHeight + eyeHeight when on surface.
     * Free to vary when airborne.
     */
    this.position = startDir.clone().multiplyScalar(moonRadius + eyeHeight);

    /**
     * Moon-local velocity (game units / second).
     */
    this.velocity = new THREE.Vector3();

    /**
     * True when the body is in contact with the terrain this frame.
     * Set by constrainToSurface().
     */
    this.onSurface = true;

    /**
     * Unit outward radial direction at the body's current position.
     * Always equals normalize(position).
     * Kept as a cached field to avoid re-normalising every getter call.
     */
    this.surfaceNormal = startDir.clone();
  }

  // ---------------------------------------------------------------------------
  // Integration
  // ---------------------------------------------------------------------------

  /**
   * Advance state by one time step using semi-implicit Euler.
   *
   * @param {THREE.Vector3} acceleration  Net acceleration (forces / mass = forces
   *                                      when mass = 1, which we assume).
   * @param {number}        dt            Frame delta time in real seconds.
   */
  integrate(acceleration, dt) {
    // Semi-implicit Euler — velocity first, THEN position.
    this.velocity.addScaledVector(acceleration, dt);
    this.position.addScaledVector(this.velocity, dt);
    // Keep surfaceNormal in sync; used for gravity direction on the next frame.
    this.surfaceNormal.copy(this.position).normalize();
  }

  /**
   * Apply an instantaneous velocity change (impulse).
   * Used for jumping, landing bounces, external forces, etc.
   * @param {THREE.Vector3} impulse
   */
  applyImpulse(impulse) {
    this.velocity.add(impulse);
  }

  // ---------------------------------------------------------------------------
  // Surface constraint
  // ---------------------------------------------------------------------------

  /**
   * Prevent the body from penetrating the terrain.
   *
   * Must be called every frame after integrate().  Reads the current terrain
   * height in the body's radial direction, pushes the body outward if needed,
   * and removes the inward velocity component (inelastic contact — no bounce).
   *
   * @param {object} terrain  Any object with getHeightAt(nx, ny, nz) → number
   * @returns {boolean} true if the body is on the surface this frame
   */
  constrainToSurface(terrain) {
    const n = this.surfaceNormal;                              // already normalised
    const terrainH  = terrain.getHeightAt(n.x, n.y, n.z);
    const minRadius = this._moonRadius + terrainH + this._eyeHeight;
    const radius    = this.position.length();

    if (radius <= minRadius) {
      // Push body to the surface, maintaining the current radial direction.
      this.position.copy(n).multiplyScalar(minRadius);

      // Kill the inward radial velocity component (inelastic contact).
      const vn = this.velocity.dot(n);
      if (vn < 0) this.velocity.addScaledVector(n, -vn);

      this.onSurface = true;
    } else {
      this.onSurface = false;
    }

    return this.onSurface;
  }

  // ---------------------------------------------------------------------------
  // Convenience accessors
  // ---------------------------------------------------------------------------

  /** Signed height above the terrain (negative = inside terrain — should never happen). */
  get altitudeAboveTerrain() {
    return this.position.length() - this._moonRadius - this._eyeHeight;
  }

  /** Speed in the tangential (horizontal) plane. */
  tangentialSpeed() {
    const vn = this.velocity.dot(this.surfaceNormal);
    return Math.sqrt(Math.max(0, this.velocity.lengthSq() - vn * vn));
  }
}
