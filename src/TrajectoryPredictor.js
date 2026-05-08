/**
 * TrajectoryPredictor — ballistic arc visualiser for RCS mode.
 *
 * When RCS is active, forward-simulates the player's current position and
 * velocity under pure lunar gravity (inverse-square, no thrusters) to show
 * an 8-second predicted flight path as a dotted arc plus a landing marker.
 *
 * Simulation matches PhysicsBody.integrate() exactly:
 *   • inverse-square gravity only (no thrust, no Earth gravity)
 *   • semi-implicit Euler  (velocity first, then position)
 *   • terrain collision stops the arc and places the landing marker
 *
 * All positions are in Moon-local space; both objects are parented to
 * moonGroup so they move with Moon spin automatically.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------

const MOON_GRAV_SURFACE = 1.62;   // game-units / s²  (must match PlayerController)
const SIM_STEPS         = 80;     // number of prediction steps
const SIM_DT            = 0.1;    // seconds per step  → 8 s prediction window
const EYE_HEIGHT        = 1.8;    // game-units  (must match PhysicsBody)

// ---------------------------------------------------------------------------
// Scratch vectors — zero allocations in update()
// ---------------------------------------------------------------------------

const _pos   = new THREE.Vector3();
const _vel   = new THREE.Vector3();
const _N     = new THREE.Vector3();
const _accel = new THREE.Vector3();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeCircleSprite(size = 64) {
  const canvas = document.createElement('canvas');
  canvas.width  = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  return new THREE.CanvasTexture(canvas);
}

// ---------------------------------------------------------------------------

export class TrajectoryPredictor {
  /**
   * @param {object}        terrain     TerrainSystem — getHeightAt(nx,ny,nz)
   * @param {number}        moonRadius
   * @param {THREE.Group}   moonGroup   Parent group; positions are Moon-local.
   */
  constructor(terrain, moonRadius, moonGroup) {
    this._terrain    = terrain;
    this._moonRadius = moonRadius;

    // ── Arc (Points) ─────────────────────────────────────────────────────────
    const posArr = new Float32Array(SIM_STEPS * 3);
    const geo    = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setDrawRange(0, 0);

    this._sprite = makeCircleSprite();

    const arcMat = new THREE.PointsMaterial({
      size:           0.3,
      map:            this._sprite,
      sizeAttenuation: true,
      transparent:    true,
      opacity:        0.80,
      depthWrite:     false,
      alphaTest:      0.1,
    });

    this._arc              = new THREE.Points(geo, arcMat);
    this._arc.frustumCulled = false;
    this._arc.visible       = false;
    moonGroup.add(this._arc);

    // ── Landing marker (Mesh) ─────────────────────────────────────────────────
    const markerGeo = new THREE.SphereGeometry(0.35, 8, 6);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xff6600 });

    this._marker              = new THREE.Mesh(markerGeo, markerMat);
    this._marker.frustumCulled = false;
    this._marker.visible       = false;
    moonGroup.add(this._marker);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Call once per frame from the main loop (player mode only).
   * Hides the arc when RCS is off; otherwise re-simulates from the player's
   * current state and updates the geometry in-place.
   *
   * @param {import('./PlayerController').PlayerController} player
   */
  update(player) {
    if (!player.rcsEnabled) {
      this._arc.visible    = false;
      this._marker.visible = false;
      return;
    }

    // Snapshot current physics state (Moon-local space).
    _pos.copy(player.body.position);
    _vel.copy(player.body.velocity);

    const posArr = this._arc.geometry.attributes.position.array;
    const R      = this._moonRadius;

    let count  = 0;
    let landed = false;
    let landX  = 0, landY = 0, landZ = 0;

    for (let i = 0; i < SIM_STEPS; i++) {
      // Inverse-square lunar gravity (radially inward).
      const dist = _pos.length();
      _N.copy(_pos).divideScalar(dist);
      const g = MOON_GRAV_SURFACE * (R / dist) * (R / dist);
      _accel.copy(_N).multiplyScalar(-g);

      // Semi-implicit Euler — matches PhysicsBody.integrate().
      _vel.addScaledVector(_accel, SIM_DT);
      _pos.addScaledVector(_vel,   SIM_DT);
      _N.copy(_pos).normalize();

      // Terrain collision — stop the arc and mark the landing point.
      const terrH = this._terrain.getHeightAt(_N.x, _N.y, _N.z);
      const minR  = R + terrH + EYE_HEIGHT;

      if (_pos.length() <= minR) {
        landX  = _N.x * (R + terrH);
        landY  = _N.y * (R + terrH);
        landZ  = _N.z * (R + terrH);
        landed = true;
        break;
      }

      posArr[i * 3]     = _pos.x;
      posArr[i * 3 + 1] = _pos.y;
      posArr[i * 3 + 2] = _pos.z;
      count++;
    }

    this._arc.geometry.attributes.position.needsUpdate = true;
    this._arc.geometry.setDrawRange(0, count);
    this._arc.visible = count > 0;

    if (landed) {
      this._marker.position.set(landX, landY, landZ);
      this._marker.visible = true;
    } else {
      this._marker.visible = false;
    }
  }

  dispose() {
    this._arc.geometry.dispose();
    this._arc.material.dispose();
    this._sprite.dispose();
    this._marker.geometry.dispose();
    this._marker.material.dispose();
  }
}
