/**
 * TrajectoryPredictor — parabolic arc or Keplerian orbital ellipse.
 *
 * When RCS is active, determines which of two visuals to show:
 *
 *   Sub-orbital  (v < v_orb, or orbit periapsis inside Moon):
 *     Forward-simulates 200 steps × 0.2 s = 40 s under lunar gravity + current
 *     thrust.  Shows a dotted arc and an orange landing marker.
 *
 *   Orbital  (v ≥ v_orb AND periapsis above Moon surface):
 *     Computes the Keplerian ellipse analytically in O(1) — no simulation.
 *     Shows a green ellipse loop.  Eccentricity is indicated by colour:
 *       near-circular (e < 0.3): bright green
 *       elliptical    (e < 0.8): yellow-green
 *       highly eccentric        : orange (orbit barely clears surface)
 *
 * All positions are in Moon-local space; objects are parented to moonGroup.
 */

import * as THREE from 'three';

// ── Configuration ────────────────────────────────────────────────────────────

const MOON_GRAV = 1.62;      // game-units / s²
const ARC_STEPS = 200;       // simulation steps for sub-orbital arc
const ARC_DT    = 0.20;      // seconds per step → 40 s lookahead
const EYE_HEIGHT = 1.8;      // must match PhysicsBody default
const ORBIT_PTS  = 96;       // vertices on the Keplerian ellipse

// ── Module-level scratch — zero allocations in update() ──────────────────────

const _pos      = new THREE.Vector3();
const _vel      = new THREE.Vector3();
const _N        = new THREE.Vector3();
const _accel    = new THREE.Vector3();
const _thrust   = new THREE.Vector3();

// Keplerian
const _h        = new THREE.Vector3();   // angular momentum  h = r × v
const _eVec     = new THREE.Vector3();   // eccentricity vector
const _pDir     = new THREE.Vector3();   // periapsis unit direction
const _qDir     = new THREE.Vector3();   // perpendicular in orbital plane
const _hN       = new THREE.Vector3();   // unit angular momentum
const _orbCen   = new THREE.Vector3();   // ellipse centre (world-space)
const _arbDir   = new THREE.Vector3();   // scratch for degenerate circular case

// ── Helper ───────────────────────────────────────────────────────────────────

function makeCircleSprite(size = 64) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  return new THREE.CanvasTexture(c);
}

// ── TrajectoryPredictor ───────────────────────────────────────────────────────

export class TrajectoryPredictor {
  constructor(terrain, moonRadius, moonGroup) {
    this._terrain    = terrain;
    this._moonRadius = moonRadius;
    this._GM         = MOON_GRAV * moonRadius * moonRadius;

    // ── Sub-orbital arc (Points) ──────────────────────────────────────────────
    const arcPosArr = new Float32Array(ARC_STEPS * 3);
    const arcGeo    = new THREE.BufferGeometry();
    this._arcPosBuf = new THREE.BufferAttribute(arcPosArr, 3)
                        .setUsage(THREE.DynamicDrawUsage);
    arcGeo.setAttribute('position', this._arcPosBuf);
    arcGeo.setDrawRange(0, 0);

    this._sprite = makeCircleSprite();
    this._arc    = new THREE.Points(arcGeo, new THREE.PointsMaterial({
      size:            0.3,
      map:             this._sprite,
      sizeAttenuation: true,
      transparent:     true,
      opacity:         0.80,
      depthWrite:      false,
      alphaTest:       0.1,
    }));
    this._arc.frustumCulled = false;
    this._arc.visible       = false;
    moonGroup.add(this._arc);

    // ── Landing marker ────────────────────────────────────────────────────────
    this._marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.35, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xff6600 }),
    );
    this._marker.frustumCulled = false;
    this._marker.visible       = false;
    moonGroup.add(this._marker);

    // ── Orbital ellipse (LineLoop) ────────────────────────────────────────────
    const orbPosArr   = new Float32Array(ORBIT_PTS * 3);
    const orbGeo      = new THREE.BufferGeometry();
    this._orbPosBuf   = new THREE.BufferAttribute(orbPosArr, 3)
                          .setUsage(THREE.DynamicDrawUsage);
    orbGeo.setAttribute('position', this._orbPosBuf);

    this._orbitMat  = new THREE.LineBasicMaterial({
      color:       0x44ff88,
      transparent: true,
      opacity:     0.75,
      depthWrite:  false,
    });
    this._orbit              = new THREE.LineLoop(orbGeo, this._orbitMat);
    this._orbit.frustumCulled = false;
    this._orbit.visible       = false;
    moonGroup.add(this._orbit);
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  update(player) {
    if (!player.rcsEnabled) {
      this._arc.visible    = false;
      this._marker.visible = false;
      this._orbit.visible  = false;
      return;
    }

    _pos.copy(player.body.position);
    _vel.copy(player.body.velocity);
    const r  = _pos.length();
    const GM = this._GM;

    // Specific orbital energy: ε = v²/2 − GM/r.
    // ε < 0 → bound orbit; ε ≥ 0 → escape.
    const eps = _vel.lengthSq() * 0.5 - GM / r;

    if (eps < 0) {
      // Bound orbit candidate — check whether periapsis clears the surface.
      const a = -GM / (2 * eps);                        // semi-major axis > 0

      _h.crossVectors(_pos, _vel);                       // angular momentum

      // Eccentricity vector  e = (v × h)/GM − r̂
      _eVec.crossVectors(_vel, _h).divideScalar(GM);
      _N.copy(_pos).normalize();
      _eVec.sub(_N);
      const e = _eVec.length();

      const rPeriapsis = a * (1 - e);

      if (rPeriapsis > this._moonRadius * 1.005) {
        // Periapsis clears the Moon — show orbital ellipse.
        this._showOrbit(a, e, r);
        this._arc.visible    = false;
        this._marker.visible = false;
        return;
      }
    }

    // Sub-orbital (or orbit that would intersect the surface) — simulate arc.
    this._showArc(player);
    this._orbit.visible = false;
  }

  dispose() {
    this._arc.geometry.dispose();
    this._arc.material.dispose();
    this._sprite.dispose();
    this._marker.geometry.dispose();
    this._marker.material.dispose();
    this._orbit.geometry.dispose();
    this._orbitMat.dispose();
  }

  // ── Private ──────────────────────────────────────────────────────────────────

  _showArc(player) {
    player.getThrustVector(_thrust);

    const R   = this._moonRadius;
    const arr = this._arcPosBuf.array;
    let count  = 0;
    let landed = false;
    let landX = 0, landY = 0, landZ = 0;

    for (let i = 0; i < ARC_STEPS; i++) {
      const dist = _pos.length();
      _N.copy(_pos).divideScalar(dist);
      const g = MOON_GRAV * (R / dist) * (R / dist);
      _accel.copy(_N).multiplyScalar(-g).add(_thrust);

      _vel.addScaledVector(_accel, ARC_DT);
      _pos.addScaledVector(_vel,   ARC_DT);
      _N.copy(_pos).normalize();

      const terrH = this._terrain.getHeightAt(_N.x, _N.y, _N.z);
      if (_pos.length() <= R + terrH + EYE_HEIGHT) {
        landX = _N.x * (R + terrH);
        landY = _N.y * (R + terrH);
        landZ = _N.z * (R + terrH);
        landed = true;
        break;
      }

      arr[i * 3]     = _pos.x;
      arr[i * 3 + 1] = _pos.y;
      arr[i * 3 + 2] = _pos.z;
      count++;
    }

    this._arcPosBuf.needsUpdate = true;
    this._arc.geometry.setDrawRange(0, count);
    this._arc.visible = count > 0;

    if (landed) {
      this._marker.position.set(landX, landY, landZ);
      this._marker.visible = true;
    } else {
      this._marker.visible = false;
    }
  }

  _showOrbit(a, e, r) {
    // _h and _eVec are already computed in update().
    // _N holds r̂ (unit position).

    // Orbital plane unit normal.
    _hN.copy(_h).normalize();

    // Periapsis direction (pDir) — direction of eccentricity vector.
    if (e < 1e-4) {
      // Near-circular: eccentricity vector is ~zero, pick arbitrary direction
      // perpendicular to the angular momentum.
      _arbDir.set(1, 0, 0);
      if (Math.abs(_hN.dot(_arbDir)) > 0.9) _arbDir.set(0, 1, 0);
      _pDir.crossVectors(_hN, _arbDir).normalize();
    } else {
      _pDir.copy(_eVec).normalize();
    }

    // Second basis vector in orbital plane.
    _qDir.crossVectors(_hN, _pDir).normalize();

    // Semi-minor axis.
    const eSafe = Math.min(e, 0.9999);
    const b     = a * Math.sqrt(1 - eSafe * eSafe);

    // Ellipse centre is displaced from the focus (Moon centre) anti-periapsis
    // by  a·e:  centre = −pDir · a·e
    _orbCen.copy(_pDir).multiplyScalar(-a * e);

    // Choose orbit-line colour by eccentricity.
    if      (e < 0.3) this._orbitMat.color.setHex(0x44ff88);  // green — near-circular
    else if (e < 0.8) this._orbitMat.color.setHex(0xaaff44);  // yellow-green — elliptical
    else              this._orbitMat.color.setHex(0xff8844);   // orange — highly eccentric

    const arr = this._orbPosBuf.array;
    for (let i = 0; i < ORBIT_PTS; i++) {
      const theta = (i / ORBIT_PTS) * Math.PI * 2;
      const cos   = Math.cos(theta);
      const sin   = Math.sin(theta);
      arr[i * 3]     = _orbCen.x + a * cos * _pDir.x + b * sin * _qDir.x;
      arr[i * 3 + 1] = _orbCen.y + a * cos * _pDir.y + b * sin * _qDir.y;
      arr[i * 3 + 2] = _orbCen.z + a * cos * _pDir.z + b * sin * _qDir.z;
    }

    this._orbPosBuf.needsUpdate = true;
    this._orbit.visible = true;
  }
}
