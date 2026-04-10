/**
 * PlayerController — first-person surface movement on a spherical Moon.
 *
 * Physics model (Phase A refactor)
 * ──────────────────────────────────
 * Movement is now entirely force-based.  Every frame:
 *
 *   1. Gravity — radial toward Moon centre (1.62 m/s² lunar gravity).
 *
 *   2. Ground response — a proportional force that drives the body toward
 *      the desired tangential velocity (WASD input) when on the surface.
 *      This same force acts as friction when no keys are pressed, bringing
 *      the player to a smooth stop without the old instant snap.
 *
 *      F_ground = GROUND_RESPONSE × (desired_vel − current_tangential_vel)
 *
 *      With GROUND_RESPONSE = 12 and dt = 1/60:
 *        lerp factor ≈ 0.20 per frame → ~80 ms to full walking speed.
 *        Same rate for deceleration → symmetric and natural-feeling.
 *
 *   3. Jump — a single velocity impulse applied the frame the player leaves
 *      the surface.  Horizontal velocity from walking is already in the
 *      physics body and is preserved for the ballistic arc automatically.
 *
 *   4. Semi-implicit Euler integration (see PhysicsBody.js).
 *
 *   5. Surface constraint — pushes the body back out of terrain and zeros
 *      the inward velocity component (see PhysicsBody.constrainToSurface).
 *
 * This replaces the old `playerDir + heightAboveTerrain + verticalVelocity`
 * scalar model that required capturing air-velocity snapshots at jump time
 * and caused visible corrections whenever LOD tiles reloaded.
 */

import * as THREE from 'three';
import { PhysicsBody } from './PhysicsBody.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Lunar surface gravity (game-units / s²). Inverse-square falloff applied at altitude. */
const MOON_GRAV_SURFACE = 1.62;

/**
 * Earth surface gravity (game-units / s²).
 * Real value ≈ 9.81 m/s²; using 8.0 at the game's Earth render radius (1500 units).
 * Must stay consistent with CelestialBody.js EARTH_RADIUS = moonRadius × 1.5.
 */
const EARTH_GRAV_SURFACE = 8.0;
const EARTH_RENDER_RADIUS = 1500;  // must match CelestialBody.js

/** Jump initial speed (game-units / s). */
const JUMP_VELOCITY = 3.5;

/**
 * Proportional gain for the ground-response force.
 * Acts as both acceleration (when pressing keys) and friction (when not).
 * Units: 1/s  (acceleration = GAIN × velocity_error)
 * With dt = 1/60:  lerp_factor = GAIN/60 ≈ 0.20 → ~80 ms response.
 */
const GROUND_RESPONSE = 12;

/**
 * RCS thruster acceleration (game-units / s²).
 * Keys apply a constant force while held — velocity accumulates freely.
 * WASD = camera-relative lateral, Space = up, Ctrl = down.
 */
const RCS_ACCEL_LATERAL = 2.0;
const RCS_ACCEL_VERTICAL = 3.5;  // Stronger lift thrust for reliable launch/descent

/**
 * Translational rate damping — GNC auto-fires thrusters opposite to the
 * body's current velocity when no thrust keys are held, nulling drift.
 * Uses the same RCS_ACCEL budget so damping feels identical to manual thrust.
 * Toggled independently from RCS via T key.  Real-world equivalent:
 * Dragon's "station-keeping" hold mode / KSP translational SAS.
 */
const RCS_DAMPING_KEY = 'KeyT';

/** Exponential smoothing rate for camera radial height (units/s). Higher = tighter. */
const CAM_HEIGHT_SMOOTH = 20;

/**
 * Exponential smoothing rate for camera up-vector (surface normal).
 * Lower than height so slope transitions feel like a natural head tilt.
 */
const CAM_UP_SMOOTH = 8;

// ---------------------------------------------------------------------------
// Module-level scratch vectors (zero allocation per frame)
// ---------------------------------------------------------------------------

const WORLD_UP       = new THREE.Vector3(0, 1, 0);
const _POLE_FALLBACK = new THREE.Vector3(1, 0, 0);
const _ZERO_POS      = new THREE.Vector3(0, 0, 0);
const _IDENTITY_Q    = new THREE.Quaternion();

// Scratch for tangent-basis computation
const _east  = new THREE.Vector3();
const _north = new THREE.Vector3();

// Scratch for force accumulation
const _move        = new THREE.Vector3();
const _tanVel      = new THREE.Vector3();

// Scratch for RCS + Earth gravity (Phase D)
const _toEarth     = new THREE.Vector3();
const _earthML     = new THREE.Vector3();  // Earth position in Moon-local space
const _rcsFwd      = new THREE.Vector3();
const _rcsRight    = new THREE.Vector3();
const _rcsUp       = new THREE.Vector3();
const _qInv        = new THREE.Quaternion();

// Scratch for camera placement
const _fwd         = new THREE.Vector3();
const _right       = new THREE.Vector3();
const _pitchedFwd  = new THREE.Vector3();
const _lookTarget  = new THREE.Vector3();
const _worldDir    = new THREE.Vector3();   // surface normal in world space
const _worldBodyP  = new THREE.Vector3();   // body position in world space

// Scratch for force result
const _accel       = new THREE.Vector3();

// ---------------------------------------------------------------------------
// PlayerController
// ---------------------------------------------------------------------------

export class PlayerController {
  /**
   * @param {THREE.PerspectiveCamera}  camera
   * @param {object}                   terrainSystem  getHeightAt(nx,ny,nz)→number
   * @param {number}                   moonRadius
   * @param {THREE.Group|null}         moonGroup      For world-space camera placement
   */
  constructor(camera, terrainSystem, moonRadius = 1000, moonGroup = null) {
    this.camera        = camera;
    this.terrainSystem = terrainSystem;
    this.moonRadius    = moonRadius;
    this._moonGroup    = moonGroup;

    // Movement parameters
    this.playerHeight     = 1.8;
    this.moveSpeed        = 3.5;
    this.sprintSpeed      = 8.0;
    this.mouseSensitivity = 0.0018;

    // Physics body — owns position, velocity, contact state.
    this.body = new PhysicsBody(moonRadius, this.playerHeight);

    // ── Camera spring state ────────────────────────────────────────────────
    // Kept in Moon-local space so Moon spin is tracked exactly — only the
    // radial height and the surface-normal tilt are smoothed.

    /** Smoothed radial distance from Moon centre (eliminates LOD height snaps). */
    this._camRadius = this.body.position.length();

    /**
     * Smoothed surface normal in Moon-local space (eliminates abrupt tilt when
     * walking onto a slope or when the terrain normal flips at an LOD boundary).
     */
    this._camUp = this.body.surfaceNormal.clone();

    // Camera orientation (look direction)
    this.yaw   = 0;
    this.pitch = 0;

    // Jump state — queued once per keypress, consumed on the next grounded frame.
    this._jumpQueued = false;

    /** RCS toggle — press R to enable, R again to disable. */
    this.rcsEnabled = false;

    /**
     * Translational rate damping toggle — press T to enable/disable.
     * When on, GNC auto-fires thrusters to null velocity whenever no thrust
     * keys are held.  Only active while rcsEnabled is also true.
     */
    this.rcsDampingEnabled = false;

    // Input state
    this.keys = {
      w: false, a: false, s: false, d: false,
      shift: false,
      space: false,  // Space held state — used for RCS up-thrust in space
      ctrl:  false,  // Ctrl held state  — RCS down-thrust
    };
    this.isLocked  = false;
    this.currentSpeed = 0;

    /**
     * Earth's current render-space position.
     * Set each frame from main.js via setEarthRenderPos().
     * Used for Earth gravity when the player is in space.
     */
    this._earthRenderPos = new THREE.Vector3(0, 25000, 0);

    this._setupInputListeners();
    this._updateCameraTransform();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Call every frame with real delta-time in seconds. */
  update(dt) {
    this._applyJumpImpulse();
    this._integrateForces(dt);
    this.body.constrainToSurface(this.terrainSystem);
    this._smoothCameraState(dt);
    this._updateCameraTransform();
    // Track speed for HUD
    this.currentSpeed = this.body.tangentialSpeed();
  }

  /** World-space camera position. */
  getPosition() { return this.camera.position.clone(); }

  getSpeed() { return this.currentSpeed; }

  /** Latitude/longitude in degrees derived from the surface normal direction. */
  getCoordinates() {
    const p = this.body.surfaceNormal;
    return {
      latitude:  Math.asin(Math.max(-1, Math.min(1, p.y))) * (180 / Math.PI),
      longitude: Math.atan2(p.z, p.x)                      * (180 / Math.PI),
    };
  }

  /** Compass heading in degrees (0 = north, 90 = east). */
  getHeading() {
    return ((this.yaw * (180 / Math.PI)) % 360 + 360) % 360;
  }

  /**
   * Call before update() each frame with Earth's render-space position so
   * the physics can apply Earth gravity when the player is in space.
   * @param {THREE.Vector3} pos
   */
  setEarthRenderPos(pos) { this._earthRenderPos.copy(pos); }

  /** Current altitude above the Moon surface (game-units). */
  getAltitude() { return Math.max(0, this.body.position.length() - this.moonRadius); }

  /**
   * Velocity breakdown in surface-relative terms.
   *   vertical   — positive = moving away from Moon, negative = falling
   *   horizontal — speed parallel to Moon surface
   *   total      — magnitude
   */
  getVelocityInfo() {
    const N       = this.body.surfaceNormal;
    const vVert   = this.body.velocity.dot(N);
    const vHoriz  = Math.sqrt(Math.max(0, this.body.velocity.lengthSq() - vVert * vVert));
    return {
      total:      this.body.velocity.length(),
      vertical:   vVert,
      horizontal: vHoriz,
      altitude:   this.getAltitude(),
      rcs:        this.rcsEnabled,
      rcsDamping: this.rcsDampingEnabled,
      grounded:   this.body.onSurface,
    };
  }

  // ---------------------------------------------------------------------------
  // Internal — jump
  // ---------------------------------------------------------------------------

  /**
   * If a jump has been queued and the body is grounded, apply the velocity
   * impulse and consume the queue.
   *
   * The horizontal walking velocity already stored in body.velocity is
   * automatically preserved — no need for a snapshot like the old _airVelocity.
   */
  _applyJumpImpulse() {
    // Jump impulse simulates legs pushing off the ground.
    // When RCS is on, Space is an up-thruster (continuous force), not a jump.
    if (this._jumpQueued && this.body.onSurface && !this.rcsEnabled) {
      this.body.applyImpulse(
        this.body.surfaceNormal.clone().multiplyScalar(JUMP_VELOCITY),
      );
      this._jumpQueued = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Internal — force integration
  // ---------------------------------------------------------------------------

  /**
   * Accumulate all accelerations and advance the physics body.
   *
   *   RCS OFF
   *     Grounded:      ground-response force drives WASD walking (leg thrust).
   *     Airborne:      pure ballistic — gravity only, no player control.
   *
   *   RCS ON
   *     Any state:     WASD / Space / Ctrl apply camera-relative continuous
   *                    thrust (rocket booster model).  Ground response is
   *                    replaced entirely; the surface constraint still prevents
   *                    penetration but the player glides freely.
   *     Grounded:      upward Space thrust lifts the player off the surface
   *                    without a jump impulse.
   */
  _integrateForces(dt) {
    const N        = this.body.surfaceNormal;
    const mgq      = this._moonGroup ? this._moonGroup.quaternion : _IDENTITY_Q;

    _accel.set(0, 0, 0);

    // ── 1. Gravity — always applied (RCS thrusters fight against it when pressed) ──
    const moonDist    = this.body.position.length();
    const moonGravAcc = MOON_GRAV_SURFACE * (this.moonRadius / moonDist) ** 2;
    _accel.addScaledVector(N, -moonGravAcc);

    if (!this.body.onSurface) {
      _qInv.copy(mgq).invert();
      _earthML.copy(this._earthRenderPos).applyQuaternion(_qInv);
      _toEarth.copy(_earthML).sub(this.body.position);
      const earthDist    = _toEarth.length();
      const earthGravAcc = EARTH_GRAV_SURFACE * (EARTH_RENDER_RADIUS / earthDist) ** 2;
      _accel.addScaledVector(_toEarth.normalize(), earthGravAcc);
    }

    if (this.rcsEnabled) {
      // ── 2a. RCS — continuous camera-relative thrust ───────────────────────
      // Keys apply a constant force while held; velocity accumulates freely.
      // When no thrust keys are held and damping is on, GNC auto-fires opposite
      // to current velocity (translational rate damping / velocity kill).
      if (this.isLocked) {
        _qInv.copy(mgq).invert();
        _rcsFwd  .set(0,  0, -1).applyQuaternion(this.camera.quaternion).applyQuaternion(_qInv);
        _rcsRight.set(1,  0,  0).applyQuaternion(this.camera.quaternion).applyQuaternion(_qInv);

        const fb = Number(this.keys.w)     - Number(this.keys.s);
        const lr = Number(this.keys.d)     - Number(this.keys.a);
        const ud = Number(this.keys.space) - Number(this.keys.ctrl);
        const anyThrustKey = fb !== 0 || lr !== 0 || ud !== 0;

        if (fb !== 0) _accel.addScaledVector(_rcsFwd,               fb * RCS_ACCEL_LATERAL);
        if (lr !== 0) _accel.addScaledVector(_rcsRight,              lr * RCS_ACCEL_LATERAL);
        // Space/Ctrl thrust along surface normal — reliable lift-off and descent
        // regardless of camera pitch.  Stronger vertical thrust for better launch.
        if (ud !== 0) _accel.addScaledVector(this.body.surfaceNormal, ud * RCS_ACCEL_VERTICAL);

        // ── 2a-ii. Translational rate damping (GNC velocity kill) ─────────
        // When T-damping is on and no thrust keys are held, auto-fire thrusters
        // opposite to current velocity.  Capped at max lateral thrust so it never
        // over-shoots zero (minimum of available thrust vs. what's needed to
        // kill remaining speed in this frame).
        if (this.rcsDampingEnabled && !anyThrustKey && !this.body.onSurface) {
          const speed = this.body.velocity.length();
          if (speed > 0.001) {
            // Max deceleration we can apply this frame without overshooting zero:
            const maxDamp = Math.min(RCS_ACCEL_LATERAL, speed / dt);
            _accel.addScaledVector(this.body.velocity, -maxDamp / speed);
          }
        }
      }
    } else if (this.body.onSurface && this.isLocked) {
      // ── 2b. Ground response — leg-force WASD walking (RCS off only) ───────
      // Proportional controller drives body toward the desired tangential
      // velocity.  Acts as friction when no keys are pressed.
      this._buildTangentBasis(this.body.surfaceNormal);

      const speed = this.keys.shift ? this.sprintSpeed : this.moveSpeed;
      const fb    = Number(this.keys.w) - Number(this.keys.s);
      const lr    = Number(this.keys.d) - Number(this.keys.a);

      _fwd.copy(_north).multiplyScalar(Math.cos(this.yaw))
          .addScaledVector(_east, Math.sin(this.yaw));
      _right.copy(_east).multiplyScalar(Math.cos(this.yaw))
            .addScaledVector(_north, -Math.sin(this.yaw));

      _move.set(0, 0, 0);
      if (fb !== 0 || lr !== 0) {
        _move.addScaledVector(_fwd, fb).addScaledVector(_right, lr);
        if (_move.lengthSq() > 0) _move.normalize().multiplyScalar(speed);
      }

      const vn = this.body.velocity.dot(N);
      _tanVel.copy(this.body.velocity).addScaledVector(N, -vn);
      _accel.addScaledVector(_move,   GROUND_RESPONSE);
      _accel.addScaledVector(_tanVel, -GROUND_RESPONSE);
    }

    this.body.integrate(_accel, dt);
  }

  // ---------------------------------------------------------------------------
  // Internal — camera spring smoothing
  // ---------------------------------------------------------------------------

  /**
   * Exponentially approach the body's current radial height and surface
   * normal.  Done in Moon-local space so the Moon's spin is tracked
   * instantly while only height variation and normal tilting are smoothed.
   *
   * Using exponential decay: factor = 1 − exp(−rate × dt)
   * This is frame-rate independent, unlike a plain lerp by a constant.
   *
   *   CAM_HEIGHT_SMOOTH = 20 → half-life ≈ 35 ms → LOD snaps invisible
   *   CAM_UP_SMOOTH     =  8 → half-life ≈ 87 ms → slope tilt feels natural
   */
  _smoothCameraState(dt) {
    const fH = 1 - Math.exp(-CAM_HEIGHT_SMOOTH * dt);
    this._camRadius += (this.body.position.length() - this._camRadius) * fH;

    const fU = 1 - Math.exp(-CAM_UP_SMOOTH * dt);
    this._camUp.lerp(this.body.surfaceNormal, fU).normalize();
  }

  // ---------------------------------------------------------------------------
  // Internal — camera placement and orientation
  // ---------------------------------------------------------------------------

  _updateCameraTransform() {
    // Use smoothed normal for both the camera-up vector and the tangent basis
    // so that look-direction, up-vector and position all tilt together.
    const N   = this._camUp;   // smoothed Moon-local surface normal
    const mgp = this._moonGroup ? this._moonGroup.position   : _ZERO_POS;
    const mgq = this._moonGroup ? this._moonGroup.quaternion  : _IDENTITY_Q;

    // Camera sits at smoothed radius along the physics body's current direction.
    // Using body.surfaceNormal (raw) for direction keeps the lateral position exact;
    // _camRadius (smoothed) removes LOD height discontinuities.
    _worldBodyP.copy(this.body.surfaceNormal)
               .multiplyScalar(this._camRadius)
               .applyQuaternion(mgq);
    this.camera.position.set(
      mgp.x + _worldBodyP.x,
      mgp.y + _worldBodyP.y,
      mgp.z + _worldBodyP.z,
    );

    // World-space camera up — smoothed, so slope transitions feel like a
    // natural head tilt rather than an instantaneous flip.
    _worldDir.copy(N).applyQuaternion(mgq);

    // Look direction built from the smoothed tangent basis.
    this._buildTangentBasis();   // uses _camUp via N

    _fwd.copy(_north).multiplyScalar(Math.cos(this.yaw))
        .addScaledVector(_east, Math.sin(this.yaw));

    _pitchedFwd.copy(_fwd).multiplyScalar(Math.cos(this.pitch))
               .addScaledVector(N, Math.sin(this.pitch));
    _pitchedFwd.normalize().applyQuaternion(mgq);

    this.camera.up.copy(_worldDir);
    _lookTarget.copy(this.camera.position).add(_pitchedFwd);
    this.camera.lookAt(_lookTarget);
  }

  // ---------------------------------------------------------------------------
  // Internal — tangent plane basis
  // ---------------------------------------------------------------------------

  /**
   * Builds _east and _north tangent vectors at the given surface direction p.
   * Uses the smoothed _camUp (via caller) for camera orientation so the
   * tangent basis, up-vector and look direction all tilt together.
   * Called from _updateCameraTransform() with N = this._camUp, and from
   * _integrateForces() with the raw body.surfaceNormal for physics.
   */
  _buildTangentBasis(p = this._camUp) {
    _east.crossVectors(WORLD_UP, p);
    if (_east.lengthSq() < 1e-6) {
      _east.crossVectors(_POLE_FALLBACK, p);
    }
    _east.normalize();
    _north.crossVectors(p, _east).normalize();
  }

  // ---------------------------------------------------------------------------
  // Internal — input
  // ---------------------------------------------------------------------------

  _setupInputListeners() {
    document.addEventListener('keydown',           e => this._onKeyDown(e));
    document.addEventListener('keyup',             e => this._onKeyUp(e));
    document.addEventListener('mousemove',         e => this._onMouseMove(e));
    document.addEventListener('pointerlockchange', () => this._onPointerLockChange());
    document.addEventListener('click', () => {
      if (!this.isLocked) document.body.requestPointerLock();
    });
  }

  _onKeyDown(e) {
    switch (e.code) {
      case 'KeyW':     this.keys.w     = true;  break;
      case 'KeyA':     this.keys.a     = true;  break;
      case 'KeyS':     this.keys.s     = true;  break;
      case 'KeyD':     this.keys.d     = true;  break;
      case 'ShiftLeft': case 'ShiftRight': this.keys.shift = true; break;
      case 'ControlLeft': case 'ControlRight':
        this.keys.ctrl = true;
        e.preventDefault();
        break;
      case 'KeyR':
        this.rcsEnabled = !this.rcsEnabled;
        this._jumpQueued = false;
        break;
      case RCS_DAMPING_KEY:
        this.rcsDampingEnabled = !this.rcsDampingEnabled;
        break;
      case 'Space':
        this.keys.space = true;
        if (!this.rcsEnabled && !this._jumpQueued) this._jumpQueued = true;
        e.preventDefault();
        break;
    }
  }

  _onKeyUp(e) {
    switch (e.code) {
      case 'KeyW':     this.keys.w     = false; break;
      case 'KeyA':     this.keys.a     = false; break;
      case 'KeyS':     this.keys.s     = false; break;
      case 'KeyD':     this.keys.d     = false; break;
      case 'ShiftLeft': case 'ShiftRight': this.keys.shift = false; break;
      case 'ControlLeft': case 'ControlRight': this.keys.ctrl = false; break;
      case 'Space':
        this.keys.space = false;
        if (!this.body.onSurface && !this.rcsEnabled) this._jumpQueued = false;
        break;
    }
  }

  _onMouseMove(e) {
    if (!this.isLocked) return;
    this.yaw   += e.movementX * this.mouseSensitivity;
    this.pitch -= e.movementY * this.mouseSensitivity;
    this.pitch  = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, this.pitch));
  }

  _onPointerLockChange() {
    const wasLocked = this.isLocked;
    this.isLocked = document.pointerLockElement === document.body;

    // Clear all key state whenever pointer-lock changes.  If the browser
    // swallows a keyup while lock was being released/acquired, keys can get
    // stuck permanently.  Resetting here is the standard fix.
    if (wasLocked !== this.isLocked) {
      for (const k of Object.keys(this.keys)) this.keys[k] = false;
      this._jumpQueued = false;
    }

    const startScreen = document.getElementById('start-screen');
    if (startScreen) startScreen.classList.toggle('hidden', this.isLocked);
  }
}
