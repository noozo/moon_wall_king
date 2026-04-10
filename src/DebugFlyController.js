/**
 * DebugFlyController — true 6DOF free-fly camera for terrain/LOD inspection.
 *
 * Orientation is stored as a quaternion and all rotations are applied in the
 * camera's LOCAL frame (post-multiply).  This prevents gimbal coupling: mouse
 * left/right is always pure yaw around the camera's current up axis, and
 * mouse up/down is always pure pitch — regardless of pitch or roll state.
 *
 * Controls (click window to lock pointer):
 *   W / S         forward / backward  (camera local -Z / +Z)
 *   A / D         strafe left / right  (camera local -X / +X)
 *   Space         fly up               (camera local +Y)
 *   Ctrl          fly down             (camera local -Y)
 *   Shift         5× speed multiplier
 *   Scroll wheel  halve / double base speed
 *   Q             roll left
 *   E             roll right
 *   Mouse         yaw (local Y) + pitch (local X)
 *   Tab           return to player  (handled in main.js)
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MOUSE_SENS  = 0.002;
const ROLL_SPEED  = 1.2;   // rad/s
const SPEED_MIN   = 0.5;
const SPEED_MAX   = 50_000;
const SHIFT_MULT  = 5;

// Pre-allocated axes in camera LOCAL space for zero-allocation rotations.
const LOCAL_X  = new THREE.Vector3(1,  0,  0);
const LOCAL_Y  = new THREE.Vector3(0,  1,  0);
const LOCAL_Z  = new THREE.Vector3(0,  0,  1);  // +Z = camera backward; roll around this

// Scratch objects reused every frame.
const _dq  = new THREE.Quaternion();
const _fwd = new THREE.Vector3();
const _rgt = new THREE.Vector3();
const _up  = new THREE.Vector3();

// ---------------------------------------------------------------------------

export class DebugFlyController {
  /** @param {THREE.PerspectiveCamera} camera */
  constructor(camera) {
    this.camera   = camera;
    this.isActive = false;

    // World-space position.
    this._pos = new THREE.Vector3();

    // Orientation stored as quaternion.  All rotations are post-multiplied
    // (camera-local frame) so they never couple with each other.
    this._rot = new THREE.Quaternion();

    this._speed    = 100;
    this._isLocked = false;

    this._keys = {
      w: false, s: false, a: false, d: false,
      up: false, down: false,
      rollLeft: false, rollRight: false,
      shift: false,
    };

    this._setupListeners();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Take over from wherever the camera currently is.
   * Copies position and quaternion directly — no Euler decomposition, so
   * there is no risk of inheriting roll from PlayerController's lookAt().
   */
  activate() {
    this.isActive = true;
    this._pos.copy(this.camera.position);
    this._rot.copy(this.camera.quaternion);
    this._apply();
  }

  deactivate() {
    this.isActive = false;
  }

  /** @param {number} dt  Seconds since last frame */
  update(dt) {
    if (!this.isActive || !this._isLocked) return;

    const speed = this._speed * (this._keys.shift ? SHIFT_MULT : 1);

    // Build camera-local direction vectors from current orientation.
    _fwd.set(0,  0, -1).applyQuaternion(this._rot);
    _rgt.set(1,  0,  0).applyQuaternion(this._rot);
    _up .set(0,  1,  0).applyQuaternion(this._rot);

    if (this._keys.w)    this._pos.addScaledVector(_fwd,   speed * dt);
    if (this._keys.s)    this._pos.addScaledVector(_fwd,  -speed * dt);
    if (this._keys.d)    this._pos.addScaledVector(_rgt,   speed * dt);
    if (this._keys.a)    this._pos.addScaledVector(_rgt,  -speed * dt);
    if (this._keys.up)   this._pos.addScaledVector(_up,    speed * dt);
    if (this._keys.down) this._pos.addScaledVector(_up,   -speed * dt);

    // Roll — post-multiply keeps it in local frame.
    // +LOCAL_Z = camera backward; positive angle rotates counter-clockwise
    // when looking along +Z (i.e. toward you) = clockwise when looking forward
    // = right-roll.  So Q (rollLeft) needs negative angle, E needs positive.
    // Previous code had the signs swapped → inverted roll.
    if (this._keys.rollLeft)  {
      _dq.setFromAxisAngle(LOCAL_Z,  ROLL_SPEED * dt);
      this._rot.multiply(_dq);
    }
    if (this._keys.rollRight) {
      _dq.setFromAxisAngle(LOCAL_Z, -ROLL_SPEED * dt);
      this._rot.multiply(_dq);
    }

    this._apply();
  }

  getPosition() { return this._pos.clone(); }
  getSpeed()    { return this._speed; }

  /**
   * Teleport the fly controller's internal position to newPos and sync the
   * camera.  Used by the floating-origin system each frame to keep _pos near
   * (0,0,0) while the real world position is tracked in float64 externally.
   * @param {THREE.Vector3} newPos
   */
  resetPosition(newPos) {
    this._pos.copy(newPos);
    this._apply();
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _apply() {
    this.camera.position.copy(this._pos);
    this.camera.quaternion.copy(this._rot);
  }

  _setupListeners() {
    document.addEventListener('keydown', e => this._onKeyDown(e));
    document.addEventListener('keyup',   e => this._onKeyUp(e));
    document.addEventListener('mousemove', e => this._onMouseMove(e));
    document.addEventListener('pointerlockchange', () => {
      this._isLocked = document.pointerLockElement === document.body;
    });
    document.addEventListener('wheel', e => this._onWheel(e), { passive: true });
    document.addEventListener('click', () => {
      if (this.isActive && !this._isLocked) document.body.requestPointerLock();
    });
  }

  _onMouseMove(e) {
    if (!this.isActive || !this._isLocked) return;

    // Yaw: rotate around camera's LOCAL Y.
    // Post-multiply = applied in camera-local frame, so this is always the
    // camera's own up axis — never bleeds into roll regardless of camera tilt.
    if (e.movementX !== 0) {
      _dq.setFromAxisAngle(LOCAL_Y, -e.movementX * MOUSE_SENS);
      this._rot.multiply(_dq);
    }

    // Pitch: rotate around camera's LOCAL X.
    if (e.movementY !== 0) {
      _dq.setFromAxisAngle(LOCAL_X, -e.movementY * MOUSE_SENS);
      this._rot.multiply(_dq);
    }

    this._apply();
  }

  _onKeyDown(e) {
    if (!this.isActive) return;
    switch (e.code) {
      case 'KeyW': this._keys.w         = true; break;
      case 'KeyS': this._keys.s         = true; break;
      case 'KeyA': this._keys.a         = true; break;
      case 'KeyD': this._keys.d         = true; break;
      case 'KeyQ': this._keys.rollLeft  = true; break;
      case 'KeyE': this._keys.rollRight = true; break;
      case 'Space':
        this._keys.up = true;
        e.preventDefault();
        break;
      case 'ControlLeft': case 'ControlRight':
        this._keys.down = true;
        e.preventDefault();
        break;
      case 'ShiftLeft': case 'ShiftRight':
        this._keys.shift = true;
        break;
    }
  }

  _onKeyUp(e) {
    if (!this.isActive) return;
    switch (e.code) {
      case 'KeyW': this._keys.w         = false; break;
      case 'KeyS': this._keys.s         = false; break;
      case 'KeyA': this._keys.a         = false; break;
      case 'KeyD': this._keys.d         = false; break;
      case 'KeyQ': this._keys.rollLeft  = false; break;
      case 'KeyE': this._keys.rollRight = false; break;
      case 'Space':
        this._keys.up   = false; break;
      case 'ControlLeft': case 'ControlRight':
        this._keys.down = false; break;
      case 'ShiftLeft': case 'ShiftRight':
        this._keys.shift = false; break;
    }
  }

  _onWheel(e) {
    if (!this.isActive) return;
    this._speed = Math.max(SPEED_MIN, Math.min(SPEED_MAX,
      this._speed * (e.deltaY > 0 ? 0.5 : 2.0)
    ));
  }
}
