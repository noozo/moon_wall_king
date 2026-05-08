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
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {import('./InputManager').InputManager} input
   */
  constructor(camera, input) {
    this.camera   = camera;
    this._input   = input;
    this.isActive = false;

    this._pos   = new THREE.Vector3();
    this._rot   = new THREE.Quaternion();
    this._speed = 100;
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
    // InputManager resets held state on pointer-lock changes; nothing to clear here.
  }

  dispose() {
    // No InputManager subscriptions to remove.
  }

  /** @param {number} dt  Seconds since last frame */
  update(dt) {
    if (!this.isActive || !this._input.isLocked) return;

    const speed = this._speed * (this._input.isDown('sprint') ? SHIFT_MULT : 1);

    _fwd.set(0,  0, -1).applyQuaternion(this._rot);
    _rgt.set(1,  0,  0).applyQuaternion(this._rot);
    _up .set(0,  1,  0).applyQuaternion(this._rot);

    if (this._input.isDown('forward'))  this._pos.addScaledVector(_fwd,   speed * dt);
    if (this._input.isDown('back'))     this._pos.addScaledVector(_fwd,  -speed * dt);
    if (this._input.isDown('right'))    this._pos.addScaledVector(_rgt,   speed * dt);
    if (this._input.isDown('left'))     this._pos.addScaledVector(_rgt,  -speed * dt);
    if (this._input.isDown('up'))       this._pos.addScaledVector(_up,    speed * dt);
    if (this._input.isDown('down'))     this._pos.addScaledVector(_up,   -speed * dt);

    // Roll — post-multiply keeps it in camera-local frame.
    if (this._input.isDown('rollLeft'))  {
      _dq.setFromAxisAngle(LOCAL_Z,  ROLL_SPEED * dt);
      this._rot.multiply(_dq);
    }
    if (this._input.isDown('rollRight')) {
      _dq.setFromAxisAngle(LOCAL_Z, -ROLL_SPEED * dt);
      this._rot.multiply(_dq);
    }

    // Mouse look — consume accumulated delta from InputManager.
    const { x: dx, y: dy } = this._input.consumeMouseDelta();
    if (dx !== 0) { _dq.setFromAxisAngle(LOCAL_Y, -dx * MOUSE_SENS); this._rot.multiply(_dq); }
    if (dy !== 0) { _dq.setFromAxisAngle(LOCAL_X, -dy * MOUSE_SENS); this._rot.multiply(_dq); }

    // Scroll — halve or double speed.
    const scroll = this._input.consumeScrollDelta();
    if (scroll !== 0) {
      this._speed = Math.max(SPEED_MIN, Math.min(SPEED_MAX,
        this._speed * (scroll > 0 ? 0.5 : 2.0)
      ));
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
}
