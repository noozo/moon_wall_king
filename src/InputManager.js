/**
 * InputManager — single source of truth for all player input.
 *
 * Responsibilities:
 *   - Own every DOM event listener (keydown/up, mousemove, wheel,
 *     pointerlockchange, click).
 *   - Maintain held-action state (isDown) for continuous key checks.
 *   - Accumulate mouse and scroll deltas; callers consume them once per frame.
 *   - Emit named one-shot events for toggles and impulses.
 *
 * Controllers and main.js read from here instead of adding their own listeners.
 * Keybinding changes only need to happen in KEY_TO_ACTION / _onKeyDown below.
 */

// ---------------------------------------------------------------------------
// Key → held-action mapping
// ---------------------------------------------------------------------------

const KEY_TO_ACTION = {
  'KeyW':         'forward',
  'KeyS':         'back',
  'KeyA':         'left',
  'KeyD':         'right',
  'ShiftLeft':    'sprint',
  'ShiftRight':   'sprint',
  'Space':        'up',
  'ControlLeft':  'down',
  'ControlRight': 'down',
  'KeyQ':         'rollLeft',
  'KeyE':         'rollRight',
};

// ---------------------------------------------------------------------------

export class InputManager {
  constructor() {
    /** Held-action booleans — read with isDown(). */
    this._actions = {};

    /** Accumulated mouse movement since last consumeMouseDelta(). */
    this._mouseDX = 0;
    this._mouseDY = 0;

    /** Accumulated wheel movement since last consumeScrollDelta(). */
    this._scrollDelta = 0;

    this._isLocked = false;

    /** Event listeners: eventName → Set<fn> */
    this._listeners = {};

    this._setupListeners();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Returns true while the named action key is held. */
  isDown(action) {
    return this._actions[action] === true;
  }

  /**
   * Returns accumulated mouse movement since the last call and resets to zero.
   * Call once per frame from the active controller's update().
   */
  consumeMouseDelta() {
    const x = this._mouseDX, y = this._mouseDY;
    this._mouseDX = 0;
    this._mouseDY = 0;
    return { x, y };
  }

  /**
   * Returns accumulated scroll wheel movement since the last call and resets.
   * Positive = scroll down (zoom out / speed down), negative = scroll up.
   */
  consumeScrollDelta() {
    const d = this._scrollDelta;
    this._scrollDelta = 0;
    return d;
  }

  /** True when the browser pointer lock is active on document.body. */
  get isLocked() { return this._isLocked; }

  /** Request pointer lock if not already locked. */
  requestPointerLock() {
    if (!this._isLocked) document.body.requestPointerLock();
  }

  /**
   * Subscribe to a named event.
   *
   * One-shot events (emitted on keydown):
   *   'jump'                — Space pressed
   *   'toggleRCS'           — R pressed
   *   'toggleDamping'       — T pressed
   *   'toggleFlashlight'    — L pressed
   *   'toggleWireframe'     — X pressed
   *   'toggleMode'          — Tab pressed
   *   'flashlightRangeInc'  — ] pressed
   *   'flashlightRangeDec'  — [ pressed
   *   'flashlightAngleInc'  — = pressed
   *   'flashlightAngleDec'  — − pressed
   *
   * State events:
   *   'lockChange'  — payload: { locked: boolean }
   */
  on(event, fn) {
    if (!this._listeners[event]) this._listeners[event] = new Set();
    this._listeners[event].add(fn);
  }

  off(event, fn) {
    this._listeners[event]?.delete(fn);
  }

  dispose() {
    document.removeEventListener('keydown',           this._boundKeyDown);
    document.removeEventListener('keyup',             this._boundKeyUp);
    document.removeEventListener('mousemove',         this._boundMouseMove);
    document.removeEventListener('pointerlockchange', this._boundLockChange);
    document.removeEventListener('wheel',             this._boundWheel);
    document.removeEventListener('click',             this._boundClick);
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _emit(event, payload) {
    const set = this._listeners[event];
    if (set) for (const fn of set) fn(payload);
  }

  _setupListeners() {
    this._boundKeyDown    = e => this._onKeyDown(e);
    this._boundKeyUp      = e => this._onKeyUp(e);
    this._boundMouseMove  = e => this._onMouseMove(e);
    this._boundLockChange = () => this._onLockChange();
    this._boundWheel      = e => this._onWheel(e);
    this._boundClick      = () => this.requestPointerLock();

    document.addEventListener('keydown',           this._boundKeyDown);
    document.addEventListener('keyup',             this._boundKeyUp);
    document.addEventListener('mousemove',         this._boundMouseMove);
    document.addEventListener('pointerlockchange', this._boundLockChange);
    document.addEventListener('wheel',             this._boundWheel, { passive: true });
    document.addEventListener('click',             this._boundClick);
  }

  _onKeyDown(e) {
    const action = KEY_TO_ACTION[e.code];
    if (action) this._actions[action] = true;

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        this._emit('jump');
        break;
      case 'ControlLeft': case 'ControlRight':
        e.preventDefault();
        break;
      case 'Tab':
        e.preventDefault();
        this._emit('toggleMode');
        break;
      case 'KeyR': this._emit('toggleRCS');             break;
      case 'KeyT': this._emit('toggleDamping');         break;
      case 'KeyL': this._emit('toggleFlashlight');      break;
      case 'KeyX': this._emit('toggleWireframe');       break;
      case 'BracketRight': this._emit('flashlightRangeInc'); break;
      case 'BracketLeft':  this._emit('flashlightRangeDec'); break;
      case 'Equal':        this._emit('flashlightAngleInc'); break;
      case 'Minus':        this._emit('flashlightAngleDec'); break;
    }
  }

  _onKeyUp(e) {
    const action = KEY_TO_ACTION[e.code];
    if (action) this._actions[action] = false;

    if (e.code === 'Space') this._emit('jumpRelease');
  }

  _onMouseMove(e) {
    if (!this._isLocked) return;
    this._mouseDX += e.movementX;
    this._mouseDY += e.movementY;
  }

  _onLockChange() {
    const locked = document.pointerLockElement === document.body;
    if (locked === this._isLocked) return;
    this._isLocked = locked;
    // Reset all held state on lock change — prevents stuck keys when the
    // browser swallows a keyup event during lock acquire/release.
    for (const k of Object.keys(this._actions)) this._actions[k] = false;
    this._mouseDX = 0;
    this._mouseDY = 0;
    this._emit('lockChange', { locked });
  }

  _onWheel(e) {
    this._scrollDelta += e.deltaY;
  }
}
