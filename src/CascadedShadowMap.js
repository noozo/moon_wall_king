/**
 * CascadedShadowMap
 *
 * Implements Cascaded Shadow Maps (CSM) for the directional sun light.
 * Reference: https://learnopengl.com/Guest-Articles/2021/CSM
 *
 * Algorithm (per frame):
 *   For each of NUM_CASCADES sub-frustums:
 *     1. Compute the 8 world-space corners of the view frustum slice [near_i, far_i]
 *        by inverting (projMatrix * viewMatrix) applied to the NDC cube corners.
 *     2. Average the corners to find the frustum centre.
 *     3. Position the shadow camera at centre + sunDir * SHADOW_CAM_DIST,
 *        looking toward the centre.
 *     4. Transform all 8 corners into light view space; find the AABB.
 *     5. Build a tight orthographic projection from that AABB, expanded in Z
 *        by Z_MULT to include geometry outside the view that can still cast shadows.
 *     6. Render shadow-caster objects (layer 1 = rocks) into a depth texture.
 *     7. Compute the bias-adjusted shadow matrix:
 *          shadowMatrix = biasMatrix × projMatrix × viewMatrix
 *
 * In the fragment shader, gl_FragCoord.z is linearised to view-space depth
 * to select the correct cascade, then Poisson-disk PCF with per-fragment
 * random rotation samples the depth texture.
 */

import * as THREE from 'three';

// ── Configuration ────────────────────────────────────────────────────────────

/** Number of shadow cascades.  Shader has 3 fixed samplers — keep in sync. */
export const NUM_CASCADES = 3;

/**
 * View-space distances (world units from camera) at which each cascade ends.
 * Cascade 0: [camera.near, SPLITS[0]]
 * Cascade 1: [SPLITS[0],   SPLITS[1]]
 * Cascade 2: [SPLITS[1],   SPLITS[2]]
 */
export const CASCADE_SPLITS = [20, 200, 2000];

/** Shadow map resolution.  All cascades use the same size for simplicity. */
const MAP_SIZE = 1024;

/**
 * Z-axis expansion multiplier for the shadow frustum.
 * Small buffer ensures geometry just outside the view slice still casts shadows.
 * Keep low: large values compress depth range, making rock vs terrain depth
 * differences smaller than the bias → no shadows visible.
 */
const Z_MULT = 2.0;

/** Distance from frustum centre to shadow camera (along sun direction). */
const SHADOW_CAM_DIST = 60;

// ── Module-level scratch objects (zero allocation per frame) ─────────────────

// Bias matrix: remaps NDC [-1,1]³  →  texture UV+depth [0,1]³
const _biasMatrix = new THREE.Matrix4().set(
  0.5, 0,   0,   0.5,
  0,   0.5, 0,   0.5,
  0,   0,   0.5, 0.5,
  0,   0,   0,   1
);

// All 8 NDC cube corner points
const _NDC_CORNERS = (() => {
  const pts = [];
  for (let x = -1; x <= 1; x += 2)
    for (let y = -1; y <= 1; y += 2)
      for (let z = -1; z <= 1; z += 2)
        pts.push(new THREE.Vector4(x, y, z, 1));
  return pts;
})();

const _invPV  = new THREE.Matrix4();
const _ndcPt  = new THREE.Vector4();
const _center = new THREE.Vector3();
const _lvPt   = new THREE.Vector3();

// ── CascadedShadowMap ────────────────────────────────────────────────────────

export class CascadedShadowMap {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene}         scene
   */
  constructor(renderer, scene) {
    this._renderer = renderer;
    this._scene    = scene;

    /**
     * Depth textures (one per cascade).
     * Set as uShadowMap0/1/2 on the terrain material once at startup — the
     * content is updated in-place each frame by renderShadows().
     */
    this.shadowTextures = [];

    /**
     * Bias-adjusted shadow matrices (one per cascade).
     * Set as uShadowMatrix0/1/2 on the terrain material once — modified
     * in-place each frame, Three.js reads the latest values on render.
     */
    this.shadowMatrices = Array.from({ length: NUM_CASCADES },
      () => new THREE.Matrix4());

    this._rts        = [];   // WebGLRenderTarget[]
    this._shadowCams = [];   // OrthographicCamera[]

    // depth-only override material — works with InstancedMesh automatically
    this._depthMat = new THREE.MeshBasicMaterial({ colorWrite: false });

    // 8 world-space corner positions, reused across cascades
    this._corners = Array.from({ length: 8 }, () => new THREE.Vector3());

    this._initResources();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Fit cascade frustums, render all shadow depth maps.
   * Call once per frame BEFORE the main scene render.
   *
   * @param {THREE.PerspectiveCamera} viewCamera  The player/fly camera
   * @param {THREE.Vector3}           sunDirNorm  Unit vector toward the Sun (render space)
   */
  renderShadows(viewCamera, sunDirNorm) {
    const splits = [viewCamera.near, ...CASCADE_SPLITS];

    for (let i = 0; i < NUM_CASCADES; i++) {
      this._fitCascade(viewCamera, sunDirNorm, splits[i], splits[i + 1], i);
      this._renderCascade(i);
    }
  }

  dispose() {
    for (let i = 0; i < NUM_CASCADES; i++) {
      this._rts[i].dispose();
      this._scene.remove(this._shadowCams[i]);
    }
    this._depthMat.dispose();
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  _initResources() {
    for (let i = 0; i < NUM_CASCADES; i++) {
      const depthTex = new THREE.DepthTexture(MAP_SIZE, MAP_SIZE);
      depthTex.type      = THREE.UnsignedShortType;
      depthTex.format    = THREE.DepthFormat;
      depthTex.minFilter = THREE.NearestFilter;
      depthTex.magFilter = THREE.NearestFilter;

      const rt = new THREE.WebGLRenderTarget(MAP_SIZE, MAP_SIZE, {
        depthTexture: depthTex,
        depthBuffer:  true,
      });

      // Layer 1 only: rocks cast shadows.  Earth, Sun, Starfield stay on
      // layer 0 and are invisible to the shadow camera.
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      cam.layers.set(1);
      this._scene.add(cam);

      this._rts.push(rt);
      this._shadowCams.push(cam);
      this.shadowTextures.push(depthTex);
    }
  }

  /**
   * Fit the i-th shadow camera's orthographic frustum tightly around the
   * camera's view sub-frustum [near, far] and update shadowMatrices[idx].
   */
  _fitCascade(viewCamera, sunDirNorm, near, far, idx) {
    // 1. Get the 8 world-space corners for this frustum slice.
    this._getFrustumCornersWorld(viewCamera, near, far, this._corners);

    // 2. Frustum centre = average of all 8 corners.
    _center.set(0, 0, 0);
    for (const c of this._corners) _center.add(c);
    _center.divideScalar(8);

    // 3. Shadow camera: positioned along sun direction from centre.
    //    Use a stable up-vector — if sunDir is close to world-Y (sun overhead),
    //    fall back to world-X to avoid lookAt gimbal lock.
    const cam = this._shadowCams[idx];
    cam.position.copy(_center).addScaledVector(sunDirNorm, SHADOW_CAM_DIST);
    const up = (Math.abs(sunDirNorm.y) > 0.9)
      ? new THREE.Vector3(1, 0, 0)
      : new THREE.Vector3(0, 1, 0);
    cam.up.copy(up);
    cam.lookAt(_center);
    cam.updateMatrixWorld();

    // 4. AABB of frustum corners in light-view space.
    let minX = Infinity,  maxX = -Infinity;
    let minY = Infinity,  maxY = -Infinity;
    let minZ = Infinity,  maxZ = -Infinity;

    for (const c of this._corners) {
      _lvPt.copy(c).applyMatrix4(cam.matrixWorldInverse);
      if (_lvPt.x < minX) minX = _lvPt.x;
      if (_lvPt.x > maxX) maxX = _lvPt.x;
      if (_lvPt.y < minY) minY = _lvPt.y;
      if (_lvPt.y > maxY) maxY = _lvPt.y;
      if (_lvPt.z < minZ) minZ = _lvPt.z;
      if (_lvPt.z > maxZ) maxZ = _lvPt.z;
    }

    // 5. Set tight orthographic frustum, extended in Z for out-of-view casters.
    //    Three.js camera space: forward = -Z.  Objects in front are at z < 0.
    //    near = -maxZ  (closest to camera, least negative z)
    //    far  = -minZ  (farthest from camera, most negative z)
    const rawNear = -maxZ;
    const rawFar  = -minZ;
    cam.left   = minX;
    cam.right  = maxX;
    cam.top    = maxY;
    cam.bottom = minY;
    cam.near   = Math.max(0.1, rawNear / Z_MULT);
    cam.far    = rawFar * Z_MULT;
    cam.updateProjectionMatrix();

    // 6. Shadow matrix: biasMatrix × projMatrix × viewMatrix
    //    Maps world-space position → [0,1]³ UV + depth.
    this.shadowMatrices[idx]
      .copy(_biasMatrix)
      .multiply(cam.projectionMatrix)
      .multiply(cam.matrixWorldInverse);
  }

  /**
   * Compute the 8 world-space corners of the camera's view frustum sliced
   * between [near, far] and write them into the `out` array.
   *
   * Technique: transform the 8 NDC cube corners through the inverse of
   * (projMatrix × viewMatrix) with near/far overridden for this slice.
   */
  _getFrustumCornersWorld(camera, near, far, out) {
    // Clone camera to override near/far without mutating the original.
    // Allocation here is acceptable — called only NUM_CASCADES (3) times/frame.
    const sliceCam = camera.clone();
    sliceCam.near = near;
    sliceCam.far  = far;
    sliceCam.updateProjectionMatrix();

    _invPV
      .multiplyMatrices(sliceCam.projectionMatrix, sliceCam.matrixWorldInverse)
      .invert();

    for (let i = 0; i < 8; i++) {
      _ndcPt.copy(_NDC_CORNERS[i]).applyMatrix4(_invPV);
      out[i].set(_ndcPt.x / _ndcPt.w, _ndcPt.y / _ndcPt.w, _ndcPt.z / _ndcPt.w);
    }
  }

  /** Render shadow-caster geometry into cascade i's depth texture. */
  _renderCascade(idx) {
    const r = this._renderer;
    const s = this._scene;

    s.overrideMaterial = this._depthMat;
    r.setRenderTarget(this._rts[idx]);
    r.clear(false, true, false);          // clear depth only (default = 1.0)
    r.render(s, this._shadowCams[idx]);
    r.setRenderTarget(null);
    s.overrideMaterial = null;
  }
}
