/**
 * CascadedShadowMap
 *
 * Implements Cascaded Shadow Maps (CSM) for the directional sun light.
 * Reference: https://learnopengl.com/Guest-Articles/2021/CSM
 *
 * Stable cascades: frustum slices are bounded by a rotation-invariant bounding
 * sphere so the shadow frustum size stays constant as the camera rotates.
 * Texel snapping moves the shadow camera only in whole-texel increments,
 * eliminating sub-texel shimmer as the camera translates.
 *
 * Algorithm (per frame):
 *   For each of NUM_CASCADES sub-frustums:
 *     1. Compute the 8 world-space corners of the view frustum slice [near_i, far_i]
 *        directly from the camera's FOV and aspect ratio (no clone).
 *     2. Fit a bounding sphere around the 8 corners (rotation-invariant).
 *     3. Position the shadow camera at sphere-centre + sunDir * SHADOW_CAM_DIST,
 *        looking toward the centre.
 *     4. Build a fixed-size orthographic projection from the sphere radius.
 *     5. Texel-snap the shadow camera so it only moves in whole-texel steps.
 *     6. Render shadow-caster objects (layer 1 = rocks) into a depth texture.
 *     7. Compute the bias-adjusted shadow matrix:
 *          shadowMatrix = biasMatrix × projMatrix × viewMatrix
 *
 * In the fragment shader, gl_FragCoord.z is linearised to view-space depth
 * to select the correct cascade, then Poisson-disk PCF with per-fragment
 * random rotation samples the depth texture.
 */

import * as THREE from 'three';
import { CASCADE_SPLITS } from './SimConfig.js';

// ── Configuration ────────────────────────────────────────────────────────────

/** Number of shadow cascades.  Shader has 3 fixed samplers — keep in sync. */
export const NUM_CASCADES = 3;

// CASCADE_SPLITS imported from SimConfig — [20, 200, 2000]
export { CASCADE_SPLITS };

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

const _center = new THREE.Vector3();
const _lvPt   = new THREE.Vector3();
const _tmpUp  = new THREE.Vector3();

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
    // 1. World-space frustum corners (no allocation)
    this._getFrustumCornersWorld(viewCamera, near, far, this._corners);

    // 2. Bounding sphere — rotation-stable: radius doesn't change when the
    //    camera rotates in place, so shadow frustum size stays constant.
    _center.set(0, 0, 0);
    for (const c of this._corners) _center.add(c);
    _center.divideScalar(8);
    let radius = 0;
    for (const c of this._corners) {
      const d = _center.distanceTo(c);
      if (d > radius) radius = d;
    }
    // Round up to 1/64-unit grid to suppress float noise in radius.
    radius = Math.ceil(radius * 64) / 64;

    // 3. Shadow camera — positioned along sun direction from sphere centre.
    const cam = this._shadowCams[idx];
    if (Math.abs(sunDirNorm.y) > 0.9) _tmpUp.set(1, 0, 0);
    else                               _tmpUp.set(0, 1, 0);
    cam.position.copy(_center).addScaledVector(sunDirNorm, SHADOW_CAM_DIST);
    cam.up.copy(_tmpUp);
    cam.lookAt(_center);
    cam.updateMatrixWorld();

    // 4. Fixed-size orthographic frustum from sphere radius (same size every
    //    frame for a given cascade when the camera only rotates).
    cam.left   = -radius;
    cam.right  =  radius;
    cam.top    =  radius;
    cam.bottom = -radius;
    cam.near   = Math.max(0.1, (SHADOW_CAM_DIST - radius) / Z_MULT);
    cam.far    = (SHADOW_CAM_DIST + radius) * Z_MULT;
    cam.updateProjectionMatrix();

    // 5. Texel-snap: shift the shadow camera in its local XY plane so the
    //    world origin always maps to the same shadow-map texel.  Eliminates
    //    sub-texel shimmer as the camera translates between frames.
    const texelSize = (2.0 * radius) / MAP_SIZE;
    _lvPt.set(0, 0, 0).applyMatrix4(cam.matrixWorldInverse);
    const snapX = Math.round(_lvPt.x / texelSize) * texelSize - _lvPt.x;
    const snapY = Math.round(_lvPt.y / texelSize) * texelSize - _lvPt.y;
    const e = cam.matrixWorld.elements;
    cam.position.x += e[0] * snapX + e[4] * snapY;
    cam.position.y += e[1] * snapX + e[5] * snapY;
    cam.position.z += e[2] * snapX + e[6] * snapY;
    cam.updateMatrixWorld();

    // 6. Final shadow matrix: biasMatrix × projMatrix × viewMatrix
    this.shadowMatrices[idx]
      .copy(_biasMatrix)
      .multiply(cam.projectionMatrix)
      .multiply(cam.matrixWorldInverse);
  }

  /**
   * Compute the 8 world-space corners of the camera's view frustum sliced
   * between [near, far] and write them into the `out` array.
   *
   * Derived directly from the camera's FOV and aspect ratio — no clone needed.
   * In Three.js camera space, forward = −Z, so a point at view-distance z is
   * at (x, y, −z).  applyMatrix4(camera.matrixWorld) converts to world space.
   */
  _getFrustumCornersWorld(camera, near, far, out) {
    const tanH   = Math.tan(camera.fov * Math.PI / 360);
    const aspect = camera.aspect;
    let k = 0;
    for (const z of [near, far]) {
      const h  = tanH * z;
      const w  = h * aspect;
      out[k++].set(-w, -h, -z).applyMatrix4(camera.matrixWorld);
      out[k++].set( w, -h, -z).applyMatrix4(camera.matrixWorld);
      out[k++].set(-w,  h, -z).applyMatrix4(camera.matrixWorld);
      out[k++].set( w,  h, -z).applyMatrix4(camera.matrixWorld);
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
