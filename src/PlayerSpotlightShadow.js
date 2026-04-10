/**
 * PlayerSpotlightShadow — shadow casting for the player's flashlight.
 *
 * Unlike the sun (directional, infinite distance), the spotlight is attached
 * to the camera and has finite range + angle.  We render rocks as shadow
 * casters into a depth texture from the spotlight's perspective.
 *
 * The spotlight shadows are added on top of the existing sun CSM in the
 * fragment shader.
 */

import * as THREE from 'three';

const SPOT_MAP_SIZE = 512;

const _biasMatrix = new THREE.Matrix4().set(
  0.5, 0,   0,   0.5,
  0,   0.5, 0,   0.5,
  0,   0,   0.5, 0.5,
  0,   0,   0,   1
);

export class PlayerSpotlightShadow {
  constructor(renderer, scene) {
    this._renderer = renderer;
    this._scene = scene;

    this.depthTexture = null;
    this.shadowMatrix = new THREE.Matrix4();

    this._rt = null;
    this._spotCamera = null;
    this._depthMat = new THREE.MeshBasicMaterial({ colorWrite: false });

    this._initResources();
  }

  _initResources() {
    this.depthTexture = new THREE.DepthTexture(SPOT_MAP_SIZE, SPOT_MAP_SIZE);
    this.depthTexture.type = THREE.UnsignedShortType;
    this.depthTexture.format = THREE.DepthFormat;
    this.depthTexture.minFilter = THREE.NearestFilter;
    this.depthTexture.magFilter = THREE.NearestFilter;

    this._rt = new THREE.WebGLRenderTarget(SPOT_MAP_SIZE, SPOT_MAP_SIZE, {
      depthTexture: this.depthTexture,
      depthBuffer: true,
    });

    this._spotCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
    this._spotCamera.layers.set(1); // rocks only
    this._scene.add(this._spotCamera);
  }

  /**
   * Configure spotlight parameters.
   * @param {number} angle — spotlight half-angle in radians (aperture)
   * @param {number} range — maximum distance the light reaches
   */
  configure(angle, range) {
    this._spotCamera.fov = angle * 2 * (180 / Math.PI);
    // Near plane set to 10% of range — excludes rocks within arm's reach
    // of the player that would project huge silhouettes on the terrain below.
    // At default range=30 this is 3 units, safely beyond walking altitude (~2.5).
    this._spotCamera.near = Math.max(2.0, range * 0.10);
    this._spotCamera.far = range;
    this._spotCamera.updateProjectionMatrix();
  }

  /**
   * Render spotlight shadows. Call before main render when flashlight is on.
   * @param {THREE.Camera} viewCamera — the player's camera
   * @param {THREE.Vector3} spotDir — normalized direction the camera is facing
   */
  render(viewCamera, spotDir) {
    const cam = this._spotCamera;

    cam.position.copy(viewCamera.position);
    cam.up.set(0, 1, 0);
    cam.lookAt(
      viewCamera.position.x + spotDir.x,
      viewCamera.position.y + spotDir.y,
      viewCamera.position.z + spotDir.z
    );
    cam.updateMatrixWorld();

    this.shadowMatrix
      .copy(_biasMatrix)
      .multiply(cam.projectionMatrix)
      .multiply(cam.matrixWorldInverse);

    const r = this._renderer;
    const s = this._scene;

    s.overrideMaterial = this._depthMat;
    r.setRenderTarget(this._rt);
    r.clear(false, true, false);
    r.render(s, cam);
    r.setRenderTarget(null);
    s.overrideMaterial = null;
  }

  dispose() {
    this._rt.dispose();
    this._scene.remove(this._spotCamera);
    this._depthMat.dispose();
  }
}