/**
 * Starfield — procedural star field rendered as a sky sphere.
 *
 * Technique:
 *   A large sphere (BackSide) encloses the scene.  It is rendered with
 *   renderOrder=-1, depthTest=false, depthWrite=false so it is always drawn
 *   first and is overwritten by any opaque geometry in front of it.
 *
 *   The fragment shader places stars using a 3-D cell hash: the normalised
 *   fragment direction is scaled to divide the unit sphere into ~250³ cells,
 *   each cell independently decides whether it contains a star and where
 *   inside the cell the star centre sits.  A smoothstep on the angular
 *   distance produces a soft circular disc per star.
 *
 *   No Point sprites, no vertex buffers — zero geometry updates per frame.
 *   The sphere follows the camera each tick so the sky is always centred.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const SKY_VERT = /* glsl */`
  varying vec3 vDir;

  void main() {
    // Pass local-space vertex position as the sky direction.
    // The sphere is centred on the camera, so local position == direction.
    vDir = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SKY_FRAG = /* glsl */`
  precision highp float;

  varying vec3 vDir;

  // ---------------------------------------------------------------------------
  // Hash helpers
  // ---------------------------------------------------------------------------

  float hash1(vec3 p) {
    p = fract(p * vec3(127.1, 311.7, 74.7));
    p += dot(p, p.yxz + 19.19);
    return fract((p.x + p.y) * p.z);
  }

  // Returns a vec3 with each component independently hashed in [0,1].
  vec3 hash3(vec3 p) {
    return vec3(
      hash1(p),
      hash1(p + vec3(1.7, 9.2, 6.3)),
      hash1(p + vec3(8.3, 2.8, 5.4))
    );
  }

  // ---------------------------------------------------------------------------
  // Star field
  // ---------------------------------------------------------------------------

  // Cell density: 250 cells per unit → ~250² cells per hemisphere arc.
  // Probability 0.5% per cell → ~3900 stars on the full sphere.
  const float CELL_SCALE  = 250.0;
  const float STAR_PROB   = 0.005;
  // Angular radius of each star disc (in cos-distance units ≈ radians for small values).
  const float STAR_RADIUS = 0.006;

  vec3 starField(vec3 dir) {
    vec3 d = normalize(dir);
    vec3 scaled = d * CELL_SCALE;
    vec3 cell   = floor(scaled);

    float rnd = hash1(cell);
    if (rnd > STAR_PROB) return vec3(0.0);

    // Jitter the star centre within the cell then re-normalise to sphere.
    vec3 jitter  = hash3(cell + 17.3);
    vec3 starPos = normalize(cell + jitter);

    // Angular distance (both vectors are unit length).
    float cosA = dot(d, starPos);
    // Convert to approximate linear angular distance for small angles.
    float dist = sqrt(max(0.0, 1.0 - cosA));
    float brightness = smoothstep(STAR_RADIUS, 0.0, dist);

    if (brightness <= 0.0) return vec3(0.0);

    // Stellar colour classification driven by a second hash.
    float colorRnd = hash1(cell + 7.3);
    vec3 color;
    if      (colorRnd < 0.60) color = vec3(1.00, 1.00, 1.00);   // white  G/F
    else if (colorRnd < 0.80) color = vec3(0.72, 0.85, 1.00);   // blue-white B/A
    else if (colorRnd < 0.93) color = vec3(1.00, 0.94, 0.70);   // yellow K
    else                      color = vec3(1.00, 0.65, 0.40);   // orange M

    // Magnitude variation: 50–100 % brightness.
    float mag = 0.5 + hash1(cell + 3.1) * 0.5;

    return brightness * mag * color;
  }

  void main() {
    vec3 col = starField(vDir);
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Starfield class
// ---------------------------------------------------------------------------

export class Starfield {
  /**
   * @param {THREE.Scene}  scene
   * @param {THREE.Camera} camera  – the sky sphere follows the camera each frame
   */
  constructor(scene, camera) {
    this._scene  = scene;
    this._camera = camera;
    this._mesh   = null;
    this._create();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Call once per frame to keep the sky sphere centred on the camera. */
  update(_dt) {
    if (this._mesh) {
      this._mesh.position.copy(this._camera.position);
    }
  }

  dispose() {
    if (!this._mesh) return;
    this._mesh.geometry.dispose();
    this._mesh.material.dispose();
    this._scene.remove(this._mesh);
    this._mesh = null;
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _create() {
    // Radius must be > camera near (0.5) and < camera far (350000).
    // 100 is arbitrary — stars look identical at any radius because
    // the shader operates purely on the fragment direction vector.
    const geometry = new THREE.SphereGeometry(100, 32, 32);

    const material = new THREE.ShaderMaterial({
      vertexShader:   SKY_VERT,
      fragmentShader: SKY_FRAG,
      side:           THREE.BackSide,  // render inside of sphere
      depthTest:      false,           // always draws, never blocked
      depthWrite:     false,           // does not pollute depth buffer
    });

    this._mesh = new THREE.Mesh(geometry, material);
    this._mesh.renderOrder  = -1;       // render before all scene geometry
    this._mesh.frustumCulled = false;   // never cull the sky

    this._scene.add(this._mesh);
  }
}
