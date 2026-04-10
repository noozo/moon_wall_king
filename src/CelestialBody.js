/**
 * CelestialBody — textured sphere with Keplerian orbital mechanics,
 * optional corona halo (Sun) and atmospheric rim scattering (Earth).
 *
 * Orbital model — Moon-centric frame
 * ───────────────────────────────────
 * The Moon terrain is fixed at world origin.  All other bodies are sky objects
 * whose apparent motion is described by Keplerian orbits.
 *
 *   Earth: follows the Moon's real orbital elements around Earth.
 *          From Moon's surface, Earth traces this path in the sky.
 *          Period ≈ 27.32 days (sidereal month).
 *
 *   Sun:   follows Earth's real orbital elements around the Sun.
 *          From Moon's surface, the Sun traces this path.
 *          Period ≈ 365.25 days (tropical year).
 *
 *   Moon (sky body): small sphere orbiting the Earth sky object; visible as a
 *          tiny companion when looking at Earth. Uses same orbital elements but
 *          orbits the Earth sky body's position, not world origin.
 *
 * Visual scale:
 *   Earth radius 1500 @ 25 000 → ~6.9° apparent diameter (real ≈ 1.9°, scaled for clarity)
 *   Sun   radius 5000 @ 200 000 → ~2.9° apparent diameter (real ≈ 0.53°, scaled for clarity)
 *   Moon-sky radius 400 @ 5000 from Earth → ~2.3° apparent near Earth
 *
 * TIME_SCALE = 1/3000:  1 real second = 50 simulation minutes.
 *   Earth orbit 27.32 d → ~787 s real  (~13 min per lap)
 *   Sun   orbit 365.25 d → ~10 522 s real (~2.9 h per lap)
 */

import * as THREE from 'three';
import { sunDirection } from './SunDirection.js';
import { SIM_TIME_SCALE } from './SimConfig.js';

// ============================================================================
// Sun-lit planet shader (used for Earth)
// ============================================================================

/**
 * Lambertian diffuse shading from the dynamic Sun direction.
 * World-space normals are used so the correct face is lit even when the
 * planet is rotating (modelMatrix includes axial tilt + day rotation).
 */
const PLANET_VERT = /* glsl */`
  varying vec3 vWorldNormal;
  varying vec2 vUv;

  void main() {
    // mat3(modelMatrix) strips translation; correct for normals on a sphere
    // with only rotation applied (no non-uniform scale).
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    vUv          = uv;
    gl_Position  = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const PLANET_FRAG = /* glsl */`
  precision mediump float;

  uniform sampler2D uColorMap;
  uniform vec3      uSunDir;   // world-space direction toward Sun (unit vector)

  varying vec3 vWorldNormal;
  varying vec2 vUv;

  void main() {
    vec3  albedo = texture2D(uColorMap, vUv).rgb;
    float NdotL  = dot(normalize(vWorldNormal), normalize(uSunDir));

    // Soft terminator: smoothstep turns the sharp mathematical terminator
    // into a gentle twilight band, matching real planetary photography.
    float diffuse = smoothstep(-0.12, 0.30, NdotL);

    // Tiny ambient so the night side is dark but not pure black.
    float ambient = 0.04;

    gl_FragColor = vec4(albedo * (diffuse + ambient), 1.0);
  }
`;

// ============================================================================
// Atmosphere rim-scattering shader
// ============================================================================

/**
 * Thin atmosphere sphere rendered with additive blending.
 *
 * Three physical constraints drive the shader:
 *   1. Rim term:     only the limb (edge) of the disc glows — not the full disc.
 *   2. Sun term:     only the sun-facing hemisphere is lit; the night-side limb
 *                    is dark (the atmosphere is in shadow on that arc).
 *   3. Thin scale:   the geometry is only 4 % larger than the planet, so depth
 *                    testing against the opaque planet naturally hides the glow
 *                    everywhere except the true outer rim.
 *
 * Normal and view direction are computed in view space to avoid coordinate
 * system mismatch.  Sun direction is passed in world space and transformed to
 * view space in the vertex shader.
 */
const ATMOS_VERT = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vSunViewDir;   // Sun direction in view space

  uniform vec3 uSunDir;       // world-space unit vector toward Sun

  void main() {
    vNormal     = normalize(normalMatrix * normal);
    vec4 vp     = modelViewMatrix * vec4(position, 1.0);
    vViewDir    = normalize(-vp.xyz);
    // Transform world-space sun direction into view space (no translation)
    vSunViewDir = normalize(mat3(viewMatrix) * uSunDir);
    gl_Position = projectionMatrix * vp;
  }
`;

const ATMOS_FRAG = /* glsl */`
  precision mediump float;

  uniform vec3  uAtmosColor;
  uniform float uDensity;

  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vSunViewDir;

  void main() {
    vec3  N    = normalize(vNormal);
    vec3  V    = normalize(vViewDir);
    vec3  S    = normalize(vSunViewDir);

    // Rim: tight power keeps glow only at the geometric limb.
    float rim  = 1.0 - max(0.0, dot(N, V));
    float glow = pow(rim, 6.0) * 0.85 + pow(rim, 14.0) * 0.40;

    // Sun-facing: only the lit arc glows.
    // Soft transition around the terminator so the atmosphere tapers off
    // gradually rather than cutting off sharply.
    float NdotS  = dot(N, S);
    float sunLit = smoothstep(-0.35, 0.35, NdotS);

    float alpha = clamp(glow * sunLit * uDensity, 0.0, 1.0);
    gl_FragColor = vec4(uAtmosColor, alpha);
  }
`;

// ============================================================================
// Keplerian orbital mechanics
// ============================================================================

class OrbitalElements {
  constructor({
    semiMajorAxis,
    eccentricity      = 0,
    inclination       = 0,
    longAscendingNode = 0,
    argPeriapsis      = 0,
    meanAnomalyEpoch  = 0,
    orbitalPeriod,
    axialTilt         = 0,
    rotationPeriod    = null,
  }) {
    this.a     = semiMajorAxis;
    this.e     = eccentricity;
    this.i     = inclination;
    this.omega = longAscendingNode;
    this.w     = argPeriapsis;
    this.M0    = meanAnomalyEpoch;
    this.P     = orbitalPeriod;
    this.axialTilt      = axialTilt;
    this.rotationPeriod = rotationPeriod;
  }
}

/** Solve Kepler's equation M = E − e·sin(E) via Newton–Raphson. */
function solveKepler(M, e) {
  let E = M;
  for (let i = 0; i < 10; i++) {
    E -= (E - e * Math.sin(E) - M) / (1.0 - e * Math.cos(E));
  }
  return E;
}

/** In-plane position: x along periapsis, z 90° ahead. */
function orbitalPlanePos(E, a, e) {
  return {
    x: a * (Math.cos(E) - e),
    z: a * Math.sqrt(1.0 - e * e) * Math.sin(E),
  };
}

/**
 * Rotate orbital-plane position (px, pz) to world space.
 * Standard Euler rotation: R_z(−Ω) · R_x(−i) · R_z(−ω)
 */
function orbitalToWorld(px, pz, e) {
  const cosW = Math.cos(e.w),  sinW = Math.sin(e.w);
  const cosI = Math.cos(e.i),  sinI = Math.sin(e.i);
  const cosO = Math.cos(e.omega), sinO = Math.sin(e.omega);

  // Rotate by argument of periapsis ω in orbital plane
  const q1x =  px * cosW - pz * sinW;
  const q1z =  px * sinW + pz * cosW;

  // Tilt by inclination i — raises out of reference plane
  const q2x =  q1x;
  const q2y = -q1z * sinI;
  const q2z =  q1z * cosI;

  // Rotate by longitude of ascending node Ω around world-Y
  return new THREE.Vector3(
     q2x * cosO - q2z * sinO,
     q2y,
     q2x * sinO + q2z * cosO,
  );
}

// ============================================================================
// CelestialBody
// ============================================================================

export class CelestialBody {
  /**
   * @param {THREE.Scene} scene
   * @param {object}  config
   * @param {number}         config.radius
   * @param {string|null}    config.texturePath
   * @param {number}         config.color      – hex fallback while texture loads
   * @param {OrbitalElements|null} config.orbit
   * @param {object|null}    config.halo       – { radiusScale, color } sun corona
   * @param {object|null}    config.atmosphere – { color, scale, density }
   */
  constructor(scene, config) {
    this._scene = scene;

    this.radius   = config.radius;
    this.orbit    = config.orbit ?? null;
    this.position = new THREE.Vector3();
    this._time    = 0;

    // ── Main sphere ────────────────────────────────────────────────────────
    // litBySun: ShaderMaterial with Lambertian shading from sunDirection.
    // Otherwise: MeshBasicMaterial (self-lit, used for the Sun).
    let mat;
    if (config.litBySun && config.texturePath) {
      const tex = new THREE.TextureLoader().load(config.texturePath);
      tex.colorSpace = THREE.SRGBColorSpace;
      mat = new THREE.ShaderMaterial({
        vertexShader:   PLANET_VERT,
        fragmentShader: PLANET_FRAG,
        uniforms: {
          uColorMap: { value: tex },
          uSunDir:   { value: sunDirection }, // shared reference, auto-updates
        },
      });
    } else {
      mat = new THREE.MeshBasicMaterial({ color: config.color ?? 0xffffff });
      if (config.texturePath) {
        const tex = new THREE.TextureLoader().load(config.texturePath);
        tex.colorSpace = THREE.SRGBColorSpace;
        mat.map = tex;
      }
    }

    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(this.radius, 64, 64),
      mat,
    );
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);

    // ── Optional corona halo (Sun) ─────────────────────────────────────────
    if (config.halo) this._addHalo(config.halo);

    // ── Optional atmosphere glow (Earth, planets) ──────────────────────────
    if (config.atmosphere) this._addAtmosphere(config.atmosphere);
  }

  // ---------------------------------------------------------------------------
  // Optional effects
  // ---------------------------------------------------------------------------

  /**
   * Sun corona / halo — a THREE.Sprite with a soft radial gradient texture,
   * rendered with additive blending so it brightens anything behind it.
   * The sprite auto-billboards toward the camera.
   *
   * Two nested sprites: inner bright halo + outer faint corona.
   */
  _addHalo({ radiusScale = 4, innerScale = 1.6 }) {
    const makeHaloTexture = (isInner) => {
      const SIZE = 512;
      const canvas = document.createElement('canvas');
      canvas.width  = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext('2d');
      const cx  = SIZE / 2;
      const g   = ctx.createRadialGradient(cx, cx, 0, cx, cx, cx);

      if (isInner) {
        // Inner disc: bright white-yellow core with a tight orange fade
        g.addColorStop(0.00, 'rgba(255,255,240,1.00)');
        g.addColorStop(0.10, 'rgba(255,250,200,1.00)');
        g.addColorStop(0.30, 'rgba(255,230,120,0.90)');
        g.addColorStop(0.55, 'rgba(255,180, 50,0.40)');
        g.addColorStop(0.80, 'rgba(255,120, 20,0.10)');
        g.addColorStop(1.00, 'rgba(220, 80,  0,0.00)');
      } else {
        // Outer corona: wide warm glow, fades to transparent
        g.addColorStop(0.00, 'rgba(255,200, 80,0.55)');
        g.addColorStop(0.20, 'rgba(255,150, 40,0.30)');
        g.addColorStop(0.50, 'rgba(220, 90, 10,0.10)');
        g.addColorStop(0.80, 'rgba(180, 50,  0,0.03)');
        g.addColorStop(1.00, 'rgba(120, 20,  0,0.00)');
      }
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, SIZE, SIZE);
      return new THREE.CanvasTexture(canvas);
    };

    // Inner bright halo
    const innerMat = new THREE.SpriteMaterial({
      map:         makeHaloTexture(true),
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
      transparent: true,
    });
    const innerSprite = new THREE.Sprite(innerMat);
    const s = this.radius * innerScale * 2;
    innerSprite.scale.set(s, s, 1);
    innerSprite.renderOrder = 1;
    this.mesh.add(innerSprite);

    // Outer corona
    const outerMat = new THREE.SpriteMaterial({
      map:         makeHaloTexture(false),
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
      transparent: true,
    });
    const outerSprite = new THREE.Sprite(outerMat);
    const sc = this.radius * radiusScale * 2;
    outerSprite.scale.set(sc, sc, 1);
    outerSprite.renderOrder = 1;
    this.mesh.add(outerSprite);

    this._haloSprites = [innerSprite, outerSprite];
  }

  /**
   * Planet atmosphere — a slightly larger sphere rendered with additive
   * blending and a rim-scattering shader that glows at the limb.
   * Parented to this.mesh so it moves/rotates with the planet automatically.
   */
  _addAtmosphere({ color = new THREE.Color(0x4488ff), scale = 1.04, density = 1.2 }) {
    const mat = new THREE.ShaderMaterial({
      vertexShader:   ATMOS_VERT,
      fragmentShader: ATMOS_FRAG,
      uniforms: {
        uAtmosColor: { value: new THREE.Color(color) },
        uDensity:    { value: density },
        uSunDir:     { value: sunDirection },  // shared singleton, auto-updates
      },
      side:        THREE.FrontSide,
      transparent: true,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });

    const atmosMesh = new THREE.Mesh(
      new THREE.SphereGeometry(this.radius * scale, 32, 32),
      mat,
    );
    atmosMesh.renderOrder = 2; // after the planet itself
    this.mesh.add(atmosMesh); // inherits planet transform automatically
    this._atmosMesh = atmosMesh;
  }

  // ---------------------------------------------------------------------------
  // Per-frame update
  // ---------------------------------------------------------------------------

  /**
   * Advance orbital simulation and sync mesh position.
   * @param {number}           dt         – seconds since last frame
   * @param {THREE.Vector3|null} centerPos – world position of the orbit focus
   *                                         (null → orbit around world origin)
   * @returns {THREE.Vector3} current world position
   */
  update(dt, centerPos = null) {
    this._time += dt;

    if (this.orbit) {
      const { a, e, M0, P, axialTilt, rotationPeriod } = this.orbit;

      const M = M0 + (2 * Math.PI * this._time) / P;
      const E = solveKepler(M, e);
      const { x, z } = orbitalPlanePos(E, a, e);
      const offset = orbitalToWorld(x, z, this.orbit);

      if (centerPos) {
        this.position.copy(centerPos).add(offset);
      } else {
        this.position.copy(offset);
      }

      this.mesh.position.copy(this.position);

      if (rotationPeriod) {
        this.mesh.rotation.y = (2 * Math.PI * this._time) / rotationPeriod;
        this.mesh.rotation.z = axialTilt;
      }
    }

    return this.position;
  }

  /**
   * Set render-space position from an external source (WorldSimulator in Phase 3).
   * Updates both the internal position Vector3 and mesh.position.
   * @param {THREE.Vector3} renderPos
   */
  setPosition(renderPos) {
    this.position.copy(renderPos);
    this.mesh.position.copy(renderPos);
  }

  /**
   * Advance time-based animations (axial rotation, halo effects) without
   * running the internal Keplerian orbit.  Used in Phase 3 where WorldSimulator
   * owns all orbital positions.
   * @param {number} dt
   */
  updateRotationOnly(dt) {
    this._time += dt;
    if (this.orbit?.rotationPeriod) {
      this.mesh.rotation.y = (2 * Math.PI * this._time) / this.orbit.rotationPeriod;
      this.mesh.rotation.z = this.orbit.axialTilt ?? 0;
    }
  }

  getPosition() { return this.position; }

  dispose() {
    this._haloSprites?.forEach(s => {
      s.material.map?.dispose();
      s.material.dispose();
    });
    this._atmosMesh?.geometry.dispose();
    this._atmosMesh?.material.uniforms?.uAtmosColor && (this._atmosMesh.material.dispose());
    this._atmosMesh?.material.dispose();
    const mainMap = this.mesh.material.uniforms?.uColorMap?.value
                 ?? this.mesh.material.map;
    mainMap?.dispose();
    this.mesh.material.dispose();
    this._scene.remove(this.mesh);
  }
}

// ============================================================================
// Solar system factory
// ============================================================================

/**
 * Create Earth and Sun as sky bodies visible from the Moon terrain.
 * The Moon terrain at world origin is NOT recreated here.
 *
 * @param {THREE.Scene} scene
 * @param {number}      moonRadius  default 1000
 * @returns {{ sun: CelestialBody, earth: CelestialBody }}
 */
export function createSolarSystem(scene, moonRadius = 1000) {
  // ── Distances & radii ─────────────────────────────────────────────────────
  const EARTH_DIST = moonRadius * 25;   // 25 000
  const SUN_DIST   = moonRadius * 200;  // 200 000

  const EARTH_RADIUS = moonRadius * 1.5; // 1 500
  const SUN_RADIUS   = moonRadius * 5;   // 5 000

  // ── Time scale ────────────────────────────────────────────────────────────
  // 1 real second = 3 000 simulated seconds.
  const DAY_S      = 86400;
  const TIME_SCALE = SIM_TIME_SCALE;

  // ── Orbital elements ──────────────────────────────────────────────────────

  // Earth in Moon-centric frame — Moon's real orbital elements around Earth
  const earthOrbit = new OrbitalElements({
    semiMajorAxis:     EARTH_DIST,
    eccentricity:      0.0549,
    inclination:       0.0898,
    longAscendingNode: 0,
    argPeriapsis:      0,
    meanAnomalyEpoch:  0,
    orbitalPeriod:     27.32 * DAY_S * TIME_SCALE,
    axialTilt:         0.4093,
    rotationPeriod:    DAY_S * TIME_SCALE,
  });

  // Sun in Moon-centric frame — Earth's real orbital elements around Sun
  const sunOrbit = new OrbitalElements({
    semiMajorAxis:     SUN_DIST,
    eccentricity:      0.0167,
    inclination:       0.0,
    longAscendingNode: 0,
    argPeriapsis:      0,
    meanAnomalyEpoch:  0,
    orbitalPeriod:     365.25 * DAY_S * TIME_SCALE,
    axialTilt:         0.1222,
    rotationPeriod:    25 * DAY_S * TIME_SCALE,
  });

  // ── Bodies ────────────────────────────────────────────────────────────────

  // Sun: no texture — from space the Sun disc is a near-uniform white/yellow.
  // The halo sprites provide all the visual interest.
  const sun = new CelestialBody(scene, {
    radius: SUN_RADIUS,
    color:  0xfffce8,   // bright warm white, slightly yellow
    orbit:  sunOrbit,
    halo: {
      radiusScale: 6,
      innerScale:  2.0,
    },
  });

  const earth = new CelestialBody(scene, {
    radius:      EARTH_RADIUS,
    texturePath: '/textures/earth_color_4k.jpg',
    orbit:       earthOrbit,
    litBySun:    true,          // Lambertian shading from actual Sun direction
    atmosphere: {
      color:   new THREE.Color(0.55, 0.78, 1.0),  // light sky-blue, not deep navy
      scale:   1.04,                               // 4 % thicker than Earth — physically thin
      density: 1.8,                                // bright at the actual limb edge
    },
  });

  // Compute initial positions so meshes aren't at origin on frame 0.
  sun.update(0);
  earth.update(0);

  return { sun, earth };
}
