/**
 * SceneManager — owns the Three.js renderer, scene, camera, and post-processing.
 *
 * Responsibilities (SRP):
 *   - Create and own: WebGLRenderer, Scene, PerspectiveCamera, EffectComposer
 *   - Drive the render loop delta-time (authoritative clock)
 *   - Delegate per-frame subsystem updates to registered objects
 *   - Handle resize
 *
 * Shadow mapping is handled externally by CascadedShadowMap.js, which renders
 * its depth passes during the subsystem update phase (before composer.render).
 */

import * as THREE from 'three';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass }      from 'three/addons/postprocessing/ShaderPass.js';

const FilmGrainShader = {
  uniforms: {
    tDiffuse:   { value: null },
    uTime:      { value: 0 },
    uIntensity: { value: 0.035 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uIntensity;
    varying vec2 vUv;
    float rand(vec2 co) { return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      float grain = rand(vUv + fract(uTime)) * uIntensity * 2.0 - uIntensity;
      color.rgb = clamp(color.rgb + grain, 0.0, 1.0);
      vec2 c = vUv - 0.5;
      float vig = 1.0 - dot(c, c) * 0.9;
      color.rgb *= clamp(vig, 0.0, 1.0);
      gl_FragColor = color;
    }
  `,
};

export class SceneManager {
  constructor() {
    this._clock      = new THREE.Clock();
    this._subsystems = [];

    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.composer = null;
    this._filmGrainPass = null;

    this._init();
  }

  register(subsystem) {
    this._subsystems.push(subsystem);
  }

  tick() {
    const dt = Math.min(this._clock.getDelta(), 0.1);

    for (const s of this._subsystems) {
      s.update(dt);
    }

    if (this._filmGrainPass) {
      this._filmGrainPass.uniforms.uTime.value += dt;
    }

    this.composer.render();
    return dt;
  }

  dispose() {
    window.removeEventListener('resize', this._resizeHandler);
    this.composer.dispose();
    this.renderer.dispose();
  }

  _init() {
    this._createScene();
    this._createCamera();
    this._createRenderer();
    this._createLighting();
    this._createPostProcessing();
    this._resizeHandler = () => this._onResize();
    window.addEventListener('resize', this._resizeHandler);
  }

  _createScene() {
    this.scene = new THREE.Scene();
  }

  _createCamera() {
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.5,
      350000
    );
  }

  _createRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      antialias:       true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping         = THREE.ReinhardToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    document.getElementById('canvas-container').appendChild(this.renderer.domElement);
  }

  _createLighting() {
    // Low ambient — actual directional shading done in custom shaders via
    // uSunDirection.  This just prevents pitch-black non-custom objects.
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.05));
  }

  _createPostProcessing() {
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    const filmGrain = new ShaderPass(FilmGrainShader);
    this.composer.addPass(filmGrain);
    this._filmGrainPass = filmGrain;
  }

  _onResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    for (const s of this._subsystems) {
      if (typeof s.resize === 'function') s.resize(w, h);
    }
  }
}
