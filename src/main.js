/**
 * main.js — Application entry point for Moon Wall King.
 *
 * Coordinate system (Phase 1 of the floating-origin architecture):
 *
 *   moonGroup  — THREE.Group that is the single parent for ALL Moon-surface
 *                content: terrain tiles, rocks, (future) surface assets.
 *                In Phase 1 it sits at world origin.  In Phase 3 it follows
 *                the Moon's Keplerian orbit around Earth.
 *
 *   worldSim   — WorldSimulator tracks the inertial (float64) state.  In
 *                Phase 1 everything is at zero; the class exists to provide
 *                the right interface for Phase 3 without refactoring.
 *
 *   All terrain/rock LOD queries receive Moon-local camera coords
 *   (camera − moonGroup.position) so they remain correct when the Moon moves.
 *
 * Two controllers share the same camera:
 *   PlayerController   — surface walking with gravity / jumping
 *   DebugFlyController — free-fly for terrain/LOD inspection
 *
 * Tab toggles between them at any time.
 */

import * as THREE from 'three';
import { MoonData }           from './MoonData.js';
import { SceneManager }       from './SceneManager.js';
import { TerrainSystem }      from './TerrainSystem.js';
import { RockSystem }         from './RockSystem.js';
import { PlayerController }   from './PlayerController.js';
import { DebugFlyController } from './DebugFlyController.js';
import { Starfield }          from './Starfield.js';
import { UI }                 from './UI.js';
import { createSolarSystem }  from './CelestialBody.js';
import { updateSunDirection, updateEarthDirection } from './SunDirection.js';
import { WorldSimulator }     from './WorldSimulator.js';
import { CascadedShadowMap } from './CascadedShadowMap.js';
import { PlayerSpotlightShadow } from './PlayerSpotlightShadow.js';
import { SIM_TIME_SCALE }     from './SimConfig.js';

const MOON_RADIUS = 1000;

// ---------------------------------------------------------------------------
// Moon spin — tidal locking
// ---------------------------------------------------------------------------

// Time scale — imported from SimConfig so WorldSimulator and MOON_SPIN_RATE
// stay in sync automatically.
const TIME_SCALE = SIM_TIME_SCALE;

// Moon sidereal rotation period = 27.32 days.
// Because the rotation matches the orbital period (tidal locking), the Earth
// CelestialBody orbital period and this spin rate are identical → Earth stays
// roughly overhead.
const MOON_SIDEREAL_S = 27.32 * 86400;              // real seconds
const MOON_SPIN_PERIOD = MOON_SIDEREAL_S * TIME_SCALE;  // real seconds at our time scale
const MOON_SPIN_RATE   = (2 * Math.PI) / MOON_SPIN_PERIOD; // radians / real second

// ---------------------------------------------------------------------------
// Debug config
// ---------------------------------------------------------------------------

const DEBUG = false;

// ---------------------------------------------------------------------------
// Fly HUD helpers
// ---------------------------------------------------------------------------

const flyHud   = () => document.getElementById('fly-hud');
const flyAlt   = () => document.getElementById('fly-alt');
const flyDist  = () => document.getElementById('fly-dist');
const flySpeed = () => document.getElementById('fly-speed');

// ---------------------------------------------------------------------------
// Velocity panel (always visible, updated in both player + fly modes)
// ---------------------------------------------------------------------------

const vpAlt   = () => document.getElementById('vp-alt');
const vpVert  = () => document.getElementById('vp-vert');
const vpHoriz = () => document.getElementById('vp-horiz');
const vpTotal = () => document.getElementById('vp-total');
const vpRcs   = () => document.getElementById('vp-rcs');
const vpDamp  = () => document.getElementById('vp-damp');

function updateVelPanel({ altitude, vertical, horizontal, total, rcs, rcsDamping, grounded }) {
  vpAlt().textContent   = altitude.toFixed(1);
  vpVert().textContent  = (vertical  >= 0 ? '+' : '') + vertical.toFixed(2);
  vpVert().className    = 'vv ' + (vertical > 0.05 ? 'pos' : vertical < -0.05 ? 'neg' : '');
  vpHoriz().textContent = horizontal.toFixed(2);
  vpTotal().textContent = total.toFixed(2);
  const rcsEl = vpRcs();
  rcsEl.textContent = rcs ? 'ON' : 'OFF';
  rcsEl.className   = 'vv ' + (rcs ? 'rcs-on' : 'rcs-off');
  const dampEl = vpDamp();
  if (dampEl) {
    dampEl.textContent = rcsDamping ? 'ON' : 'OFF';
    dampEl.className   = 'vv ' + (rcsDamping ? 'rcs-on' : 'rcs-off');
  }
}

// Fly HUD now receives true world-space camera→Moon distance so it stays
// accurate even when the fly camera is the floating origin (camera at 0,0,0).
function updateFlyHud(flyCtrl, camWorldX, camWorldY, camWorldZ, worldSim) {
  const dx = camWorldX - worldSim.moonWorldPos.x;
  const dy = camWorldY - worldSim.moonWorldPos.y;
  const dz = camWorldZ - worldSim.moonWorldPos.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  flyAlt().textContent   = (dist - MOON_RADIUS).toFixed(1);
  flyDist().textContent  = dist.toFixed(1);
  flySpeed().textContent = flyCtrl.getSpeed().toFixed(0);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  const startTitle = document.querySelector('#start-screen h1');
  if (startTitle) startTitle.textContent = 'LOADING…';

  // 1. Three.js plumbing.
  const sceneManager = new SceneManager();
  const { scene, camera, renderer } = sceneManager;

  // ── Floating-origin world simulator ──────────────────────────────────────
  const worldSim = new WorldSimulator(MOON_RADIUS);

  // ── Moon body group ────────────────────────────────────────────────────────
  // ALL Moon-surface content (terrain, rocks, future assets) is parented here.
  // Phase 1: position stays at (0,0,0).
  // Phase 3: position = worldSim.getMoonRenderPos() each frame.
  const moonGroup = new THREE.Group();
  moonGroup.name  = 'moonGroup';
  scene.add(moonGroup);

  // 2. NASA datasets.
  const moonData = new MoonData(MOON_RADIUS);
  await moonData.load(renderer);

  if (startTitle) startTitle.textContent = 'MOON WALL KING';

  // 3. Terrain — parented to moonGroup, not directly to scene.
  const terrain = new TerrainSystem(scene, MOON_RADIUS, moonData, moonGroup);

  // 4. Rocks — parented to moonGroup so they orbit with the Moon in Phase 3.
  const rocks = new RockSystem(
    scene, MOON_RADIUS, terrain, null, moonData.detailTexture, moonGroup,
  );

  // 5. Cascaded Shadow Maps — created after terrain + rocks so the scene
  //    already has the shadow-caster objects (rocks on layer 1).
  const csm = new CascadedShadowMap(renderer, scene);

  // Bind CSM depth textures + matrices to terrain shader uniforms.
  // The texture objects and matrix objects are updated in-place each frame
  // by csm.renderShadows(), so the terrain shader sees fresh data automatically.
  {
    const u = terrain._material.uniforms;
    u.uShadowMap0.value    = csm.shadowTextures[0];
    u.uShadowMap1.value    = csm.shadowTextures[1];
    u.uShadowMap2.value    = csm.shadowTextures[2];
    u.uShadowMatrix0.value = csm.shadowMatrices[0];
    u.uShadowMatrix1.value = csm.shadowMatrices[1];
    u.uShadowMatrix2.value = csm.shadowMatrices[2];

    // Bind to rocks too
    const ru = rocks._material.uniforms;
    ru.uShadowMap0.value    = csm.shadowTextures[0];
    ru.uShadowMap1.value    = csm.shadowTextures[1];
    ru.uShadowMap2.value    = csm.shadowTextures[2];
    ru.uShadowMatrix0.value = csm.shadowMatrices[0];
    ru.uShadowMatrix1.value = csm.shadowMatrices[1];
    ru.uShadowMatrix2.value = csm.shadowMatrices[2];
  }

  // 5b. Player spotlight (flashlight)
  const spotlight = new PlayerSpotlightShadow(renderer, scene);
  let spotlightOn = false;
  let spotlightAngle = 0.5;   // radians (~30° half-angle)
  let spotlightRange = 30.0;  // world units

  // 6. Solar system.
  const { sun, earth } = createSolarSystem(scene, MOON_RADIUS);

  // ── Float64 fly-camera world position ─────────────────────────────────────
  // Tracks the fly camera's true heliocentric world position in float64.
  // The fly controller's internal _pos (float32) is reset to (0,0,0) every
  // frame; the delta it produces is accumulated here instead.  This is the
  // "Krakensbane" / floating-origin pattern applied to the camera itself.
  let camWorldX = worldSim.moonWorldPos.x;
  let camWorldY = worldSim.moonWorldPos.y + MOON_RADIUS * 3;   // start above Moon
  let camWorldZ = worldSim.moonWorldPos.z;

  // Sync the fly controller to this initial world position, expressed in
  // Moon-centric render space (Moon is at render origin at this point).
  const _initFlyRender = new THREE.Vector3(0, MOON_RADIUS * 3, 0);

  /**
   * Call whenever the fly controller is activated.
   * Converts the current camera render-space position to a float64 world
   * position and resets the fly controller to render-space origin so
   * _pos stays small every frame.
   */
  function enterFlyMode() {
    // camera.position is in Moon-centric render space at this moment
    // (floatingOrigin = moonWorldPos)
    camWorldX = worldSim.floatingOrigin.x + camera.position.x;
    camWorldY = worldSim.floatingOrigin.y + camera.position.y;
    camWorldZ = worldSim.floatingOrigin.z + camera.position.z;
    // Fly controller keeps its own _pos; reset it so it starts at render origin
    flyCtrl.resetPosition(new THREE.Vector3(0, 0, 0));
  }

  // 6. Controllers.
  const player  = new PlayerController(camera, terrain, MOON_RADIUS, moonGroup);
  const flyCtrl = new DebugFlyController(camera);
  let   mode    = 'player';

  if (DEBUG) {
    mode = 'fly';
    flyCtrl.activate();
    enterFlyMode();
    flyHud().classList.add('active');
    document.body.requestPointerLock?.();
    document.getElementById('start-screen')?.classList.add('hidden');
  }

  // X key — toggle wireframe on terrain + rocks for LOD inspection.
  // L key — toggle player flashlight
  let wireframe = false;
  document.addEventListener('keydown', e => {
    if (e.code === 'KeyX') {
      wireframe = !wireframe;
      terrain.setWireframe(wireframe);
      rocks.setWireframe(wireframe);
      return;
    }
    if (e.code === 'KeyL') {
      spotlightOn = !spotlightOn;
      console.log(`Flashlight: ${spotlightOn ? 'ON' : 'OFF'}`);
      return;
    }
    // Spotlight config: [ / ] for range, - / = for angle
    if (spotlightOn) {
      if (e.code === 'BracketRight') {   // ] increase range
        spotlightRange = Math.min(100, spotlightRange + 5);
        console.log(`Flashlight range: ${spotlightRange.toFixed(1)}`);
        return;
      }
      if (e.code === 'BracketLeft') {    // [ decrease range
        spotlightRange = Math.max(5, spotlightRange - 5);
        console.log(`Flashlight range: ${spotlightRange.toFixed(1)}`);
        return;
      }
      if (e.code === 'Equal') {           // = increase angle
        spotlightAngle = Math.min(1.5, spotlightAngle + 0.1);
        console.log(`Flashlight angle: ${(spotlightAngle * 180 / Math.PI).toFixed(1)}°`);
        return;
      }
      if (e.code === 'Minus') {           // - decrease angle
        spotlightAngle = Math.max(0.1, spotlightAngle - 0.1);
        console.log(`Flashlight angle: ${(spotlightAngle * 180 / Math.PI).toFixed(1)}°`);
        return;
      }
    }
  });

  document.addEventListener('keydown', e => {
    if (e.code !== 'Tab') return;
    e.preventDefault();
    if (mode === 'player') {
      mode = 'fly';
      flyCtrl.activate();
      enterFlyMode();
      flyHud().classList.add('active');
    } else {
      mode = 'player';
      flyCtrl.deactivate();
      flyHud().classList.remove('active');
      // Restore Moon-centric floating origin for player mode
      worldSim.floatingOrigin.x = worldSim.moonWorldPos.x;
      worldSim.floatingOrigin.y = worldSim.moonWorldPos.y;
      worldSim.floatingOrigin.z = worldSim.moonWorldPos.z;
    }
  });

  // 7. Sky + HUD.
  const starfield = new Starfield(scene, camera);
  const ui        = new UI(player);

  // 8. Subsystem registration (order matters).

  // 8a. Active controller + Moon spin.
  //
  //     CRITICAL ORDER: Moon spin must happen BEFORE player.update() so that
  //     the camera is placed using the same moonGroup.quaternion that Three.js
  //     will use when rendering all moonGroup children (terrain, rocks).
  //     If player.update() ran first, the camera would be one Δθ behind the
  //     terrain/rocks at render time → variable-frame-rate jitter (~1.75 units
  //     at r=1000 with a 60-second Moon orbit).
  //
  //     Fly mode uses float64 floating-origin tracking (camera always at render
  //     origin; movement accumulated in camWorldX/Y/Z).
  sceneManager.register({
    update(dt) {
      // Spin first — camera computed below will then match moonGroup at render.
      moonGroup.rotation.y += MOON_SPIN_RATE * dt;

      if (mode === 'player') {
        player.update(dt);
      } else {
        flyCtrl.update(dt);
        const p = flyCtrl.getPosition();
        camWorldX += p.x;
        camWorldY += p.y;
        camWorldZ += p.z;
        flyCtrl.resetPosition(new THREE.Vector3(0, 0, 0));
      }
    },
  });

  // 8b. World simulation — single authoritative orbital update.
  //
  //   Floating origin:
  //     player mode → moonWorldPos   (Moon terrain at render origin)
  //     fly mode    → camWorldPos    (camera at render origin, Moon moves visibly)
  sceneManager.register({
    update(dt) {
      // 1. Advance the heliocentric simulation (float64).
      worldSim.advance(dt);

      // 2. Set floating origin based on active mode.
      if (mode === 'player') {
        worldSim.floatingOrigin.x = worldSim.moonWorldPos.x;
        worldSim.floatingOrigin.y = worldSim.moonWorldPos.y;
        worldSim.floatingOrigin.z = worldSim.moonWorldPos.z;
      } else {
        // Camera is the origin — all other bodies move relative to it.
        worldSim.floatingOrigin.x = camWorldX;
        worldSim.floatingOrigin.y = camWorldY;
        worldSim.floatingOrigin.z = camWorldZ;
      }

      // 3. Moon render position.
      //    Moon spin is now done in 8a so the camera and moonGroup are always
      //    consistent within the same frame at render time.
      worldSim.getMoonRenderPos(moonGroup.position);

      // 4. Earth and Sun positions in render space.
      earth.setPosition(worldSim.getEarthRenderPos());
      earth.updateRotationOnly(dt);
      sun.setPosition(worldSim.getSunRenderPos());
      sun.updateRotationOnly(dt);

      // Feed Earth render position to PlayerController for space gravity.
      // Done after computing earth render pos so it's always fresh.
      if (mode === 'player') player.setEarthRenderPos(earth.getPosition());

      // 5. Lighting direction singletons.
      const camRenderPos = mode === 'player' ? player.getPosition() : camera.position;
      updateSunDirection(sun.getPosition());
      updateEarthDirection(earth.getPosition());

      // 6. Render CSM shadow passes.
      //    Must happen after sun direction is known and before composer.render().
      const sunDirNorm = sun.getPosition().clone().normalize();
      csm.renderShadows(camera, sunDirNorm);

      // 7. Update player spotlight uniforms if flashlight is on.
      if (spotlightOn) {
        const camDir = new THREE.Vector3();
        camera.getWorldDirection(camDir);

        terrain.setSpotlightUniforms(
          true, camera.position, camDir,
          spotlightAngle, spotlightRange,
          null, new THREE.Matrix4()
        );
        rocks.setSpotlightUniforms(
          true, camera.position, camDir,
          spotlightAngle, spotlightRange,
          null, new THREE.Matrix4()
        );
      } else {
        terrain.setSpotlightUniforms(false, camera.position, new THREE.Vector3(), 0, 0, null, new THREE.Matrix4());
        rocks.setSpotlightUniforms(false, camera.position, new THREE.Vector3(), 0, 0, null, new THREE.Matrix4());
      }
    },
  });

  // 8c. Terrain + rocks LOD.
  //     In player mode: camera.position is Moon-local render pos → worldToLocal undoes spin.
  //     In fly mode:    camera.position = (0,0,0) render origin → worldToLocal gives
  //                     the camera's Moon-local position correctly (Moon has moved).
  sceneManager.register({
    update(_dt) {
      const camRenderPos = mode === 'player' ? player.getPosition() : camera.position;
      const moonLocalCam = moonGroup.worldToLocal(camRenderPos.clone());
      terrain.update(moonLocalCam);
      rocks.update(moonLocalCam);
    },
  });

  // 8d. Sky sphere follows camera.
  sceneManager.register({ update(dt) { starfield.update(dt); } });

  // 8e. HUD.
  sceneManager.register({
    update() {
      if (mode === 'player') {
        ui.update();
        updateVelPanel(player.getVelocityInfo());
      } else {
        updateFlyHud(flyCtrl, camWorldX, camWorldY, camWorldZ, worldSim);
        // Show fly camera's distance-from-Moon as altitude in the vel panel
        const dx = camWorldX - worldSim.moonWorldPos.x;
        const dy = camWorldY - worldSim.moonWorldPos.y;
        const dz = camWorldZ - worldSim.moonWorldPos.z;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        updateVelPanel({
          altitude: dist - MOON_RADIUS,
          vertical: 0, horizontal: 0,
          total: flyCtrl.getSpeed(),
          rcs: false, grounded: false,
        });
      }
    },
  });

  // 9. Render loop.
  let running = true;
  (function loop() {
    if (!running) return;
    requestAnimationFrame(loop);
    sceneManager.tick();
  })();

  window.addEventListener('beforeunload', () => {
    running = false;
    terrain.dispose();
    rocks.dispose();
    sun.dispose();
    earth.dispose();
    starfield.dispose();
    sceneManager.dispose();
  });
}

window.addEventListener('DOMContentLoaded', () => { boot(); });
