/**
 * SimConfig — shared simulation constants.
 *
 * All values that appear in more than one file live here so there is a single
 * source of truth.  Derived values (e.g. CAM_FAR relative to orbital radii)
 * are computed below rather than hardcoded independently in each consumer.
 */

// ---------------------------------------------------------------------------
// World scale
// ---------------------------------------------------------------------------

/** Moon sphere radius in game units. Drives all orbital distances. */
export const MOON_RADIUS = 1000;

// ---------------------------------------------------------------------------
// Camera clip planes
// ---------------------------------------------------------------------------

/**
 * Near/far clip distances (game units).
 * CAM_FAR must comfortably exceed the Sun–Moon distance at its furthest
 * (Sun–Earth ≈ MOON_RADIUS×200, Earth–Moon ≈ MOON_RADIUS×25 → ≈ 225 000).
 * 350 000 gives ~56% margin.
 * Used in: SceneManager (camera), TerrainSystem (uCamNear/uCamFar), RockSystem.
 */
export const CAM_NEAR =    0.5;
export const CAM_FAR  = 350000;

// ---------------------------------------------------------------------------
// Cascaded Shadow Map splits
// ---------------------------------------------------------------------------

/**
 * View-space depth (game units) at which each shadow cascade ends.
 *   Cascade 0: [camera.near, CASCADE_SPLITS[0]]
 *   Cascade 1: [CASCADE_SPLITS[0], CASCADE_SPLITS[1]]
 *   Cascade 2: [CASCADE_SPLITS[1], CASCADE_SPLITS[2]]
 *
 * Must stay in sync with the fragment shaders (terrain.frag, RockSystem inline)
 * which read these values from the uCascadeSplits uniform.
 */
export const CASCADE_SPLITS = [20, 200, 2000];

// ---------------------------------------------------------------------------
// Simulation time scale
// ---------------------------------------------------------------------------

/**
 * Set MOON_ORBIT_REAL_S to the desired real-world duration (seconds) of one
 * complete Moon–Earth orbit.  Everything else derives from that.
 *
 *  MOON_ORBIT_REAL_S │  Moon orbit  │  Sun day/night  │  Earth year
 *  ──────────────────┼──────────────┼─────────────────┼─────────────
 *           10       │   10 s       │    10.8 s       │   2.2 min
 *           30       │   30 s       │    32 s         │   6.7 min
 *           60       │    1 min     │    65 s         │  13.4 min
 *
 * Formula:
 *   SIM_TIME_SCALE = MOON_ORBIT_REAL_S / (27.32 * 86400)
 */
const MOON_ORBIT_REAL_S = 240;   // ← change this to tune simulation speed

export const SIM_TIME_SCALE = MOON_ORBIT_REAL_S / (27.32 * 86400);
