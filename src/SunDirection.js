/**
 * SunDirection / EarthDirection — mutable singleton vectors shared by all
 * shaders and systems that need to know the current Sun or Earth position.
 *
 * Every shader that references lighting direction holds a reference to one
 * of these Vector3 objects (not a copy), so mutating them here is instantly
 * visible to all uniforms without needing to reach into individual materials.
 *
 * updateSunDirection   — call once per frame with Sun world position + camera position
 * updateEarthDirection — call once per frame with Earth world position
 */

import * as THREE from 'three';

/** Unit vector pointing from Moon surface toward the Sun (world space). */
export const sunDirection = new THREE.Vector3(1, 0.4, 0.6).normalize();

/** Unit vector pointing from Moon origin toward Earth (world space). */
export const earthDirection = new THREE.Vector3(0, 1, 0);

/**
 * Recompute sunDirection so it points from camPos toward sunPos.
 * @param {THREE.Vector3} sunPos
 * @param {THREE.Vector3} camPos
 */
export function updateSunDirection(sunPos) {
  sunDirection.copy(sunPos).normalize();
}

/**
 * Recompute earthDirection so it points from the Moon (origin) toward Earth.
 * Since the Moon terrain is at world origin, this is just the normalised
 * Earth position.
 * @param {THREE.Vector3} earthPos
 */
export function updateEarthDirection(earthPos) {
  earthDirection.copy(earthPos).normalize();
}
