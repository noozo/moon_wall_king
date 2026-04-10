/**
 * SimConfig — shared simulation time scale.
 *
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
 *   Orbital period (real s) = realDays × 86400 × SIM_TIME_SCALE
 *                           = realDays × 86400 × MOON_ORBIT_REAL_S / (27.32 × 86400)
 *                           = realDays / 27.32 × MOON_ORBIT_REAL_S
 *
 * Note: a SMALLER denominator in 1/x means a LARGER fraction and SLOWER orbit.
 *       Use this constant instead of writing 1/x directly.
 */

const MOON_ORBIT_REAL_S = 240;   // ← change this to tune simulation speed

export const SIM_TIME_SCALE = MOON_ORBIT_REAL_S / (27.32 * 86400);
