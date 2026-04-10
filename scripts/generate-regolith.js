/**
 * generate-regolith.js
 *
 * Procedurally generates a 1024×1024 tileable lunar-regolith detail texture
 * saved to public/textures/regolith_detail.png.
 *
 * The texture is a MODULATION MAP centred at 0.5 (= no change):
 *   value < 0.5 → darken the NASA albedo  (crevices between rocks, shadow)
 *   value > 0.5 → brighten slightly       (sunlit rock faces, coarse grain)
 *
 * In the terrain shader it is sampled triplanarly and blended as:
 *   albedo += (detail - 0.5) * strength * closeT
 *
 * Algorithm:
 *   1. Two-octave FBM for large albedo patches (~tile/4 size).
 *   2. Grid-based Worley F1 noise at two rock scales (tile/16, tile/32).
 *      The Worley F1 distance is small near cell boundaries → darkened crease.
 *   3. High-frequency FBM for fine regolith grain (tile/64 features).
 *   All noise functions are perfectly tileable (hash indices wrapped by period).
 *
 * Usage:
 *   node --experimental-vm-modules scripts/generate-regolith.js
 *   (or add "type":"module" to package.json and run with: node scripts/generate-regolith.js)
 */

import sharp from 'sharp';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_PATH   = path.join(__dirname, '..', 'public', 'textures', 'regolith_detail.png');

const W = 1024, H = 1024;

// ---------------------------------------------------------------------------
// Hash helpers — fast, repeatable, no dependencies
// ---------------------------------------------------------------------------

/** Unsigned 16-bit hash of two integers → [0, 1). */
function hash1(x, y) {
  // PCG-ish mixing on 32-bit integers (JavaScript handles bit ops as int32)
  let h = (Math.imul(x | 0, 0x4b51_1117 | 0) ^ Math.imul(y | 0, 0x8e3a_1789 | 0)) | 0;
  h ^= h >>> 14;
  h  = Math.imul(h, 0x9e37_79b9 | 0) | 0;
  h ^= h >>> 16;
  return ((h >>> 0) & 0xFFFF) / 65536.0;
}

/** Two independent hash channels for Worley point jitter. */
function hashJitterX(cx, cy) { return hash1(cx,         cy + 0x1234); }
function hashJitterY(cx, cy) { return hash1(cx + 0xABCD, cy        ); }

// ---------------------------------------------------------------------------
// Tileable value noise
//
// px, py are already in "noise-cell" coordinates.
// period  — number of cells before the noise repeats (must be integer ≥ 1).
// ---------------------------------------------------------------------------

function valueNoise(px, py, period) {
  const ix = Math.floor(px), iy = Math.floor(py);
  const fx = px - ix,        fy = py - iy;

  // Wrap cell indices so the hash table is periodic
  const p = period | 0;
  const x0 = ((ix % p) + p) % p,  y0 = ((iy % p) + p) % p;
  const x1 = (x0 + 1) % p,        y1 = (y0 + 1) % p;

  const a = hash1(x0, y0), b = hash1(x1, y0);
  const c = hash1(x0, y1), d = hash1(x1, y1);

  // Quintic smoothstep
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  return a*(1-ux)*(1-uy) + b*ux*(1-uy) + c*(1-ux)*uy + d*ux*uy;
}

// ---------------------------------------------------------------------------
// Tileable FBM
//
// nx, ny  — normalised texture coordinates [0, 1)
// basePeriod — number of noise cells that span [0,1] at octave 0
// ---------------------------------------------------------------------------

function fbm(nx, ny, octaves, basePeriod) {
  let v = 0, amp = 0.5, period = basePeriod | 0;
  for (let i = 0; i < octaves; i++) {
    v += amp * valueNoise(nx * period, ny * period, period);
    period *= 2;
    amp   *= 0.5;
  }
  return v;  // range ≈ [0, 1] before centering
}

// ---------------------------------------------------------------------------
// Tileable Worley F1 noise (grid-based)
//
// Each grid cell contains one jittered point.  The hash indices are wrapped
// by `period` so the entire noise field tiles perfectly.
//
// Returns the distance to the nearest cell point, scaled so the typical
// maximum distance (≈ half cell diagonal) is ≈ 1.0.
// ---------------------------------------------------------------------------

function worleyF1(nx, ny, period) {
  const sx = nx * period, sy = ny * period;
  const ix = Math.floor(sx), iy = Math.floor(sy);
  const p  = period | 0;

  let minDist = 1e10;

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = ((ix + dx) % p + p) % p;
      const cy = ((iy + dy) % p + p) % p;

      // Point position within the neighbour cell (in [0,1)^2 cell-local coords)
      const ptX = (ix + dx + hashJitterX(cx, cy)) / period;
      const ptY = (iy + dy + hashJitterY(cx, cy)) / period;

      // Distance, accounting for torus wrap
      let ddx = ptX - nx, ddy = ptY - ny;
      if (ddx >  0.5) ddx -= 1.0;
      if (ddx < -0.5) ddx += 1.0;
      if (ddy >  0.5) ddy -= 1.0;
      if (ddy < -0.5) ddy += 1.0;

      const dist = Math.sqrt(ddx * ddx + ddy * ddy);
      if (dist < minDist) minDist = dist;
    }
  }

  // Normalise: maximum F1 for a jittered grid ≈ 0.5/sqrt(density)
  // Multiply by period so "1.0" ≈ half cell-width distance
  return minDist * period;
}

// ---------------------------------------------------------------------------
// Main generation loop
// ---------------------------------------------------------------------------

console.log(`Generating ${W}×${H} regolith detail texture…`);
const t0 = Date.now();

const buf = new Uint8Array(W * H);

for (let py = 0; py < H; py++) {
  for (let px = 0; px < W; px++) {
    const nx = px / W;
    const ny = py / H;

    // ---- Large-scale albedo patches (4 periods across the tile) -------------
    // FBM with 3 octaves: coarse feature size ≈ tile/4.
    const largePatch = fbm(nx, ny, 3, 4) - 0.5;   // ≈ [-0.5, +0.5]

    // ---- Medium rocks (16 cells) ≈ tile/16 per rock ------------------------
    // F1 is small (~0) at cell boundaries (crevices between rocks).
    // crease16 → 1 at the boundary, 0 when 1/6-cell inside a rock.
    const f1_16   = worleyF1(nx, ny, 16);
    const crease16 = Math.max(0.0, 1.0 - f1_16 * 6.0);

    // ---- Small rocks (32 cells) ≈ tile/32 per rock -------------------------
    const f1_32   = worleyF1(nx, ny, 32);
    const crease32 = Math.max(0.0, 1.0 - f1_32 * 8.0);

    // ---- Fine regolith grain (64 periods at octave 0) ----------------------
    const fineGrain = fbm(nx, ny, 4, 32) - 0.5;   // ≈ [-0.5, +0.5]

    // ---- Compose -----------------------------------------------------------
    // Centre at 0.5 (= neutral modulation in the shader).
    // Large patches:  ±0.14   (gentle regional brightness variation)
    // Crevices:      -0.28 / -0.18  (dark crevice shadow at rock edges)
    // Fine grain:    ±0.07   (micro surface texture)
    let val = 0.5
            + largePatch  * 0.14
            - crease16    * 0.28
            - crease32    * 0.18
            + fineGrain   * 0.07;

    buf[py * W + px] = Math.min(255, Math.max(0, Math.round(val * 255))) | 0;
  }

  // Progress indicator every 5%
  if (py % Math.floor(H / 20) === 0) {
    process.stdout.write(`\r  ${Math.round(py / H * 100)}%   `);
  }
}

console.log('\r  100%  \nSaving PNG…');

await sharp(Buffer.from(buf.buffer), {
  raw: { width: W, height: H, channels: 1 },
})
  .png({ compressionLevel: 8 })
  .toFile(OUT_PATH);

console.log(`Saved → ${OUT_PATH}   (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
