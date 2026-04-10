/**
 * Simplex noise implementation + terrain height helpers.
 *
 * Key design decisions:
 *  - Module-level singleton avoids re-building the permutation table on every
 *    terrain-vertex evaluation (was being reconstructed per-call before).
 *  - Terrain is sampled in 3-D Cartesian sphere-space rather than lat/lon to
 *    avoid polar distortion and the ±π seam on longitude.
 */

// ---------------------------------------------------------------------------
// SimplexNoise
// ---------------------------------------------------------------------------

export class SimplexNoise {
  constructor(seed = 1337) {
    this._p = new Uint8Array(256);
    this._perm = new Uint8Array(512);
    this._permMod12 = new Uint8Array(512);

    // Seed-deterministic Fisher-Yates shuffle.
    for (let i = 0; i < 256; i++) this._p[i] = i;
    let s = (seed ^ (seed >>> 17)) | 1; // LCG-like seed
    for (let i = 255; i > 0; i--) {
      s = Math.imul(s, 1664525) + 1013904223 | 0;
      const j = (s >>> 0) % (i + 1);
      const tmp = this._p[i]; this._p[i] = this._p[j]; this._p[j] = tmp;
    }
    for (let i = 0; i < 512; i++) {
      this._perm[i] = this._p[i & 255];
      this._permMod12[i] = this._perm[i] % 12;
    }
  }

  // Gradient table
  static _GRAD3 = [
    [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
    [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
    [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
  ];

  _dot3(g, x, y, z) { return g[0]*x + g[1]*y + g[2]*z; }

  noise3D(xin, yin, zin) {
    const F3 = 1/3, G3 = 1/6;
    const s  = (xin + yin + zin) * F3;
    const i  = Math.floor(xin + s);
    const j  = Math.floor(yin + s);
    const k  = Math.floor(zin + s);
    const t  = (i + j + k) * G3;

    const x0 = xin - (i - t), y0 = yin - (j - t), z0 = zin - (k - t);

    let i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
      if (y0 >= z0)      { i1=1;j1=0;k1=0; i2=1;j2=1;k2=0; }
      else if (x0 >= z0) { i1=1;j1=0;k1=0; i2=1;j2=0;k2=1; }
      else               { i1=0;j1=0;k1=1; i2=1;j2=0;k2=1; }
    } else {
      if (y0 < z0)       { i1=0;j1=0;k1=1; i2=0;j2=1;k2=1; }
      else if (x0 < z0)  { i1=0;j1=1;k1=0; i2=0;j2=1;k2=1; }
      else               { i1=0;j1=1;k1=0; i2=1;j2=1;k2=0; }
    }

    const x1 = x0-i1+G3, y1 = y0-j1+G3, z1 = z0-k1+G3;
    const x2 = x0-i2+2*G3, y2 = y0-j2+2*G3, z2 = z0-k2+2*G3;
    const x3 = x0-1+3*G3, y3 = y0-1+3*G3, z3 = z0-1+3*G3;

    const ii = i&255, jj = j&255, kk = k&255;
    const g3 = SimplexNoise._GRAD3;

    const gi0 = this._permMod12[ii    + this._perm[jj    + this._perm[kk   ]]];
    const gi1 = this._permMod12[ii+i1 + this._perm[jj+j1 + this._perm[kk+k1]]];
    const gi2 = this._permMod12[ii+i2 + this._perm[jj+j2 + this._perm[kk+k2]]];
    const gi3 = this._permMod12[ii+1  + this._perm[jj+1  + this._perm[kk+1 ]]];

    let n0=0, n1=0, n2=0, n3=0;
    let t0 = 0.6-x0*x0-y0*y0-z0*z0; if (t0>0) { t0*=t0; n0=t0*t0*this._dot3(g3[gi0],x0,y0,z0); }
    let t1 = 0.6-x1*x1-y1*y1-z1*z1; if (t1>0) { t1*=t1; n1=t1*t1*this._dot3(g3[gi1],x1,y1,z1); }
    let t2 = 0.6-x2*x2-y2*y2-z2*z2; if (t2>0) { t2*=t2; n2=t2*t2*this._dot3(g3[gi2],x2,y2,z2); }
    let t3 = 0.6-x3*x3-y3*y3-z3*z3; if (t3>0) { t3*=t3; n3=t3*t3*this._dot3(g3[gi3],x3,y3,z3); }

    return 32*(n0+n1+n2+n3); // [-1, 1]
  }

  /**
   * Fractal Brownian Motion — returns value in roughly [-1, 1].
   */
  fbm(x, y, z, octaves = 4, persistence = 0.5, lacunarity = 2.0) {
    let total = 0, amplitude = 1, frequency = 1, maxValue = 0;
    for (let i = 0; i < octaves; i++) {
      total    += this.noise3D(x*frequency, y*frequency, z*frequency) * amplitude;
      maxValue += amplitude;
      amplitude  *= persistence;
      frequency  *= lacunarity;
    }
    return total / maxValue;
  }
}

// ---------------------------------------------------------------------------
// Singleton used by TerrainSystem + PlayerController
// ---------------------------------------------------------------------------

/** Shared noise instance — created once per module load. */
const _terrainNoise = new SimplexNoise(42);

/**
 * Returns the terrain displacement height (metres above the sphere surface)
 * for a point identified by its unit-sphere 3-D coordinates.
 *
 * Using 3-D Cartesian sphere-space instead of lat/lon avoids:
 *  - Polar distortion
 *  - The ±π seam on longitude
 *
 * @param {number} lat  Latitude in radians [-π/2, π/2]
 * @param {number} lon  Longitude in radians [-π, π]
 * @param {number} amplitude  Maximum displacement in world units (default 15)
 * @returns {number}
 */
export function sampleTerrainHeight(lat, lon, amplitude = 15) {
  // Convert to unit-sphere Cartesian
  const nx = Math.cos(lat) * Math.cos(lon);
  const ny = Math.sin(lat);
  const nz = Math.cos(lat) * Math.sin(lon);

  // Large-scale terrain (hills, gentle rolling plains)
  const base = _terrainNoise.fbm(nx * 2.5, ny * 2.5, nz * 2.5, 4, 0.5, 2.0);

  // Fine detail (small rocks, regolith texture)
  const detail = _terrainNoise.fbm(nx * 8, ny * 8, nz * 8, 2, 0.5, 2.0) * 0.15;

  return (base + detail) * amplitude;
}
