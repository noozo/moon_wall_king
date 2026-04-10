// terrain.frag
//
// Close-up quality strategy:
//
//   1. Analytical-gradient FBM bump normals
//   2. Triplanar detail texture (uDetailMap)
//   3. 70% desaturation of NASA LROC albedo
//
// Lighting: Lambertian diffuse + earthshine ambient.
// Shadows:  Cascaded Shadow Maps (CSM) — 3 cascades.
//   Cascade selection by linearised gl_FragCoord.z depth.
//   Poisson-disk PCF with per-fragment random rotation.

#define PI 3.14159265358979

varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec3 vLocalPos;
varying vec2 vUv;

uniform vec3      uSunDirection;
uniform vec3      uEarthDirection;
uniform sampler2D uColorMapLOD0;
uniform sampler2D uColorMapLOD1;
uniform float     uLOD1Distance;
uniform vec3      uCameraPos;
uniform sampler2D uDetailMap;
uniform float     uDetailTiling;

// ── CSM uniforms ─────────────────────────────────────────────────────────────
uniform sampler2D uShadowMap0;
uniform sampler2D uShadowMap1;
uniform sampler2D uShadowMap2;
uniform mat4      uShadowMatrix0;
uniform mat4      uShadowMatrix1;
uniform mat4      uShadowMatrix2;
uniform vec3      uCascadeSplits;   // x=end0, y=end1, z=end2 (view-space depth)
uniform float     uCamNear;
uniform float     uCamFar;
uniform float     uWireframe;

// Player spotlight uniforms
uniform float     uSpotlightOn;
uniform vec3      uSpotlightPos;
uniform vec3      uSpotlightDir;
uniform float     uSpotlightAngle;
uniform float     uSpotlightRange;
uniform sampler2D uSpotlightShadowMap;
uniform mat4      uSpotlightMatrix;

// 16-tap Poisson disk (used by both CSM and spotlight shadows)
const vec2 POISSON_DISK[16] = vec2[16](
  vec2(-0.94201624, -0.39906216),
  vec2( 0.94558609, -0.76890725),
  vec2(-0.09418410, -0.92938870),
  vec2( 0.34495938,  0.29387760),
  vec2(-0.91588581,  0.45771432),
  vec2(-0.81544232, -0.87912464),
  vec2(-0.38277543,  0.27676845),
  vec2( 0.97484398,  0.75648379),
  vec2( 0.44323325, -0.97511554),
  vec2( 0.53742981, -0.47373420),
  vec2(-0.26496911, -0.41893023),
  vec2( 0.79197514,  0.19090188),
  vec2(-0.24188840,  0.99706507),
  vec2(-0.81409955,  0.91437590),
  vec2( 0.19984126,  0.78641367),
  vec2( 0.14383161, -0.14100790)
);

// ---------------------------------------------------------------------------
// Hash (fast, uniform-ish distribution over [0,1))
// ---------------------------------------------------------------------------

float hash(vec2 p) {
  p  = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

// ---------------------------------------------------------------------------
// Analytical-gradient value noise
// Returns vec3(dv/dp.x, dv/dp.y, value)
// ---------------------------------------------------------------------------

vec3 noiseGV(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);

  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));

  vec2 u  = f * f * (3.0 - 2.0 * f);
  vec2 du = 6.0 * f * (1.0 - f);

  float val = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  float gx  = du.x * ((b - a) * (1.0 - u.y) + (d - c) * u.y);
  float gy  = du.y * ((c - a) * (1.0 - u.x) + (d - b) * u.x);

  return vec3(gx, gy, val);
}

// ---------------------------------------------------------------------------
// FBM with accumulated analytical gradient
// 30° rotation per octave breaks axis alignment, suppresses grid artefacts.
// Returns vec3(gradX, gradY, value)
// ---------------------------------------------------------------------------

const mat2 FBM_ROT = mat2(0.86602540378, 0.5, -0.5, 0.86602540378);

vec3 fbmGV4(vec2 p) {
  float val  = 0.0;
  vec2  grad = vec2(0.0);
  float amp  = 0.5;
  vec3  n;
  n = noiseGV(p); val += amp * n.z; grad += amp * n.xy; p = FBM_ROT * p * 2.17; amp *= 0.5;
  n = noiseGV(p); val += amp * n.z; grad += amp * n.xy; p = FBM_ROT * p * 2.17; amp *= 0.5;
  n = noiseGV(p); val += amp * n.z; grad += amp * n.xy; p = FBM_ROT * p * 2.17; amp *= 0.5;
  n = noiseGV(p); val += amp * n.z; grad += amp * n.xy;
  return vec3(grad, val);
}

// ---------------------------------------------------------------------------
// Triplanar detail-texture sampler (greyscale modulation map)
//
// Three axis-aligned UV planes blended by pow(|N|, 3) — sharpens the blend
// transition without creating visible hard edges at projection seams.
// ---------------------------------------------------------------------------

float sampleDetailMono(vec3 pos, vec3 absNormal, float tiling) {
  vec3 w = pow(absNormal, vec3(3.0));
  w /= (w.x + w.y + w.z + 1e-5);
  float sX = texture2D(uDetailMap, pos.yz * tiling).r;
  float sY = texture2D(uDetailMap, pos.xz * tiling).r;
  float sZ = texture2D(uDetailMap, pos.xy * tiling).r;
  return sX * w.x + sY * w.y + sZ * w.z;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
  vec3 N = normalize(vNormal);   // world-space shading normal

  // Use Moon-local position for UV and noise so the texture/detail pattern
  // stays anchored to the terrain geography and never drifts as moonGroup
  // rotates.  vLocalPos == vWorldPos in Phase 1 (no rotation) — no change.
  vec3 d = normalize(vLocalPos);

  // Equirectangular UV from Moon-local sphere direction.
  float u_coord = atan(d.z, d.x) / (2.0 * PI) + 0.5;
  float v_coord = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;

  // Fix the ±180° equirectangular seam.
  //
  // atan(d.z, d.x) jumps from +π to -π as longitude crosses 180°E, making
  // u_coord jump from ≈1.0 to ≈0.0 within a single terrain triangle.
  // The GPU uses screen-space UV derivatives to select the mip level; with a
  // gradient of ≈1.0, it picks the coarsest mip → blurry stripe at 180°E.
  //
  // Fix: if the screen-space derivative in EITHER direction exceeds 0.3
  // (covers both vertical-seam and horizontal-seam screen orientations)
  // AND u_coord < 0.5 (this fragment wrapped from ≈1 down to ≈0),
  // shift u_coord up by 1.  The texture uses RepeatWrapping so u > 1 samples
  // the same texel as u-1, and the gradient is now correct for mip selection.
  float u_du = dFdx(u_coord);
  float u_dv = dFdy(u_coord);
  if ((abs(u_du) > 0.3 || abs(u_dv) > 0.3) && u_coord < 0.5) u_coord += 1.0;

  vec2  uv      = vec2(u_coord, v_coord);

  // World-space distance to camera (for LOD blend).
  float dist    = length(vWorldPos - uCameraPos);
  float closeT  = 1.0 - smoothstep(0.0, uLOD1Distance, dist);
  float closeT2 = closeT * closeT;

  // ---------------------------------------------------------------------------
  // Base albedo from NASA LROC textures (no sharpening — plain bilinear)
  //
  // The 8K texture is blended in at close range.  USM sharpening was removed
  // because it amplified the subtle colour tints in the raw data, making the
  // surface look unnaturally contrasty and blue/brown at close range.
  //
  // Partial desaturation (70% towards luminance grey) neutralises regional
  // colour casts while preserving the real brightness variation from the data.
  // ---------------------------------------------------------------------------

  vec3 colorFar   = texture2D(uColorMapLOD0, uv).rgb;
  vec3 colorClose = texture2D(uColorMapLOD1, uv).rgb;
  vec3 albedo     = mix(colorFar, colorClose, closeT);

  // Desaturate 70% → kills blue/brown tints, keeps luminance structure
  float lum = dot(albedo, vec3(0.299, 0.587, 0.114));
  albedo = mix(albedo, vec3(lum), 0.70);

  // ---------------------------------------------------------------------------
  // Analytical bump normals (shape detail, not colour)
  // ---------------------------------------------------------------------------

  vec3 bN = N;

  if (closeT > 0.001) {
    // Tangent basis from world-space N so the bump perturbation is applied in
    // world space.  (Using `d` here would give Moon-local tangent vectors,
    // which would be in a different space from N when moonGroup is rotating.)
    vec3 east;
    if (abs(N.y) < 0.999) {
      east = normalize(cross(vec3(0.0, 1.0, 0.0), N));
    } else {
      east = normalize(cross(vec3(1.0, 0.0, 0.0), N));
    }
    vec3 north = normalize(cross(N, east));

    vec3 b1 = fbmGV4(d.xz *   4.0);   // ~1110/4   ≈ 280 m  — rolling undulations
    vec3 b2 = fbmGV4(d.xz *  50.0);   // ~1110/50  ≈  22 m  — medium rocky bumps
    vec3 b3 = fbmGV4(d.xz * 300.0);   // ~1110/300 ≈   4 m  — fine surface texture

    // Combine gradients with distance-dependent strength.
    // b1: visible from moderate range, gentle rolling character.
    // b2: medium bumps, fade with closeT2 (quadratic).
    // b3: smallest scale — only active when very close (cubic closeT3 fade).
    float closeT3 = closeT2 * closeT;
    vec2 bumpGrad = b1.xy * (0.35 * closeT)
                  + b2.xy * (0.15 * closeT2)
                  + b3.xy * (0.07 * closeT3);
    bN = normalize(N - east * bumpGrad.x - north * bumpGrad.y);
  }

  // ---------------------------------------------------------------------------
  // Triplanar detail texture
  //
  // Additive modulation centred at 0.5 — strength 0.20 gives ±0.10 swing,
  // enough to show rock/crevice structure without deep blacks or blown whites.
  // Two tiling scales are mixed 60/40 (coarse/fine) for layered detail.
  // ---------------------------------------------------------------------------

  if (closeT2 > 0.001) {
    // Use Moon-local position so the triplanar UVs are stable under moonGroup
    // rotation — same fix as for the FBM noise above.
    float detailCoarse = sampleDetailMono(vLocalPos, abs(bN), uDetailTiling);
    float detailFine   = sampleDetailMono(vLocalPos, abs(bN), uDetailTiling * 4.0);
    float detail       = mix(detailCoarse, detailFine, 0.4);
    albedo = clamp(albedo + (detail - 0.5) * 0.20 * closeT2, 0.0, 1.0);
  }

  // ---------------------------------------------------------------------------
  // Lighting — Lambertian diffuse + earthshine ambient
  //
  // Earthshine: Earth reflects ~37% of sunlight back onto the Moon night side.
  // Apparent earthshine illuminance ≈ 1.5% of direct sunlight, with a faint
  // blue tint (Earth's blue-ocean/white-cloud albedo).
  //
  // A small constant ambient (0.02) accounts for interplanetary scatter
  // and keeps deep-shadow surfaces from clipping to pure black.
  // ---------------------------------------------------------------------------

  vec3  sun  = normalize(uSunDirection);
  vec3  eDir = normalize(uEarthDirection);

  float diff = max(dot(bN, sun), 0.0);

  // ── Cascaded Shadow Maps ───────────────────────────────────────────────────
  // 1. Linearise gl_FragCoord.z to view-space depth (world units from camera).
  // 2. Select cascade based on that depth.
  // 3. Transform fragment world position into shadow UV space using the
  //    cascade's precomputed bias matrix.
  // 4. 16-tap Poisson-disk PCF, disk rotated randomly per fragment to break
  //    the fixed-pattern banding that would otherwise tile across the terrain.
  // ──────────────────────────────────────────────────────────────────────────
  float z_ndc     = gl_FragCoord.z * 2.0 - 1.0;
  float viewDepth = (2.0 * uCamNear * uCamFar)
                  / (uCamFar + uCamNear - z_ndc * (uCamFar - uCamNear));

  // Select cascade and shadow matrix
  mat4 shadowMatrix;
  int  cascade;
  if (viewDepth < uCascadeSplits.x) {
    cascade      = 0;
    shadowMatrix = uShadowMatrix0;
  } else if (viewDepth < uCascadeSplits.y) {
    cascade      = 1;
    shadowMatrix = uShadowMatrix1;
  } else {
    cascade      = 2;
    shadowMatrix = uShadowMatrix2;
  }

  vec4 shadowUVW = shadowMatrix * vec4(vWorldPos, 1.0);
  vec2 shadowUV  = shadowUVW.xy;

  float shadow = 1.0;
  if (shadowUV.x > 0.001 && shadowUV.x < 0.999 &&
      shadowUV.y > 0.001 && shadowUV.y < 0.999) {

    float currentDepth = shadowUVW.z;
    float bias = max(0.0008 * (1.0 - diff), 0.0002);

    // Per-fragment random rotation — breaks repeating banding pattern
    float phi    = fract(sin(dot(shadowUV, vec2(127.1, 311.7))) * 43758.5453) * 6.28318;
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float spread = 5.0 / 1024.0;

    float sum = 0.0;
    for (int i = 0; i < 16; i++) {
      vec2 rotated  = vec2(
        POISSON_DISK[i].x * cosPhi - POISSON_DISK[i].y * sinPhi,
        POISSON_DISK[i].x * sinPhi + POISSON_DISK[i].y * cosPhi
      );
      vec2 sampleUV = clamp(shadowUV + rotated * spread, 0.001, 0.999);

      float d;
      if      (cascade == 0) d = texture2D(uShadowMap0, sampleUV).r;
      else if (cascade == 1) d = texture2D(uShadowMap1, sampleUV).r;
      else                   d = texture2D(uShadowMap2, sampleUV).r;

      sum += (currentDepth - bias > d) ? 0.0 : 1.0;
    }
    shadow = mix(0.15, 1.0, sum / 16.0);
  }

  float earthFace  = max(dot(bN, eDir), 0.0);

  // Earthshine: blue-tinted, visible on night side
  vec3 earthshine = vec3(0.45, 0.65, 1.0) * earthFace * 0.018;

  float ambient = 0.02;

  // --- Player spotlight ---------------------------------------------------
  // Light comes FROM the camera — no shadow map (co-located light/camera
  // always produces near-geometry occlusion artifacts). The cone attenuation
  // and distance falloff give the visual flashlight effect.
  float spotLight = 0.0;
  if (uSpotlightOn > 0.5) {
    vec3  toFrag   = vWorldPos - uSpotlightPos;
    float dist     = length(toFrag);
    vec3  fragDir  = toFrag / dist;
    vec3  L        = -fragDir;

    float cosAngle = dot(fragDir, uSpotlightDir);
    float spotCos  = cos(uSpotlightAngle);

    if (cosAngle > spotCos && dist < uSpotlightRange) {
      float spotAtten = smoothstep(spotCos, spotCos + 0.05, cosAngle);
      float distAtten = 1.0 - smoothstep(0.0, uSpotlightRange, dist);
      spotLight = max(dot(bN, L), 0.0) * spotAtten * distAtten * 3.0;
    }
  }

  vec3 totalLight = vec3(diff * shadow + ambient) + vec3(spotLight);
  gl_FragColor = vec4(albedo * totalLight + albedo * earthshine, 1.0);

  // Wireframe diagnostic mode (X key): override output with flat green.
  // Bypasses all lighting so the mesh is fully visible even on the night side.
  if (uWireframe > 0.5) gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}
