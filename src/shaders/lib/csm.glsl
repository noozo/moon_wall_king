// Cascaded Shadow Map sampling — 3 cascades, 16-tap Poisson PCF.
// Included by terrain.frag and RockSystem inline fragment shader.
//
// Requires these uniforms in the enclosing shader:
//   sampler2D uShadowMap0, uShadowMap1, uShadowMap2
//   mat4      uShadowMatrix0, uShadowMatrix1, uShadowMatrix2
//   vec3      uCascadeSplits   (view-space end depths: x=cascade0, y=cascade1, z=cascade2)
//   float     uCamNear, uCamFar

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

// Returns shadow factor in [0.15, 1.0]. 1.0 = fully lit.
// worldPos — fragment world-space position projected into shadow space.
// diff     — NdotL (sun), used to scale the depth bias.
float sampleCSMShadow(vec3 worldPos, float diff) {
  // Linearise gl_FragCoord.z to view-space depth for cascade selection.
  float z_ndc     = gl_FragCoord.z * 2.0 - 1.0;
  float viewDepth = (2.0 * uCamNear * uCamFar)
                  / (uCamFar + uCamNear - z_ndc * (uCamFar - uCamNear));

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

  vec4 shadowUVW = shadowMatrix * vec4(worldPos, 1.0);
  vec2 shadowUV  = shadowUVW.xy;

  // Fragment is outside all shadow map bounds — no shadow data, assume lit.
  if (shadowUV.x <= 0.001 || shadowUV.x >= 0.999 ||
      shadowUV.y <= 0.001 || shadowUV.y >= 0.999) {
    return 1.0;
  }

  float currentDepth = shadowUVW.z;
  float bias = max(0.0008 * (1.0 - diff), 0.0002);

  // 16-tap Poisson disk with per-fragment random rotation to break banding.
  float phi    = fract(sin(dot(shadowUV, vec2(127.1, 311.7))) * 43758.5453) * 6.28318;
  float cosPhi = cos(phi);
  float sinPhi = sin(phi);
  float spread = 5.0 / 1024.0;

  float sum = 0.0;
  for (int i = 0; i < 16; i++) {
    vec2 rotated = vec2(
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
  return mix(0.15, 1.0, sum / 16.0);
}
