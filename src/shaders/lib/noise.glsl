// Analytical-gradient 2D value noise and 4-octave FBM.
// Included by terrain.frag and RockSystem inline fragment shader.

float hash(vec2 p) {
  p  = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

// Returns vec3(dv/dp.x, dv/dp.y, value)
vec3 noiseGV(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash(i),                  b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
  vec2 u  = f * f * (3.0 - 2.0 * f);
  vec2 du = 6.0 * f * (1.0 - f);
  float val = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  float gx  = du.x * ((b - a) * (1.0 - u.y) + (d - c) * u.y);
  float gy  = du.y * ((c - a) * (1.0 - u.x) + (d - b) * u.x);
  return vec3(gx, gy, val);
}

// 30° rotation per octave — breaks axis alignment, suppresses grid artefacts.
const mat2 FBM_ROT = mat2(0.86602540378, 0.5, -0.5, 0.86602540378);

// 4-octave FBM with accumulated analytical gradient.
// Returns vec3(gradX, gradY, value).
vec3 fbmGV4(vec2 p) {
  float val = 0.0; vec2 grad = vec2(0.0); float amp = 0.5;
  vec3 n;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = FBM_ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = FBM_ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy; p = FBM_ROT*p*2.17; amp *= 0.5;
  n = noiseGV(p); val += amp*n.z; grad += amp*n.xy;
  return vec3(grad, val);
}
