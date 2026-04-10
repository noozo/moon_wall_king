// terrain.vert
// Displacement is baked into geometry on the CPU side (TerrainSystem.js).
// This shader is a clean pass-through — just transforms vertices and
// forwards the data needed by the fragment stage.
//
// Normal transform:
//   Terrain geometry is baked in Moon-local space (relative to moonGroup).
//   moonGroup may carry a rotation (Phase 2 Moon spin).
//   mat3(modelMatrix) extracts the rotation part of the full model matrix,
//   correctly transforming Moon-local normals into world space without
//   introducing any view-space dependency.  This avoids the camera-rotation
//   lighting bug that would occur if we used normalMatrix (which is view-space).
//
// No shadow coord here — the fragment shader computes it directly from
// vWorldPos using uShadowMatrix, avoiding an extra varying.

varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec3 vLocalPos;   // Moon-local position — stable under moonGroup rotation
varying vec2 vUv;

void main() {
  vNormal   = normalize(mat3(modelMatrix) * normal);
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  vLocalPos = position;
  vUv       = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
