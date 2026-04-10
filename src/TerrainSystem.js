/**
 * TerrainSystem — Cube-sphere quadtree LOD terrain for a planet.
 *
 * Algorithm (inspired by PlanetaryTerrain & the referenced paper):
 *
 *  1. Represent the sphere as a "spherified cube": 6 axis-aligned cube faces.
 *  2. Each face is a quad-tree root.  A QuadNode stores:
 *       trPos     – cube-space centre of the node (e.g. (0,1,0) for top root)
 *       rotation  – quaternion that rotates a flat upward-facing grid onto the face
 *       scale     – edge length in cube-space (1.0 for root, halved per split)
 *       level     – tree depth (0 = root)
 *  3. Each frame, a recursive traversal decides to split or combine nodes
 *     based on the 3-D Euclidean distance from the camera to the node's
 *     sphere-surface centre.
 *  4. Leaf nodes own a Three.js Mesh.  Parent meshes are kept visible until
 *     every child is built, then swapped out (preventing holes/flickering).
 *  5. Geometry is built by:
 *       a. Grid vertex in flat plane:    (u, 0, v)
 *       b. Rotate to cube face:          applyQuaternion(rotation)
 *       c. Offset to cube face centre:   += trPos
 *       d. Normalise → projects to unit sphere (cube → sphere)
 *       e. Query height at that 3-D direction (3-D noise, no pole distortion)
 *       f. Scale:  position = dir * (moonRadius + height)
 *  6. Normals are recomputed via computeVertexNormals() for smooth shading.
 *  7. Mesh generation is throttled to MAX_BUILDS_PER_FRAME to prevent hitches.
 */

import * as THREE from 'three';
import terrainVert from './shaders/terrain.vert?raw';
import terrainFrag from './shaders/terrain.frag?raw';
import { sunDirection, earthDirection } from './SunDirection.js';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Per-level split distance (world units).  A leaf node at level N splits
 * into 4 children when the camera is closer than LOD_SPLIT_DIST[N].
 */
const LOD_SPLIT_DIST = [
  1400, 620, 350, 180, 92, 46, 23, 12, 6,
];

/** Combine when camera moves 40% beyond the split distance (hysteresis). */
const LOD_COMBINE_FACTOR = 1.4;

/** Max depth of the quadtree. */
const MAX_LEVEL = LOD_SPLIT_DIST.length; // 9

/** Segments per quad side. */
const QUAD_SEGMENTS = 32;

/** Maximum number of new geometries built synchronously per frame. */
const MAX_BUILDS_PER_FRAME = 2;

// ---------------------------------------------------------------------------
// Face definitions for the spherified cube
// ---------------------------------------------------------------------------

/**
 * Pre-computed quaternions for the 6 cube faces.
 * Each rotation transforms an upward-facing flat grid (vertices at (u,0,v),
 * spanning u,v ∈ [-scale, scale]) onto the correct cube face.
 *
 * Winding-order derivation (cross-product check):
 *   For every face, the triangle (a, b=row+1, a+1=col+1) must produce a normal
 *   pointing OUTWARD from the sphere centre.  The rotation must map the base
 *   "up" vector (0,1,0) to the face's outward direction.
 *
 *   +Y : identity          → base-up (0,1,0) maps to +Y ✓
 *   -Y : Euler(+180°,0,0)  → (0,1,0) → (0,-1,0) ✓
 *   +Z : Euler(+90°,0,0)   → (0,1,0) → (0,0,+1) ✓  (was -90°, gave wrong winding)
 *   -Z : Euler(-90°,0,0)   → (0,1,0) → (0,0,-1) ✓  (was +90°)
 *   +X : Euler(0,0,-90°)   → (0,1,0) → (+1,0,0) ✓  (was +90°)
 *   -X : Euler(0,0,+90°)   → (0,1,0) → (-1,0,0) ✓  (was -90°)
 */
const FACE_DEFS = (() => {
  const deg = THREE.MathUtils.degToRad;
  return [
    //  name     trPos              Euler(XYZ)         base-up → outward
    { name: '+Y', trPos: [0,  1,  0], euler: [        0, 0,       0] },  // (0,1,0)→(0,+1,0)
    { name: '-Y', trPos: [0, -1,  0], euler: [ deg(180), 0,       0] },  // (0,1,0)→(0,-1,0)
    { name: '+Z', trPos: [0,  0,  1], euler: [  deg(90), 0,       0] },  // (0,1,0)→(0,0,+1)
    { name: '-Z', trPos: [0,  0, -1], euler: [ deg(-90), 0,       0] },  // (0,1,0)→(0,0,-1)
    { name: '+X', trPos: [ 1, 0,  0], euler: [        0, 0, deg(-90)] }, // (0,1,0)→(+1,0,0)
    { name: '-X', trPos: [-1, 0,  0], euler: [        0, 0,  deg(90)] }, // (0,1,0)→(-1,0,0)
  ].map(def => ({
    name:     def.name,
    trPos:    new THREE.Vector3(...def.trPos),
    rotation: new THREE.Quaternion().setFromEuler(
      new THREE.Euler(...def.euler, 'XYZ')
    ),
  }));
})();

// ---------------------------------------------------------------------------
// QuadNode — internal data class
// ---------------------------------------------------------------------------

let _nodeId = 0;

class QuadNode {
  /**
   * @param {THREE.Vector3} trPos   – cube-space centre of this node
   * @param {THREE.Quaternion} rotation – face rotation quaternion (shared ref, never mutated)
   * @param {number} scale          – cube-space edge length
   * @param {number} level          – tree depth
   * @param {QuadNode|null} parent
   */
  constructor(trPos, rotation, scale, level, parent) {
    this.id = _nodeId++;
    this.trPos    = trPos.clone();
    this.rotation = rotation;      // shared across all nodes of the same face
    this.scale    = scale;
    this.level    = level;
    this.parent   = parent;

    this.children = null;    // QuadNode[4] when split, null = leaf
    this.mesh     = null;    // THREE.Mesh (present when this leaf has a built geometry)
    this.built    = false;   // geometry has been generated
    this.inScene  = false;   // mesh is currently added to the scene

    /**
     * Sphere-surface position of this node's centre.
     * Computed once by normalising trPos.  Used for LOD distance checks.
     */
    this.sphereCenter = trPos.clone().normalize();
  }

  isLeaf() { return this.children === null; }

  /**
   * Squared Euclidean distance from cameraPos to this node's sphere-surface centre.
   * @param {THREE.Vector3} cameraPos
   * @param {number} moonRadius
   */
  distanceSq(cameraPos, moonRadius) {
    const sx = this.sphereCenter.x * moonRadius;
    const sy = this.sphereCenter.y * moonRadius;
    const sz = this.sphereCenter.z * moonRadius;
    return (
      (cameraPos.x - sx) * (cameraPos.x - sx) +
      (cameraPos.y - sy) * (cameraPos.y - sy) +
      (cameraPos.z - sz) * (cameraPos.z - sz)
    );
  }

  /** Subdivide: create 4 children by halving this node's cube-space area. */
  split() {
    if (!this.isLeaf()) return;

    const half = this.scale * 0.5;
    const { x, y, z } = this.trPos;

    // Determine which two axes vary for this face (the third is "frozen").
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    let childOffsets;

    if (ay >= ax && ay >= az) {
      // Y-face: X and Z vary
      childOffsets = [
        [-half, 0, -half], [ half, 0, -half],
        [ half, 0,  half], [-half, 0,  half],
      ];
    } else if (az >= ax) {
      // Z-face: X and Y vary
      childOffsets = [
        [-half, -half, 0], [ half, -half, 0],
        [ half,  half, 0], [-half,  half, 0],
      ];
    } else {
      // X-face: Y and Z vary
      childOffsets = [
        [0, -half, -half], [0,  half, -half],
        [0,  half,  half], [0, -half,  half],
      ];
    }

    this.children = childOffsets.map(([dx, dy, dz]) =>
      new QuadNode(
        new THREE.Vector3(x + dx, y + dy, z + dz),
        this.rotation,
        half,
        this.level + 1,
        this
      )
    );
  }

  /**
   * Mark this node as abandoned so it is skipped by the build queue processor.
   * Called by TerrainSystem when a node is consumed by a combine operation.
   */
  abandon() {
    this._abandoned = true;
  }
}

// ---------------------------------------------------------------------------
// TerrainSystem
// ---------------------------------------------------------------------------

export class TerrainSystem {
  /**
   * @param {THREE.Scene}   scene
   * @param {number}        moonRadius  World-space sphere radius
   * @param {import('./MoonData').MoonData} moonData  Loaded NASA dataset
   * @param {THREE.Group}   moonGroup   Parent group for all Moon surface meshes.
   *                                    Terrain's internal group is added here
   *                                    instead of directly to the scene, so the
   *                                    entire Moon body can be repositioned as
   *                                    a unit when Phase 3 orbital motion lands.
   */
  constructor(scene, moonRadius = 1000, moonData, moonGroup) {
    this.scene      = scene;
    this.moonRadius = moonRadius;
    this._moonData  = moonData;
    this._moonGroup = moonGroup;

    // Shared ShaderMaterial for ALL terrain quads.
    // Uses two texture LOD levels: 4K for far, 8K for close, plus a tileable
    // detail modulation map sampled triplanarly at walking range.
    // FrontSide only — back faces (inside the moon) are culled; skirt geometry
    // has corrected winding so it is always visible from outside.
    this._material = new THREE.ShaderMaterial({
      vertexShader:   terrainVert,
      fragmentShader: terrainFrag,
      side: THREE.FrontSide,
      // dFdx / dFdy are used in the fragment shader to fix the equirectangular
      // 180° seam.  Always available in WebGL 2.0; the extension enables it
      // in WebGL 1.0 fallback mode.
      extensions: { derivatives: true },
      uniforms: {
        uSunDirection:   { value: sunDirection },
        uEarthDirection: { value: earthDirection },
        uColorMapLOD0:   { value: moonData.colorTextures[0] },
        uColorMapLOD1:   { value: moonData.colorTextures[1] },
        uLOD1Distance:   { value: 150.0 },
        uCameraPos:      { value: new THREE.Vector3() },
        uDetailMap:      { value: moonData.detailTexture },
        uDetailTiling:   { value: 0.1 },
        // ── Cascaded Shadow Maps ─────────────────────────────────────────────
        // Three depth textures + three shadow matrices (one per cascade).
        // Set once from main.js after CascadedShadowMap is created; updated
        // in-place each frame by CascadedShadowMap.renderShadows().
        uShadowMap0:     { value: null },
        uShadowMap1:     { value: null },
        uShadowMap2:     { value: null },
        uShadowMatrix0:  { value: new THREE.Matrix4() },
        uShadowMatrix1:  { value: new THREE.Matrix4() },
        uShadowMatrix2:  { value: new THREE.Matrix4() },
        // Cascade end distances (x=cascade0, y=cascade1, z=cascade2) and
        // camera near/far for gl_FragCoord.z linearisation.
        uCascadeSplits:  { value: new THREE.Vector3(10, 40, 120) },
        uCamNear:        { value: 0.5 },
        uCamFar:         { value: 350000.0 },
        uWireframe:      { value: 0 },
        // Player spotlight
        uSpotlightOn:    { value: 0 },
        uSpotlightPos:   { value: new THREE.Vector3() },
        uSpotlightDir:   { value: new THREE.Vector3() },
        uSpotlightAngle: { value: 0.5 },
        uSpotlightRange: { value: 30.0 },
        uSpotlightShadowMap: { value: null },
        uSpotlightMatrix:    { value: new THREE.Matrix4() },
      },
    });

    // Scene group — all terrain meshes are children of this.
    // Parented to moonGroup (not scene directly) so the whole Moon body can
    // be moved as a unit in Phase 3.
    this._group = new THREE.Group();
    this._moonGroup.add(this._group);

    // Per-frame mesh-build budget counter (reset each update call).
    this._buildsThisFrame = 0;

    // Build queue: FIFO list of nodes awaiting geometry generation.
    // _buildSet is the O(1) membership check companion so we never double-enqueue.
    this._buildQueue = [];
    this._buildSet   = new Set();

    // Initialise the 6 face root nodes.
    this._roots = FACE_DEFS.map(def =>
      new QuadNode(def.trPos, def.rotation, 1.0, 0, null)
    );
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Query terrain elevation at a world-space unit-sphere direction.
   * @param {number} nx  x component of normalised sphere direction
   * @param {number} ny  y component
   * @param {number} nz  z component
   * @returns {number} Displacement height above the base sphere (world units)
   */
  /**
   * Sample real LOLA elevation at a normalised sphere direction.
   * Delegates to MoonData which does bilinear interpolation on the uint16 TIFF.
   */
  getHeightAt(nx, ny, nz) {
    return this._moonData.getElevationAt(nx, ny, nz);
  }

  /**
   * Per-frame update.  Traverses the quadtree, issues splits/combines, and
   * processes the build queue.
   * @param {THREE.Vector3} cameraPos  World-space camera position
   */
  update(cameraPos) {
    this._buildsThisFrame = 0;

    // Update camera position for texture LOD in shader
    this._material.uniforms.uCameraPos.value.copy(cameraPos);

    for (const root of this._roots) {
      this._traverse(root, cameraPos);
    }

    // Process pending builds (throttled to MAX_BUILDS_PER_FRAME per frame).
    while (this._buildQueue.length > 0 && this._buildsThisFrame < MAX_BUILDS_PER_FRAME) {
      const node = this._buildQueue.shift();
      this._buildSet.delete(node);

      // Skip nodes that were abandoned by a combine while queued.
      if (node._abandoned || node.built) continue;

      this._buildMesh(node);
      this._buildsThisFrame++;
    }
  }

  /** Toggle wireframe rendering on all terrain tiles (X key diagnostic). */
  setWireframe(enabled) {
    this._material.wireframe = enabled;
    this._material.uniforms.uWireframe.value = enabled ? 1 : 0;
  }

  dispose() {
    for (const root of this._roots) {
      this._disposeSubtree(root);
    }
    this._material.dispose();
    this._moonGroup.remove(this._group);
  }

  // ---------------------------------------------------------------------------
  // Spotlight uniform binding
  // ---------------------------------------------------------------------------

  setSpotlightUniforms(on, pos, dir, angle, range, shadowTex, shadowMatrix) {
    const u = this._material.uniforms;
    u.uSpotlightOn.value    = on ? 1 : 0;
    u.uSpotlightPos.value.copy(pos);
    u.uSpotlightDir.value.copy(dir);
    u.uSpotlightAngle.value = angle;
    u.uSpotlightRange.value = range;
    u.uSpotlightShadowMap.value = shadowTex;
    u.uSpotlightMatrix.value.copy(shadowMatrix);
  }

  // ---------------------------------------------------------------------------
  // Internal — LOD traversal
  // ---------------------------------------------------------------------------

  /**
   * Recursive LOD decision for a node.
   *
   * Key invariant: at every frame, at least one built mesh covers every visible
   * region.  Specifically:
   *
   *   SPLIT:   parent stays in scene until ALL 4 children have built geometry.
   *            Only then is the parent hidden and children shown.
   *
   *   COMBINE: parent is put back in scene BEFORE children are removed.
   *            This prevents even a single-frame gap.
   */
  _traverse(node, cameraPos) {
    const distSq = node.distanceSq(cameraPos, this.moonRadius);

    if (node.isLeaf()) {
      // ------------------------------------------------------------------ leaf

      // Ensure this leaf has geometry queued.
      if (!node.built) {
        this._enqueueBuildOnce(node);
        return; // Nothing to show yet — parent is still covering this region.
      }

      // Show leaf mesh.
      if (!node.inScene) this._addToScene(node);

      // Split decision.
      if (node.level < MAX_LEVEL && distSq < this._splitDistSq(node.level)) {
        node.split();
        // Enqueue children.  Parent stays visible as fallback; the non-leaf
        // branch of this same node will hide it once all children are built.
        for (const child of node.children) this._enqueueBuildOnce(child);
        // Fall through to the non-leaf branch immediately so deeper LOD can
        // already be queued this frame.
        // (node is now a non-leaf — fall into the else below)
      } else {
        return; // Leaf is final at this LOD level.
      }
    }

    // ---------------------------------------------------------------- non-leaf

    const children = node.children; // non-null at this point

    // Check whether all direct children have built geometry.
    const allBuilt = children.every(c => c.built);

    if (allBuilt) {
      // Children are ready → hide parent, ensure children are shown.
      if (node.inScene) this._removeFromScene(node);
      for (const child of children) {
        if (child.isLeaf() && !child.inScene) this._addToScene(child);
      }
    } else {
      // Children still loading → keep parent visible as placeholder.
      if (node.built && !node.inScene) this._addToScene(node);
    }

    // Combine decision: only safe when all children are leaves (no grandchildren
    // that would orphan their sub-trees if we collapsed prematurely).
    const canCombine = children.every(c => c.isLeaf());
    if (canCombine && distSq > this._combineDistSq(node.level)) {
      this._doCombine(node);
      return;
    }

    // Recurse.
    for (const child of children) this._traverse(child, cameraPos);
  }

  /**
   * Collapse all 4 leaf children back into the parent node.
   * Shows the parent FIRST so there is zero-frame gap.
   */
  _doCombine(node) {
    // Parent must be in scene before children are removed.
    if (node.built && !node.inScene) this._addToScene(node);

    for (const child of node.children) {
      // Properly remove from the THREE.Group (sets inScene = false internally).
      this._removeFromScene(child);
      // Cancel any pending build for this child.
      child.abandon();
      this._buildSet.delete(child);
      // Free GPU memory.
      if (child.mesh) {
        child.mesh.geometry.dispose();
        child.mesh = null;
      }
      child.built = false;
    }

    node.children = null; // Node is a leaf again.
  }

  _splitDistSq(level) {
    if (level >= LOD_SPLIT_DIST.length) return 0;
    const d = LOD_SPLIT_DIST[level];
    return d * d;
  }

  _combineDistSq(level) {
    if (level >= LOD_SPLIT_DIST.length) return Infinity;
    const d = LOD_SPLIT_DIST[level] * LOD_COMBINE_FACTOR;
    return d * d;
  }

  // ---------------------------------------------------------------------------
  // Internal — scene graph management
  // ---------------------------------------------------------------------------

  _addToScene(node) {
    if (!node.mesh || node.inScene) return;
    this._group.add(node.mesh);
    node.inScene = true;
  }

  _removeFromScene(node) {
    if (!node.mesh || !node.inScene) return;
    this._group.remove(node.mesh);
    node.inScene = false;
  }

  /** Enqueue a node for geometry generation (idempotent, O(1)). */
  _enqueueBuildOnce(node) {
    if (!node.built && !node._abandoned && !this._buildSet.has(node)) {
      this._buildSet.add(node);
      this._buildQueue.push(node);
    }
  }

  _disposeSubtree(node) {
    if (!node.isLeaf()) {
      for (const child of node.children) this._disposeSubtree(child);
    }
    this._removeFromScene(node);
    if (node.mesh) {
      node.mesh.geometry.dispose();
      node.mesh = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Internal — geometry generation
  // ---------------------------------------------------------------------------

  /**
   * Builds a THREE.Mesh for the given leaf node and attaches it.
   *
   * Algorithm (PlanetaryTerrain-style GetPosition):
   *   for each grid vertex (u, v):
   *     1. flat:   p = (u*scale, 0, v*scale)
   *     2. rotate: p.applyQuaternion(faceRotation)   → correct cube face
   *     3. offset: p += trPos                        → position on unit cube
   *     4. normalise: p.normalize()                  → project to unit sphere
   *     5. height: h = getHeightAt(p)                → real LOLA elevation
   *     6. scale:  p *= (moonRadius + h)             → displaced sphere surface
   *     7. normal: analytical gradient from height field (seamless across tiles)
   *
   *  Skirts: each tile edge grows a row of vertices pushed inward by skirtDepth.
   *  These curtains cover T-junction gaps that appear when adjacent tiles are at
   *  different LOD levels (a fine tile's intermediate-edge vertices have no
   *  corresponding vertex in the coarser neighbour).
   */
  _buildMesh(node) {
    const segs = QUAD_SEGMENTS;
    const n    = segs + 1;            // vertices per side
    const gridCount  = n * n;
    const skirtCount = 4 * n;         // one row per edge
    const totalVerts = gridCount + skirtCount;

    const positions = new Float32Array(totalVerts * 3);
    const normals   = new Float32Array(totalVerts * 3);
    const uvs       = new Float32Array(totalVerts * 2);

    // Reusable temporaries — allocated once, reused every vertex.
    const _p   = new THREE.Vector3();
    const _pe  = new THREE.Vector3();
    const _pn  = new THREE.Vector3();
    const _e   = new THREE.Vector3();
    const _nn  = new THREE.Vector3();
    const _te  = new THREE.Vector3();
    const _tn  = new THREE.Vector3();
    const _Y   = new THREE.Vector3(0, 1, 0);
    const _X   = new THREE.Vector3(1, 0, 0);

    const inv = 1.0 / segs;

    // Epsilon for height-gradient normal sampling.
    // CRITICAL: must be a FIXED constant, independent of LOD level.
    // If eps varies with node.scale, adjacent tiles at different LOD levels
    // sample the height gradient at different angular offsets → different normals
    // at the shared boundary → visible brightness seam (especially at cube-face
    // boundaries where the +Y and side faces meet at different LOD levels).
    // 0.002 rad ≈ 2 km at moonRadius=1000, comfortably larger than the LOLA
    // raster pixel (~0.0017 rad at 16ppd) and gives consistent normals everywhere.
    const eps = 0.002;

    // Skirt depth: proportional to tile size so every LOD gap is covered.
    const skirtDepth = this.moonRadius * node.scale * 0.02;

    // -----------------------------------------------------------------------
    // Build main grid
    // -----------------------------------------------------------------------

    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const i = row * n + col;

        // 1-4. Grid → cube → sphere direction
        const u = (col * inv * 2.0 - 1.0) * node.scale;
        const v = (row * inv * 2.0 - 1.0) * node.scale;
        _p.set(u, 0.0, v);
        _p.applyQuaternion(node.rotation);
        _p.add(node.trPos);
        _p.normalize();

        // 5-6. Height displacement
        const h = this.getHeightAt(_p.x, _p.y, _p.z);
        const r = this.moonRadius + h;

        positions[i * 3]     = _p.x * r;
        positions[i * 3 + 1] = _p.y * r;
        positions[i * 3 + 2] = _p.z * r;

        uvs[i * 2]     = col * inv;
        uvs[i * 2 + 1] = row * inv;

        // 7. Analytical normal from height field gradient.
        //    Uses the same continuous getHeightAt() on both sides of any
        //    tile boundary → normals match exactly across seams.
        //
        //    East and north tangent directions on the sphere surface:
        if (Math.abs(_p.y) < 0.999) {
          _e.crossVectors(_Y, _p).normalize();
        } else {
          _e.crossVectors(_X, _p).normalize(); // pole singularity fallback
        }
        _nn.crossVectors(_p, _e).normalize();

        // Neighbouring sphere directions
        _pe.copy(_p).addScaledVector(_e,  eps).normalize();
        _pn.copy(_p).addScaledVector(_nn, eps).normalize();
        const he = this.getHeightAt(_pe.x, _pe.y, _pe.z);
        const hn = this.getHeightAt(_pn.x, _pn.y, _pn.z);

        // Surface tangent vectors (chain rule: P(p) = p * (R + h(p)))
        _te.copy(_e ).multiplyScalar(r).addScaledVector(_p, he - h);
        _tn.copy(_nn).multiplyScalar(r).addScaledVector(_p, hn - h);

        // cross(te, tn) points outward for a right-handed surface coordinate system
        _te.cross(_tn).normalize();
        normals[i * 3]     = _te.x;
        normals[i * 3 + 1] = _te.y;
        normals[i * 3 + 2] = _te.z;
      }
    }

    // -----------------------------------------------------------------------
    // Build skirt vertices — same position/normal as edge, pulled inward
    // -----------------------------------------------------------------------

    const topStart   = gridCount;
    const botStart   = gridCount +     n;
    const leftStart  = gridCount + 2 * n;
    const rightStart = gridCount + 3 * n;

    const addSkirt = (gridIdx, skirtIdx) => {
      const nx = normals[gridIdx * 3];
      const ny = normals[gridIdx * 3 + 1];
      const nz = normals[gridIdx * 3 + 2];
      // Push toward planet centre along the surface normal direction
      positions[skirtIdx * 3]     = positions[gridIdx * 3]     - nx * skirtDepth;
      positions[skirtIdx * 3 + 1] = positions[gridIdx * 3 + 1] - ny * skirtDepth;
      positions[skirtIdx * 3 + 2] = positions[gridIdx * 3 + 2] - nz * skirtDepth;
      normals[skirtIdx * 3]     = nx;
      normals[skirtIdx * 3 + 1] = ny;
      normals[skirtIdx * 3 + 2] = nz;
      uvs[skirtIdx * 2]     = uvs[gridIdx * 2];
      uvs[skirtIdx * 2 + 1] = uvs[gridIdx * 2 + 1];
    };

    for (let col = 0; col < n; col++) {
      addSkirt(0 * n + col,    topStart  + col); // top edge    (row = 0)
      addSkirt(segs * n + col, botStart  + col); // bottom edge (row = segs)
    }
    for (let row = 0; row < n; row++) {
      addSkirt(row * n + 0,    leftStart  + row); // left edge  (col = 0)
      addSkirt(row * n + segs, rightStart + row); // right edge (col = segs)
    }

    // -----------------------------------------------------------------------
    // Triangle indices
    // -----------------------------------------------------------------------

    const gridTris  = segs * segs * 6;
    const skirtTris = segs * 4 * 6;
    const indices   = new Uint32Array(gridTris + skirtTris);
    let   idx       = 0;

    // Main grid (CCW from outside — winding verified in FACE_DEFS comments)
    for (let row = 0; row < segs; row++) {
      for (let col = 0; col < segs; col++) {
        const a = row * n + col;
        const b = a + n;
        indices[idx++] = a;     indices[idx++] = b;     indices[idx++] = a + 1;
        indices[idx++] = b;     indices[idx++] = b + 1; indices[idx++] = a + 1;
      }
    }

    // Skirts — winding auto-corrected per quad so the outward face is always front.
    //
    // Problem with the naive approach (dot trial-normal with vertex position):
    //   Skirt faces are *tangential* to the sphere — their normals are perpendicular
    //   to the radial direction, making the dot product near-zero and unreliable.
    //
    // Correct reference: the direction from the tile's centre vertex to the edge
    //   vertex. This vector always points "away from the tile interior" regardless
    //   of face orientation or LOD level, and it has a reliable non-zero dot product
    //   with the skirt face normal (which also points away from the tile interior).
    const centerIdx = Math.floor(segs / 2) * n + Math.floor(segs / 2);
    const tcx = positions[centerIdx * 3];
    const tcy = positions[centerIdx * 3 + 1];
    const tcz = positions[centerIdx * 3 + 2];

    const addSkirtQuad = (g0, g1, s0, s1) => {
      const ax = positions[g0*3], ay = positions[g0*3+1], az = positions[g0*3+2];
      const bx = positions[g1*3], by = positions[g1*3+1], bz = positions[g1*3+2];
      const cx = positions[s0*3], cy = positions[s0*3+1], cz = positions[s0*3+2];

      const e1x = bx-ax, e1y = by-ay, e1z = bz-az;
      const e2x = cx-ax, e2y = cy-ay, e2z = cz-az;
      // Trial normal = e1 × e2
      const tnx = e1y*e2z - e1z*e2y;
      const tny = e1z*e2x - e1x*e2z;
      const tnz = e1x*e2y - e1y*e2x;

      // Reference: direction from tile centre to edge vertex — always points
      // away from the tile interior for every edge on every cube face.
      const rx = ax - tcx, ry = ay - tcy, rz = az - tcz;

      if ((tnx*rx + tny*ry + tnz*rz) > 0) {
        indices[idx++] = g0; indices[idx++] = g1; indices[idx++] = s0;
        indices[idx++] = g1; indices[idx++] = s1; indices[idx++] = s0;
      } else {
        indices[idx++] = g0; indices[idx++] = s0; indices[idx++] = g1;
        indices[idx++] = g1; indices[idx++] = s0; indices[idx++] = s1;
      }
    };

    for (let col = 0; col < segs; col++) {
      // Top edge (row = 0)
      addSkirtQuad(col,          col + 1,
                   topStart+col, topStart+col+1);
      // Bottom edge (row = segs)
      addSkirtQuad(segs*n + col, segs*n + col + 1,
                   botStart+col, botStart+col+1);
    }
    for (let row = 0; row < segs; row++) {
      // Left edge (col = 0)
      addSkirtQuad(row*n,           (row+1)*n,
                   leftStart+row,  leftStart+row+1);
      // Right edge (col = segs)
      addSkirtQuad(row*n + segs,    (row+1)*n + segs,
                   rightStart+row, rightStart+row+1);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    // Note: no computeVertexNormals() — analytical normals are already seamless

    node.mesh  = new THREE.Mesh(geometry, this._material);
    // Terrain does NOT go on layer 1 — only rocks cast shadows.
    // Terrain self-shadowing is already handled by NdotL + bump normals.
    node.built = true;
  }
}
