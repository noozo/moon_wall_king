# Moon Wall King

A realistic lunar surface simulation with realistic lighting, shadows, and physics-based player movement.

## Controls

### Movement
- **WASD** — Move (forward/back/strafe)
- **Mouse** — Look around
- **Shift** — Sprint
- **Space** — Jump

### Flight & Physics
- **Tab** — Toggle debug fly mode
- **T** — RCS damping (velocity kill mode)

### Equipment
- **L** — Toggle flashlight
- **[** / **]** — Decrease/increase flashlight range
- **-** / **=** — Decrease/increase flashlight angle

### Debug
- **X** — Toggle wireframe mode (terrain/rocks)

### Fly Mode (when enabled with Tab)
- **W/S** — Forward/backward
- **A/D** — Strafe left/right
- **Space/Ctrl** — Fly up/down
- **Q/E** — Roll left/right
- **Shift** — 5× speed multiplier
- **Scroll wheel** — Adjust speed
- **Tab** — Return to player mode

## Features

- Cascaded shadow maps for proximity-based shadows
- RCS (Reaction Control System) force-based movement with gravity
- Player flashlight with cone + distance attenuation
- Realistic Moon terrain with LOD
- Rock system with displaced icosahedron geometry
- Earthshine and moon occlusion for realistic night-side lighting