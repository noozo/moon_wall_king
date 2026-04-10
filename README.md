# Moon Wall King

A realistic lunar surface simulation with realistic lighting, shadows, and physics-based player movement.

<img width="1679" height="1322" alt="image" src="https://github.com/user-attachments/assets/16340db3-f02d-4427-8e30-7dc9afb0dad8" />

<img width="1676" height="1317" alt="image" src="https://github.com/user-attachments/assets/44cf4eb2-7c8f-4d5c-a03a-ae3bfb4fd227" />

<img width="1676" height="1317" alt="image" src="https://github.com/user-attachments/assets/c3c81a0c-f9aa-4aed-9135-be2afac2ac70" />

<img width="1676" height="1317" alt="image" src="https://github.com/user-attachments/assets/23534c66-e0bb-4727-b8f6-3845a088c211" />

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
