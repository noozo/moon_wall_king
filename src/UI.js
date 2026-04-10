/**
 * UI — Head-up display: coordinates, speed, compass heading.
 */
export class UI {
  constructor(playerController) {
    this._player   = playerController;
    this._coords   = document.getElementById('coords-value');
    this._speed    = document.getElementById('speed-value');
    this._compass  = document.getElementById('compass');
  }

  update() {
    const coords  = this._player.getCoordinates();
    const speed   = this._player.getSpeed();
    const heading = this._player.getHeading();

    const latDir = coords.latitude  >= 0 ? 'N' : 'S';
    const lonDir = coords.longitude >= 0 ? 'E' : 'W';

    this._coords.textContent =
      `${Math.abs(coords.latitude).toFixed(2)}° ${latDir},  ` +
      `${Math.abs(coords.longitude).toFixed(2)}° ${lonDir}`;

    this._speed.textContent = `${speed.toFixed(1)} m/s`;

    const DIRS = ['N','NE','E','SE','S','SW','W','NW'];
    const idx  = Math.round(((heading % 360) + 360) % 360 / 45) % 8;
    this._compass.textContent = DIRS[idx];
  }

  dispose() {}
}
