/**
 * Download NASA CGI Moon Kit textures to public/textures/.
 * Run once:  node scripts/download-textures.js
 *
 * Sources (NASA SVS 4720 – public domain):
 *   https://svs.gsfc.nasa.gov/4720
 *
 * Resolution options:
 *   Color:    2K (2048x1024) → 4K (4096x2048) → 8K (8192x4096) → 16K (16384x8192)
 *   Elevation: 4ppd (1440x720) → 16ppd (5760x2880) → 64ppd (23040x11520)
 *
 * Current setup uses:
 *   - Color:     lroc_color_16bit_srgb_4k.tif (59 MB) → converted to PNG
 *   - Elevation: ldem_16_uint.tif (31.7 MB)
 */

import fs   from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEST_DIR  = path.resolve(__dirname, '../public/textures');

const BASE = 'https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720';

const FILES = [
  { name: 'lroc_color_4k.png',  url: `${BASE}/lroc_color_16bit_srgb_4k.tif`,  desc: 'Color map 4K (4096x2048, 59 MB)', convertToPng: true },
  { name: 'ldem_16_uint.tif',    url: `${BASE}/ldem_16_uint.tif`,               desc: 'Elevation 16ppd uint16 (5760x2880, 31.7 MB)' },
  { name: 'earth_color_4k.jpg', url: 'https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg', desc: 'Earth color 2K', convertToPng: false },
  { name: 'sun_color_4k.jpg',    url: 'https://www.solarsystemscope.com/textures/download/2k_sun.jpg',           desc: 'Sun texture 2K', convertToPng: false },
];

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const req = https.get(url, { headers: { 'User-Agent': 'moon-wall-king/1.0' } }, (res) => {
      if (res.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        reject(new Error(`HTTP ${res.statusCode}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(); });
    });
    req.on('error', (e) => {
      file.close();
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
      reject(e);
    });
  });
}

async function tiffToPng(tiffPath, pngPath) {
  const { default: sharp } = await import('sharp');
  await sharp(tiffPath).png().toFile(pngPath);
  fs.unlinkSync(tiffPath);
  console.log(`    converted to PNG and removed TIFF`);
}

async function main() {
  fs.mkdirSync(DEST_DIR, { recursive: true });

  for (const { name, url, desc, convertToPng } of FILES) {
    const dest = path.join(DEST_DIR, name);
    const tiffTemp = convertToPng ? dest.replace('.png', '.tif') : dest;

    if (fs.existsSync(dest)) {
      console.log(`  ✓ ${name} already present — skipping`);
      continue;
    }

    process.stdout.write(`  ↓ ${desc} … `);
    await downloadFile(url, tiffTemp);

    if (convertToPng) {
      await tiffToPng(tiffTemp, dest);
    }

    const kb = Math.round(fs.statSync(dest).size / 1024);
    console.log(`done (${kb} KB)`);
  }

  console.log('\nAll textures ready in public/textures/');
}

main().catch(e => { console.error('\nError:', e.message); process.exit(1); });
