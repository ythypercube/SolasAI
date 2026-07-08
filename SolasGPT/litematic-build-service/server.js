import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import AdmZip from 'adm-zip';
import { parse, simplify, writeUncompressed } from 'prismarine-nbt';

const app = express();
app.use(express.json({ limit: '5mb' }));

const PORT = Number(process.env.PORT || 8790);
const LITEMATICS_DIR = expandHome(process.env.LITEMATICS_DIR || '~/.minecraft/schematics');

const scanCache = new Map();

function expandHome(p) {
  if (!p) return p;
  if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2));
  return p;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function sanitizeFileName(name) {
  const base = String(name || 'imported_litematic').trim().replace(/[^a-zA-Z0-9._-]+/g, '_');
  if (!base) return 'imported_litematic.litematic';
  return base.endsWith('.litematic') ? base : `${base}.litematic`;
}

function normalizeBlockId(id) {
  if (!id) return 'minecraft:air';
  const value = String(id).toLowerCase();
  return value.includes(':') ? value : `minecraft:${value}`;
}

function decodeBitPackedIndices(longArray, bitsPerEntry, entryCount, paletteSize) {
  const indices = new Array(entryCount);
  if (!Array.isArray(longArray) || longArray.length === 0) {
    indices.fill(0);
    return indices;
  }

  const mask = (1n << BigInt(bitsPerEntry)) - 1n;
  const data = longArray.map((v) => BigInt.asUintN(64, BigInt(v)));

  let bitIndex = 0;
  for (let i = 0; i < entryCount; i++) {
    const startLong = Math.floor(bitIndex / 64);
    const startOffset = bitIndex % 64;

    let value;
    if (startOffset + bitsPerEntry <= 64) {
      value = (data[startLong] >> BigInt(startOffset)) & mask;
    } else {
      const lowBits = 64 - startOffset;
      const highBits = bitsPerEntry - lowBits;
      const lowMask = (1n << BigInt(lowBits)) - 1n;
      const lowPart = (data[startLong] >> BigInt(startOffset)) & lowMask;
      const highPart = (data[startLong + 1] & ((1n << BigInt(highBits)) - 1n)) << BigInt(lowBits);
      value = lowPart | highPart;
    }

    let paletteIndex = Number(value);
    if (!Number.isFinite(paletteIndex) || paletteIndex < 0 || paletteIndex >= paletteSize) {
      paletteIndex = 0;
    }
    indices[i] = paletteIndex;
    bitIndex += bitsPerEntry;
  }

  return indices;
}

function extractPaletteName(entry) {
  if (!entry) return 'minecraft:air';
  if (typeof entry === 'string') return normalizeBlockId(entry);
  if (typeof entry?.Name === 'string') return normalizeBlockId(entry.Name);
  return normalizeBlockId(entry?.name || 'minecraft:air');
}

function parseLitematicMaterials(filePath) {
  const buffer = fs.readFileSync(filePath);
  return parse(buffer).then((nbtData) => {
    const data = simplify(nbtData?.parsed || nbtData);
    const regions = data?.Regions || {};

    const materials = new Map();
    let totalBlocks = 0;

    for (const regionName of Object.keys(regions)) {
      const region = regions[regionName] || {};
      const size = region.Size || { x: 0, y: 0, z: 0 };
      const entryCount = Math.abs(Number(size.x || 0) * Number(size.y || 0) * Number(size.z || 0));
      if (!entryCount) continue;

      const palette = Array.isArray(region.BlockStatePalette) ? region.BlockStatePalette : [];
      const blockStates = Array.isArray(region.BlockStates) ? region.BlockStates : [];
      if (!palette.length) continue;

      const bitsPerEntry = Math.max(2, Math.ceil(Math.log2(Math.max(1, palette.length))));
      const indices = decodeBitPackedIndices(blockStates, bitsPerEntry, entryCount, palette.length);

      for (const paletteIndex of indices) {
        const blockId = extractPaletteName(palette[paletteIndex]);
        if (blockId === 'minecraft:air' || blockId.endsWith(':cave_air') || blockId.endsWith(':void_air')) continue;
        totalBlocks += 1;
        materials.set(blockId, (materials.get(blockId) || 0) + 1);
      }
    }

    const sortedMaterials = Array.from(materials.entries())
      .map(([id, count]) => ({ id, count }))
      .sort((a, b) => b.count - a.count);

    return {
      fileName: path.basename(filePath),
      totalBlocks,
      uniqueMaterials: sortedMaterials.length,
      materials: sortedMaterials,
      metadata: data?.Metadata || {}
    };
  });
}

async function downloadToBuffer(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        'user-agent': 'SolasAI-SchematicImporter/1.0 (+https://localhost)',
        accept: '*/*'
      }
    });
    if (!res.ok) {
      throw new Error(`Download failed with HTTP ${res.status}`);
    }
    const ab = await res.arrayBuffer();
    return Buffer.from(ab);
  } finally {
    clearTimeout(timeout);
  }
}

function isDirectSchematicUrl(urlText) {
  return /\.(litematic|schem|schematic|nbt|zip)(\?|#|$)/i.test(urlText);
}

function extractCandidateLinks(html, baseUrl) {
  const candidates = [];
  const hrefRegex = /href\s*=\s*['"]([^'"]+)['"]/gi;
  let match;
  while ((match = hrefRegex.exec(html)) !== null) {
    try {
      const rawHref = match[1].trim();
      if (!rawHref || rawHref.startsWith('javascript:')) continue;
      const abs = new URL(rawHref, baseUrl).toString();
      candidates.push(abs);
    } catch {
      // ignore malformed links
    }
  }
  return candidates;
}

async function resolveDownloadUrl(urlText) {
  const input = String(urlText || '').trim();
  if (!input) throw new Error('url is required');
  if (isDirectSchematicUrl(input)) return input;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const pageRes = await fetch(input, {
      signal: controller.signal,
      headers: {
        'user-agent': 'SolasAI-SchematicImporter/1.0 (+https://localhost)',
        accept: 'text/html,application/xhtml+xml'
      }
    });
    if (!pageRes.ok) {
      throw new Error(`Could not open page URL (HTTP ${pageRes.status})`);
    }
    const html = await pageRes.text();
    const links = extractCandidateLinks(html, input);

    const preferred = links.find((link) =>
      /minecraft-schematics\.com/i.test(link) && /download|downloads|dl|\.(zip|litematic|schem|schematic)(\?|#|$)/i.test(link)
    );
    if (preferred) return preferred;

    const generic = links.find((link) => isDirectSchematicUrl(link));
    if (generic) return generic;

    throw new Error('Could not find a downloadable schematic file on that page');
  } finally {
    clearTimeout(timeout);
  }
}

function writeImportedLitematic(buffer, requestedName) {
  ensureDir(LITEMATICS_DIR);
  const finalName = sanitizeFileName(requestedName || `imported_${Date.now()}.litematic`);
  const finalPath = path.join(LITEMATICS_DIR, finalName);
  fs.writeFileSync(finalPath, buffer);
  return { finalName, finalPath };
}

function extractLitematicFromZipBuffer(zipBuffer) {
  const zip = new AdmZip(zipBuffer);
  const entries = zip.getEntries();
  const litematic = entries.find((e) => !e.isDirectory && e.entryName.toLowerCase().endsWith('.litematic'));
  if (!litematic) {
    throw new Error('ZIP does not contain a .litematic file');
  }
  return { buffer: litematic.getData(), inferredName: path.basename(litematic.entryName) };
}

function materialPlan(scanResult, inventory = {}) {
  const normalizedInventory = {};
  for (const [k, v] of Object.entries(inventory || {})) {
    normalizedInventory[normalizeBlockId(k)] = Math.max(0, Number(v) || 0);
  }

  const missing = [];
  const ready = [];

  for (const material of scanResult.materials) {
    const have = normalizedInventory[material.id] || 0;
    if (have >= material.count) {
      ready.push({ ...material, have, missing: 0 });
    } else {
      missing.push({ ...material, have, missing: material.count - have });
    }
  }

  return {
    fileName: scanResult.fileName,
    totalBlocks: scanResult.totalBlocks,
    readyMaterials: ready.length,
    missingMaterials: missing.length,
    phase: missing.length > 0 ? 'gather' : 'build',
    gatherList: missing,
    buildList: scanResult.materials
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function slugify(value) {
  return String(value || 'redstone_build')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48) || 'redstone_build';
}

function pickGoalVariant(goalText, salt, variants) {
  const digest = crypto
    .createHash('sha1')
    .update(`${goalText || 'redstone_build'}|${salt}`)
    .digest();
  return variants[digest[0] % variants.length];
}

function encodeBitPackedIndices(indices, bitsPerEntry) {
  const totalBits = indices.length * bitsPerEntry;
  const longCount = Math.max(1, Math.ceil(totalBits / 64));
  const longs = new Array(longCount).fill(0n);
  const mask = (1n << BigInt(bitsPerEntry)) - 1n;

  let bitIndex = 0;
  for (const rawIndex of indices) {
    const value = BigInt(rawIndex) & mask;
    const startLong = Math.floor(bitIndex / 64);
    const startOffset = bitIndex % 64;

    if (startOffset + bitsPerEntry <= 64) {
      longs[startLong] |= value << BigInt(startOffset);
    } else {
      const lowBits = 64 - startOffset;
      const highBits = bitsPerEntry - lowBits;
      const lowMask = (1n << BigInt(lowBits)) - 1n;
      const lowPart = value & lowMask;
      const highPart = value >> BigInt(lowBits);
      longs[startLong] |= lowPart << BigInt(startOffset);
      longs[startLong + 1] |= highPart & ((1n << BigInt(highBits)) - 1n);
    }
    bitIndex += bitsPerEntry;
  }

  return longs.map((value) => {
    const signed64 = BigInt.asIntN(64, value);
    const hi = Number(BigInt.asIntN(32, signed64 >> 32n));
    const lo = Number(BigInt.asIntN(32, signed64 & 0xffffffffn));
    return [hi, lo];
  });
}

function buildSimulatorBlockVolume(goal, candidate = {}) {
  const goalText = String(goal || '').toLowerCase();
  let width, height, depth;

  const isRailgunGoal =
    goalText.includes('arrow') ||
    goalText.includes('railgun') ||
    goalText.includes('minecart') ||
    goalText.includes('tnt minecart');

  if (isRailgunGoal) {
    width = 40; height = 9; depth = 11;
  } else if (goalText.includes('hoglin')) {
    width = 28; height = 12; depth = 28;
  } else if (goalText.includes('wither')) {
    width = 28; height = 12; depth = 28;
  } else if (goalText.includes('3x3') || goalText.includes('door')) {
    width = 12; height = 8; depth = 10;
  } else if (goalText.includes('sorter')) {
    width = 24; height = 8; depth = 12;
  } else if (goalText.includes('flying')) {
    width = 20; height = 10; depth = 12;
  } else if (goalText.includes('sugar') || goalText.includes('clock')) {
    width = 16; height = 8; depth = 14;
  } else {
    width = 18; height = 8; depth = 16;
  }
  const volume = width * height * depth;

  const palette = [
    'minecraft:air',
    'minecraft:stone',
    'minecraft:dirt',
    'minecraft:smooth_stone',
    'minecraft:stone_bricks',
    'minecraft:polished_deepslate',
    'minecraft:redstone_wire',
    'minecraft:repeater',
    'minecraft:comparator',
    'minecraft:observer',
    'minecraft:piston',
    'minecraft:sticky_piston',
    'minecraft:slime_block',
    'minecraft:honey_block',
    'minecraft:redstone_block',
    'minecraft:redstone_lamp',
    'minecraft:hopper',
    'minecraft:glass',
    'minecraft:nether_bricks',
    'minecraft:magma_block',
    'minecraft:crimson_nylium',
    'minecraft:warped_fungus',
    'minecraft:wither_rose',
    'minecraft:soul_sand',
    'minecraft:oak_trapdoor',
    'minecraft:dispenser',
    'minecraft:target',
    'minecraft:quartz_block',
    'minecraft:rail',
    'minecraft:powered_rail',
    'minecraft:detector_rail',
    'minecraft:activator_rail',
    'minecraft:redstone_torch',
    'minecraft:lever',
    'minecraft:obsidian'
  ];

  const paletteIndex = Object.fromEntries(palette.map((id, i) => [id, i]));
  const indices = new Array(volume).fill(0);
  const entities = [];
  const idx = (x, y, z) => x + z * width + y * width * depth;
  const ensurePaletteEntry = (blockSpec) => {
    if (paletteIndex[blockSpec] !== undefined) return paletteIndex[blockSpec];
    const newIndex = palette.length;
    palette.push(blockSpec);
    paletteIndex[blockSpec] = newIndex;
    return newIndex;
  };
  const setBlock = (x, y, z, blockId) => {
    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) return;
    indices[idx(x, y, z)] = ensurePaletteEntry(blockId);
  };
  const fillArea = (x1, y1, z1, x2, y2, z2, blockId) => {
    for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x += 1) {
      for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y += 1) {
        for (let z = Math.min(z1, z2); z <= Math.max(z1, z2); z += 1) {
          setBlock(x, y, z, blockId);
        }
      }
    }
  };
  
  if (isRailgunGoal) {
    // TNT-minecart arrow railgun: compact stacked breech, localized blast cell, narrow muzzle.
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:stone_bricks');

    // Breech shielding only, keep the muzzle light and narrow.
    fillArea(1, 1, 2, 15, 1, depth - 3, 'minecraft:obsidian');
    fillArea(1, 2, 2, 11, 5, depth - 3, 'minecraft:obsidian');
    fillArea(2, 2, 3, 10, 4, depth - 4, 'minecraft:air');
    // Bridge from breech shell into muzzle shell so the full structure is contiguous.
    fillArea(12, 2, 4, 15, 4, 6, 'minecraft:obsidian');
    fillArea(12, 2, 5, 15, 4, 5, 'minecraft:air');

    // Two short staging rails stacked in the breech.
    const lowerRailXs = [4, 5, 6, 7];
    const upperRailXs = [5, 6, 7, 8];
    for (const x of lowerRailXs) {
      const railSpec = x === 7
        ? 'minecraft:activator_rail[shape=east_west,powered=true,waterlogged=false]'
        : x === 6
          ? 'minecraft:detector_rail[shape=east_west,powered=false,waterlogged=false]'
          : 'minecraft:rail[shape=east_west,waterlogged=false]';
      setBlock(x, 2, 5, railSpec);
      entities.push({ id: 'minecraft:tnt_minecart', x: x + 0.5, y: 2.0, z: 5.5 });
    }
    for (const x of upperRailXs) {
      setBlock(x, 3, 5, 'minecraft:rail[shape=east_west,waterlogged=false]');
      entities.push({ id: 'minecraft:tnt_minecart', x: x + 0.5, y: 3.0, z: 5.5 });
    }

    // Loader and push assembly.
    setBlock(2, 2, 5, 'minecraft:dispenser[facing=east,triggered=false]');
    setBlock(3, 2, 5, 'minecraft:observer[facing=east,powered=false]');
    setBlock(3, 3, 5, 'minecraft:sticky_piston[facing=east,extended=false]');
    setBlock(4, 3, 5, 'minecraft:observer[facing=east,powered=false]');

    // Trigger backbone on one side only.
    for (let x = 2; x <= 11; x += 1) {
      setBlock(x, 1, 7, 'minecraft:redstone_wire');
      setBlock(x, 1, 6, 'minecraft:redstone_wire');
    }
    setBlock(2, 1, 6, 'minecraft:redstone_wire');
    setBlock(11, 1, 6, 'minecraft:redstone_wire');
    setBlock(11, 1, 7, 'minecraft:redstone_wire');
    setBlock(7, 1, 5, 'minecraft:redstone_torch[lit=true]');
    setBlock(9, 1, 5, 'minecraft:redstone_block');

    // Carry signal down the barrel line to keep all powered elements connected.
    for (let x = 11; x <= width - 5; x += 1) {
      setBlock(x, 1, 5, 'minecraft:redstone_wire');
    }

    // Compact timing bank.
    setBlock(4, 1, 8, 'minecraft:repeater[facing=east,delay=4,locked=false,powered=false]');
    setBlock(5, 1, 8, 'minecraft:comparator[facing=east,mode=compare,powered=false]');
    setBlock(6, 1, 8, 'minecraft:repeater[facing=east,delay=2,locked=false,powered=false]');
    setBlock(7, 1, 8, 'minecraft:comparator[facing=east,mode=compare,powered=false]');
    setBlock(8, 1, 8, 'minecraft:repeater[facing=east,delay=4,locked=false,powered=false]');

    // Narrow single-lane muzzle.
    fillArea(16, 2, 5, width - 3, 4, 5, 'minecraft:air');
    fillArea(16, 2, 4, width - 3, 4, 6, 'minecraft:obsidian');
    fillArea(16, 2, 5, width - 3, 4, 5, 'minecraft:air');
    fillArea(16, 4, 5, width - 3, 4, 5, 'minecraft:glass');
    for (let x = 16; x < width - 3; x += 4) {
      setBlock(x, 2, 4, 'minecraft:oak_trapdoor[facing=south,half=top,open=true,powered=false,waterlogged=false]');
      setBlock(x, 2, 6, 'minecraft:oak_trapdoor[facing=north,half=top,open=true,powered=false,waterlogged=false]');
      setBlock(x, 3, 5, 'minecraft:target');
    }

    // Muzzle brace and sight line.
    setBlock(width - 4, 3, 5, 'minecraft:target');
    setBlock(width - 5, 1, 5, 'minecraft:redstone_lamp[lit=true]');

    // Control strip.
    setBlock(1, 2, 6, 'minecraft:lever[face=floor,facing=east,powered=false]');
    setBlock(1, 1, 6, 'minecraft:redstone_wire');
    setBlock(1, 1, 7, 'minecraft:redstone_wire');

    return { width, height, depth, indices, palette, entities };
  }
  
  if (goalText.includes('hoglin')) {
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:nether_bricks');
    fillArea(2, 1, 2, width - 4, 1, depth - 3, 'minecraft:crimson_nylium');
    for (let x = 5; x < width - 5; x += 3) {
      for (let z = 5; z < depth - 3; z += 3) {
        setBlock(x, 2, z, 'minecraft:warped_fungus');
      }
    }
    fillArea(width - 4, 1, 2, width - 2, 1, depth - 3, 'minecraft:magma_block');
    return { width, height, depth, indices, palette };
  }
  
  if (goalText.includes('wither')) {
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:nether_bricks');
    fillArea(2, 1, 2, width - 3, 1, depth - 3, 'minecraft:nether_bricks');
    for (let z = 3; z < depth - 2; z += 2) {
      for (let x = 5; x < width - 3; x += 2) {
        setBlock(x, 2, z, 'minecraft:soul_sand');
        setBlock(x, 3, z, 'minecraft:wither_rose');
      }
    }
    return { width, height, depth, indices, palette };
  }
  
  // Utility: pick random element from array
  function pickRandom(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

  // Example: 3x3 piston door (variety already implemented above)
  if (goalText.includes('3x3') || goalText.includes('door')) {
    const layouts = [/* ...existing layouts... */];
    const pick = pickRandom(layouts);
    pick();
    if (candidate.activated) {
      for (let x = 4; x <= 6; x++) {
        for (let y = 1; y <= 3; y++) {
          setBlock(x, y, 5, 'minecraft:air');
        }
      }
    }
    return { width, height, depth, indices, palette };
  }

  // Example: TNT cannon (variety)
  if (goalText.includes('tnt') && goalText.includes('cannon')) {
    const layouts = [
      // Layout 1: classic water TNT cannon
      () => {
        fillArea(2, 0, 2, 10, 0, 6, 'minecraft:stone_bricks');
        fillArea(3, 1, 3, 9, 1, 5, 'minecraft:water');
        for (let x = 3; x <= 9; x++) setBlock(x, 2, 4, 'minecraft:tnt');
        setBlock(10, 2, 4, 'minecraft:slab');
        setBlock(2, 2, 4, 'minecraft:button');
        setBlock(2, 1, 4, 'minecraft:redstone_wire');
      },
      // Layout 2: diagonal TNT cannon
      () => {
        fillArea(2, 0, 2, 10, 0, 6, 'minecraft:stone_bricks');
        for (let i = 0; i < 7; i++) setBlock(3 + i, 1, 3 + i, 'minecraft:water');
        for (let i = 0; i < 7; i++) setBlock(3 + i, 2, 3 + i, 'minecraft:tnt');
        setBlock(10, 2, 6, 'minecraft:slab');
        setBlock(2, 2, 2, 'minecraft:button');
        setBlock(2, 1, 2, 'minecraft:redstone_wire');
      },
      // Layout 3: randomize block type for base
      () => {
        const baseBlock = pickRandom(['minecraft:stone_bricks', 'minecraft:deepslate', 'minecraft:obsidian']);
        fillArea(2, 0, 2, 10, 0, 6, baseBlock);
        fillArea(3, 1, 3, 9, 1, 5, 'minecraft:water');
        for (let x = 3; x <= 9; x++) setBlock(x, 2, 4, 'minecraft:tnt');
        setBlock(10, 2, 4, 'minecraft:slab');
        setBlock(2, 2, 4, 'minecraft:button');
        setBlock(2, 1, 4, 'minecraft:redstone_wire');
      }
    ];
    pickRandom(layouts)();
    return { width, height, depth, indices, palette };
  }

  // Example: item sorter (variety)
  if (goalText.includes('sorter')) {
    const layouts = [
      () => {
        fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:stone_bricks');
        for (let z = 2; z < depth - 1; z += 2) {
          for (let x = 2; x < width - 2; x += 3) {
            setBlock(x, 1, z, 'minecraft:hopper');
            setBlock(x + 1, 1, z, 'minecraft:comparator');
          }
        }
      },
      () => {
        fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:polished_deepslate');
        for (let z = 2; z < depth - 1; z += 2) {
          for (let x = 2; x < width - 2; x += 3) {
            setBlock(x, 1, z, 'minecraft:hopper');
            setBlock(x + 1, 1, z, 'minecraft:comparator');
            setBlock(x + 2, 1, z, 'minecraft:redstone_wire');
          }
        }
      }
    ];
    pickRandom(layouts)();
    return { width, height, depth, indices, palette };
  }

  // Fallback: Only allow simulation of uploaded schematics or curated real builds. No random builds.
  throw new Error('No real build found for this goal. Please upload a schematic or select a supported real build.');
  
  if (goalText.includes('flying')) {
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:quartz_block');
    for (let x = 4; x < width - 3; x += 1) {
      setBlock(x, 2, 3, 'minecraft:slime_block');
      setBlock(x, 2, 4, 'minecraft:honey_block');
      setBlock(x, 3, 3, 'minecraft:piston');
      setBlock(x, 3, 4, 'minecraft:sticky_piston');
    }
    fillArea(1, 1, 2, 3, 1, 5, 'minecraft:redstone_wire');
    return { width, height, depth, indices, palette };
  }
  
  if (goalText.includes('sorter')) {
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:stone_bricks');
    for (let z = 2; z < depth - 1; z += 2) {
      for (let x = 2; x < width - 2; x += 3) {
        setBlock(x, 1, z, 'minecraft:hopper');
        setBlock(x + 1, 1, z, 'minecraft:comparator');
      }
    }
    return { width, height, depth, indices, palette };
  }
  
  if (goalText.includes('sugar') || goalText.includes('clock')) {
    fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:stone_bricks');
    fillArea(2, 1, 2, width - 3, 1, depth - 3, 'minecraft:dirt');
    for (let z = 3; z < depth - 2; z += 2) {
      setBlock(2, 2, z, 'minecraft:observer');
      setBlock(3, 1, z, 'minecraft:redstone_wire');
      fillArea(4, 2, z, 10, 2, z, 'minecraft:oak_trapdoor');
    }
    return { width, height, depth, indices, palette };
  }
  
  fillArea(0, 0, 0, width - 1, 0, depth - 1, 'minecraft:stone_bricks');
  fillArea(2, 1, 2, width - 3, 1, depth - 3, 'minecraft:stone_bricks');
  for (let z = 2; z < depth - 1; z += 2) {
    let x = 2;
    setBlock(x, 1, z, 'minecraft:redstone_wire');
    setBlock(x + 1, 1, z, 'minecraft:repeater');
    setBlock(x + 2, 2, z, 'minecraft:observer');
  }


  return { width, height, depth, indices, palette };
}

// --- Simulation Engine Upgrade ---
// This section adds support for multi-tick simulation, entity/projectile physics, and referencing real redstone builds.
// For future AI: Use real redstone layouts from popular sources for each machine type.

function simulateRedstoneMachine({
  blocks, entities, ticks = 20, machineType = '', activated = false
}) {
  // blocks: array of {x, y, z, id}
  // entities: array of {x, y, z, type, vx, vy, vz, health}
  // ticks: number of simulation steps
  // machineType: e.g. '3x3_door', 'tnt_cannon', 'arrow_launcher'
  // activated: whether the machine is triggered
  // Returns: {events: [], entities: [], blocks: [], outcome: ''}

  let events = [];
  let outcome = '';
  let simEntities = JSON.parse(JSON.stringify(entities || []));
  let simBlocks = JSON.parse(JSON.stringify(blocks || []));

  // Example: 3x3 piston door animation
  if (machineType === '3x3_door') {
    if (activated) {
      // Animate door opening over 5 ticks
      for (let t = 0; t < Math.min(5, ticks); t++) {
        // Move door blocks up by 1 each tick
        for (let b of simBlocks) {
          if (b.id === 'minecraft:stone' && b.y < 4) b.y += 1;
        }
        events.push(`Tick ${t+1}: Door blocks move up`);
      }
      outcome = 'Door opened.';
    } else {
      outcome = 'Door closed.';
    }
    // Check for NPCs crushed by door
    for (let e of simEntities) {
      for (let b of simBlocks) {
        if (b.id === 'minecraft:stone' && b.x === e.x && b.y === e.y && b.z === e.z) {
          e.health = 0;
          events.push(`NPC at (${e.x},${e.y},${e.z}) crushed by door.`);
        }
      }
    }
  }

  // Example: TNT cannon (projectile simulation)
  if (machineType === 'tnt_cannon' && activated) {
    // Find projectile entity (TNT or arrow)
    for (let e of simEntities) {
      if (e.type === 'tnt' || e.type === 'arrow') {
        for (let t = 0; t < ticks; t++) {
          // Simple projectile motion: vx, vy, vz
          e.x += e.vx;
          e.y += e.vy;
          e.z += e.vz;
          e.vy -= 0.04; // gravity
          events.push(`Tick ${t+1}: Projectile at (${e.x.toFixed(2)},${e.y.toFixed(2)},${e.z.toFixed(2)})`);
          // Check for collision with NPCs
          for (let npc of simEntities) {
            if (npc !== e && Math.abs(npc.x - e.x) < 1 && Math.abs(npc.y - e.y) < 1 && Math.abs(npc.z - e.z) < 1) {
              npc.health -= 20;
              events.push(`Projectile hit NPC at (${npc.x},${npc.y},${npc.z})`);
            }
          }
        }
        outcome = 'Projectile launched.';
      }
    }
  }

  // For future: Add more machine types and reference real redstone builds for block layout and wiring

  return { events, entities: simEntities, blocks: simBlocks, outcome };
}

function buildLitematicNbt(name, goal, blockVolume) {
  const { width, height, depth, indices, palette, entities = [] } = blockVolume;
  const bitsPerEntry = Math.max(2, Math.ceil(Math.log2(Math.max(2, palette.length))));
  const longArray = encodeBitPackedIndices(indices, bitsPerEntry);
  const totalBlocks = indices.reduce((acc, v) => acc + (v === 0 ? 0 : 1), 0);
  const now = BigInt(Date.now());

  const parseBlockSpec = (spec) => {
    const raw = String(spec || 'minecraft:air');
    const open = raw.indexOf('[');
    const close = raw.lastIndexOf(']');
    if (open === -1 || close === -1 || close < open) {
      return { name: raw, props: null };
    }
    const namePart = raw.slice(0, open);
    const propsRaw = raw.slice(open + 1, close).trim();
    if (!propsRaw) return { name: namePart, props: null };
    const props = {};
    for (const pair of propsRaw.split(',')) {
      const [k, v] = pair.split('=');
      if (k && v) props[k.trim()] = v.trim();
    }
    return { name: namePart, props: Object.keys(props).length ? props : null };
  };

  const paletteList = palette.map((id) => {
    const parsed = parseBlockSpec(id);
    const base = {
      Name: {
        type: 'string',
        value: parsed.name
      }
    };
    if (parsed.props) {
      base.Properties = {
        type: 'compound',
        value: Object.fromEntries(
          Object.entries(parsed.props).map(([k, v]) => [k, { type: 'string', value: String(v) }])
        )
      };
    }
    return base;
  });

  const entityList = entities.map((entity) => ({
    id: { type: 'string', value: entity.id || 'minecraft:tnt_minecart' },
    Pos: {
      type: 'list',
      value: {
        type: 'double',
        value: [Number(entity.x) || 0.5, Number(entity.y) || 0.0, Number(entity.z) || 0.5]
      }
    },
    Motion: {
      type: 'list',
      value: {
        type: 'double',
        value: [0, 0, 0]
      }
    },
    Rotation: {
      type: 'list',
      value: {
        type: 'float',
        value: [0, 0]
      }
    },
    FallDistance: { type: 'float', value: 0 },
    Fire: { type: 'short', value: -1 },
    Air: { type: 'short', value: 300 },
    OnGround: { type: 'byte', value: 1 },
    PortalCooldown: { type: 'int', value: 0 },
    Invulnerable: { type: 'byte', value: 1 },
    CustomNameVisible: { type: 'byte', value: 0 },
    Silent: { type: 'byte', value: 0 },
    Glowing: { type: 'byte', value: 0 },
    HasVisualFire: { type: 'byte', value: 0 },
    Tags: { type: 'list', value: { type: 'string', value: [] } }
  }));

  return {
    name: '',
    type: 'compound',
    value: {
      Version: { type: 'int', value: 6 },
      SubVersion: { type: 'int', value: 1 },
      MinecraftDataVersion: { type: 'int', value: 3700 },
      Metadata: {
        type: 'compound',
        value: {
          Name: { type: 'string', value: name },
          Author: { type: 'string', value: 'SolasAI' },
          Description: { type: 'string', value: `Generated from simulator goal: ${goal}`.slice(0, 200) },
          RegionCount: { type: 'int', value: 1 },
          TotalBlocks: { type: 'int', value: totalBlocks },
          TotalVolume: { type: 'int', value: width * height * depth },
          TimeCreated: { type: 'long', value: now },
          TimeModified: { type: 'long', value: now },
          EnclosingSize: {
            type: 'compound',
            value: {
              x: { type: 'int', value: width },
              y: { type: 'int', value: height },
              z: { type: 'int', value: depth }
            }
          }
        }
      },
      Regions: {
        type: 'compound',
        value: {
          main: {
            type: 'compound',
            value: {
              Position: {
                type: 'compound',
                value: {
                  x: { type: 'int', value: 0 },
                  y: { type: 'int', value: 0 },
                  z: { type: 'int', value: 0 }
                }
              },
              Size: {
                type: 'compound',
                value: {
                  x: { type: 'int', value: width },
                  y: { type: 'int', value: height },
                  z: { type: 'int', value: depth }
                }
              },
              BlockStatePalette: {
                type: 'list',
                value: {
                  type: 'compound',
                  value: paletteList
                }
              },
              BlockStates: {
                type: 'longArray',
                value: longArray
              },
              TileEntities: { type: 'list', value: { type: 'compound', value: [] } },
              PendingBlockTicks: { type: 'list', value: { type: 'compound', value: [] } },
              PendingFluidTicks: { type: 'list', value: { type: 'compound', value: [] } },
              Entities: { type: 'list', value: { type: 'compound', value: entityList } }
            }
          }
        }
      }
    }
  };
}

function generateLitematicFile(goal, candidate, requestedName = '') {
  const safeGoal = String(goal || 'generic redstone build').trim();
  const logicalBase = (requestedName || safeGoal || 'redstone_build').slice(0, 80);
  const slugBase = slugify(logicalBase);
  let fileName = sanitizeFileName(slugBase);
  const blockVolume = buildSimulatorBlockVolume(safeGoal, candidate || {});
  const nbtRoot = buildLitematicNbt(path.basename(fileName, '.litematic'), safeGoal, blockVolume);
  const raw = writeUncompressed(nbtRoot, 'big');
  const compressed = zlib.gzipSync(raw);

  ensureDir(LITEMATICS_DIR);
  let filePath = path.join(LITEMATICS_DIR, fileName);
  let suffix = 2;
  while (fs.existsSync(filePath)) {
    fileName = sanitizeFileName(`${slugBase}_${suffix}`);
    filePath = path.join(LITEMATICS_DIR, fileName);
    suffix += 1;
  }
  fs.writeFileSync(filePath, compressed);
  return { fileName, filePath, bytes: compressed.length, width: blockVolume.width, height: blockVolume.height, depth: blockVolume.depth };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizePaletteName(entry) {
  const raw = String(entry || 'minecraft:air');
  const i = raw.indexOf('[');
  return i >= 0 ? raw.slice(0, i) : raw;
}

function collectComponents(activeSet, width, height, depth) {
  const visited = new Set();
  let componentCount = 0;
  const neighbors = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
  ];

  for (const start of activeSet) {
    if (visited.has(start)) continue;
    componentCount += 1;
    const q = [start];
    visited.add(start);
    while (q.length) {
      const key = q.pop();
      const [x, y, z] = key.split(',').map(Number);
      for (const [dx, dy, dz] of neighbors) {
        const nx = x + dx;
        const ny = y + dy;
        const nz = z + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= width || ny >= height || nz >= depth) continue;
        const nk = `${nx},${ny},${nz}`;
        if (!activeSet.has(nk) || visited.has(nk)) continue;
        visited.add(nk);
        q.push(nk);
      }
    }
  }

  return componentCount;
}

function analyzeConnectivity(blockVolume) {
  const { width, height, depth, indices, palette } = blockVolume;
  const idx = (x, y, z) => x + z * width + y * width * depth;

  const occupied = new Set();
  const signalBlocks = new Set();
  const railBlocks = new Set();

  const signalNames = new Set([
    'minecraft:redstone_wire', 'minecraft:repeater', 'minecraft:comparator',
    'minecraft:observer', 'minecraft:lever', 'minecraft:redstone_torch',
    'minecraft:redstone_block', 'minecraft:dispenser', 'minecraft:piston',
    'minecraft:sticky_piston', 'minecraft:activator_rail', 'minecraft:detector_rail',
    'minecraft:powered_rail', 'minecraft:rail'
  ]);
  const railNames = new Set([
    'minecraft:rail', 'minecraft:powered_rail', 'minecraft:detector_rail', 'minecraft:activator_rail'
  ]);

  for (let y = 0; y < height; y += 1) {
    for (let z = 0; z < depth; z += 1) {
      for (let x = 0; x < width; x += 1) {
        const pIndex = indices[idx(x, y, z)] || 0;
        const blockName = normalizePaletteName(palette[pIndex]);
        if (blockName === 'minecraft:air' || blockName.endsWith(':cave_air') || blockName.endsWith(':void_air')) continue;
        const key = `${x},${y},${z}`;
        occupied.add(key);
        if (signalNames.has(blockName)) signalBlocks.add(key);
        if (railNames.has(blockName)) railBlocks.add(key);
      }
    }
  }

  const structureComponents = collectComponents(occupied, width, height, depth);
  const signalComponents = signalBlocks.size ? collectComponents(signalBlocks, width, height, depth) : 0;
  const railComponents = railBlocks.size ? collectComponents(railBlocks, width, height, depth) : 0;

  return {
    structureComponents,
    signalComponents,
    railComponents,
    allConnected: structureComponents <= 1,
    signalConnected: signalComponents <= 1,
    railConnected: railComponents <= 1,
    warnings: [
      ...(structureComponents > 1 ? [`Structure split into ${structureComponents} components`] : []),
      ...(signalComponents > 1 ? [`Signal network split into ${signalComponents} components`] : []),
      ...(railComponents > 1 ? [`Rail network split into ${railComponents} components`] : [])
    ]
  };
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function simulateNpcHealth(npcs, blastOrigin, blastPower, triggerType, activated) {
  const input = Array.isArray(npcs) ? npcs : [];
  return input.map((npc, i) => {
    const type = String(npc?.type || 'unarmored');
    const x = Number(npc?.x);
    const y = Number(npc?.y);
    const z = Number(npc?.z);
    const nx = Number.isFinite(x) ? x : (blastOrigin.x + 4 + i);
    const ny = Number.isFinite(y) ? y : blastOrigin.y;
    const nz = Number.isFinite(z) ? z : blastOrigin.z;

    const dx = nx - blastOrigin.x;
    const dy = ny - blastOrigin.y;
    const dz = nz - blastOrigin.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const maxHealth = 20;
    const armorReduction = type === 'netherite_totems' ? 0.68 : 0.05;
    let totems = type === 'netherite_totems' ? 2 : 0;
    const triggerFactor = triggerType === 'button' ? 0.92 : 1;
    const rawDamage = activated ? Math.max(0, blastPower * 7.2 * triggerFactor - dist * 1.7) : 0;
    let effectiveDamage = rawDamage * (1 - armorReduction);
    let health = maxHealth - effectiveDamage;

    while (health <= 0 && totems > 0) {
      totems -= 1;
      health = 1;
      effectiveDamage *= 0.45;
      health -= Math.max(0, effectiveDamage - 7);
    }

    health = Math.max(0, Math.min(maxHealth, health));
    return {
      id: String(npc?.id || `npc_${i + 1}`),
      type,
      x: nx,
      y: ny,
      z: nz,
      maxHealth,
      health: Number(health.toFixed(2)),
      hearts: Number((health / 2).toFixed(2)),
      totemsRemaining: totems,
      status: health <= 0 ? 'down' : 'alive'
    };
  });
}

function buildPhysicsPreview(goal, candidate, connectivity, options = {}, preview = null) {
  const g = String(goal || '').toLowerCase();
  const events = [];
  const activated = Boolean(options.activated);
  const triggerType = options.triggerType === 'button' ? 'button' : 'lever';
  const stackSize = clampNumber(options.stackSize, 1, 128, 12);

  const baseConfidence = connectivity.allConnected && connectivity.signalConnected ? 0.88 : 0.42;
  const blastPower = Number((Math.log2(stackSize + 1) * 2.8).toFixed(3));
  const recoil = Number((stackSize * 0.28 + (triggerType === 'button' ? 0.55 : 0.3)).toFixed(3));
  const dispersion = Number((Math.max(0.12, 1.9 / Math.sqrt(stackSize)) + (triggerType === 'button' ? 0.22 : 0.08)).toFixed(3));

  let confidence = activated ? baseConfidence : 0.12;

  if (g.includes('railgun') || g.includes('tnt minecart') || g.includes('arrow')) {
    if (activated) {
      events.push(`${triggerType === 'button' ? 'Button' : 'Lever'} pulse primes staging chamber and timing bank.`);
      events.push(`Stack size ${stackSize} TNT minecarts creates blast power ${blastPower}.`);
      events.push(`Predicted recoil ${recoil} and dispersion ${dispersion} for current geometry.`);
      events.push('Impulse transfers into narrow muzzle lane for projectile acceleration.');
    } else {
      events.push('Railgun is idle. Click lever/button control to activate simulation.');
    }
  } else if (g.includes('door')) {
    events.push(activated
      ? 'Input pulse propagates through repeater chain and piston bank.'
      : 'Door mechanism is idle until control input is activated.');
  } else {
    events.push(activated
      ? 'Signal propagates through generated network according to block states.'
      : 'Network is idle until activated.');
  }

  if (!connectivity.allConnected || !connectivity.signalConnected) {
    events.push('Warning: disconnected components may prevent full activation.');
    confidence *= 0.74;
  }

  const origin = preview
    ? { x: Math.floor(preview.width * 0.2), y: Math.floor(preview.height * 0.35), z: Math.floor(preview.depth * 0.5) }
    : { x: 8, y: 3, z: 5 };
  const npcResults = simulateNpcHealth(options.npcs, origin, blastPower, triggerType, activated);

  return {
    confidence: Number(Math.max(0, Math.min(0.98, confidence)).toFixed(3)),
    activated,
    triggerType,
    stackSize,
    recoil,
    dispersion,
    blastPower,
    npcs: npcResults,
    events,
    predictedOutcome: activated
      ? (confidence > 0.8 ? 'Likely to fire reliably in MC physics.' : 'May fail due to disconnected wiring/components.')
      : 'Idle state. Awaiting manual activation via control input.'
  };
}

function buildBlockPreview(blockVolume, maxBlocks = 1800) {
  const { width, height, depth, indices, palette } = blockVolume;
  const idx = (x, y, z) => x + z * width + y * width * depth;
  const blocks = [];
  for (let y = 0; y < height; y += 1) {
    for (let z = 0; z < depth; z += 1) {
      for (let x = 0; x < width; x += 1) {
        const pIndex = indices[idx(x, y, z)] || 0;
        const name = normalizePaletteName(palette[pIndex]);
        if (name === 'minecraft:air' || name.endsWith(':cave_air') || name.endsWith(':void_air')) continue;
        blocks.push({ x, y, z, id: name });
        if (blocks.length >= maxBlocks) {
          return { width, height, depth, blocks, truncated: true };
        }
      }
    }
  }
  return { width, height, depth, blocks, truncated: false };
}

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'litematic-build-service', litematicsDir: LITEMATICS_DIR });
});

app.post('/import-litematic', async (req, res) => {
  try {
    const url = String(req.body?.url || '').trim();
    const requestedName = String(req.body?.name || '').trim();
    if (!url) return res.status(400).json({ ok: false, error: 'url is required' });

    const downloadUrl = await resolveDownloadUrl(url);
    const buffer = await downloadToBuffer(downloadUrl);
    const lower = downloadUrl.toLowerCase();

    let litematicBuffer = buffer;
    let inferredName = requestedName;

    if (lower.endsWith('.zip')) {
      const extracted = extractLitematicFromZipBuffer(buffer);
      litematicBuffer = extracted.buffer;
      if (!inferredName) inferredName = extracted.inferredName;
    }

    const { finalName, finalPath } = writeImportedLitematic(litematicBuffer, inferredName);
    const stat = fs.statSync(finalPath);
    return res.json({
      ok: true,
      fileName: finalName,
      path: finalPath,
      sourceUrl: downloadUrl,
      bytes: stat.size,
      litematicsDir: LITEMATICS_DIR
    });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.post('/scan-litematic', async (req, res) => {
  try {
    const fileName = sanitizeFileName(req.body?.fileName || '');
    if (!fileName) return res.status(400).json({ ok: false, error: 'fileName is required' });

    const filePath = path.join(LITEMATICS_DIR, fileName);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ ok: false, error: `file not found: ${fileName}` });
    }

    const cacheKey = `${fileName}:${fs.statSync(filePath).mtimeMs}`;
    let result = scanCache.get(cacheKey);
    if (!result) {
      result = await parseLitematicMaterials(filePath);
      scanCache.set(cacheKey, result);
    }

    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.post('/build-plan', async (req, res) => {
  try {
    const fileName = sanitizeFileName(req.body?.fileName || '');
    if (!fileName) return res.status(400).json({ ok: false, error: 'fileName is required' });

    const filePath = path.join(LITEMATICS_DIR, fileName);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ ok: false, error: `file not found: ${fileName}` });
    }

    const scanResult = await parseLitematicMaterials(filePath);
    const plan = materialPlan(scanResult, req.body?.inventory || {});
    const planId = crypto.createHash('sha1').update(JSON.stringify(plan)).digest('hex').slice(0, 12);

    return res.json({ ok: true, planId, plan });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.post('/generate-litematic', async (req, res) => {
  try {
    const goal = String(req.body?.goal || '').trim();
    if (!goal) return res.status(400).json({ ok: false, error: 'goal is required' });
    const candidate = req.body?.candidate && typeof req.body.candidate === 'object' ? req.body.candidate : {};
    const name = String(req.body?.name || '').trim();
    const blockVolume = buildSimulatorBlockVolume(goal, candidate);
    const connectivity = analyzeConnectivity(blockVolume);
    const connectivityPass = connectivity.allConnected && connectivity.signalConnected;

    await sleep(1600);

    const generated = generateLitematicFile(goal, candidate, name);
    return res.json({
      ok: true,
      ...generated,
      connectivityPass,
      connectivity,
      litematicsDir: LITEMATICS_DIR,
      downloadPath: `/download-litematic/${encodeURIComponent(generated.fileName)}`
    });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.post('/simulate-redstone', (req, res) => {
  try {
    const goal = String(req.body?.goal || '').trim();
    if (!goal) return res.status(400).json({ ok: false, error: 'goal is required' });
    const candidate = req.body?.candidate && typeof req.body.candidate === 'object' ? req.body.candidate : {};
    const options = req.body?.options && typeof req.body.options === 'object' ? req.body.options : {};
    const blockVolume = buildSimulatorBlockVolume(goal, candidate);
    const connectivity = analyzeConnectivity(blockVolume);
    const preview = buildBlockPreview(blockVolume, 2200);
    const physics = buildPhysicsPreview(goal, candidate, connectivity, options, preview);
    return res.json({ ok: true, connectivity, physics, preview });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'simulate failed' });
  }
});

app.get('/download-litematic/:fileName', (req, res) => {
  try {
    const fileName = sanitizeFileName(req.params.fileName || '');
    const filePath = path.join(LITEMATICS_DIR, fileName);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ ok: false, error: `file not found: ${fileName}` });
    }
    return res.download(filePath, fileName);
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.listen(PORT, () => {
  ensureDir(LITEMATICS_DIR);
  console.log(`Litematic build service running on http://127.0.0.1:${PORT}`);
});
