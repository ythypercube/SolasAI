
import express from 'express';
import mineflayer from 'mineflayer';
import pathfinderPkg from 'mineflayer-pathfinder';
import minecraftData from 'minecraft-data';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { Vec3 } from 'vec3';
import net from 'node:net';

const { pathfinder, Movements, goals } = pathfinderPkg;
const { GoalFollow, GoalNear } = goals;

function loadPersistentUsername(instanceId) {
  try {
    const filepath = path.join(USERNAME_CACHE_DIR, `${instanceId}.json`);
    if (fs.existsSync(filepath)) {
      const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
      if (data.username && data.expiresAt > Date.now()) {
        return data.username;
      }
    }
  } catch {
    // ignore read errors
  }
  return null;
}

function savePersistentUsername(instanceId, username) {
  try {
    const filepath = path.join(USERNAME_CACHE_DIR, `${instanceId}.json`);
    const data = {
      username,
      createdAt: Date.now(),
      expiresAt: Date.now() + (30 * 24 * 60 * 60 * 1000) // 30 days
    };
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf8');
  } catch {
    // ignore write errors
  }
}


// ===== IMITATION LEARNING SYSTEM =====
const OBSERVATION_DIR = '/tmp/solasai-observations';
if (!fs.existsSync(OBSERVATION_DIR)) {
  fs.mkdirSync(OBSERVATION_DIR, { recursive: true });
}

function recordPlayerAction(botId, playerName, actionType, block, item, timestamp = Date.now()) {
  try {
    const observation = {
      botId,
      playerName,
      actionType, // 'place', 'break', 'use', 'craft', 'etc'
      blockName: block?.name || '',
      blockPos: block?.position ? { x: Math.floor(block.position.x), y: Math.floor(block.position.y), z: Math.floor(block.position.z) } : null,
      itemName: item?.name || '',
      timestamp
    };
    
    const recordPath = path.join(OBSERVATION_DIR, `${botId}-observations.json`);
    let records = [];
    try {
      if (fs.existsSync(recordPath)) {
        records = JSON.parse(fs.readFileSync(recordPath, 'utf8'));
      }
    } catch {
      records = [];
    }
    
    records.push(observation);
    if (records.length > 500) records.splice(0, records.length - 500);
    fs.writeFileSync(recordPath, JSON.stringify(records, null, 2), 'utf8');
  } catch {
    // ignore observation logging errors
  }
}


function loadObservations(botId, limit = 50) {
  try {
    const recordPath = path.join(OBSERVATION_DIR, `${botId}-observations.json`);
    if (fs.existsSync(recordPath)) {
      const records = JSON.parse(fs.readFileSync(recordPath, 'utf8'));
      return records.slice(-limit);
    }
  } catch {
    // ignore read errors
  }
  return [];
}

const app = express();
app.use(express.json({ limit: '1mb' }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(path.join(__dirname, 'public')));
const SOLASAI_SPEAK_SCRIPT = process.env.SOLASAI_SPEAK_SCRIPT
  || path.join(__dirname, '..', 'turbowarp-ai-backend', 'solasai_speak.py');

const PORT = Number(process.env.BOT_SERVICE_PORT || 8789);
const DEFAULT_BACKEND_URL = process.env.MC_AGENT_URL || 'https://solasai-backend.onrender.com/mc-agent';
const DEFAULT_BOT_USERNAME = process.env.DEFAULT_BOT_USERNAME || 'SolasAIBot';
const DEFAULT_BOT_AUTH = process.env.DEFAULT_BOT_AUTH || 'offline';

const SWARM_FIRST_NAMES = [
  'Ari', 'Nova', 'Luna', 'Kai', 'Ezra', 'Iris', 'Milo', 'Nora', 'Zane', 'Rhea', 'Theo', 'Juno', 'Axel', 'Lena', 'Orin', 'Mira'
];
const SWARM_LAST_NAMES = [
  'Stone', 'Ridge', 'Vale', 'River', 'Forge', 'Warden', 'Walker', 'Builder', 'Crafter', 'Miner', 'Scout', 'Keeper', 'Farmer', 'Smith', 'Anchor', 'Trail'
];

const state = {
  bot: null,
  connected: false,
  connecting: false,
  objective: 'general1',
  sessionId: 'bot-default',
  backendUrl: DEFAULT_BACKEND_URL,
  host: '',
  port: 25565,
  username: DEFAULT_BOT_USERNAME,
  auth: DEFAULT_BOT_AUTH,
  connectAttemptAt: 0,
  lastConnectError: '',
  lastDisconnectReason: '',
  connectTimeoutHandle: null,
  tickCounter: 0,
  controlTimer: null,
  lastDecisionAt: 0,
  lastNote: '',
  lastMode: 'general',
  stopping: false,
  movements: null,
  lastHealth: 20,
  revengeTargetId: null,
  revengeExpireAt: 0,
  lastAttackAt: 0,
  lastHelperAt: 0,
  lastDigAt: 0,
  lastStuckAt: 0,
  helperActive: false,
  decisionInFlight: false,
  lastEatAt: 0,
  retreatUntil: 0,
  lastSpawnAt: 0,
  miningInProgress: false,
  noJumpUntil: 0,
  useHeld: false,
  mcData: null,
  mcVersion: '',
  kbRecoveryUntil: 0,
  kbStrafeUntil: 0,
  kbStrafeDir: '',
  nextAttackAllowedAt: 0,
  lastWindChargeUseAt: 0,
  lastBaseScanAt: 0,
  lastBaseRoamAt: 0,
  lastRoamJumpAt: 0,
  nextPullComboAt: 0,
  lastOffhandCheckAt: 0,
  lastOffhandItemName: '',
  eatingUntil: 0,
  combatStrafeDir: '',
  lastCombatStrafeSwitchAt: 0,
  nextMicroPauseAt: 0,
  lastSprintJumpAt: 0,
  noSprintUntil: 0,
  huntCompassGranted: false,
  huntTargetKilled: false,
  huntLastSeenAt: 0,
  huntDimensionSearchStartedAt: 0,
  concurrentTasks: [],
  microPauseUntil: 0,
  strafeLockDir: '',
  strafeLockUntil: 0,
  swarmRole: 'solo',
  swarmWorkerId: '',
  teamInbox: [],
  lastTeamBroadcastAt: 0,
  lastTeamSeenAt: 0,
  maceLastY: 0,
  fallStartY: 0,
  inFallSequence: false,
  lastMaceUsedAt: 0,
  controlSmoothing: {
    forward: { actual: false, desired: false, nextChangeAt: 0 },
    back: { actual: false, desired: false, nextChangeAt: 0 },
    left: { actual: false, desired: false, nextChangeAt: 0 },
    right: { actual: false, desired: false, nextChangeAt: 0 },
    jump: { actual: false, desired: false, nextChangeAt: 0 },
    sprint: { actual: false, desired: false, nextChangeAt: 0 },
    sneak: { actual: false, desired: false, nextChangeAt: 0 }
  },
  lastYawDelta: 0,
  lastPitchDelta: 0,
  lastSwarmPlan: null,
  swarmWorkers: {},
  baseFindingsByServer: {},
  imitationEnabled: true,
  watchingPlayers: {},
  lastPlayerObservationAt: 0,
  persistentInstanceId: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  money: {}, // username -> balance
  lastTaxDay: 0,
  president: null,
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeText(value) {
  return String(value || '').trim();
}

function getChatPlainEndpoint(mcAgentUrl) {
  const base = normalizeText(mcAgentUrl || DEFAULT_BACKEND_URL).replace(/\/+$/, '');
  if (base.endsWith('/mc-agent')) return `${base.slice(0, -'/mc-agent'.length)}/chat-plain`;
  if (base.endsWith('/mc')) return `${base.slice(0, -'/mc'.length)}/chat-plain`;
  return `${base}/chat-plain`;
}

function requestChatReplyPlain(sessionId, message) {
  const endpoint = getChatPlainEndpoint(state.backendUrl);
  const payload = {
    sessionId: normalizeText(sessionId || `voice-${Date.now()}`),
    message: normalizeText(message)
  };

  return fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(async (response) => {
    const body = await response.text();
    if (!response.ok) {
      throw new Error(`chat-plain HTTP ${response.status}: ${body.slice(0, 200)}`);
    }
    return normalizeText(body);
  });
}

function speakText(text, voice = process.env.SOLASAI_VOICE || 'en-US-BrianNeural') {
  const spoken = normalizeText(text);
  if (!spoken) return;

  const args = [SOLASAI_SPEAK_SCRIPT, '--voice', normalizeText(voice), spoken];
  const child = spawn('/usr/bin/python3', args, {
    detached: true,
    stdio: 'ignore'
  });
  child.unref();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function scheduleConcurrentTask(key, taskFn, maxRunMs = 8000) {
  try {
    if (key) {
      const hasRunning = (state.concurrentTasks || []).some((t) => t.key === key);
      if (hasRunning) return null;
    }
    const id = `ct-${Date.now()}-${Math.random().toString(36).slice(2,5)}`;
    const run = async () => {
      try {
        await Promise.race([
          taskFn(),
          new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), maxRunMs))
        ]);
      } catch (e) {
        // Background tasks are best-effort.
      } finally {
        state.concurrentTasks = state.concurrentTasks.filter(t => t.id !== id);
      }
    };
    state.concurrentTasks.push({ id, key: String(key || ''), started: Date.now() });
    run();
    return id;
  } catch {
    return null;
  }
}

function trySprintJump(bot, minMs = 100, maxMs = 140) {
  if (!bot?.entity?.onGround) return false;
  if (state.miningInProgress || inNoJumpWindow()) return false;
  const now = Date.now();
  if (now - Number(state.lastSprintJumpAt || 0) < randomBetween(minMs, maxMs)) return false;
  state.lastSprintJumpAt = now;
  bot.setControlState('jump', true);
  setTimeout(() => {
    try { bot.setControlState('jump', false); } catch {}
  }, 80);
  return true;
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function randomInt(min, maxInclusive) {
  return Math.floor(randomBetween(min, maxInclusive + 1));
}

function sanitizeMcUsername(raw, fallback = 'Solas') {
  const cleaned = String(raw || '')
    .replace(/[^a-zA-Z0-9_]/g, '')
    .replace(/^_+/, '')
    .slice(0, 16);
  if (!cleaned) return fallback;
  return cleaned;
}

function generateSwarmUsername(mode, baseUsername, index, used = new Set()) {
  const safeBase = sanitizeMcUsername(baseUsername || 'Solas', 'Solas');
  let candidate = safeBase;

  if (mode === 'random_mc') {
    const alpha = 'abcdefghijklmnopqrstuvwxyz';
    const nums = '0123456789';
    const len = randomInt(8, 14);
    let name = '';
    name += alpha[randomInt(0, alpha.length - 1)].toUpperCase();
    for (let i = 1; i < len; i++) {
      const source = Math.random() > 0.72 ? nums : alpha;
      name += source[randomInt(0, source.length - 1)];
    }
    candidate = sanitizeMcUsername(name, `Solas${index + 1}`);
  } else if (mode === 'random_name') {
    const first = SWARM_FIRST_NAMES[randomInt(0, SWARM_FIRST_NAMES.length - 1)];
    const last = SWARM_LAST_NAMES[randomInt(0, SWARM_LAST_NAMES.length - 1)];
    candidate = sanitizeMcUsername(`${first}${last}`, `Solas${index + 1}`);
  } else {
    candidate = sanitizeMcUsername(`${safeBase}${index + 1}`, `Solas${index + 1}`);
  }

  if (!used.has(candidate)) {
    used.add(candidate);
    return candidate;
  }

  for (let attempt = 2; attempt <= 9999; attempt++) {
    const tryName = sanitizeMcUsername(`${candidate}${attempt}`, `Solas${index + 1}_${attempt}`);
    if (!used.has(tryName)) {
      used.add(tryName);
      return tryName;
    }
  }

  const fallback = sanitizeMcUsername(`Solas${Date.now()}${index}`, `Solas${index + 1}`);
  used.add(fallback);
  return fallback;
}

function splitSwarmJobs(rawJobs) {
  const input = String(rawJobs || '').trim();
  if (!input) return ['miner', 'builder', 'farmer', 'treasurer'];
  const jobs = input
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .slice(0, 64);
  // Remove 'warrior' if present, ensure 'treasurer' is present
  const filtered = jobs.filter(j => j !== 'warrior');
  if (!filtered.includes('treasurer')) filtered.push('treasurer');
  return filtered.length > 0 ? filtered : ['miner', 'builder', 'farmer', 'treasurer'];
}

function buildSwarmPlan(config = {}) {
  const count = clamp(Number(config.count || 1), 1, 500);
  const host = normalizeText(config.host || state.host || '');
  const port = Number(config.port || state.port || 25565);
  const auth = normalizeText(config.auth || state.auth || DEFAULT_BOT_AUTH) || 'offline';
  const objective = normalizeText(config.objective || state.objective || 'general1');
  const baseUsername = normalizeText(config.baseUsername || 'Solas');
  const mode = normalizeText(config.usernameMode || 'numbered').toLowerCase();
  const jobs = splitSwarmJobs(config.jobs);
  const autoThink = Boolean(config.autoThink ?? true);

  const used = new Set();
  const bots = [];
  for (let i = 0; i < count; i++) {
    const username = generateSwarmUsername(mode, baseUsername, i, used);
    let role, roleObjective;
    if (i === 0) {
      role = 'president';
      roleObjective = 'Lead the team, coordinate tasks, and make decisions. Assign tasks to others and ensure the civilization thrives. Buy ores from miners and use them to gear up the civilization.';
    } else {
      role = jobs[(i - 1) % jobs.length];
      switch (role) {
        case 'miner':
          roleObjective = 'Mine wood, stone, iron, gold, lapis, redstone, diamond, and netherite. Sell ores to the president and resources to builders.';
          break;
        case 'builder':
          roleObjective = 'Build structures for the team: walls, towers, and houses. Use available blocks. Buy resources from miners.';
          break;
        case 'farmer':
          roleObjective = 'Farm crops and breed animals. Ensure food supply for the team. Sell food to all.';
          break;
        case 'treasurer':
          roleObjective = 'Collect rare and elite items (mace, trident, enchanted books, netherite, etc). Explore, loot, and trade for valuables. Act as a wandering trader for rare items.';
          break;
        default:
          roleObjective = `${objective}. Role: ${role}. Think independently, plan short-term tasks yourself, collaborate with nearby bots via chat, share resources, and self-assign useful tasks.`;
      }
    }
    bots.push({
      id: `bot-${i + 1}`,
      username,
      role,
      objective: roleObjective,
      host,
      port,
      auth,
      autoThink,
      chatReadEnabled: true
    });
  }

  return {
    createdAt: Date.now(),
    host,
    port,
    auth,
    count,
    mode,
    autoThink,
    jobs,
    bots
  };
}

function rememberTeamMessage(message) {
  const text = normalizeText(message);
  if (!text) return;
  state.teamInbox.push({ text, at: Date.now() });
  if (state.teamInbox.length > 24) {
    state.teamInbox.splice(0, state.teamInbox.length - 24);
  }
}

function maybeBroadcastTeamStatus(bot) {
  // Team chat protocol disabled to keep combat behavior and notes clean.
  return;
}

function listSwarmWorkers() {
  return Object.values(state.swarmWorkers || {}).sort((a, b) => a.id.localeCompare(b.id));
}

async function stopSwarmWorkers() {
  const workers = listSwarmWorkers();
  if (workers.length === 0) return { stopped: 0 };

  for (const worker of workers) {
    try {
      process.kill(worker.pid, 'SIGTERM');
      worker.status = 'stopping';
    } catch {
      worker.status = 'stopped';
    }
  }

  await sleep(700);

  for (const worker of workers) {
    if (worker.status === 'stopped') continue;
    try {
      process.kill(worker.pid, 0);
      process.kill(worker.pid, 'SIGKILL');
      worker.status = 'stopped';
    } catch {
      worker.status = 'stopped';
    }
  }

  state.swarmWorkers = {};
  return { stopped: workers.length };
}

async function launchSwarmWorkers(plan, options = {}) {
  const launchCount = clamp(Number(options.launchCount || plan.count || 1), 1, plan.count || 1);
  const basePort = Number(options.basePort || 8800);
  const backendUrl = normalizeText(options.backendUrl || state.backendUrl || DEFAULT_BACKEND_URL);

  const startedWorkers = [];
  const indexPath = path.join(__dirname, 'index.js');

  for (let i = 0; i < launchCount; i++) {
    const bot = plan.bots[i];
    const workerPort = basePort + i;
    const workerId = bot.id || `bot-${i + 1}`;
    const safeUser = sanitizeMcUsername(bot.username || `Solas${i + 1}`, `Solas${i + 1}`);
    const sessionId = `swarm-${safeUser}-${Date.now()}-${i}`;
    const logFile = `/tmp/solasai-swarm-worker-${workerPort}-${safeUser}.log`;

    const outStream = fs.createWriteStream(logFile, { flags: 'a' });
    const child = spawn(process.execPath, [indexPath], {
      env: {
        ...process.env,
        BOT_SERVICE_PORT: String(workerPort),
        MC_AGENT_URL: backendUrl,
        DEFAULT_BOT_USERNAME: safeUser,
        DEFAULT_BOT_AUTH: String(bot.auth || DEFAULT_BOT_AUTH),
        SOLASAI_AUTOSTART: '1',
        SOLASAI_AUTOSTART_HOST: String(bot.host || plan.host || ''),
        SOLASAI_AUTOSTART_PORT: String(bot.port || plan.port || 25565),
        SOLASAI_AUTOSTART_USERNAME: safeUser,
        SOLASAI_AUTOSTART_AUTH: String(bot.auth || DEFAULT_BOT_AUTH),
        SOLASAI_AUTOSTART_OBJECTIVE: String(bot.objective || plan.objective || state.objective || 'general1'),
        SOLASAI_AUTOSTART_BACKEND_URL: backendUrl,
        SOLASAI_AUTOSTART_SESSION_ID: sessionId,
        SOLASAI_AUTOSTART_ROLE: String(bot.role || 'worker'),
        SOLASAI_AUTOSTART_WORKER_ID: String(workerId)
      },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    child.stdout?.pipe(outStream);
    child.stderr?.pipe(outStream);

    const record = {
      id: workerId,
      pid: child.pid,
      port: workerPort,
      username: safeUser,
      role: bot.role || 'worker',
      host: bot.host || plan.host,
      gamePort: bot.port || plan.port,
      status: 'starting',
      startedAt: Date.now(),
      logFile
    };
    state.swarmWorkers[workerId] = record;

    child.on('exit', (code, signal) => {
      const current = state.swarmWorkers[workerId];
      if (!current) return;
      current.status = 'stopped';
      current.exitCode = code;
      current.exitSignal = signal;
      current.stoppedAt = Date.now();
    });

    startedWorkers.push(record);
    await sleep(120);
  }

  await sleep(900);
  for (const worker of startedWorkers) {
    try {
      process.kill(worker.pid, 0);
      worker.status = 'running';
    } catch {
      worker.status = 'failed';
    }
  }

  return startedWorkers;
}

function inKbRecoveryWindow() {
  return Date.now() < state.kbRecoveryUntil;
}

function inNoJumpWindow() {
  return Date.now() < Number(state.noJumpUntil || 0);
}

function clearMovementControls(bot) {
  const controls = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'];
  for (const key of controls) {
    bot.setControlState(key, false);
    const memory = state.controlSmoothing?.[key];
    if (memory) {
      memory.actual = false;
      memory.desired = false;
      memory.nextChangeAt = 0;
    }
  }
}

function setControlStateSmoothed(bot, key, targetValue, holdOnMs = 120, holdOffMs = 90) {
  if (!state.controlSmoothing[key]) {
    state.controlSmoothing[key] = { actual: false, desired: false, nextChangeAt: 0 };
  }

  const control = state.controlSmoothing[key];
  const now = Date.now();
  const target = Boolean(targetValue);

  if (control.desired !== target) {
    control.desired = target;
    control.nextChangeAt = now + (target ? holdOnMs : holdOffMs);
  }

  if (control.actual !== control.desired && now >= control.nextChangeAt) {
    control.actual = control.desired;
    bot.setControlState(key, control.actual);
  }
}

function objectiveText() {
  return normalizeText(state.objective).toLowerCase();
}

function parseHuntObjective() {
  const raw = normalizeText(state.objective);
  const match = raw.match(/\bhunt\s+([A-Za-z0-9_]+)/i);
  if (!match) return null;
  return normalizeText(match[1]);
}

function isNamedHuntObjective() {
  return Boolean(parseHuntObjective());
}

function resetHuntState() {
  state.huntCompassGranted = false;
  state.huntTargetKilled = false;
  state.huntLastSeenAt = 0;
  state.huntDimensionSearchStartedAt = 0;
}

function isBasefinderObjective() {
  const text = objectiveText();
  return /\bbasefinder\b/.test(text);
}

function currentServerKey() {
  const host = normalizeText(state.host || '').toLowerCase();
  const port = Number(state.port || 25565);
  if (!host) return 'unknown:25565';
  return `${host}:${port}`;
}

function getBaseFindingsForServer(serverKey = currentServerKey()) {
  if (!state.baseFindingsByServer[serverKey]) {
    state.baseFindingsByServer[serverKey] = [];
  }
  return state.baseFindingsByServer[serverKey];
}

function recordBaseCandidate(kind, position, confidence = 0.5, details = '') {
  if (!position || !Number.isFinite(position.x) || !Number.isFinite(position.y) || !Number.isFinite(position.z)) {
    return;
  }

  const serverKey = currentServerKey();
  const list = getBaseFindingsForServer(serverKey);
  const x = Math.floor(position.x);
  const y = Math.floor(position.y);
  const z = Math.floor(position.z);
  const now = Date.now();
  const baseClusterRadiusBlocks = 64; // 4 chunks

  const existing = list.find((entry) =>
    Math.hypot(entry.x - x, entry.z - z) <= baseClusterRadiusBlocks
  );

  if (existing) {
    if (Number(confidence || 0) >= existing.confidence) {
      existing.x = x;
      existing.y = y;
      existing.z = z;
      existing.kind = kind;
    }
    existing.confidence = Math.max(existing.confidence, Number(confidence || 0));
    existing.lastSeenAt = now;
    if (details && !existing.details) {
      existing.details = details;
    } else if (details && existing.details && !existing.details.includes(details)) {
      existing.details = `${existing.details}; ${details}`;
    }
    if (kind && existing.kind && kind !== existing.kind && !String(existing.kind).includes('+')) {
      existing.kind = `${existing.kind}+${kind}`;
    }
    return;
  }

  list.push({
    kind,
    x,
    y,
    z,
    confidence: Number(confidence || 0),
    details: String(details || ''),
    firstSeenAt: now,
    lastSeenAt: now
  });

  if (list.length > 300) {
    list.splice(0, list.length - 300);
  }
}

function getBaseCandidates(serverKey = currentServerKey()) {
  const list = getBaseFindingsForServer(serverKey).slice();
  list.sort((a, b) => b.confidence - a.confidence || b.lastSeenAt - a.lastSeenAt);
  return list;
}

function isGeneral1Objective() {
  const text = objectiveText();
  return /\bgeneral\s*1\b|\bgeneral1\b/.test(text);
}

function isCollectObjective() {
  const text = objectiveText();
  return /\bcollect\b|\bgather\b|\bfarm\b|\bresource\b|\bmaterials?\b|\bloot\b/.test(text);
}

function isTreeObjective() {
  const text = objectiveText();
  return /(\btree\b|\blog\b|\bwood\b|\bchop\b)/.test(text);
}

function isStoneObjective() {
  const text = objectiveText();
  return /(\bstone\b|\bcobble\b|\bcobbled\b|\bdeepslate\b|\bmine stone\b)/.test(text);
}

function hasCraftObjective() {
  const text = objectiveText();
  return /\bcraft|plank|stick|crafting table|workbench\b/.test(text);
}

function isCombatObjective() {
  const text = objectiveText();
  return /\battack\b|\bkill\b|\bhunt\b|\bpvp\b|\bfight\b|\bcombat\b|\braid\b/.test(text);
}

function getPlayerEntityByName(bot, username) {
  const wanted = normalizeText(username).toLowerCase();
  if (!wanted) return null;
  return getNearestEntity(bot, (entity) =>
    entity?.type === 'player'
      && normalizeText(entity?.username).toLowerCase() === wanted
      && entity?.id !== bot.entity?.id
      && entity?.position
  ).entity;
}

function isPrisonEscapeObjective() {
  const text = objectiveText();
  return /\bescape\b.*\bprison\b|\bprison\b.*\bescape\b|\bbreak\s*out\b|\bescape\b.*\bjail\b|\bjail\b.*\bescape\b/.test(text);
}

function isAirBlock(block) {
  return !block || String(block.name || '') === 'air';
}

function likelyOpenAround(bot) {
  const base = bot.entity.position.floored();
  const dirs = [
    new Vec3(1, 0, 0),
    new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1),
    new Vec3(0, 0, -1)
  ];
  let openSides = 0;
  for (const dir of dirs) {
    const feet = bot.blockAt(base.offset(dir.x, 0, dir.z));
    const head = bot.blockAt(base.offset(dir.x, 1, dir.z));
    if (isAirBlock(feet) && isAirBlock(head)) {
      openSides += 1;
    }
  }
  return openSides;
}

function scoreEscapeBlock(name) {
  const n = String(name || '').toLowerCase();
  if (!n || n === 'air') return -1;
  if (/bedrock|barrier|command_block|structure_block|end_portal_frame/.test(n)) return -1;
  if (/iron_bars|glass_pane|glass|trapdoor|door|fence|fence_gate|ladder/.test(n)) return 95;
  if (/wool|planks|wood|log|leaves|hay/.test(n)) return 88;
  if (/dirt|sand|gravel|clay|snow/.test(n)) return 84;
  if (/stone|cobblestone|deepslate|brick|terracotta/.test(n)) return 76;
  if (/obsidian|ancient_debris/.test(n)) return 8;
  return 55;
}

function findPrisonEscapeBlock(bot, radius = 5) {
  const base = bot.entity.position.floored();
  let best = null;
  let bestScore = -1;

  for (let dx = -radius; dx <= radius; dx += 1) {
    for (let dz = -radius; dz <= radius; dz += 1) {
      for (let dy = -1; dy <= 2; dy += 1) {
        const pos = base.offset(dx, dy, dz);
        const block = bot.blockAt(pos);
        if (!block || block.type === 0) continue;
        if (typeof bot.canDigBlock === 'function' && !bot.canDigBlock(block)) continue;

        const nameScore = scoreEscapeBlock(block.name);
        if (nameScore < 0) continue;

        const dist = bot.entity.position.distanceTo(block.position);
        if (dist > radius + 1.8 || dist < 0.9) continue;

        let openingBonus = 0;
        const neighbors = [
          block.position.offset(1, 0, 0),
          block.position.offset(-1, 0, 0),
          block.position.offset(0, 0, 1),
          block.position.offset(0, 0, -1)
        ];
        for (const nPos of neighbors) {
          const nb = bot.blockAt(nPos);
          if (isAirBlock(nb)) openingBonus += 6;
        }

        const score = nameScore + openingBonus - (dist * 2.2) + (Math.abs(dy) <= 1 ? 6 : 0);
        if (score > bestScore) {
          bestScore = score;
          best = block;
        }
      }
    }
  }

  return best;
}

function findAdjacentBlockingEscapeBlock(bot) {
  const base = bot.entity.position.floored();
  const probes = [
    base.offset(1, 0, 0), base.offset(-1, 0, 0), base.offset(0, 0, 1), base.offset(0, 0, -1),
    base.offset(1, 1, 0), base.offset(-1, 1, 0), base.offset(0, 1, 1), base.offset(0, 1, -1),
    base.offset(0, 2, 0)
  ];
  let best = null;
  let bestScore = -1;
  for (const pos of probes) {
    const block = bot.blockAt(pos);
    if (!block || block.type === 0) continue;
    if (typeof bot.canDigBlock === 'function' && !bot.canDigBlock(block)) continue;
    const score = scoreEscapeBlock(block.name);
    if (score > bestScore) {
      bestScore = score;
      best = block;
    }
  }
  return best;
}

function escapeToolNamesForBlock(blockName) {
  const n = String(blockName || '').toLowerCase();
  if (/wood|log|planks|fence|door|trapdoor|ladder/.test(n)) {
    return ['netherite_axe', 'diamond_axe', 'iron_axe', 'stone_axe', 'wooden_axe'];
  }
  if (/dirt|sand|gravel|clay|snow/.test(n)) {
    return ['netherite_shovel', 'diamond_shovel', 'iron_shovel', 'stone_shovel', 'wooden_shovel'];
  }
  return ['netherite_pickaxe', 'diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe'];
}

async function runPrisonEscapeMode(bot) {
  const immediateEscapeBlock = findAdjacentBlockingEscapeBlock(bot);
  if (immediateEscapeBlock) {
    state.lastNote = `prison_escape: breaking nearby ${immediateEscapeBlock.name}`;
    await mineBlockWithTool(bot, immediateEscapeBlock, escapeToolNamesForBlock(immediateEscapeBlock.name));
    return true;
  }

  const escapeBlock = findPrisonEscapeBlock(bot, 6);
  if (escapeBlock) {
    state.lastNote = `prison_escape: breaking ${escapeBlock.name}`;
    await mineBlockWithTool(bot, escapeBlock, escapeToolNamesForBlock(escapeBlock.name));
    return true;
  }

  const openSides = likelyOpenAround(bot);
  if (openSides <= 1 && await tryPillarUp(bot)) {
    state.lastNote = 'prison_escape: pillar up';
    return true;
  }

  if (bot.pathfinder && state.movements) {
    const roamX = bot.entity.position.x + (Math.random() * 32 - 16);
    const roamZ = bot.entity.position.z + (Math.random() * 32 - 16);
    bot.pathfinder.setMovements(state.movements);
    bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
  }
  bot.setControlState('sprint', true);
  if (bot.entity.onGround && Date.now() - Number(state.lastRoamJumpAt || 0) > randomBetween(320, 560)) {
    state.lastRoamJumpAt = Date.now();
    bot.setControlState('jump', true);
    setTimeout(() => {
      try { bot.setControlState('jump', false); } catch {}
    }, 110);
  }
  state.lastNote = 'prison_escape: sprinting to open terrain';
  return true;
}

function isHostileMob(entity) {
  if (!entity) return false;
  const name = String(entity.name || '').toLowerCase();
  return /zombie|skeleton|creeper|spider|enderman|witch|phantom|blaze|piglin|hoglin|ravager|warden|ghast|guardian|pillager|vindicator|evoker|shulker|magma_cube|silverfish|endermite|vex|bee|wolf|cave_spider|drowned|husk|stray|bogged|breeze/.test(name);
}

function getNearestEntity(bot, predicate) {
  let best = null;
  let bestDist = Infinity;
  for (const entity of Object.values(bot.entities)) {
    if (!entity || !entity.position) continue;
    if (!predicate(entity)) continue;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist < bestDist) {
      best = entity;
      bestDist = dist;
    }
  }
  return { entity: best, distance: Number.isFinite(bestDist) ? bestDist : -1 };
}

function isLikelyPlaceableBlockName(name) {
  const lower = String(name || '').toLowerCase();
  if (!lower) return false;
  return /(_log|_wood|_planks|_slab|_stairs|_wall|_fence|_door|_trapdoor|_block|_bricks)$/.test(lower)
    || /stone|cobblestone|cobbled_deepslate|deepslate|dirt|grass_block|sand|gravel|netherrack|obsidian|furnace|crafting_table|chest/.test(lower);
}

function getEntityById(bot, id) {
  if (id == null) return null;
  for (const entity of Object.values(bot.entities)) {
    if (entity && entity.id === id) return entity;
  }
  return null;
}

function canLikelyWinFight(bot, entity, distance) {
  const myHealth = Number(bot.health || 20);
  const enemyHealth = Number(entity?.health || 10);
  const closeEnough = Number(distance) > 0 && Number(distance) < 10;
  const myGear = estimateBotCombatPower(bot);
  const enemyGear = estimateEntityCombatPower(entity);
  const healthEdge = myHealth - enemyHealth;
  const gearEdge = myGear - enemyGear;
  return closeEnough && myHealth >= 7 && (healthEdge >= -1 || myHealth >= 14) && (gearEdge >= -8 || healthEdge >= 4);
}

function armorTierScore(name) {
  const lower = String(name || '').toLowerCase();
  const materialScores = {
    netherite: 7,
    diamond: 6,
    iron: 5,
    chainmail: 4,
    golden: 3,
    gold: 3,
    leather: 2
  };
  const tier = Object.keys(materialScores).find((material) => lower.startsWith(`${material}_`));
  return tier ? materialScores[tier] : 0;
}

function estimateEntityCombatPower(entity) {
  if (!entity) return 0;
  let score = meleeWeaponScore(entity.heldItem?.name);
  const equipment = Array.isArray(entity.equipment) ? entity.equipment : [];
  for (const item of equipment) {
    const name = String(item?.name || '').toLowerCase();
    if (/_helmet|_chestplate|_leggings|_boots/.test(name)) {
      score += armorTierScore(name) * 12;
    }
  }
  return score;
}

function estimateBotCombatPower(bot) {
  if (!bot?.inventory) return meleeWeaponScore(bot?.heldItem?.name);
  let score = meleeWeaponScore(bot.heldItem?.name);
  for (const slot of [5, 6, 7, 8]) {
    const name = String(bot.inventory?.slots?.[slot]?.name || '').toLowerCase();
    if (/_helmet|_chestplate|_leggings|_boots/.test(name)) {
      score += armorTierScore(name) * 12;
    }
  }
  return score;
}

function isThreatReachable(bot, entity, maxYDiff = 3.25) {
  if (!bot?.entity?.position || !entity?.position) return false;
  const yDiff = Math.abs(Number(entity.position.y || 0) - Number(bot.entity.position.y || 0));
  if (yDiff > maxYDiff) return false;
  if (typeof bot.canSeeEntity === 'function') {
    try {
      return Boolean(bot.canSeeEntity(entity));
    } catch {
      return true;
    }
  }
  return true;
}

function isEntityRunningAway(bot, entity) {
  if (!bot?.entity?.position || !entity?.position) return false;

  const velX = Number(entity.velocity?.x || 0);
  const velZ = Number(entity.velocity?.z || 0);
  const speed = Math.hypot(velX, velZ);
  if (speed < 0.06) return false;

  const toTargetX = entity.position.x - bot.entity.position.x;
  const toTargetZ = entity.position.z - bot.entity.position.z;
  const dist = Math.hypot(toTargetX, toTargetZ);
  if (dist < 0.001) return false;

  const awayDir = ((velX / speed) * (toTargetX / dist)) + ((velZ / speed) * (toTargetZ / dist));
  return awayDir > 0.58;
}

async function primePullCombo(bot, target) {
  if (!target?.position) return false;
  const now = Date.now();
  if (now < Number(state.nextPullComboAt || 0)) return false;

  const dist = bot.entity.position.distanceTo(target.position);
  if (dist < 2.0 || dist > 4.4) return false;
  if (!isEntityRunningAway(bot, target)) return false;

  state.nextPullComboAt = now + randomBetween(1200, 1900);
  state.lastNote = 'combat: pull combo on fleeing target';

  try {
    const backYaw = bot.entity.yaw + Math.PI + randomBetween(-0.12, 0.12);
    const backPitch = clamp(bot.entity.pitch + randomBetween(-0.04, 0.05), -Math.PI / 2, Math.PI / 2);
    await bot.look(backYaw, backPitch, true);
    await sleep(randomBetween(24, 48));
    await bot.lookAt(target.position.offset(0, target.height ? target.height * 0.62 : 1.2, 0), true);
    await sleep(randomBetween(16, 38));
    return true;
  } catch {
    return false;
  }
}

function nearestThreat(bot) {
  const playerThreat = getNearestEntity(bot, (entity) =>
    entity.type === 'player' &&
    entity.username &&
    entity.username !== bot.username &&
    isThreatReachable(bot, entity, 4.5)
  );
  if (playerThreat.entity && playerThreat.distance > 0 && playerThreat.distance < 12) {
    return playerThreat;
  }
  return getNearestEntity(bot, (entity) => isHostileMob(entity) && entity.position && isThreatReachable(bot, entity, 3.25));
}

function selectCombatTarget(bot, allowPlayerAggro = false) {
  const revengeTarget = getEntityById(bot, state.revengeTargetId);
  if (revengeTarget && Date.now() < Number(state.revengeExpireAt || 0) && revengeTarget.position) {
    return revengeTarget;
  }

  if (allowPlayerAggro) {
    const playerThreat = getNearestEntity(bot, (entity) =>
      entity?.type === 'player'
        && entity?.username
        && entity?.username !== bot.username
        && entity?.position
        && isThreatReachable(bot, entity, 4.5)
        && canLikelyWinFight(bot, entity, bot.entity.position.distanceTo(entity.position))
    ).entity;
    if (playerThreat) return playerThreat;
  }

  const hostileThreat = getNearestEntity(bot, (entity) => isHostileMob(entity) && entity.position && isThreatReachable(bot, entity, 3.25)).entity;
  if (hostileThreat) return hostileThreat;

  return allowPlayerAggro ? nearestThreat(bot).entity : null;
}

function nearestDroppedItemEntity(bot, maxDistance = 10) {
  return getNearestEntity(bot, (entity) => entity.name === 'item' && entity.position && bot.entity.position.distanceTo(entity.position) <= maxDistance);
}

function nearestFoodAnimal(bot, maxDistance = 28) {
  return getNearestEntity(bot, (entity) => {
    if (!entity?.position) return false;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist > maxDistance) return false;
    const name = String(entity.name || '').toLowerCase();
    return /cow|pig|sheep|chicken|rabbit|salmon|cod|tropical_fish/.test(name);
  });
}

function nearestLogBlock(bot, maxDistance = 24) {
  try {
    return bot.findBlock({
      maxDistance,
      matching: (block) => Boolean(block?.name) && /(_log|_wood)$/.test(block.name)
    });
  } catch {
    return null;
  }
}

function getMcData(bot) {
  if (!state.mcData || state.mcVersion !== bot.version) {
    state.mcData = minecraftData(bot.version);
    state.mcVersion = bot.version;
  }
  return state.mcData;
}

function countItemsByName(bot, names) {
  const wanted = new Set(names);
  let total = 0;
  for (const item of bot.inventory?.items?.() || []) {
    if (wanted.has(String(item.name || ''))) {
      total += Number(item.count || 0);
    }
  }
  return total;
}

function countItemsByRegex(bot, regex) {
  let total = 0;
  for (const item of bot.inventory?.items?.() || []) {
    if (regex.test(String(item.name || ''))) {
      total += Number(item.count || 0);
    }
  }
  return total;
}

function inventoryHasItem(bot, name) {
  return countItemsByName(bot, [name]) > 0;
}

function findInventoryItem(bot, matcher) {
  for (const item of bot.inventory?.items?.() || []) {
    if (matcher(item)) return item;
  }
  return null;
}

async function equipToolByNames(bot, toolNames) {
  const tool = findInventoryItem(bot, (item) => toolNames.includes(String(item.name || '')));
  if (!tool) return false;
  try {
    await bot.equip(tool, 'hand');
    return true;
  } catch {
    return false;
  }
}

// ===== MACE SELECTION =====
function selectMaceForFallingDamage(bot) {
  // When falling from height, use density mace for maximum damage
  const facts = collectInventoryFacts(bot);
  if (facts.densityMaceSlot >= 0) {
    return facts.densityMaceSlot;
  }
  if (facts.maceSlot >= 0) {
    return facts.maceSlot;
  }
  return -1;
}

function selectMaceForBreachSwap(bot) {
  // Use breach mace for block breaking and breach swapping
  const facts = collectInventoryFacts(bot);
  if (facts.breachMaceSlot >= 0) {
    return facts.breachMaceSlot;
  }
  if (facts.maceSlot >= 0) {
    return facts.maceSlot;
  }
  return -1;
}

function weaponTierScore(name) {
  const lower = String(name || '').toLowerCase();
  const materialScores = {
    netherite: 7,
    diamond: 6,
    iron: 5,
    stone: 4,
    golden: 3,
    gold: 3,
    wooden: 2,
    wood: 2
  };
  const tier = Object.keys(materialScores).find((material) => lower.startsWith(`${material}_`));
  return tier ? materialScores[tier] : 1;
}

function meleeWeaponScore(name) {
  const lower = String(name || '').toLowerCase();
  if (lower.endsWith('_sword')) return 100 + weaponTierScore(lower);
  if (lower === 'density_mace') return 92;
  if (lower === 'breach_mace') return 91;
  if (lower.endsWith('mace')) return 90;
  if (lower.endsWith('_axe')) return 70 + weaponTierScore(lower);
  if (lower.endsWith('_trident')) return 60;
  return -1;
}

function selectBestMeleeWeapon(bot) {
  let bestSlot = -1;
  let bestScore = -1;
  for (const item of bot.inventory?.items?.() || []) {
    const invSlot = Number(item?.slot);
    if (invSlot < 36 || invSlot > 44) continue;
    const score = meleeWeaponScore(item?.name);
    if (score > bestScore) {
      bestScore = score;
      bestSlot = invSlot - 36;
    }
  }
  return bestSlot;
}

async function equipBestMeleeWeapon(bot) {
  // Prefer axe if sword is lower or equal tier
  let bestAxe = null, bestSword = null, bestAxeScore = -1, bestSwordScore = -1;
  for (const item of bot.inventory?.items?.() || []) {
    const name = String(item?.name || '');
    if (name.endsWith('_axe')) {
      const score = weaponTierScore(name);
      if (score > bestAxeScore) {
        bestAxeScore = score;
        bestAxe = item;
      }
    } else if (name.endsWith('_sword')) {
      const score = weaponTierScore(name);
      if (score > bestSwordScore) {
        bestSwordScore = score;
        bestSword = item;
      }
    }
  }
  let bestItem = null;
  if (bestAxe && (!bestSword || bestAxeScore >= bestSwordScore)) {
    bestItem = bestAxe;
  } else if (bestSword) {
    bestItem = bestSword;
  }
  if (!bestItem) return false;
  try {
    await bot.equip(bestItem, 'hand');
    return true;
  } catch {
    return false;
  }
}

function shouldBlockWithShield(bot, threat) {
  // Block if low health or enemy is aiming/charging
  if (!bot) return false;
  const health = Number(bot.health || 20);
  if (health < 13) return true;
  if (threat && threat.type === 'player' && threat.heldItem && /bow|crossbow|trident/.test(String(threat.heldItem.name || ''))) return true;
  // Randomly block sometimes in melee
  if (threat && Math.random() < 0.18) return true;
  return false;
}

function enemyHasShield(entity) {
  if (!entity || !entity.heldItem) return false;
  return /shield/.test(String(entity.heldItem.name || ''));
}

async function equipMaceForFalling(bot) {
  // Equip density mace for optimal falling + mace damage
  const slot = selectMaceForFallingDamage(bot);
  if (slot < 0) return false;
  try {
    bot.setQuickBarSlot(slot);
    await sleep(40);
    return true;
  } catch {
    return false;
  }
}

function findLikelyRecentAttacker(bot, maxDistance = 6.5) {
  if (!bot?.entity?.position) return null;
  const playerCandidate = getNearestEntity(bot, (entity) =>
    entity?.type === 'player'
      && entity?.id !== bot.entity?.id
      && entity?.username
      && entity?.username !== bot.username
      && entity?.position
      && bot.entity.position.distanceTo(entity.position) <= maxDistance
      && isThreatReachable(bot, entity, 4.5)
  ).entity;
  if (playerCandidate) return playerCandidate;

  return getNearestEntity(bot, (entity) =>
    isHostileMob(entity)
      && entity?.position
      && bot.entity.position.distanceTo(entity.position) <= maxDistance
      && isThreatReachable(bot, entity, 3.25)
  ).entity;
}

function targetAimAligned(bot, target, minDot = 0.78) {
  if (!bot?.entity?.position || !target?.position) return false;
  const eyeY = Number(bot.entity.position.y || 0) + 1.62;
  const lookX = -Math.sin(bot.entity.yaw) * Math.cos(bot.entity.pitch);
  const lookY = -Math.sin(bot.entity.pitch);
  const lookZ = Math.cos(bot.entity.yaw) * Math.cos(bot.entity.pitch);

  const targetX = Number(target.position.x || 0) - Number(bot.entity.position.x || 0);
  const targetY = (Number(target.position.y || 0) + (target.height ? target.height * 0.62 : 1.2)) - eyeY;
  const targetZ = Number(target.position.z || 0) - Number(bot.entity.position.z || 0);
  const targetLen = Math.hypot(targetX, targetY, targetZ);
  if (targetLen < 0.001) return true;
  const dot = ((lookX * targetX) + (lookY * targetY) + (lookZ * targetZ)) / targetLen;
  return dot >= minDot;
}

async function equipMaceForBreach(bot) {
  // Equip breach mace for block swapping and breaking
  const slot = selectMaceForBreachSwap(bot);
  if (slot < 0) return false;
  try {
    bot.setQuickBarSlot(slot);
    await sleep(40);
    return true;
  } catch {
    return false;
  }
}

function findPlacedCraftingTable(bot, maxDistance = 24) {
  try {
    return bot.findBlock({
      maxDistance,
      matching: (block) => String(block?.name || '') === 'crafting_table'
    });
  } catch {
    return null;
  }
}

function findNearestBlockByNames(bot, names, maxDistance = 32) {
  const wanted = new Set(names);
  try {
    return bot.findBlock({
      maxDistance,
      matching: (block) => wanted.has(String(block?.name || ''))
    });
  } catch {
    return null;
  }
}

function buildingLikeBlockScore(name) {
  if (!name) return 0;
  if (/_planks$/.test(name)) return 1;
  if (/cobblestone|stone_bricks|brick|glass|terracotta|concrete|quartz|purpur|obsidian|deepslate_tiles|deepslate_bricks/.test(name)) return 1;
  return 0;
}

const SUSPICIOUS_NON_NATURAL_BLOCKS = new Set([
  'redstone_block',
  'piston',
  'sticky_piston',
  'observer',
  'repeater',
  'comparator',
  'target',
  'note_block',
  'redstone_lamp',
  'dispenser',
  'dropper',
  'hopper',
  'lever',
  'tripwire_hook',
  'daylight_detector',
  'tnt',
  'scaffolding',
  'respawn_anchor'
]);

async function forageFood(bot) {
  const foodLevel = Number(bot.food || 20);
  if (foodLevel > 14) return false;

  const ate = await consumeFoodIfNeeded(bot, true);
  if (ate) {
    state.lastNote = 'food: eating from inventory';
    return true;
  }

  const melon = findNearestBlockByNames(bot, ['melon', 'melon_stem', 'pumpkin'], 28);
  if (melon) {
    state.lastNote = `food: harvesting ${melon.name}`;
    await mineBlockWithTool(bot, melon, []);
    return true;
  }

  const animal = nearestFoodAnimal(bot, 30);
  if (animal.entity) {
    state.lastNote = `food: hunting ${animal.entity.name}`;
    return chaseAndAttack(bot, animal.entity);
  }

  if (bot.pathfinder && state.movements) {
    const roamX = bot.entity.position.x + (Math.random() * 18 - 9);
    const roamZ = bot.entity.position.z + (Math.random() * 18 - 9);
    bot.pathfinder.setMovements(state.movements);
    bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
  }
  state.lastNote = 'food: searching for animals/melons/fish';
  return true;
}

function scanForBasefinderClues(bot) {
  const now = Date.now();
  if (now - state.lastBaseScanAt < 1200) return;
  state.lastBaseScanAt = now;

  for (const entity of Object.values(bot.entities)) {
    if (!entity?.position) continue;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist > 80) continue;
    const lower = String(entity.name || '').toLowerCase();
    if (lower === 'villager') {
      recordBaseCandidate('villager', entity.position, 0.74, 'villager spotted');
    }
  }

  const enderChest = findNearestBlockByNames(bot, ['ender_chest'], 64);
  if (enderChest?.position) {
    recordBaseCandidate('ender_chest', enderChest.position, 0.95, 'ender chest block');
  }

  try {
    if (typeof bot.findBlocks === 'function') {
      const suspiciousPositions = bot.findBlocks({
        matching: (block) => SUSPICIOUS_NON_NATURAL_BLOCKS.has(String(block?.name || '')),
        maxDistance: 72,
        count: 48
      }) || [];

      for (const pos of suspiciousPositions) {
        const b = bot.blockAt(pos);
        if (!b) continue;
        const blockName = String(b.name || 'unknown');
        const confidence = blockName === 'redstone_block' || blockName.includes('piston') ? 0.96 : 0.86;
        recordBaseCandidate('non_natural_block', pos, confidence, blockName);
      }
    }
  } catch {
    // ignore suspicious block scan failures
  }

  try {
    if (typeof bot.findBlocks === 'function') {
      const chestPositions = bot.findBlocks({
        matching: (block) => String(block?.name || '') === 'chest',
        maxDistance: 64,
        count: 20
      }) || [];

      for (const pos of chestPositions) {
        const chestBlock = bot.blockAt(pos);
        if (!chestBlock || String(chestBlock.name || '') !== 'chest') continue;
        const adjacent = [
          bot.blockAt(pos.offset(1, 0, 0)),
          bot.blockAt(pos.offset(-1, 0, 0)),
          bot.blockAt(pos.offset(0, 0, 1)),
          bot.blockAt(pos.offset(0, 0, -1))
        ].some((neighbor) => String(neighbor?.name || '') === 'chest');
        if (adjacent) {
          recordBaseCandidate('double_chest', pos, 0.9, 'adjacent chest pair');
        } else {
          recordBaseCandidate('chest', pos, 0.62, 'single chest');
        }
      }
    }
  } catch {
    // ignore chest scan issues
  }

  const center = bot.entity.position.floored();
  let buildLike = 0;
  let cobbledDeepslateCount = 0;
  let lowYCobbledDeepslateCount = 0;
  for (let dx = -6; dx <= 6; dx++) {
    for (let dz = -6; dz <= 6; dz++) {
      for (let dy = -1; dy <= 3; dy++) {
        const block = bot.blockAt(center.offset(dx, dy, dz));
        if (!block) continue;
        const blockName = String(block.name || '');
        buildLike += buildingLikeBlockScore(blockName);
        if (blockName === 'cobbled_deepslate') {
          cobbledDeepslateCount += 1;
          if (block.position?.y <= 30) {
            lowYCobbledDeepslateCount += 1;
          }
        }
      }
    }
  }

  if (cobbledDeepslateCount >= 35) {
    const conf = lowYCobbledDeepslateCount >= 20 ? 0.95 : 0.82;
    recordBaseCandidate(
      'cobbled_deepslate_cluster',
      center,
      conf,
      `cluster=${cobbledDeepslateCount},lowY=${lowYCobbledDeepslateCount},y=${center.y}`
    );
  }

  if (lowYCobbledDeepslateCount >= 20) {
    recordBaseCandidate('deep_base_level', center, 0.97, `lowY cobbled_deepslate at y=${center.y}`);
  }

  if (buildLike >= 110) {
    recordBaseCandidate('large_cubic_area', center, Math.min(0.99, buildLike / 180), `build-score=${buildLike}`);
  }
}

// ===== INTELLIGENT BUILDING SYSTEM =====
async function learnAndBuildStructure(bot) {
  // Load recent observations of how other players build
  const observations = loadObservations(state.swarmWorkerId || state.username, 30);
  const placeActions = observations.filter(o => o.actionType === 'place');
  
  if (placeActions.length === 0) return false;
  
  // Check for common building patterns
  const blockTypes = {};
  for (const action of placeActions) {
    blockTypes[action.blockName] = (blockTypes[action.blockName] || 0) + 1;
  }
  
  const mostCommonBlock = Object.entries(blockTypes).sort((a, b) => b[1] - a[1])[0];
  if (!mostCommonBlock) return false;
  
  const [blockName, frequency] = mostCommonBlock;
  
  // If we see a pattern of building with a certain block, try to build with it
  if (frequency >= 4) {
    const facts = collectInventoryFacts(bot);
    const inventory = bot.inventory?.items?.() || [];
    
    // Find matching block in inventory
    const matchingItem = inventory.find(item => 
      String(item.name || '').toLowerCase() === blockName.toLowerCase()
    );
    
    if (matchingItem && facts.hasBlocks) {
      // Look for a good building spot near other players
      for (const entity of Object.values(bot.entities)) {
        if (!entity || entity.type !== 'player') continue;
        const dist = bot.entity.position.distanceTo(entity.position);
        if (dist < 3 && dist > 0.5) {
          // Try to build near other players (learning by proximity)
          const reference = bot.blockAt(entity.position.offset(0, -1, 0));
          if (reference && String(reference.name || '') !== 'air') {
            try {
              await bot.equip(matchingItem, 'hand');
              await bot.placeBlock(reference, new Vec3(-1, 0, 0));
              recordPlayerAction(state.swarmWorkerId || state.username, bot.username, 'place', null, matchingItem);
              state.lastNote = `learning: building with ${blockName} near player`;
              return true;
            } catch {
              // ignore placement failure
            }
          }
        }
      }
    }
  }
  
  return false;
}

// ===== STRUCTURE TYPES (houses, farms, etc) =====
async function buildSimpleStructure(bot, structureType = 'wall') {
  // Teaches bot to build basic structures
  const facts = collectInventoryFacts(bot);
  if (!facts.hasBlocks) return false;
  
  const blockSlot = facts.blockSlot;
  if (blockSlot < 0) return false;
  
  try {
    bot.setQuickBarSlot(blockSlot);
    
    // Find ground to build on
    const reference = bot.blockAt(bot.entity.position.offset(0, -1, 0));
    if (!reference || String(reference.name || '') === 'air') return false;
    
    if (structureType === 'tower') {
      // Build a tower straight up
      for (let i = 0; i < 5; i++) {
        try {
          const placeRef = bot.blockAt(bot.entity.position.offset(0, i - 1, 0));
          if (placeRef && String(placeRef.name || '') !== 'air') {
            await bot.placeBlock(placeRef, new Vec3(0, 1, 0));
            await sleep(150);
          }
        } catch {
          break;
        }
      }
    } else if (structureType === 'wall') {
      // Build a horizontal wall
      const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
      for (const [dx, dz] of dirs) {
        try {
          const placeRef = bot.blockAt(bot.entity.position.offset(dx, -1, dz));
          if (placeRef && String(placeRef.name || '') !== 'air') {
            await bot.placeBlock(placeRef, new Vec3(0, 1, 0));
            await sleep(150);
          }
        } catch {
          // ignore placement failures
        }
      }
    } else if (structureType === 'farm') {
      // Build a 3x3 farm bed
      for (let dx = -1; dx <= 1; dx++) {
        for (let dz = -1; dz <= 1; dz++) {
          try {
            const placeRef = bot.blockAt(bot.entity.position.offset(dx, -1, dz));
            if (placeRef && String(placeRef.name || '') !== 'air') {
              await bot.placeBlock(placeRef, new Vec3(0, 1, 0));
              await sleep(100);
            }
          } catch {
            // ignore placement failures
          }
        }
      }
    }
    
    // Track buildings built by builder
    if (!state.buildingsBuilt) state.buildingsBuilt = {};
    state.buildingsBuilt[bot.username] = (state.buildingsBuilt[bot.username] || 0) + 1;
    state.lastNote = `building: constructed ${structureType}`;
    return true;
  } catch {
    return false;
  }
}

async function runBasefinderMode(bot) {
  scanForBasefinderClues(bot);

  const hungry = Number(bot.food || 20) <= 13;
  if (hungry) {
    const foraged = await forageFood(bot);
    if (foraged) return true;
  }

  const now = Date.now();
  if (bot.pathfinder && state.movements && (now - state.lastBaseRoamAt > 1300)) {
    state.lastBaseRoamAt = now;
    const roamX = bot.entity.position.x + (Math.random() * 150 - 75);
    const roamZ = bot.entity.position.z + (Math.random() * 150 - 75);
    bot.pathfinder.setMovements(state.movements);
    bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 3));
  }

  bot.setControlState('sprint', true);
  bot.setControlState('forward', true);
  const nowJump = Date.now();
  const shouldHop = bot.entity.onGround
    && !state.miningInProgress
    && !inNoJumpWindow()
    && (bot.entity.isCollidedHorizontally || Math.random() > 0.94)
    && (nowJump - state.lastRoamJumpAt > 900);
  bot.setControlState('jump', Boolean(shouldHop));
  if (shouldHop) state.lastRoamJumpAt = nowJump;

  const findings = getBaseCandidates();
  state.lastNote = findings.length > 0
    ? `basefinder: roaming + scanning (${findings.length} candidates)`
    : 'basefinder: roaming + scanning';
  return true;
}

async function craftItemByName(bot, itemName, count = 1, craftingTable = null) {
  const mc = getMcData(bot);
  const itemDef = mc.itemsByName[itemName];
  if (!itemDef) return false;

  const recipes = bot.recipesFor(itemDef.id, null, count, craftingTable || null);
  if (!recipes || recipes.length === 0) return false;

  try {
    await bot.craft(recipes[0], count, craftingTable || null);
    return true;
  } catch {
    return false;
  }
}

async function craftFirstAvailable(bot, itemNames, count = 1, craftingTable = null) {
  for (const name of itemNames) {
    const crafted = await craftItemByName(bot, name, count, craftingTable);
    if (crafted) return true;
  }
  return false;
}

async function ensureCombatHandReady(bot) {
  const heldName = String(bot?.heldItem?.name || '').toLowerCase();
  if (Date.now() < Number(state.eatingUntil || 0)) return !isLikelyPlaceableBlockName(heldName);
  const equipped = await equipBestMeleeWeapon(bot);
  if (equipped) return true;
  if (isLikelyPlaceableBlockName(heldName) || heldName === 'shield') {
    try {
      if (typeof bot.unequip === 'function') {
        await bot.unequip('hand');
      }
    } catch {
      // ignore unequip failure
    }
  }
  return true;
}

async function ensureCraftingTablePlaced(bot) {
  const existing = findPlacedCraftingTable(bot, 20);
  if (existing) return existing;

  const tableItem = findInventoryItem(bot, (item) => String(item.name || '') === 'crafting_table');
  if (!tableItem) return null;

  let reference = bot.blockAt(bot.entity.position.offset(0, -1, 0));
  if (!reference || String(reference.name || '') === 'air') {
    try {
      reference = bot.findBlock({
        maxDistance: 6,
        matching: (block) => block && String(block.name || '') !== 'air' && !String(block.name || '').includes('water') && !String(block.name || '').includes('lava')
      });
    } catch {
      reference = null;
    }
  }

  if (!reference) return null;

  const dist = bot.entity.position.distanceTo(reference.position);
  if (dist > 4.5) {
    if (bot.pathfinder && state.movements) {
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(reference.position.x, reference.position.y, reference.position.z, 2));
      return null;
    }
  }

  try {
    await bot.equip(tableItem, 'hand');
    await bot.placeBlock(reference, new Vec3(0, 1, 0));
    await sleep(120);
  } catch {
    return null;
  }

  return findPlacedCraftingTable(bot, 8);
}

function chooseApproachPointNearBlock(bot, blockPos) {
  const offsets = [
    { x: 2, z: 0 }, { x: -2, z: 0 }, { x: 0, z: 2 }, { x: 0, z: -2 },
    { x: 2, z: 2 }, { x: 2, z: -2 }, { x: -2, z: 2 }, { x: -2, z: -2 }
  ];

  let best = { x: blockPos.x, y: blockPos.y, z: blockPos.z, radius: 1.6 };
  let bestDist = Infinity;

  for (const offset of offsets) {
    const x = blockPos.x + offset.x;
    const y = blockPos.y;
    const z = blockPos.z + offset.z;
    const feet = bot.blockAt(new Vec3(Math.floor(x), Math.floor(y), Math.floor(z)));
    const below = bot.blockAt(new Vec3(Math.floor(x), Math.floor(y - 1), Math.floor(z)));
    const blocked = feet && String(feet.name || '') !== 'air';
    const hasFloor = below && String(below.name || '') !== 'air';
    if (blocked || !hasFloor) continue;

    const dist = bot.entity.position.distanceTo(new Vec3(x, y, z));
    if (dist < bestDist) {
      bestDist = dist;
      best = { x, y, z, radius: 1.4 };
    }
  }

  return best;
}

async function approachAndAimAtBlock(bot, block, reachDistance = 4.2, goalRadius = 1) {
  if (!block?.position) return false;
  const freshBlock = bot.blockAt(block.position) || block;
  if (!freshBlock?.position || freshBlock.type === 0) return false;

  const dist = bot.entity.position.distanceTo(freshBlock.position);
  if (dist > reachDistance) {
    if (bot.pathfinder && state.movements) {
      const approach = chooseApproachPointNearBlock(bot, freshBlock.position);
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(approach.x, approach.y, approach.z, approach.radius || goalRadius));
    }
    try {
      await bot.lookAt(freshBlock.position.offset(0.5, 0.5, 0.5), true);
    } catch {
      // ignore look failures while approaching
    }
    return true;
  }

  try {
    await bot.lookAt(freshBlock.position.offset(0.5, 0.5, 0.5), true);
  } catch {
    // ignore look failure
  }

  return false;
}

async function mineBlockWithTool(bot, block, toolNames = []) {
  if (!block || !block.position) return false;

  const movingToBlock = await approachAndAimAtBlock(bot, block, 4.2, 1);
  if (movingToBlock) return true;

  if (state.miningInProgress) return true;

  state.miningInProgress = true;
  state.noJumpUntil = Date.now() + 1800;

  try {
    if (bot.pathfinder) {
      bot.pathfinder.setGoal(null);
    }
    clearMovementControls(bot);

    if (toolNames.length > 0) {
      const equipped = await equipToolByNames(bot, toolNames);
      if (!equipped) {
        try {
          if (typeof bot.unequip === 'function') {
            await bot.unequip('hand');
          }
        } catch {
          // ignore unequip failure; keep current item if fist cannot be forced
        }
        state.lastNote = 'general1: no better tool found, using fist';
      }
    }

    const refreshedForCheck = bot.blockAt(block.position) || block;
    if (typeof bot.canDigBlock === 'function' && !bot.canDigBlock(refreshedForCheck)) {
      return true;
    }

    // Re-fetch the block right before digging — the original reference may be
    // stale if the block was already broken, causing the server-side
    // "Mismatch in destroy block pos: is{0,0,0}" warning.
    const freshBlock = bot.blockAt(block.position);
    if (!freshBlock || freshBlock.type === 0) return true;

    try {
      await bot.lookAt(freshBlock.position.offset(0.5, 0.5, 0.5), true);
    } catch {
      // ignore look failure
    }

    const mineDeadline = Date.now() + 5500;
    while (Date.now() < mineDeadline) {
      const currentBlock = bot.blockAt(block.position);
      if (!currentBlock || currentBlock.type === 0) return true;

      state.lastDigAt = Date.now();
      try {
        await bot.dig(currentBlock, true);
      } catch {
        // retry while the block still exists
      }
      await sleep(80);
    }
    return true;
  } catch {
    return true;
  } finally {
    clearMovementControls(bot);
    state.miningInProgress = false;
    state.noJumpUntil = Date.now() + 650;
  }
}

async function runGeneral1Progression(bot) {
    // --- Money System Core ---
    // Initialize money for all bots
    if (!state.money[bot.username]) {
      if (bot.role === 'president') {
        state.money[bot.username] = 100;
      } else {
        state.money[bot.username] = 10;
      }
    }
    // Elect president if not set
    if (!state.president) {
      const allBots = Object.values(bot.entities).filter(e => e.type === 'player' && e.username);
      if (allBots.length > 0) {
        state.president = allBots.map(e => e.username).sort()[0];
      }
    }
      // Show balances in log every 60s
      if (!state.lastBalanceLog || now - state.lastBalanceLog > 60000) {
        state.lastBalanceLog = now;
        console.log('[MONEY] Balances:', JSON.stringify(state.money));
      }
    // Tax and builder payout every Minecraft day (20 min = 24000 ticks, ~1200s)
      const now = Date.now();
      if (!state.lastTaxDay || now - state.lastTaxDay > 1200 * 1000) {
        state.lastTaxDay = now;
        // Tax collection
        for (const user in state.money) {
          if (user !== state.president) {
            const tax = Math.floor(state.money[user] * 0.05);
            state.money[user] -= tax;
            state.money[state.president] = (state.money[state.president] || 0) + tax;
          }
        }
        // President gives $4 to each builder
        for (const e of Object.values(bot.entities)) {
          if (e.username && e.username !== state.president && e.role === 'builder') {
            if ((state.money[state.president] || 0) >= 4) {
              state.money[state.president] -= 4;
              state.money[e.username] = (state.money[e.username] || 0) + 4;
            }
          }
        }
        // President receives $10 * buildings built * progression level
        if (!state.buildingsBuilt) state.buildingsBuilt = {};
        if (!state.progressionLevel) state.progressionLevel = 1;
        for (const e of Object.values(bot.entities)) {
          if (e.username && e.role === 'builder') {
            const built = state.buildingsBuilt[e.username] || 0;
            const payout = 10 * built * state.progressionLevel;
            if (payout > 0) {
              state.money[state.president] = (state.money[state.president] || 0) + payout;
              console.log(`[MONEY] President received $${payout} for ${built} buildings by ${e.username} at progression level ${state.progressionLevel}`);
            }
          }
        }
      }

    // --- Trading Logic ---
    // Miners sell cobblestone/logs to builders for $1 each, and ores to president at set prices
    if (bot.role === 'miner') {
      const builders = Object.values(bot.entities).filter(e => e.type === 'player' && e.role === 'builder');
      for (const builder of builders) {
        const dist = bot.entity.position.distanceTo(builder.position);
        if (dist < 4) {
          const cobble = findInventoryItem(bot, i => /cobble/.test(i.name));
          const logs = findInventoryItem(bot, i => /_log/.test(i.name));
          if (cobble && cobble.count > 0 && (state.money[builder.username] || 0) >= 1) {
            state.money[bot.username] += 1;
            state.money[builder.username] -= 1;
            cobble.count -= 1;
          }
          if (logs && logs.count > 0 && (state.money[builder.username] || 0) >= 1) {
            state.money[bot.username] += 1;
            state.money[builder.username] -= 1;
            logs.count -= 1;
          }
        }
      }
      // Sell ores to president
      const president = Object.values(bot.entities).find(e => e.type === 'player' && e.role === 'president');
      if (president && bot.entity.position.distanceTo(president.position) < 4) {
        const orePrices = {
          coal: 1, iron: 2, gold: 3, lapis: 1, redstone: 1, diamond: 5, netherite: 12
        };
        for (const ore in orePrices) {
          const item = findInventoryItem(bot, i => i.name && i.name.includes(ore));
          if (item && item.count > 0 && (state.money[president.username] || 0) >= orePrices[ore]) {
            state.money[bot.username] += orePrices[ore];
            state.money[president.username] -= orePrices[ore];
            item.count -= 1;
          }
        }
      }
    }
      // Treasurer: wandering trader with random trades
      if (bot.role === 'treasurer') {
        // Example: offer random trades to nearby bots
        const offers = [
          { give: 'emerald', get: 'diamond', price: 4 },
          { give: 'book', get: 'enchanted_book', price: 8 },
          { give: 'iron_ingot', get: 'gold_ingot', price: 2 },
          { give: 'bread', get: 'emerald', price: 1 },
          { give: 'apple', get: 'golden_apple', price: 6 }
        ];
        const others = Object.values(bot.entities).filter(e => e.type === 'player' && e.username !== bot.username);
        for (const other of others) {
          if (bot.entity.position.distanceTo(other.position) < 4) {
            const offer = offers[Math.floor(Math.random() * offers.length)];
            // If other has enough money, do trade (simulate)
            if ((state.money[other.username] || 0) >= offer.price) {
              state.money[other.username] -= offer.price;
              state.money[bot.username] += offer.price;
              // (simulate item transfer)
            }
          }
        }
      }
      // President: use ores to gear up civilization (simulate)
      if (bot.role === 'president') {
        // If president has diamond/iron/netherite gear, equip it and distribute extras to others
        const gearTypes = [
          { slot: 'head', names: ['diamond_helmet', 'netherite_helmet', 'iron_helmet'] },
          { slot: 'torso', names: ['diamond_chestplate', 'netherite_chestplate', 'iron_chestplate'] },
          { slot: 'legs', names: ['diamond_leggings', 'netherite_leggings', 'iron_leggings'] },
          { slot: 'feet', names: ['diamond_boots', 'netherite_boots', 'iron_boots'] },
          { slot: 'hand', names: ['diamond_sword', 'netherite_sword', 'iron_sword'] }
        ];
        for (const gear of gearTypes) {
          const item = findInventoryItem(bot, i => gear.names.includes(i.name));
          if (item) {
            try { await bot.equip(item, gear.slot); } catch {}
          }
        }
        // Distribute extra gear to teammates nearby
        const teammates = Object.values(bot.entities).filter(e => e.type === 'player' && e.username !== bot.username && bot.entity.position.distanceTo(e.position) < 5);
        for (const gear of gearTypes) {
          let extra = bot.inventory?.items?.().filter(i => gear.names.includes(i.name)).slice(1) || [];
          for (const teammate of teammates) {
            if (extra.length === 0) break;
            // Simulate giving gear (real implementation: drop or toss to teammate)
            // Here, just decrement count for simulation
            extra[0].count -= 1;
            extra = extra.filter(i => i.count > 0);
          }
        }
          // If miner doesn't have enough logs, cobble, iron, or diamonds, go mine for them
          const needs = [];
          if (progress.logCount < 4) needs.push('logs');
          if (progress.cobbleCount < 6) needs.push('cobble');
          if ((progress.rawIronCount + progress.ironIngotCount) < 3) needs.push('iron');
          if (progress.diamondCount < 1) needs.push('diamond');
          if (needs.length > 0) {
            // Prioritize mining missing resources
            if (needs.includes('logs')) await general1CollectLogs();
            else if (needs.includes('cobble')) await general1MineStone();
            else if (needs.includes('iron')) await general1MineIron();
            else if (needs.includes('diamond')) await general1Diamonds();
          }
        }
        // Farmer logic: plant, harvest, breed, kill, cook
        if (bot.role === 'farmer') {
            // Treasurer logic: seek and collect rare/elite items
            if (bot.role === 'treasurer') {
              // List of rare/elite items to seek
              const rareItems = [
                'trident', 'mace', 'netherite_ingot', 'netherite_scrap', 'enchanted_golden_apple',
                'elytra', 'totem_of_undying', 'beacon', 'dragon_egg', 'nether_star',
                'diamond', 'diamond_block', 'ancient_debris', 'enchanting_table', 'shulker_shell',
                'heart_of_the_sea', 'conduit', 'music_disc', 'saddle', 'name_tag', 'enchanted_book'
              ];
              // Search inventory for missing rare items
              const missing = rareItems.filter(name => !inventoryHasItem(bot, name));
              if (missing.length > 0) {
                // Look for dropped rare items nearby
                const dropped = nearestDroppedItemEntity(bot, 12);
                if (dropped && rareItems.includes(dropped.metadata?.itemId ? getMcData(bot).items[dropped.metadata.itemId]?.name : '')) {
                  await moveToDroppedItem(bot, dropped);
                }
                // Try to loot chests, ruins, or structures (not implemented: placeholder for future expansion)
                // Could add logic to pathfind to structures, open chests, etc.
              }
              // If has extra rare items, offer trades to other bots (simulate)
              for (const itemName of rareItems) {
                const count = countItemsByName(bot, [itemName]);
                if (count > 1) {
                  // Simulate offering trade to nearby bots
                  const others = Object.values(bot.entities).filter(e => e.type === 'player' && e.username !== bot.username && bot.entity.position.distanceTo(e.position) < 5);
                  for (const other of others) {
                    // Simulate giving one item
                    // (real implementation: drop or toss to teammate)
                  }
                }
              }
            }
          // 1. Plant crops if seeds available and farmland nearby
          const seeds = findInventoryItem(bot, i => /_seeds|carrot|potato/.test(i.name));
          if (seeds && seeds.count > 0) {
            // Find farmland or dirt/grass to plant on
            const farmland = bot.findBlock({ maxDistance: 8, matching: b => ['farmland', 'dirt', 'grass_block'].includes(b.name) });
            if (farmland) {
              try {
                await bot.equip(seeds, 'hand');
                await bot.placeBlock(farmland, new Vec3(0, 1, 0));
                await sleep(200);
              } catch {}
            }
          }
          // 2. Harvest mature crops
          const matureCrop = bot.findBlock({ maxDistance: 8, matching: b => /wheat|carrots|potatoes/.test(b.name) && b.metadata === 7 });
          if (matureCrop) {
            await mineBlockWithTool(bot, matureCrop, []);
          }
          // 3. Breed cows and pigs if food available
          const wheat = findInventoryItem(bot, i => i.name === 'wheat');
          const carrot = findInventoryItem(bot, i => i.name === 'carrot');
          const nearbyCow = Object.values(bot.entities).find(e => e.name === 'cow' && bot.entity.position.distanceTo(e.position) < 5);
          const nearbyPig = Object.values(bot.entities).find(e => e.name === 'pig' && bot.entity.position.distanceTo(e.position) < 5);
          if (wheat && wheat.count > 1 && nearbyCow) {
            try { await bot.equip(wheat, 'hand'); await bot.activateEntity(nearbyCow); } catch {}
          }
          if (carrot && carrot.count > 1 && nearbyPig) {
            try { await bot.equip(carrot, 'hand'); await bot.activateEntity(nearbyPig); } catch {}
          }
          // 4. Kill cows and pigs for food
          const foodAnimal = nearestFoodAnimal(bot, 8);
          if (foodAnimal && foodAnimal.entity) {
            await chaseAndAttack(bot, foodAnimal.entity);
          }
          // 5. Cook food in furnace if raw food and furnace available
          const rawFood = findInventoryItem(bot, i => /raw_beef|raw_porkchop|raw_mutton|raw_chicken|raw_cod|raw_salmon/.test(i.name));
          if (rawFood) {
            const furnace = findNearestBlockByNames(bot, ['furnace'], 8);
            if (furnace) {
              try {
                const furnaceWindow = await bot.openFurnace(furnace);
                await furnaceWindow.putInput(rawFood.type, null, Math.min(rawFood.count, 4));
                const fuel = findInventoryItem(bot, i => /coal|charcoal|_log|_planks|stick/.test(i.name));
                if (fuel) await furnaceWindow.putFuel(fuel.type, null, Math.min(fuel.count, 8));
                await sleep(1200);
                try { await furnaceWindow.takeOutput(); } catch {}
                furnaceWindow.close();
              } catch {}
            }
          }
      }
    // Builders buy cobble/logs from miners, buy food from farmers for $1
    if (bot.role === 'builder') {
      const miners = Object.values(bot.entities).filter(e => e.type === 'player' && e.role === 'miner');
      const farmers = Object.values(bot.entities).filter(e => e.type === 'player' && e.role === 'farmer');
      for (const farmer of farmers) {
        const dist = bot.entity.position.distanceTo(farmer.position);
        if (dist < 4) {
          const food = findInventoryItem(farmer, i => /beef|pork|carrot|potato|bread|apple|wheat|melon|chicken|fish|mutton|stew|soup|berry|cookie|cake|pie/.test(i.name));
          if (food && food.count > 0 && (state.money[bot.username] || 0) >= 1) {
            state.money[bot.username] -= 1;
            state.money[farmer.username] = (state.money[farmer.username] || 0) + 1;
            food.count -= 1;
          }
        }
      }
    }
    // All bots buy food from farmers for $1
    if (bot.role !== 'farmer') {
      const farmers = Object.values(bot.entities).filter(e => e.type === 'player' && e.role === 'farmer');
      for (const farmer of farmers) {
        const dist = bot.entity.position.distanceTo(farmer.position);
        if (dist < 4) {
          const food = findInventoryItem(farmer, i => /beef|pork|carrot|potato|bread|apple|wheat|melon|chicken|fish|mutton|stew|soup|berry|cookie|cake|pie/.test(i.name));
          if (food && food.count > 0 && (state.money[bot.username] || 0) >= 1) {
            state.money[bot.username] -= 1;
            state.money[farmer.username] = (state.money[farmer.username] || 0) + 1;
            food.count -= 1;
          }
        }
      }
    }
    // --- End Money System ---
  const progress = {
    logCount: countItemsByRegex(bot, /(_log|_wood)$/),
    plankCount: countItemsByRegex(bot, /_planks$/),
    stickCount: countItemsByName(bot, ['stick']),
    cobbleCount: countItemsByName(bot, ['cobblestone', 'cobbled_deepslate']),
    rawIronCount: countItemsByName(bot, ['raw_iron']),
    ironIngotCount: countItemsByName(bot, ['iron_ingot']),
    diamondCount: countItemsByName(bot, ['diamond'])
  };

  // Make bots stay near each other (swarm behavior)
  const roamNearby = (radius = 12) => {
    if (!bot.pathfinder || !state.movements) return;
    // Find average position of other bots
    const teammates = Object.values(bot.entities).filter(e => e.type === 'player' && e.username && e.username !== bot.username && bot.entity.position.distanceTo(e.position) < 48);
    if (teammates.length > 0) {
      // Move toward the average teammate position
      const avg = teammates.reduce((acc, e) => ({
        x: acc.x + e.position.x,
        y: acc.y + e.position.y,
        z: acc.z + e.position.z
      }), {x:0, y:0, z:0});
      avg.x /= teammates.length;
      avg.y /= teammates.length;
      avg.z /= teammates.length;
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(avg.x, avg.y, avg.z, radius));
    } else {
      // Default to random roam if no teammates nearby
      const roamX = bot.entity.position.x + (Math.random() * (radius * 2) - radius);
      const roamZ = bot.entity.position.z + (Math.random() * (radius * 2) - radius);
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
    }
  };

  const general1CollectLogs = async () => {
    if (progress.logCount >= 4) return false;
    state.lastNote = `general1/logs: ${progress.logCount}/4`;
    const logBlock = nearestLogBlock(bot, 32);
    if (!logBlock) {
      roamNearby(8);
      return true;
    }
    await mineBlockWithTool(bot, logBlock, []);
    return true;
  };

  const general1CraftPlanks = async () => {
    if (progress.plankCount >= 12) return false;
    state.lastNote = 'general1/planks: crafting';
    await craftFirstAvailable(bot, [
      'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks',
      'mangrove_planks', 'cherry_planks', 'pale_oak_planks', 'bamboo_planks', 'crimson_planks', 'warped_planks'
    ], 1, null);
    return true;
  };

  const general1EnsureTable = async () => {
    if (!inventoryHasItem(bot, 'crafting_table') && !findPlacedCraftingTable(bot, 16)) {
      if (progress.plankCount < 4) {
        return { handled: false, table: null };
      }
      state.lastNote = 'general1/table: crafting';
      const crafted = await craftItemByName(bot, 'crafting_table', 1, null);
      if (!crafted) {
        return { handled: false, table: null };
      }
      return { handled: true, table: null };
    }
    const table = await ensureCraftingTablePlaced(bot);
    if (!table) {
      state.lastNote = 'general1/table: placing/moving';
      return { handled: true, table: null };
    }
    return { handled: false, table };
  };

  const general1WoodPick = async () => {
    const tableStep = await general1EnsureTable();
    if (tableStep.handled) return true;
    const table = tableStep.table;
    if (inventoryHasItem(bot, 'wooden_pickaxe')) return false;
    state.lastNote = 'general1/wood_pick: crafting';
    if (progress.stickCount < 2) {
      await craftItemByName(bot, 'stick', 1, null);
      return true;
    }
    await craftItemByName(bot, 'wooden_pickaxe', 1, table);
    return true;
  };

  const general1MineStone = async () => {
    if (progress.cobbleCount >= 6) return false;
    state.lastNote = `general1/stone: ${progress.cobbleCount}/6`;
    const stoneBlock = findNearestBlockByNames(bot, ['stone', 'deepslate', 'cobblestone', 'cobbled_deepslate'], 36);
    if (!stoneBlock) {
      roamNearby(10);
      return true;
    }
    await mineBlockWithTool(bot, stoneBlock, ['wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe', 'netherite_pickaxe']);
    return true;
  };

  const general1StoneTools = async () => {
    const tableStep = await general1EnsureTable();
    if (tableStep.handled) return true;
    const table = tableStep.table;
    const hasStonePickaxe = inventoryHasItem(bot, 'stone_pickaxe');
    const hasStoneAxe = inventoryHasItem(bot, 'stone_axe');
    if (hasStonePickaxe && hasStoneAxe) return false;
    state.lastNote = 'general1/stone_tools: crafting';
    if (progress.stickCount < 4) {
      await craftItemByName(bot, 'stick', 1, null);
      return true;
    }
    if (!hasStonePickaxe) {
      await craftItemByName(bot, 'stone_pickaxe', 1, table);
      return true;
    }
    if (!hasStoneAxe) {
      await craftItemByName(bot, 'stone_axe', 1, table);
      return true;
    }
    return true;
  };

  const general1MineIron = async () => {
    if ((progress.rawIronCount + progress.ironIngotCount) >= 3) return false;
    state.lastNote = `general1/iron_ore: ${progress.rawIronCount + progress.ironIngotCount}/3`;
    const ironOre = findNearestBlockByNames(bot, ['iron_ore', 'deepslate_iron_ore'], 48);
    if (!ironOre) {
      roamNearby(14);
      return true;
    }
    await mineBlockWithTool(bot, ironOre, ['stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe', 'netherite_pickaxe']);
    return true;
  };

  const general1SmeltIron = async () => {
    if (progress.ironIngotCount >= 3) return false;
    state.lastNote = 'general1/smelt: iron';
    const furnace = findNearestBlockByNames(bot, ['furnace'], 16);
    if (!furnace) {
      const tableStep = await general1EnsureTable();
      if (tableStep.handled) return true;
      const table = tableStep.table;
      if (!inventoryHasItem(bot, 'furnace')) {
        await craftItemByName(bot, 'furnace', 1, table);
        return true;
      }

      const furnaceItem = findInventoryItem(bot, (item) => String(item.name || '') === 'furnace');
      if (furnaceItem) {
        const reference = bot.blockAt(bot.entity.position.offset(0, -1, 0));
        if (reference && String(reference.name || '') !== 'air') {
          try {
            await bot.equip(furnaceItem, 'hand');
            await bot.placeBlock(reference, new Vec3(0, 1, 0));
            await sleep(120);
            return true;
          } catch {
            // ignore placement failure
          }
        }
      }
      return true;
    }

    const furnaceDist = bot.entity.position.distanceTo(furnace.position);
    if (furnaceDist > 4.5 && bot.pathfinder && state.movements) {
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(furnace.position.x, furnace.position.y, furnace.position.z, 1));
      return true;
    }

    try {
      const furnaceWindow = await bot.openFurnace(furnace);
      const rawIron = findInventoryItem(bot, (item) => String(item.name || '') === 'raw_iron');
      const fuel = findInventoryItem(bot, (item) => ['coal', 'charcoal', 'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks', 'crimson_planks', 'warped_planks', 'stick', 'oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log', 'dark_oak_log'].includes(String(item.name || '')));

      if (rawIron) {
        await furnaceWindow.putInput(rawIron.type, null, Math.min(rawIron.count, 3));
      }
      if (fuel) {
        await furnaceWindow.putFuel(fuel.type, null, Math.min(fuel.count, 8));
      }

      await sleep(2400);
      try {
        await furnaceWindow.takeOutput();
      } catch {
        // ignore if not ready yet
      }
      furnaceWindow.close();
      return true;
    } catch {
      return true;
    }
  };

  const general1IronPick = async () => {
    const tableStep = await general1EnsureTable();
    if (tableStep.handled) return true;
    const table = tableStep.table;
    if (inventoryHasItem(bot, 'iron_pickaxe')) return false;
    state.lastNote = 'general1/iron_pick: crafting';
    if (progress.stickCount < 2) {
      await craftItemByName(bot, 'stick', 1, null);
      return true;
    }
    await craftItemByName(bot, 'iron_pickaxe', 1, table);
    return true;
  };

  const general1Diamonds = async () => {
    state.lastNote = progress.diamondCount > 0
      ? `general1/diamond: acquired (${progress.diamondCount})`
      : 'general1/diamond: mining';
    const diamondOre = findNearestBlockByNames(bot, ['diamond_ore', 'deepslate_diamond_ore'], 64);
    if (diamondOre) {
      await mineBlockWithTool(bot, diamondOre, ['iron_pickaxe', 'diamond_pickaxe', 'netherite_pickaxe']);
      return true;
    }
    roamNearby(12);
    return true;
  };

  const steps = [
    general1CollectLogs,
    general1CraftPlanks,
    general1WoodPick,
    general1MineStone,
    general1StoneTools,
    general1MineIron,
    general1SmeltIron,
    general1IronPick,
    general1Diamonds
  ];

  for (const step of steps) {
    if (await step()) return true;
  }

  return true;
}

async function tryGrantHuntCompass(bot, targetName) {
  if (state.huntCompassGranted || !bot?.chat || !targetName) return false;
  state.huntCompassGranted = true;
  try {
    bot.chat(`/give ${bot.username} compass 1`);
    state.lastNote = `hunt: requested compass for ${targetName}`;
    return true;
  } catch {
    return false;
  }
}

async function runHuntPlayerMode(bot) {
  const targetName = parseHuntObjective();
  if (!targetName) return false;

  const now = Date.now();
  const target = getPlayerEntityByName(bot, targetName);
  const sameNameRevenge = normalizeText(target?.username || '').toLowerCase() === targetName.toLowerCase();

  await tryGrantHuntCompass(bot, targetName);

  if (target?.position) {
    state.huntLastSeenAt = now;
    state.huntDimensionSearchStartedAt = 0;
    state.huntTargetKilled = false;
    state.lastNote = `hunt: tracking ${targetName}`;

    const dist = bot.entity.position.distanceTo(target.position);
    const revengeActive = state.revengeTargetId === target.id && now < Number(state.revengeExpireAt || 0);
    const canTakeFight = canLikelyWinFight(bot, target, dist);
    if (!canTakeFight && !revengeActive) {
      state.lastNote = `hunt: gearing for ${targetName}`;
      return runGeneral1Progression(bot);
    }

    state.helperActive = true;
    return chaseAndAttack(bot, target, sameNameRevenge);
  }

  // If we lost the player, keep gearing up and start dimension search heuristics.
  const huntGearScore = estimateBotCombatPower(bot);
  if (huntGearScore < 140) {
    state.lastNote = `hunt: gearing before re-engaging ${targetName}`;
    return runGeneral1Progression(bot);
  }

  if (!state.huntDimensionSearchStartedAt) {
    state.huntDimensionSearchStartedAt = now;
  }

  const dimension = String(bot.game?.dimension || 'overworld').toLowerCase();
  const netherPortal = findNearestBlockByNames(bot, ['nether_portal'], 48);
  const endPortal = findNearestBlockByNames(bot, ['end_portal_frame', 'end_portal'], 64);

  if (!dimension.includes('nether') && now - state.huntDimensionSearchStartedAt < 45000) {
    if (netherPortal && bot.pathfinder && state.movements) {
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(netherPortal.position.x, netherPortal.position.y, netherPortal.position.z, 1));
      state.lastNote = `hunt: searching nether for ${targetName}`;
      return true;
    }
  }

  if (!dimension.includes('end') && now - state.huntDimensionSearchStartedAt >= 45000) {
    if (endPortal && bot.pathfinder && state.movements) {
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(endPortal.position.x, endPortal.position.y, endPortal.position.z, 2));
      state.lastNote = `hunt: searching end for ${targetName}`;
      return true;
    }
  }

  if (bot.pathfinder && state.movements) {
    const roamX = bot.entity.position.x + (Math.random() * 42 - 21);
    const roamZ = bot.entity.position.z + (Math.random() * 42 - 21);
    bot.pathfinder.setMovements(state.movements);
    bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
  }
  state.lastNote = `hunt: searching for ${targetName}`;
  return true;
}

function setupMovements(bot) {
  if (!bot.pathfinder) return;
  const movements = new Movements(bot);
  movements.allowSprinting = true;
  movements.canDig = true;
  movements.allow1by1towers = true;
  movements.allowParkour = true;
  movements.allowFreeMotion = true;
  movements.maxDropDown = 2;
  movements.infiniteLiquidDropdownDistance = false;
  state.movements = movements;
  bot.pathfinder.setMovements(movements);
}

async function chaseAndAttack(bot, target) {
  if (!target || !target.position || !bot.pathfinder || !state.movements) return false;
  // Never attack self, but allow attacking enemy players.
  if (target.id === bot.entity?.id) {
    return false;
  }
  bot.pathfinder.setMovements(state.movements);
  bot.pathfinder.setGoal(new GoalFollow(target, 1.5), true);
  const dist = bot.entity.position.distanceTo(target.position);
  await ensureCombatHandReady(bot);

  // Only swap weapons when not eating — equip() cancels the eat animation
  const isEating = Date.now() < Number(state.eatingUntil || 0);
  if (!isEating) {
    // If enemy has shield, use best axe if available
    if (enemyHasShield(target)) {
      const items = bot.inventory?.items?.() || [];
      let bestAxe = null, bestAxeScore = -1;
      for (const item of items) {
        const name = String(item?.name || '');
        if (name.endsWith('_axe')) {
          const score = weaponTierScore(name);
          if (score > bestAxeScore) {
            bestAxeScore = score;
            bestAxe = item;
          }
        }
      }
      if (bestAxe && (!bot.heldItem || bot.heldItem.name !== bestAxe.name)) {
        try { await bot.equip(bestAxe, 'hand'); } catch {}
      }
    } else {
      await equipBestMeleeWeapon(bot);
    }
  }

  // Shield block logic: always block if any threat is near
  const threat = getNearestEntity(bot, (entity) => entity.type === 'player' && entity.id !== bot.entity.id && entity.position && isThreatReachable(bot, entity, 4.5));
  const offhandName = String(bot.inventory?.slots?.[45]?.name || '').toLowerCase();
  if (threat.entity || shouldBlockWithShield(bot, target)) {
    if (offhandName === 'shield') {
      try { bot.activateItem(true); state.useHeld = true; } catch {}
    }
  } else if (state.useHeld) {
    try { if (typeof bot.deactivateItem === 'function') bot.deactivateItem(); } catch {};
    state.useHeld = false;
  }

  const now = Date.now();
  if (now - Number(state.lastCombatStrafeSwitchAt || 0) > randomBetween(220, 420)) {
    state.lastCombatStrafeSwitchAt = now;
    state.combatStrafeDir = Math.random() > 0.5 ? 'left' : 'right';
  }

  // Always look at the target while fighting, clamp pitch to avoid sky bug
  try {
    const tgt = target.position.offset(0, target.height ? target.height * 0.6 : 1.2, 0);
    const clampPitch = (pitch) => Math.max(-Math.PI/2, Math.min(Math.PI/2, pitch));
    const botPos = bot.entity.position;
    const dx = tgt.x - botPos.x;
    const dy = tgt.y - (botPos.y + 1.62);
    const dz = tgt.z - botPos.z;
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const yaw = Math.atan2(-dx, dz);
    const pitch = clampPitch(-Math.asin(dy / dist));
    await bot.look(yaw, pitch, true);
  } catch {
    // ignore look failure
  }

  bot.setControlState('sprint', dist > 2.2 && now >= Number(state.noSprintUntil || 0));
  bot.setControlState('forward', dist > 1.55);
  bot.setControlState('back', dist < 1.15);
  const shouldStrafe = dist > 1.8 && dist < 4.8;
  // 45-degree diagonal: alternate strafe side every 600ms — forward+strafe = sqrt(2)x speed
  const chaseSide = Math.floor(Date.now() / 600) % 2 === 0 ? 'left' : 'right';
  const strafeSide = (state.combatStrafeDir === 'left' || state.combatStrafeDir === 'right')
    ? state.combatStrafeDir
    : chaseSide;
  bot.setControlState('left', shouldStrafe && strafeSide === 'left');
  bot.setControlState('right', shouldStrafe && strafeSide === 'right');
  if (dist > 1.8) trySprintJump(bot, 100, 140);

  if (dist <= 4.2 && Date.now() - state.lastAttackAt >= 300) {
    // If enemy has shield and we are not eating, swap to best axe
    const notEating = Date.now() >= Number(state.eatingUntil || 0);
    if (notEating && enemyHasShield(target)) {
      const held = String(bot.heldItem?.name || '');
      if (!held.endsWith('_axe')) {
        const items = bot.inventory?.items?.() || [];
        let bestAxe = null, bestAxeScore = -1;
        for (const item of items) {
          const name = String(item?.name || '');
          if (name.endsWith('_axe')) {
            const score = weaponTierScore(name);
            if (score > bestAxeScore) {
              bestAxeScore = score;
              bestAxe = item;
            }
          }
        }
        if (bestAxe && (!bot.heldItem || bot.heldItem.name !== bestAxe.name)) {
          try { await bot.equip(bestAxe, 'hand'); } catch {}
        }
      }
    }
    // Snap look to target then attack — use lookAt for reliable aim
    try {
      bot.setControlState('sprint', false);
      await bot.lookAt(target.position.offset(0, target.height ? target.height * 0.6 : 1.2, 0), true);
    } catch {}
    // Always attack if in range, even if aim is not perfect
    try {
      bot.attack(target);
      state.lastAttackAt = Date.now();
      state.noSprintUntil = Date.now() + 180;
    } catch {}
  }
  return true;
}

async function moveToDroppedItem(bot, itemEntity) {
  if (!itemEntity || !itemEntity.position || !bot.pathfinder || !state.movements) return false;
  bot.pathfinder.setMovements(state.movements);
  bot.pathfinder.setGoal(new GoalNear(itemEntity.position.x, itemEntity.position.y, itemEntity.position.z, 1));
  return true;
}

async function handleTreeTask(bot) {
  const block = nearestLogBlock(bot, 28);
  if (!block) return false;

  const freshBlock = bot.blockAt(block.position);
  if (!freshBlock || !freshBlock.position || !/(_log|_wood)$/.test(String(freshBlock.name || ''))) {
    return false;
  }

  const dist = bot.entity.position.distanceTo(freshBlock.position);
  if (dist > 3.5) {
    if (bot.pathfinder && state.movements) {
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(freshBlock.position.x, freshBlock.position.y, freshBlock.position.z, 2));
      return true;
    }
    return false;
  }

  if (Date.now() - state.lastDigAt < 1200) {
    return true;
  }

  if (state.miningInProgress) {
    return true;
  }

  try {
    await bot.lookAt(freshBlock.position.offset(0.5, 0.5, 0.5), true);
  } catch {
    // ignore look failure
  }

  if (typeof bot.canDigBlock === 'function' && !bot.canDigBlock(freshBlock)) {
    return true;
  }

  try {
    state.miningInProgress = true;
    state.noJumpUntil = Date.now() + 2000;
    clearMovementControls(bot);
    
    state.lastDigAt = Date.now();
    await bot.dig(freshBlock, true);
    return true;
  } catch {
    return true;
  } finally {
    state.miningInProgress = false;
    state.noJumpUntil = Date.now() + 700;
  }
}

function nearestSafeStoneBlock(bot, maxDistance = 20) {
  const standingX = Math.floor(bot.entity.position.x);
  const standingY = Math.floor(bot.entity.position.y);
  const standingZ = Math.floor(bot.entity.position.z);
  try {
    return bot.findBlock({
      maxDistance,
      matching: (block) => {
        const name = String(block?.name || '');
        if (!/^(stone|deepslate|cobblestone|cobbled_deepslate)$/.test(name)) return false;
        if (!block?.position) return false;
        const bx = Math.floor(block.position.x);
        const by = Math.floor(block.position.y);
        const bz = Math.floor(block.position.z);
        if (bx === standingX && bz === standingZ && by <= standingY) return false;
        if (by < standingY - 1) return false;
        return true;
      }
    });
  } catch {
    return null;
  }
}

async function handleStoneTask(bot) {
  const stoneBlock = nearestSafeStoneBlock(bot, 24);
  state.lastMode = 'stone';
  if (!stoneBlock) {
    if (bot.pathfinder && state.movements) {
      const roamX = bot.entity.position.x + (Math.random() * 18 - 9);
      const roamZ = bot.entity.position.z + (Math.random() * 18 - 9);
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
    }
    state.lastNote = 'stone: searching stone vein';
    return true;
  }

  state.lastNote = 'stone: positioning + mining target';
  await mineBlockWithTool(bot, stoneBlock, ['wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe', 'netherite_pickaxe']);
  return true;
}

async function keepOffhandClearOfBlocks(bot) {
  if (!bot?.inventory) return false;
  const now = Date.now();
  // Never touch inventory while eating — it cancels the eat animation
  if (now < Number(state.eatingUntil || 0)) return false;
  if (now - Number(state.lastOffhandCheckAt || 0) < 150) return false;
  state.lastOffhandCheckAt = now;

  const offhandItemName = String(bot.inventory?.slots?.[45]?.name || '').toLowerCase();
  const justPoppedTotem = state.lastOffhandItemName === 'totem_of_undying' && offhandItemName !== 'totem_of_undying';
  state.lastOffhandItemName = offhandItemName;

  const threat = nearestThreat(bot);
  const preferShield = threat.entity && threat.distance > 0 && threat.distance < 8 && Number(bot.health || 20) > 8;
  const forceTotem = justPoppedTotem || Number(bot.health || 20) <= 14 || (threat.entity && threat.distance > 0 && threat.distance <= 7.5);
  const desiredNames = forceTotem
    ? ['totem_of_undying', 'shield']
    : (preferShield ? ['shield', 'totem_of_undying'] : ['totem_of_undying', 'shield']);

  if (justPoppedTotem) {
    const facts = collectInventoryFacts(bot);
    if (facts.totemSlot >= 0) {
      try {
        bot.setQuickBarSlot(facts.totemSlot);
      } catch {
        // ignore hotbar slot change failures
      }
    }
    state.nextAttackAllowedAt = Math.max(state.nextAttackAllowedAt, now + 280);
  }

  for (const name of desiredNames) {
    const item = findInventoryItem(bot, (entry) => String(entry?.name || '') === name);
    if (!item) continue;
    try {
      await bot.equip(item, 'off-hand');
      return true;
    } catch {
      // try next candidate
    }
  }

  return false;
}

async function ensureBestArmor(bot) {
  if (!bot?.inventory) return false;
  // Never touch inventory while eating — it cancels the eat animation
  if (Date.now() < Number(state.eatingUntil || 0)) return false;

  const armorPriority = {
    netherite: 6,
    diamond: 5,
    iron: 4,
    chainmail: 3,
    golden: 2,
    leather: 1
  };
  const gearTypes = [
    { slot: 'head', suffix: '_helmet' },
    { slot: 'torso', suffix: '_chestplate' },
    { slot: 'legs', suffix: '_leggings' },
    { slot: 'feet', suffix: '_boots' }
  ];

  let changed = false;
  for (const gear of gearTypes) {
    let item = null;
    let bestScore = -1;
    for (const entry of bot.inventory?.items?.() || []) {
      const name = String(entry?.name || '');
      if (!name.endsWith(gear.suffix)) continue;
      const tier = Object.keys(armorPriority).find((candidate) => name.startsWith(candidate + '_'));
      const score = tier ? armorPriority[tier] : 0;
      if (score > bestScore) {
        bestScore = score;
        item = entry;
      }
    }
    if (!item) continue;

    try {
      await bot.equip(item, gear.slot);
      changed = true;
    } catch {
      // ignore failed equip for this slot
    }
  }
  return changed;
}

function applyWaterRecovery(bot) {
  if (!bot?.entity?.isInWater) return false;
  bot.setControlState('jump', true);
  bot.setControlState('forward', true);
  bot.setControlState('sprint', false);
  if (Number(bot.oxygenLevel || 20) < 18) {
    bot.look(bot.entity.yaw, -Math.PI / 6, true).catch(() => {});
  }
  return true;
}

async function tryPillarUp(bot) {
  if (!bot?.entity || !bot?.inventory || !bot.entity.onGround) return false;
  const placeItem = (bot.inventory.items() || []).find((item) => item && /(_planks|_cobblestone|_stone|_deepslate|_dirt|_sandstone|_netherrack|_obsidian|scaffolding)$/.test(String(item.name || '')));
  if (!placeItem) return false;

  const support = bot.blockAt(bot.entity.position.offset(0, -1, 0));
  const feet = bot.blockAt(bot.entity.position);
  if (!support || String(support.name || '') === 'air') return false;
  if (feet && String(feet.name || '') !== 'air') return false;

  try {
    clearMovementControls(bot);
    await bot.equip(placeItem, 'hand');
    await bot.look(bot.entity.yaw, -Math.PI / 8, true).catch(() => {});
    await bot.placeBlock(support, new Vec3(0, 1, 0));
    bot.setControlState('jump', true);
    bot.setControlState('forward', true);
    state.lastNote = 'general1: pillar up';
    await sleep(90);
    return true;
  } catch {
    return false;
  } finally {
    bot.setControlState('jump', false);
  }
}

async function consumeFoodIfNeeded(bot, urgent = false) {
  const now = Date.now();
  if (now - state.lastEatAt < 1200) return false;
  if (now < Number(state.eatingUntil || 0)) return true;

  const health = Number(bot.health || 20);
  const food = Number(bot.food || 20);
  if (!urgent && health > 15 && food > 14) return false;

  const facts = collectInventoryFacts(bot);
  if (facts.utilityFoodSlot < 0) return false;

  try {
    // Switch hotbar slot but keep moving — eat while running/retreating
    if (bot.quickBarSlot !== facts.utilityFoodSlot) {
      bot.setQuickBarSlot(facts.utilityFoodSlot);
      await sleep(60);
    }
    state.eatingUntil = Date.now() + 1650;
    bot.activateItem(false); // hold right-click to eat
    // Sprint forward while eating so movement continues
    bot.setControlState('sprint', true);
    bot.setControlState('forward', true);
    await sleep(1550);
    if (typeof bot.deactivateItem === 'function') {
      bot.deactivateItem();
    }
    state.eatingUntil = 0;
    state.lastEatAt = Date.now();
    return true;
  } catch {
    state.eatingUntil = 0;
    return false;
  }
}

function retreatFromThreat(bot, threatEntity) {
  if (!threatEntity?.position || !bot.pathfinder || !state.movements) return false;
  const dx = bot.entity.position.x - threatEntity.position.x;
  const dz = bot.entity.position.z - threatEntity.position.z;
  const len = Math.hypot(dx, dz);
  if (len < 0.01) return false;

  const scale = 14 / len;
  const targetX = bot.entity.position.x + (dx * scale);
  const targetZ = bot.entity.position.z + (dz * scale);
  bot.pathfinder.setMovements(state.movements);
  bot.pathfinder.setGoal(new GoalNear(targetX, bot.entity.position.y, targetZ, 2));
  bot.setControlState('sprint', true);
  bot.setControlState('forward', true);
  trySprintJump(bot, 160, 260);
  state.retreatUntil = Date.now() + 1800;
  return true;
}

async function handleCraftPrep(bot) {
  if (!hasCraftObjective()) return false;
  return handleTreeTask(bot);
}

async function runAutonomousHelpers(bot) {
  if (!bot || !state.connected) return false;
  state.helperActive = false;

  const now = Date.now();

  // Non-blocking upkeep tasks so movement/combat can continue concurrently.
  scheduleConcurrentTask('armor', async () => {
    await ensureBestArmor(bot);
  }, 5000);
  scheduleConcurrentTask('offhand', async () => {
    await keepOffhandClearOfBlocks(bot);
  }, 2500);
  scheduleConcurrentTask('teamStatus', async () => {
    maybeBroadcastTeamStatus(bot);
  }, 1200);

  if (applyWaterRecovery(bot)) {
    state.helperActive = true;
    state.lastNote = 'survival: swimming to surface';
    return true;
  }

  if (await tryPillarUp(bot)) {
    state.helperActive = true;
    return true;
  }

  if (now < state.kbRecoveryUntil) {
    state.helperActive = true;
    clearMovementControls(bot);
    if (now < state.kbStrafeUntil && (state.kbStrafeDir === 'left' || state.kbStrafeDir === 'right')) {
      bot.setControlState(state.kbStrafeDir, true);
    }
    return true;
  }


  const healthNow = Number(bot.health || 20);
  const threat = nearestThreat(bot);
  const tookDamage = healthNow < state.lastHealth;
  const offhandName = String(bot.inventory?.slots?.[45]?.name || '').toLowerCase();
  const hasTotemInOffhand = offhandName === 'totem_of_undying';

  // Always update lastHealth immediately after checking for damage
  if (tookDamage) {
    const attacker = findLikelyRecentAttacker(bot, 7.5) || threat.entity;
    if (attacker) {
      state.revengeTargetId = attacker.id;
      state.revengeExpireAt = now + 18000;
    }
    state.kbRecoveryUntil = Math.max(state.kbRecoveryUntil, now + randomBetween(380, 760));
    state.kbStrafeUntil = Math.max(state.kbStrafeUntil, now + randomBetween(220, 480));
    state.kbStrafeDir = Math.random() > 0.5 ? 'left' : 'right';
  }
  state.lastHealth = healthNow;

  // Eat proactively when not in close combat — keeps health regen running
  const foodLevel = Number(bot.food || 20);
  if (foodLevel <= 16 && (!threat.entity || threat.distance < 0 || threat.distance > 8)) {
    state.helperActive = true;
    const foraged = await forageFood(bot);
    if (foraged) {
      state.lastHealth = healthNow;
      return true;
    }
  } else if (foodLevel <= 11) {
    // Critically hungry — eat even in combat (while moving)
    state.helperActive = true;
    const foraged = await forageFood(bot);
    if (foraged) {
      state.lastHealth = healthNow;
      return true;
    }
  }

  if (healthNow <= 10 && threat.entity && threat.distance > 0 && threat.distance < 10) {
    state.helperActive = true;
    await keepOffhandClearOfBlocks(bot);
    const freshOffhandName = String(bot.inventory?.slots?.[45]?.name || '').toLowerCase();
    if (freshOffhandName !== 'totem_of_undying') {
      state.lastNote = 'survival: prioritize totem';
      // Retreat AND try to eat simultaneously while running
      retreatFromThreat(bot, threat.entity);
      consumeFoodIfNeeded(bot, true).catch(() => {});
      return true;
    }
    // Retreat and eat simultaneously
    retreatFromThreat(bot, threat.entity);
    consumeFoodIfNeeded(bot, true).catch(() => {});
    state.lastNote = 'survival: retreat + eat';
    return true;
  }

  if (healthNow <= 14 && (!threat.entity || threat.distance < 0 || threat.distance > 6)) {
    // Not in immediate melee range — eat while continuing to move
    const ate = await consumeFoodIfNeeded(bot, true);
    if (ate) {
      state.helperActive = true;
      return true;
    }
  } else if (healthNow <= 12 && hasTotemInOffhand && threat.entity && threat.distance > 0 && threat.distance <= 6) {
    // In close combat but critically low — eat while strafing away
    retreatFromThreat(bot, threat.entity);
    consumeFoodIfNeeded(bot, true).catch(() => {});
    state.helperActive = true;
    return true;
  }


  const combatObjective = isCombatObjective();
  const general1Objective = isGeneral1Objective();
  const prisonEscapeObjective = isPrisonEscapeObjective();
  const revengeTarget = getEntityById(bot, state.revengeTargetId);
  // Always prioritize revenge target if set and not expired
  if (revengeTarget && now < state.revengeExpireAt) {
    state.helperActive = true;
    return chaseAndAttack(bot, revengeTarget, true);
  }

  const proactiveHostile = getNearestEntity(bot, (entity) => isHostileMob(entity) && entity.position && isThreatReachable(bot, entity, 3.25));
  if (proactiveHostile.entity && proactiveHostile.distance > 0 && proactiveHostile.distance <= 10.5) {
    state.helperActive = true;
    state.lastNote = 'combat: proactive hostile engage';
    return chaseAndAttack(bot, proactiveHostile.entity);
  }

  if (combatObjective) {
    const objectiveTarget = selectCombatTarget(bot, true);
    if (objectiveTarget?.position) {
      state.helperActive = true;
      state.lastNote = `combat: engaging ${String(objectiveTarget.username || objectiveTarget.name || 'target')}`;
      return chaseAndAttack(bot, objectiveTarget);
    }

    // No immediate target: keep moving and patrolling instead of standing still.
    if (bot.pathfinder && state.movements && (now - Number(state.lastBaseRoamAt || 0) > 1200)) {
      state.lastBaseRoamAt = now;
      const roamX = bot.entity.position.x + (Math.random() * 52 - 26);
      const roamZ = bot.entity.position.z + (Math.random() * 52 - 26);
      bot.pathfinder.setMovements(state.movements);
      bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
      state.lastNote = 'combat: patrol roam';
      state.helperActive = true;
      return true;
    }

    // Keep momentum between pathfinder updates.
    bot.setControlState('sprint', true);
    bot.setControlState('forward', true);
    trySprintJump(bot, 120, 170);
    state.helperActive = true;
    return true;
  }

  if (general1Objective) {
    state.helperActive = true;
    return runGeneral1Progression(bot);
  }

  // Mild anti-idle fallback: if no threat and no recent backend decision, nudge exploration.
  if (!prisonEscapeObjective && !threat.entity && bot.pathfinder && state.movements && (now - Number(state.lastDecisionAt || 0) > 1800) && (now - Number(state.lastBaseRoamAt || 0) > 1500)) {
    state.lastBaseRoamAt = now;
    const roamX = bot.entity.position.x + (Math.random() * 34 - 17);
    const roamZ = bot.entity.position.z + (Math.random() * 34 - 17);
    bot.pathfinder.setMovements(state.movements);
    bot.pathfinder.setGoal(new GoalNear(roamX, bot.entity.position.y, roamZ, 2));
    state.lastNote = 'general: anti-idle roam';
    state.helperActive = true;
    return true;
  }

  return false;
}

function getForwardProbe(bot, distance = 1.1) {
  const yaw = Number(bot.entity?.yaw || 0);
  const origin = bot.entity.position.offset(0, 0, 0);
  const dx = -Math.sin(yaw) * distance;
  const dz = Math.cos(yaw) * distance;
  return {
    feet: origin.offset(dx, 0, dz).floored(),
    head: origin.offset(dx, 1, dz).floored(),
    left: origin.offset(dx + Math.cos(yaw) * 0.9, 0, dz + Math.sin(yaw) * 0.9).floored(),
    right: origin.offset(dx - Math.cos(yaw) * 0.9, 0, dz - Math.sin(yaw) * 0.9).floored()
  };
}

function applyMoveAngleIntent(action = {}) {
  const moveAngle = Number(action.moveAngle);
  if (!Number.isFinite(moveAngle)) return action;
  if (action.forward || action.back || action.left || action.right) return action;

  const normalized = ((((moveAngle + 180) % 360) + 360) % 360) - 180;
  return {
    ...action,
    forward: Math.abs(normalized) <= 67.5,
    back: Math.abs(normalized) >= 112.5,
    left: normalized < -22.5 && normalized > -157.5,
    right: normalized > 22.5 && normalized < 157.5
  };
}

function collectInventoryFacts(bot) {
  const items = bot.inventory?.items?.() || [];
  const facts = {
    selectedItem: '',
    selectedItemCount: 0,
    swordSlot: -1,
    axeSlot: -1,
    pickaxeSlot: -1,
    blockSlot: -1,
    waterBucketSlot: -1,
    utilityFoodSlot: -1,
    cobwebSlot: -1,
    obsidianSlot: -1,
    endCrystalSlot: -1,
    respawnAnchorSlot: -1,
    glowstoneSlot: -1,
    totemSlot: -1,
    pearlSlot: -1,
    densityMaceSlot: -1,
    breachMaceSlot: -1,
    maceSlot: -1,
    maceBreachLevel: 0,
    bowSlot: -1,
    windChargeSlot: -1,
    windChargeCount: 0,
    shieldSlot: -1,
    combatPotionSlot: -1,
    combatPotionCount: 0,
    fireballSlot: -1,
    fireballCount: 0,
    tntSlot: -1,
    tntCount: 0,
    boatSlot: -1,
    boatCount: 0,
    hotbarBlocks: 0,
    hasBlocks: false,
    hasWaterBucket: false,
    hasMeleeWeapon: false,
    hasElytra: false,
    ironCount: 0,
    redstoneCount: 0,
    diamondCount: 0,
    goldCount: 0,
    emeraldCount: 0,
    netheriteIngotCount: 0,
    netheriteScrapCount: 0,
    ancientDebrisCount: 0,
    netheriteUpgradeTemplateCount: 0,
    enchantedBookCount: 0,
    cobwebCount: 0,
    obsidianCount: 0,
    endCrystalCount: 0,
    respawnAnchorCount: 0,
    glowstoneCount: 0,
    totemCount: 0,
    pearlCount: 0,
    densityMaceCount: 0,
    breachMaceCount: 0,
    maceCount: 0,
    flintAndSteelSlot: -1,
    flintAndSteelCount: 0,
    blazeRodCount: 0,
    eyeOfEnderCount: 0
  };

  for (const item of items) {
    const name = String(item.name || '');
    const count = Number(item.count || 0);
    const inventoryIndex = Number(item.slot);
    const isHotbar = inventoryIndex >= 36 && inventoryIndex <= 44;
    const hotbarSlot = isHotbar ? (inventoryIndex - 36) : -1;

    if (name.endsWith('_sword') && hotbarSlot >= 0 && facts.swordSlot < 0) facts.swordSlot = hotbarSlot;
    if (name.endsWith('_axe') && hotbarSlot >= 0 && facts.axeSlot < 0) facts.axeSlot = hotbarSlot;
    if (name.endsWith('_pickaxe') && hotbarSlot >= 0 && facts.pickaxeSlot < 0) facts.pickaxeSlot = hotbarSlot;
    if (name === 'water_bucket' && hotbarSlot >= 0) facts.waterBucketSlot = hotbarSlot;
    if (name === 'totem_of_undying' && hotbarSlot >= 0 && facts.totemSlot < 0) facts.totemSlot = hotbarSlot;
    if (name === 'ender_pearl' && hotbarSlot >= 0 && facts.pearlSlot < 0) facts.pearlSlot = hotbarSlot;
    if (name === 'shield' && hotbarSlot >= 0 && facts.shieldSlot < 0) facts.shieldSlot = hotbarSlot;
    if (name.endsWith('_bow') && hotbarSlot >= 0 && facts.bowSlot < 0) facts.bowSlot = hotbarSlot;
    if (name === 'wind_charge' && hotbarSlot >= 0 && facts.windChargeSlot < 0) facts.windChargeSlot = hotbarSlot;
    if (name === 'cobweb' && hotbarSlot >= 0 && facts.cobwebSlot < 0) facts.cobwebSlot = hotbarSlot;
    if (name === 'obsidian' && hotbarSlot >= 0 && facts.obsidianSlot < 0) facts.obsidianSlot = hotbarSlot;
    if (name === 'end_crystal' && hotbarSlot >= 0 && facts.endCrystalSlot < 0) facts.endCrystalSlot = hotbarSlot;
    if (name === 'respawn_anchor' && hotbarSlot >= 0 && facts.respawnAnchorSlot < 0) facts.respawnAnchorSlot = hotbarSlot;
    if (name === 'glowstone' && hotbarSlot >= 0 && facts.glowstoneSlot < 0) facts.glowstoneSlot = hotbarSlot;
    if (name === 'flint_and_steel' && hotbarSlot >= 0 && facts.flintAndSteelSlot < 0) facts.flintAndSteelSlot = hotbarSlot;
    if (name === 'fire_charge' && hotbarSlot >= 0 && facts.fireballSlot < 0) facts.fireballSlot = hotbarSlot;
    if (name === 'tnt' && hotbarSlot >= 0 && facts.tntSlot < 0) facts.tntSlot = hotbarSlot;
    if (name.endsWith('_boat') && hotbarSlot >= 0 && facts.boatSlot < 0) facts.boatSlot = hotbarSlot;
    if (/potion/.test(name) && hotbarSlot >= 0 && facts.combatPotionSlot < 0) facts.combatPotionSlot = hotbarSlot;
    if (name === 'density_mace' && hotbarSlot >= 0 && facts.densityMaceSlot < 0) facts.densityMaceSlot = hotbarSlot;
    if (name === 'breach_mace' && hotbarSlot >= 0 && facts.breachMaceSlot < 0) facts.breachMaceSlot = hotbarSlot;
    if (name.endsWith('mace') && hotbarSlot >= 0 && facts.maceSlot < 0) facts.maceSlot = hotbarSlot;

    // Recognize building blocks anywhere in inventory, not just hotbar
    if (/(planks|stone|cobblestone|wool|dirt|sandstone|deepslate|netherrack|obsidian|log|wood|bricks|concrete|terracotta|glass|slab|stairs|wall|fence|door|trapdoor|block)/.test(name)) {
      facts.hasBlocks = true;
      // Prefer hotbar slot for blockSlot, but fallback to any inventory slot
      if (facts.blockSlot < 0 && isHotbar) facts.blockSlot = hotbarSlot;
      if (facts.blockSlot < 0 && inventoryIndex >= 0) facts.blockSlot = hotbarSlot >= 0 ? hotbarSlot : 0;
    }

    if (isHotbar && /(planks|stone|cobblestone|wool|dirt|sandstone|deepslate|netherrack|obsidian)/.test(name)) {
      facts.hotbarBlocks += count;
    }

    if (isHotbar && /(bread|steak|porkchop|carrot|potato|golden_apple|apple|cooked)/.test(name) && facts.utilityFoodSlot < 0) {
      facts.utilityFoodSlot = hotbarSlot;
    }

    if (name.endsWith('_sword') || name.endsWith('_axe') || name.endsWith('_trident') || name.endsWith('mace')) {
      facts.hasMeleeWeapon = true;
    }
    if (name === 'water_bucket') facts.hasWaterBucket = true;
    if (name === 'elytra') facts.hasElytra = true;

    if (name === 'iron_ingot' || name === 'raw_iron') facts.ironCount += count;
    if (name === 'redstone' || name === 'redstone_dust') facts.redstoneCount += count;
    if (name === 'diamond') facts.diamondCount += count;
    if (name === 'gold_ingot' || name === 'raw_gold') facts.goldCount += count;
    if (name === 'emerald') facts.emeraldCount += count;
    if (name === 'netherite_ingot') facts.netheriteIngotCount += count;
    if (name === 'netherite_scrap') facts.netheriteScrapCount += count;
    if (name === 'ancient_debris') facts.ancientDebrisCount += count;
    if (name.includes('netherite_upgrade_smithing_template')) facts.netheriteUpgradeTemplateCount += count;
    if (name === 'enchanted_book') facts.enchantedBookCount += count;
    if (name === 'cobweb') facts.cobwebCount += count;
    if (name === 'obsidian') facts.obsidianCount += count;
    if (name === 'end_crystal') facts.endCrystalCount += count;
    if (name === 'respawn_anchor') facts.respawnAnchorCount += count;
    if (name === 'glowstone') facts.glowstoneCount += count;
    if (name === 'totem_of_undying') facts.totemCount += count;
    if (name === 'ender_pearl') facts.pearlCount += count;
    if (name === 'density_mace') facts.densityMaceCount += count;
    if (name === 'breach_mace') facts.breachMaceCount += count;
    if (name.endsWith('mace')) facts.maceCount += count;
    if (/potion/.test(name)) facts.combatPotionCount += count;
    if (name === 'wind_charge') facts.windChargeCount += count;
    if (name === 'fire_charge') facts.fireballCount += count;
    if (name === 'tnt') facts.tntCount += count;
    if (name.endsWith('_boat')) facts.boatCount += count;
    if (name === 'flint_and_steel') facts.flintAndSteelCount += count;
    if (name === 'blaze_rod') facts.blazeRodCount += count;
    if (name === 'ender_eye') facts.eyeOfEnderCount += count;
  }

  const selectedHotbarSlot = Number(bot.quickBarSlot || 0);
  const selectedItem = bot.heldItem;
  facts.selectedItem = selectedItem?.name || '';
  facts.selectedItemCount = selectedItem?.count || 0;

  if (facts.blockSlot < 0 && selectedItem && /(planks|stone|cobblestone|wool|dirt|sandstone|deepslate|netherrack|obsidian)/.test(selectedItem.name)) {
    facts.blockSlot = selectedHotbarSlot;
  }

  return facts;
}

function collectDroppedLootFacts(bot) {
  let nearestDroppedItemDistance = -1;
  let nearestDroppedItemDx = 0;
  let nearestDroppedItemDz = 0;
  let nearbyDroppedCount = 0;

  for (const entity of Object.values(bot.entities)) {
    if (!entity || entity.name !== 'item' || !entity.position) continue;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist <= 12) nearbyDroppedCount += 1;
    if (nearestDroppedItemDistance < 0 || dist < nearestDroppedItemDistance) {
      nearestDroppedItemDistance = dist;
      nearestDroppedItemDx = entity.position.x - bot.entity.position.x;
      nearestDroppedItemDz = entity.position.z - bot.entity.position.z;
    }
  }

  return {
    nearbyDroppedTotemCount: 0,
    nearbyDroppedPearlCount: 0,
    nearbyDroppedPotionCount: 0,
    nearbyDroppedGappleCount: nearbyDroppedCount,
    nearbyDroppedCrystalCount: 0,
    nearestDroppedItemDistance,
    nearestDroppedItemDx,
    nearestDroppedItemDz
  };
}

function collectState(bot) {
  const inventoryFacts = collectInventoryFacts(bot);
  const droppedFacts = collectDroppedLootFacts(bot);

  const nearestEnemy = getNearestEntity(bot, (entity) =>
    entity.type === 'player' &&
    entity.username &&
    entity.username !== bot.username &&
    isThreatReachable(bot, entity, 4.5)
  );
  const nearestHostile = getNearestEntity(bot, (entity) => isHostileMob(entity) && isThreatReachable(bot, entity, 3.25));

  const focusedEntity = nearestEnemy.entity || nearestHostile.entity || null;
  const focusedDistance = focusedEntity ? bot.entity.position.distanceTo(focusedEntity.position) : -1;

  const look = bot.entity.yaw;
  const pitch = bot.entity.pitch;
  const lookX = -Math.sin(look) * Math.cos(pitch);
  const lookY = -Math.sin(pitch);
  const lookZ = Math.cos(look) * Math.cos(pitch);

  const nearestEnemyDx = nearestEnemy.entity ? nearestEnemy.entity.position.x - bot.entity.position.x : 0;
  const nearestEnemyDy = nearestEnemy.entity ? nearestEnemy.entity.position.y - bot.entity.position.y : 0;
  const nearestEnemyDz = nearestEnemy.entity ? nearestEnemy.entity.position.z - bot.entity.position.z : 0;
  const nearestHostileDx = nearestHostile.entity ? nearestHostile.entity.position.x - bot.entity.position.x : 0;
  const nearestHostileDz = nearestHostile.entity ? nearestHostile.entity.position.z - bot.entity.position.z : 0;

  return {
    x: bot.entity.position.x,
    y: bot.entity.position.y,
    z: bot.entity.position.z,
    yaw: bot.entity.yaw * (180 / Math.PI),
    pitch: bot.entity.pitch * (180 / Math.PI),
    health: Number(bot.health || 20),
    food: Number(bot.food || 20),
    onGround: Boolean(bot.entity.onGround),
    isSprinting: false,
    isSneaking: false,
    isTouchingWater: Boolean(bot.entity.isInWater),
    verticalSpeed: Number(bot.entity.velocity?.y || 0),
    horizontalSpeed: Math.hypot(Number(bot.entity.velocity?.x || 0), Number(bot.entity.velocity?.z || 0)),
    fallDistance: 0,
    worldTime: Number(bot.time?.timeOfDay || 0),
    facing: 'north',
    lookX,
    lookY,
    lookZ,
    nearestEnemyName: nearestEnemy.entity?.username || '',
    nearestEnemyDistance: nearestEnemy.distance,
    nearestEnemyHealth: Number(nearestEnemy.entity?.health || 0),
    nearestEnemyMainItem: '',
    nearestEnemyArmorPieces: 0,
    nearestEnemyHasMeleeWeapon: true,
    nearestEnemyHasShield: false,
    nearestEnemyVelX: Number(nearestEnemy.entity?.velocity?.x || 0),
    nearestEnemyVelY: Number(nearestEnemy.entity?.velocity?.y || 0),
    nearestEnemyVelZ: Number(nearestEnemy.entity?.velocity?.z || 0),
    nearestEnemyDy,
    nearestEnemyDx,
    nearestEnemyDz,
    nearestHostile: nearestHostile.entity?.name || '',
    nearestHostileDistance: nearestHostile.distance,
    nearestHostileDx,
    nearestHostileDz,
    focusedEntity: focusedEntity?.name || '',
    focusedDistance,
    hasSpeedEffect: false,
    hasStrengthEffect: false,
    villagerNearbyCount: 0,
    bedNearby: false,
    nearestBedDistance: -1,
    nearestBedDefenseScore: 0,
    nearestBedDefenseBlock: '',
    strongholdEstX: 0,
    strongholdEstZ: 0,
    strongholdTriangulated: false,
    dimensionId: String(bot.game?.dimension || 'overworld'),
    fallHeight: state.inFallSequence ? state.maceLastY - bot.entity.position.y : 0,
    inFalling: !bot.entity.onGround && Number(bot.entity.velocity?.y || 0) < -0.1,
    densityMaceSlotForFalling: selectMaceForFallingDamage(bot),
    breachMaceSlotForBreak: selectMaceForBreachSwap(bot),
    ...inventoryFacts,
    ...droppedFacts
  };
}

function applyAction(bot, action = {}) {
  action = applyMoveAngleIntent(action);

  // Do NOT freeze movement while eating — bot eats while running/retreating

  if (inKbRecoveryWindow()) {
    clearMovementControls(bot);
    if (Date.now() < state.kbStrafeUntil && (state.kbStrafeDir === 'left' || state.kbStrafeDir === 'right')) {
      bot.setControlState(state.kbStrafeDir, true);
    }
    return;
  }

  // Track falling for density mace usage
  const isCurrentlyFalling = !bot.entity.onGround && Number(bot.entity.velocity?.y || 0) < -0.1;
  if (isCurrentlyFalling && !state.inFallSequence) {
    state.inFallSequence = true;
    state.fallStartY = bot.entity.position.y;
    state.maceLastY = bot.entity.position.y;
  } else if (!isCurrentlyFalling && state.inFallSequence) {
    // Fell enough to use density mace?
    const fallDamage = state.maceLastY - bot.entity.position.y;
    if (fallDamage >= 3.5) {
      // Auto-equip density mace on landing if we fell enough
      equipMaceForFalling(bot).catch(() => {});
    }
    state.inFallSequence = false;
  }
  if (state.inFallSequence) {
    state.maceLastY = Math.max(state.maceLastY, bot.entity.position.y);
  }

  if (state.miningInProgress || inNoJumpWindow()) {
    try {
      if (bot.pathfinder) {
        bot.pathfinder.setGoal(null);
      }
    } catch {
      // ignore pathfinder clear failures
    }
    clearMovementControls(bot);
    return;
  }

  if (bot.pathfinder && !state.helperActive && Date.now() >= state.retreatUntil) {
    bot.pathfinder.setGoal(null);
  }

  const now = Date.now();
  const engagedTarget = Number(action.focusedDistance || 99) <= 6.5 || Boolean(action.attack);
  if (!engagedTarget && !state.miningInProgress) {
    if (state.nextMicroPauseAt <= 0) {
      state.nextMicroPauseAt = now + randomBetween(1800, 3600);
    }
    if (now >= state.nextMicroPauseAt) {
      state.microPauseUntil = now + randomBetween(70, 150);
      state.nextMicroPauseAt = now + randomBetween(2100, 3900);
    }
  } else {
    state.microPauseUntil = 0;
    state.nextMicroPauseAt = now + randomBetween(1400, 2400);
  }

  if (now >= state.strafeLockUntil) {
    if (Boolean(action.left) !== Boolean(action.right)) {
      state.strafeLockDir = Boolean(action.left) ? 'left' : 'right';
      state.strafeLockUntil = now + randomBetween(160, 340);
    } else {
      state.strafeLockDir = '';
      state.strafeLockUntil = now + randomBetween(90, 180);
    }
  }

  const microPaused = now < state.microPauseUntil;
  const forwardProbe = getForwardProbe(bot, action.sprint ? 1.35 : 1.05);
  const obstacleAhead = hasSolidBlockNear(bot, forwardProbe.feet) || hasSolidBlockNear(bot, forwardProbe.head);
  const leftBlocked = hasSolidBlockNear(bot, forwardProbe.left);
  const rightBlocked = hasSolidBlockNear(bot, forwardProbe.right);

  if (action.forward && obstacleAhead && bot.entity.onGround) {
    action.jump = true;
    if (!action.left && !action.right) {
      action.left = !leftBlocked && (rightBlocked || (Math.floor(now / 180) % 2 === 0));
      action.right = !action.left && !rightBlocked;
    }
  }

  const controls = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'];
  for (const key of controls) {
    if (key === 'jump' && (state.miningInProgress || inNoJumpWindow())) {
      bot.setControlState(key, false);
      continue;
    }
    let desired = Boolean(action[key]);
    if (microPaused && (key === 'forward' || key === 'back' || key === 'left' || key === 'right' || key === 'sprint')) {
      desired = false;
    }
    if ((key === 'left' || key === 'right') && state.strafeLockDir) {
      desired = key === state.strafeLockDir;
    }
    if (key === 'forward' || key === 'back' || key === 'left' || key === 'right') {
      setControlStateSmoothed(bot, key, desired, 170, 110);
      continue;
    }
    if (key === 'sprint' && now < Number(state.noSprintUntil || 0)) {
      desired = false;
    }
    if (key === 'sprint' || key === 'sneak') {
      setControlStateSmoothed(bot, key, desired, 140, 100);
      continue;
    }
    // Jump should stay responsive, but not hyper-twitchy.
    setControlStateSmoothed(bot, key, desired, 75, 75);
  }

  if (
    Boolean(action.forward)
    && Boolean(action.sprint)
    && !Boolean(action.back)
    && bot.entity.onGround
    && !state.miningInProgress
    && Date.now() - Number(state.lastSprintJumpAt || 0) > randomBetween(100, 140)
  ) {
    trySprintJump(bot, 100, 140);
  }

  // 45-degree diagonal strafe while sprinting forward for ~2% ground speed gain
  if (Boolean(action.forward) && Boolean(action.sprint) && !Boolean(action.back) && !microPaused) {
    const diagSide = state.strafeLockDir || (Math.floor(now / 600) % 2 === 0 ? 'left' : 'right');
    if (diagSide === 'left' && !leftBlocked && !Boolean(action.right)) bot.setControlState('left', true);
    if (diagSide === 'right' && !rightBlocked && !Boolean(action.left)) bot.setControlState('right', true);
  }

  applyUnstuckLogic(bot, action);

  // Smart mace selection based on action hints
  let finalSlot = Number(action.hotbarSlot);
  const maceHint = String(action.maceType || '').toLowerCase();
  
  if (maceHint === 'density' && !Number.isFinite(finalSlot)) {
    finalSlot = selectMaceForFallingDamage(bot);
  } else if (maceHint === 'breach' && !Number.isFinite(finalSlot)) {
    finalSlot = selectMaceForBreachSwap(bot);
  }
  
  if (Number.isFinite(finalSlot) && finalSlot >= 0 && finalSlot < 9) {
    bot.setQuickBarSlot(finalSlot);
  }

  const rawYawDelta = Number(action.yawDelta || 0);
  const rawPitchDelta = Number(action.pitchDelta || 0);
  const targetYawDelta = clamp(rawYawDelta, -7, 7);
  const targetPitchDelta = clamp(rawPitchDelta, -5, 5);
  const yawDelta = (state.lastYawDelta * 0.58) + (targetYawDelta * 0.42) + randomBetween(-0.18, 0.18);
  const pitchDelta = (state.lastPitchDelta * 0.6) + (targetPitchDelta * 0.4) + randomBetween(-0.12, 0.12);
  const filteredYawDelta = Math.abs(yawDelta) < 0.22 ? 0 : yawDelta;
  const filteredPitchDelta = Math.abs(pitchDelta) < 0.16 ? 0 : pitchDelta;

  // When action.attack is true and a target is within melee range, snap look directly
  // at them. This prevents the yaw-delta smoothing from drifting aim away every tick.
  if (action.attack) {
    const snapTarget = selectCombatTarget(bot, isCombatObjective());
    if (snapTarget?.position && bot.entity.position.distanceTo(snapTarget.position) <= 5.0) {
      const snapPos = snapTarget.position.offset(0, snapTarget.height ? snapTarget.height * 0.6 : 1.2, 0);
      bot.lookAt(snapPos, true).catch(() => {});
      state.lastYawDelta = 0;
      state.lastPitchDelta = 0;
    } else {
      state.lastYawDelta = filteredYawDelta;
      state.lastPitchDelta = filteredPitchDelta;
      if (filteredYawDelta !== 0 || filteredPitchDelta !== 0) {
        const newYaw = bot.entity.yaw + (filteredYawDelta * Math.PI / 180);
        const newPitch = clamp(bot.entity.pitch + (filteredPitchDelta * Math.PI / 180), -Math.PI / 2, Math.PI / 2);
        bot.look(newYaw, newPitch, true).catch(() => {});
      }
    }
  } else {
    state.lastYawDelta = filteredYawDelta;
    state.lastPitchDelta = filteredPitchDelta;
    if (filteredYawDelta !== 0 || filteredPitchDelta !== 0) {
      const newYaw = bot.entity.yaw + (filteredYawDelta * Math.PI / 180);
      const newPitch = clamp(bot.entity.pitch + (filteredPitchDelta * Math.PI / 180), -Math.PI / 2, Math.PI / 2);
      bot.look(newYaw, newPitch, true).catch(() => {});
    }
  }

  if (action.attack && Number(bot.health || 20) > 6 && Date.now() >= state.nextAttackAllowedAt) {
    const combatObjective = isCombatObjective();
    let target = selectCombatTarget(bot, combatObjective);
    const objectiveText = String(state.objective || '').toLowerCase();
    const explicitPlayerAttack = /attack|kill|hunt|pvp|fight|combat/.test(objectiveText);
    
    // Only attack players if they hit us first (revenge) or if no hostile mob nearby
    if (target && target.type === 'player') {
      const hostileMob = getNearestEntity(bot, (entity) => isHostileMob(entity) && entity.position && isThreatReachable(bot, entity, 3.25)).entity;
      const isRevengeTarget = target.id === state.revengeTargetId && Date.now() < state.revengeExpireAt;
      if (!explicitPlayerAttack && hostileMob && !isRevengeTarget) {
        target = hostileMob; // Attack hostile mob instead of player
      } else if (!explicitPlayerAttack && !isRevengeTarget) {
        target = null; // Don't attack player unless they hit us first
      }
    }

    const now = Date.now();
    if (target && target.position && now - Number(state.lastCombatStrafeSwitchAt || 0) > randomBetween(220, 420)) {
      state.lastCombatStrafeSwitchAt = now;
      state.combatStrafeDir = Math.random() > 0.5 ? 'left' : 'right';
    }
    if (target && target.position && bot.entity.position.distanceTo(target.position) <= 4.4) {
      const closeDist = bot.entity.position.distanceTo(target.position);
      const strafeNow = closeDist > 1.8 && closeDist < 4.8;
      action.left = strafeNow && state.combatStrafeDir === 'left';
      action.right = strafeNow && state.combatStrafeDir === 'right';
      action.forward = closeDist > 1.45;
      action.back = closeDist < 1.05;
    }

    const attackCd = Number(action.attackCooldownMs || 0);
    const minDelay = attackCd > 0 ? clamp(attackCd, 525, 900) : randomBetween(540, 830);
    if (target && bot.entity.position.distanceTo(target.position) <= 4.1 && Date.now() - state.lastAttackAt >= minDelay) {
      // Don't equip mid-eat — that cancels eating
      if (Date.now() >= Number(state.eatingUntil || 0)) {
        equipBestMeleeWeapon(bot).catch(() => {});
      }
      // Always snap look to target before deciding to attack
      const aimPos = target.position.offset(0, target.height ? target.height * 0.6 : 1.2, 0);
      bot.lookAt(aimPos, true).catch(() => {});
      if (!targetAimAligned(bot, target, 0.94)) {
        // still rotating — give it one tick before attacking
        state.nextAttackAllowedAt = Date.now() + randomBetween(30, 60);
      } else {
        bot.setControlState('sprint', false);
        bot.attack(target);
        state.lastAttackAt = Date.now();
        state.noSprintUntil = Date.now() + 180;
        state.nextAttackAllowedAt = Date.now() + randomBetween(90, 180);
      }
    }
  }

  const heldName = String(bot.heldItem?.name || '').toLowerCase();
  
  // Don't place blocks if holding placeable blocks
  const suppressUse = isLikelyPlaceableBlockName(heldName);
  
  // Allow use (right-click) for special items
  const isSpecialItem = /wind_charge|mace|ender_pearl|bow|trident|flint_and_steel/.test(heldName);
  const allowUse = isSpecialItem || !suppressUse;

  const offhandItemName = String(bot.inventory?.slots?.[45]?.name || '').toLowerCase();
  const shouldUseOffhand = offhandItemName === 'shield';

  if (action.use && !state.useHeld && allowUse) {
    try {
      bot.activateItem(shouldUseOffhand);
      state.useHeld = true;
    } catch {
      // ignore use failures
    }
    if (heldName.includes('wind_charge')) {
      state.lastWindChargeUseAt = Date.now();
      state.useHeld = false;
      try {
        if (typeof bot.deactivateItem === 'function') {
          bot.deactivateItem();
        }
      } catch {
        // ignore deactivate failures for one-shot items
      }
    }
  } else if (action.use && suppressUse) {
    state.useHeld = false;
  } else if (!action.use && state.useHeld) {
    try {
      if (typeof bot.deactivateItem === 'function') {
        bot.deactivateItem();
      }
    } catch {
      // ignore deactivate failures
    }
    state.useHeld = false;
  }
}

async function requestDecision() {
  if (!state.bot || !state.connected || state.stopping || state.decisionInFlight) return;
  if (Date.now() - state.lastSpawnAt < 450) return;
  state.decisionInFlight = true;
  const bot = state.bot;
  try {
    const helperHandled = await runAutonomousHelpers(bot);
    if (helperHandled) {
      state.lastDecisionAt = Date.now();
      return;
    }
    const payload = {
      sessionId: state.sessionId,
      objective: state.objective,
      state: collectState(bot)
    };

    const abortCtrl = new AbortController();
    const abortTimer = setTimeout(() => abortCtrl.abort(), 8000);
    let response;
    try {
      response = await fetch(state.backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortCtrl.signal
      });
    } finally {
      clearTimeout(abortTimer);
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`mc-agent HTTP ${response.status}: ${text.slice(0, 200)}`);
    }

    const data = await response.json();
    state.lastDecisionAt = Date.now();
    state.lastNote = String(data?.note || '');
    state.lastMode = String(data?.mode || 'general');
    applyAction(bot, data?.action || {});
  } finally {
    state.decisionInFlight = false;
  }
}

function startDecisionLoop() {
  stopDecisionLoop();
  const tick = () => {
    requestDecision().catch((error) => {
      console.error('[bot-service] decision error:', error.message || error);
    }).finally(() => {
      if (!state.controlTimer) return;
      state.controlTimer = setTimeout(tick, randomBetween(220, 340));
    });
  };
  state.controlTimer = setTimeout(tick, randomBetween(140, 240));
}

function stopDecisionLoop() {
  if (state.controlTimer) {
    clearTimeout(state.controlTimer);
    state.controlTimer = null;
  }
}

function clearConnectTimeout() {
  if (state.connectTimeoutHandle) {
    clearTimeout(state.connectTimeoutHandle);
    state.connectTimeoutHandle = null;
  }
}

let wireBotEvents;

wireBotEvents = function wireBotEvents(bot) {
  bot.on('spawn', () => {
    clearConnectTimeout();
    state.connected = true;
    state.connecting = false;
    state.lastConnectError = '';
    state.lastDisconnectReason = '';
    state.tickCounter = 0;
    state.lastHealth = Number(bot.health || 20);
    state.lastSpawnAt = Date.now();
    state.retreatUntil = 0;
    state.kbRecoveryUntil = 0;
    state.kbStrafeUntil = 0;
    state.kbStrafeDir = '';
    state.combatStrafeDir = '';
    state.lastCombatStrafeSwitchAt = 0;
    state.lastOffhandItemName = '';
    state.eatingUntil = 0;
    state.nextAttackAllowedAt = 0;
    clearMovementControls(bot);
    setupMovements(bot);
    startDecisionLoop();
    console.log('[bot-service] bot spawned');
  });

  bot.on('entityVelocity', (entity) => {
    if (!entity || entity.id !== bot.entity?.id) return;
    const now = Date.now();
    if (now - Number(state.lastWindChargeUseAt || 0) < 450) {
      return;
    }
    const vx = Number(entity.velocity?.x || 0);
    const vy = Number(entity.velocity?.y || 0);
    const vz = Number(entity.velocity?.z || 0);
    const kbMagnitude = Math.hypot(vx, vz, vy * 0.75);
    if (kbMagnitude < 0.09) return;
    state.kbRecoveryUntil = Math.max(state.kbRecoveryUntil, now + randomBetween(450, 900));
    state.kbStrafeUntil = now + randomBetween(200, 460);
    state.kbStrafeDir = Math.random() > 0.5 ? 'left' : 'right';
    state.noSprintUntil = Math.max(Number(state.noSprintUntil || 0), now + randomBetween(480, 760));
    state.nextAttackAllowedAt = Math.max(state.nextAttackAllowedAt, now + randomBetween(220, 420));

    try {
      if (bot.pathfinder && state.movements) {
        bot.pathfinder.setMovements(state.movements);
        bot.pathfinder.setGoal(null);
      }
    } catch {
      // ignore path clear failures
    }
  });

  bot.on('death', () => {
    state.revengeTargetId = null;
    state.revengeExpireAt = 0;
    state.retreatUntil = Date.now() + 1500;
    state.miningInProgress = false;
    state.useHeld = false;
    state.kbRecoveryUntil = 0;
    state.kbStrafeUntil = 0;
    state.kbStrafeDir = '';
    state.combatStrafeDir = '';
    state.lastCombatStrafeSwitchAt = 0;
    state.lastOffhandItemName = '';
    state.eatingUntil = 0;
    state.nextAttackAllowedAt = 0;
    clearMovementControls(bot);
    // Without respawning the bot sits on the death screen and stops
    // sending packets, causing the server to "Timed out" the connection.
    setTimeout(() => {
      try { bot.respawn(); } catch { /* ignore if already respawned */ }
    }, 1000);
  });

  // Guard so both 'kicked' and 'end' (which both fire on a server kick)
  // don't each trigger a full disconnect handling path.
  let disconnectHandled = false;

  bot.on('kicked', (reason) => {
    console.log('[bot-service] kicked:', reason);
    if (disconnectHandled) return;
    disconnectHandled = true;
    clearConnectTimeout();
    state.connecting = false;
    state.connected = false;
    state.lastDisconnectReason = normalizeText(reason?.toString?.() || reason || 'kicked');
    if (state.bot === bot) {
      state.bot = null;
      state.movements = null;
    }
    stopDecisionLoop();
  });

  bot.on('end', () => {
    if (disconnectHandled) {
      console.log('[bot-service] bot disconnected (end after kick)');
      return;
    }
    disconnectHandled = true;
    clearConnectTimeout();
    state.connecting = false;
    state.connected = false;
    if (!state.lastDisconnectReason) {
      state.lastDisconnectReason = 'connection ended';
    }
    if (state.bot === bot) {
      state.bot = null;
      state.movements = null;
    }
    stopDecisionLoop();
    console.log('[bot-service] bot disconnected');
  });

  bot.on('error', (err) => {
    const msg = normalizeText(err?.message || String(err || 'unknown bot error'));
    const code = normalizeText(err?.code || '');
    console.error('[bot-service] bot error:', msg);
    state.lastConnectError = msg;
    if (!state.lastDisconnectReason) {
      state.lastDisconnectReason = code || 'socket error';
    }
    if (!state.connected) {
      state.connecting = false;
    }

    // Some network failures (ECONNRESET/ETIMEDOUT/ECONNREFUSED) may emit only
    // 'error' without a later 'end'. Ensure state is cleaned up immediately.
    if (['ECONNRESET', 'ETIMEDOUT', 'ECONNREFUSED', 'EPIPE'].includes(code)) {
      clearConnectTimeout();
      state.connecting = false;
      state.connected = false;
      if (state.bot === bot) {
        state.bot = null;
        state.movements = null;
      }
      stopDecisionLoop();
    }
  });

  bot.on('chat', (username, message) => {
    if (!username || !message) return;
    if (String(username).toLowerCase() === String(bot.username || '').toLowerCase()) return;
    const msg = String(message || '');
    if (!msg.includes('[SolasTeam]')) return;
    // Ignore SolasTeam protocol chatter entirely.
    return;
    state.lastTeamSeenAt = Date.now();
    rememberTeamMessage(`${username}: ${msg}`);
    const latest = state.teamInbox[state.teamInbox.length - 1];
    if (latest) {
      state.lastNote = `team: ${latest.text.slice(0, 80)}`;
    }
  });

  // ===== IMITATION LEARNING =====
  // Watch how other players interact with the world and record their actions
  bot.on('blockBreak', (block) => {
    if (!state.imitationEnabled || !block) return;
    const now = Date.now();
    if (now - state.lastPlayerObservationAt < 800) return;
    
    for (const entity of Object.values(bot.entities)) {
      if (!entity || !entity.position) continue;
      if (entity.type !== 'player' || !entity.username) continue;
      const dist = bot.entity.position.distanceTo(entity.position);
      if (dist > 12) continue;
      
      state.lastPlayerObservationAt = now;
      recordPlayerAction(state.swarmWorkerId || state.username, entity.username, 'break', block, null, now);
    }
  });

  bot.on('blockPlace', (oldBlock, newBlock) => {
    if (!state.imitationEnabled || !newBlock) return;
    const now = Date.now();
    if (now - state.lastPlayerObservationAt < 800) return;
    
    for (const entity of Object.values(bot.entities)) {
      if (!entity || !entity.position) continue;
      if (entity.type !== 'player' || !entity.username) continue;
      const dist = bot.entity.position.distanceTo(entity.position);
      if (dist > 12) continue;
      
      state.lastPlayerObservationAt = now;
      recordPlayerAction(state.swarmWorkerId || state.username, entity.username, 'place', newBlock, null, now);
    }
  });
};

function createBot(config) {
  const host = normalizeText(config.host || state.host);
  const port = Number(config.port || state.port || 25565);
  
  // Always use the username from config if provided, otherwise fall back to persistent username, then default
  let username;
  if (config.username) {
    username = normalizeText(config.username);
  } else {
    const persistentUsername = loadPersistentUsername(state.persistentInstanceId);
    username = persistentUsername || normalizeText(state.username || DEFAULT_BOT_USERNAME);
  }
  const auth = normalizeText(config.auth || state.auth || DEFAULT_BOT_AUTH) || 'offline';

  if (!host) {
    throw new Error('host is required');
  }

  // Save the username for future sessions
  savePersistentUsername(state.persistentInstanceId, username);

  const bot = mineflayer.createBot({
    host,
    port,
    username,
    auth,
    respawn: true
  });
  bot.loadPlugin(pathfinder);

  state.host = host;
  state.port = port;
  state.username = username;
  state.auth = auth;
  state.bot = bot;
  state.connected = false;
  state.connecting = true;
  state.connectAttemptAt = Date.now();
  state.lastConnectError = '';
  state.lastDisconnectReason = '';
  state.movements = null;
  state.lastHealth = 20;
  state.revengeTargetId = null;
  state.revengeExpireAt = 0;
  state.lastAttackAt = 0;
  state.lastDigAt = 0;
  state.lastStuckAt = 0;
  state.helperActive = false;
  state.decisionInFlight = false;
  state.lastEatAt = 0;
  state.retreatUntil = 0;
  state.lastSpawnAt = 0;
  state.miningInProgress = false;
  state.useHeld = false;
  state.kbRecoveryUntil = 0;
  state.kbStrafeUntil = 0;
  state.kbStrafeDir = '';
  state.combatStrafeDir = '';
  state.lastCombatStrafeSwitchAt = 0;
  state.lastOffhandItemName = '';
  state.eatingUntil = 0;
  state.nextAttackAllowedAt = 0;
  state.lastBaseScanAt = 0;
  state.lastBaseRoamAt = 0;
  state.swarmRole = normalizeText(config.role || process.env.SOLASAI_AUTOSTART_ROLE || 'solo');
  state.swarmWorkerId = normalizeText(config.workerId || process.env.SOLASAI_AUTOSTART_WORKER_ID || '');
  state.teamInbox = [];
  state.lastTeamBroadcastAt = 0;
  state.lastTeamSeenAt = 0;

  clearConnectTimeout();
  state.connectTimeoutHandle = setTimeout(() => {
    if (state.bot !== bot || state.connected) return;
    state.connecting = false;
    if (!state.lastConnectError) {
      state.lastConnectError = 'spawn timeout: server did not complete login in 20s';
    }
  }, 20000);

  wireBotEvents(bot);
}

async function stopBot() {
  state.stopping = true;
  clearConnectTimeout();
  stopDecisionLoop();
  if (state.bot) {
    try {
      clearMovementControls(state.bot);
      state.bot.quit('SolasAI bot service stop');
    } catch {
      // ignore quit failures
    }
    state.bot = null;
  }
  state.connected = false;
  state.connecting = false;
  state.lastConnectError = '';
  state.lastDisconnectReason = 'stopped by api';
  state.movements = null;
  state.revengeTargetId = null;
  state.revengeExpireAt = 0;
  state.helperActive = false;
  state.decisionInFlight = false;
  state.retreatUntil = 0;
  state.miningInProgress = false;
  state.useHeld = false;
  state.kbRecoveryUntil = 0;
  state.kbStrafeUntil = 0;
  state.kbStrafeDir = '';
  state.combatStrafeDir = '';
  state.lastCombatStrafeSwitchAt = 0;
  state.lastOffhandItemName = '';
  state.eatingUntil = 0;
  state.nextAttackAllowedAt = 0;
  state.lastBaseScanAt = 0;
  state.lastBaseRoamAt = 0;
  state.stopping = false;
}

app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'minecraft-bot-service' });
});

app.get('/backend', (req, res) => {
  res.json({ ok: true, backendUrl: state.backendUrl });
});

app.post('/backend', (req, res) => {
  const body = req.body && typeof req.body === 'object' ? req.body : {};
  const backendUrl = normalizeText(body.backendUrl || '');
  if (!backendUrl) {
    return res.status(400).json({ ok: false, error: 'backendUrl is required' });
  }
  state.backendUrl = backendUrl;
  return res.json({ ok: true, backendUrl: state.backendUrl });
});

app.post('/voice/speak', (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const text = normalizeText(body.text || '');
    const voice = normalizeText(body.voice || process.env.SOLASAI_VOICE || 'en-US-BrianNeural');
    if (!text) {
      return res.status(400).json({ ok: false, error: 'text is required' });
    }
    speakText(text, voice);
    return res.json({ ok: true, spoken: true, voice, text });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'voice speak failed' });
  }
});

app.post('/voice/listen', async (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const message = normalizeText(body.message || '');
    const sessionId = normalizeText(body.sessionId || `voice-${Date.now()}`);
    const speak = body.speak !== false;
    const voice = normalizeText(body.voice || process.env.SOLASAI_VOICE || 'en-US-BrianNeural');

    if (!message) {
      return res.status(400).json({ ok: false, error: 'message is required' });
    }

    const humanPrompt = 'Reply like a normal Minecraft player in one short message. '
      + 'No AI intro or model identity. Message: ' + message;

    const reply = await requestChatReplyPlain(sessionId, humanPrompt);
    if (!reply) {
      return res.status(502).json({ ok: false, error: 'empty backend reply' });
    }

    if (speak) {
      speakText(reply, voice);
    }

    return res.json({ ok: true, heard: message, reply, spoken: speak, voice });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'voice listen failed' });
  }
});

app.get('/status', (req, res) => {
  const serverKey = currentServerKey();
  const connectionState = state.connected ? 'connected' : (state.connecting ? 'connecting' : (state.lastConnectError ? 'error' : 'disconnected'));
  res.json({
    ok: true,
    connected: state.connected,
    connecting: state.connecting,
    connectionState,
    connectAttemptAt: state.connectAttemptAt,
    lastConnectError: state.lastConnectError,
    lastDisconnectReason: state.lastDisconnectReason,
    host: state.host,
    port: state.port,
    username: state.username,
    objective: state.objective,
    backendUrl: state.backendUrl,
    lastMode: state.lastMode,
    lastNote: state.lastNote,
    lastDecisionAt: state.lastDecisionAt,
    hasBot: Boolean(state.bot),
    baseServerKey: serverKey,
    baseCandidateCount: getBaseCandidates(serverKey).length
  });
});

app.get('/bases', (req, res) => {
  const queryServer = normalizeText(req.query?.server || '');
  const serverKey = queryServer || currentServerKey();
  const bases = getBaseCandidates(serverKey);
  res.json({
    ok: true,
    server: serverKey,
    count: bases.length,
    bases
  });
});

app.post('/start', async (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    if (state.bot) {
      await stopBot();
    }

    state.objective = normalizeText(body.objective || state.objective || 'general1');
    state.sessionId = normalizeText(body.sessionId || `bot-${Date.now()}`);
    state.backendUrl = normalizeText(body.backendUrl || state.backendUrl || DEFAULT_BACKEND_URL);

    createBot(body);

    res.json({
      ok: true,
      connecting: state.connecting,
      connected: state.connected,
      connectionState: state.connecting ? 'connecting' : (state.connected ? 'connected' : 'disconnected'),
      host: state.host,
      port: state.port,
      username: state.username,
      objective: state.objective,
      backendUrl: state.backendUrl,
      sessionId: state.sessionId
    });
  } catch (error) {
    res.status(400).json({ ok: false, error: error instanceof Error ? error.message : 'start failed' });
  }
});

app.post('/swarm/start', async (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const plan = buildSwarmPlan(body);
    if (!plan.host) {
      return res.status(400).json({ ok: false, error: 'host is required for swarm start' });
    }

    state.lastSwarmPlan = plan;

    if (plan.count === 1) {
      if (state.bot) {
        await stopBot();
      }

      const single = plan.bots[0];
      state.objective = normalizeText(single.objective || state.objective || 'general1');
      state.sessionId = normalizeText(`swarm-${single.username}-${Date.now()}`);
      state.backendUrl = normalizeText(body.backendUrl || state.backendUrl || DEFAULT_BACKEND_URL);

      createBot({
        host: single.host,
        port: single.port,
        username: single.username,
        auth: single.auth
      });

      return res.json({ ok: true, accepted: true, liveStarted: true, count: 1, bot: single, note: 'single bot started immediately' });
    }

    const shouldLaunchWorkers = body.launch !== false;
    if (shouldLaunchWorkers) {
      if (state.bot) {
        await stopBot();
      }
      await stopSwarmWorkers();
      const started = await launchSwarmWorkers(plan, {
        launchCount: Number(body.launchCount || plan.count),
        basePort: Number(body.basePort || 8800),
        backendUrl: normalizeText(body.backendUrl || state.backendUrl || DEFAULT_BACKEND_URL)
      });
      return res.json({
        ok: true,
        accepted: true,
        liveStarted: true,
        count: plan.count,
        launched: started.length,
        workers: started,
        note: 'swarm workers launched'
      });
    }

    return res.json({
      ok: true,
      accepted: true,
      liveStarted: false,
      count: plan.count,
      note: 'swarm plan generated. this node runs one live bot instance; use multiple service instances/workers for full 500 concurrent joins.',
      usernames: plan.bots.map((bot) => bot.username),
      jobs: plan.jobs
    });
  } catch (error) {
    return res.status(400).json({ ok: false, error: error instanceof Error ? error.message : 'swarm start failed' });
  }
});

app.get('/swarm/plan', (req, res) => {
  res.json({
    ok: true,
    hasPlan: Boolean(state.lastSwarmPlan),
    plan: state.lastSwarmPlan
  });
});

app.get('/swarm/status', (req, res) => {
  const workers = listSwarmWorkers();
  res.json({
    ok: true,
    activeWorkers: workers.filter((w) => w.status === 'running' || w.status === 'starting').length,
    workers
  });
});

app.post('/swarm/stop', async (req, res) => {
  const result = await stopSwarmWorkers();
  res.json({ ok: true, ...result });
});

app.post('/objective', (req, res) => {
  const body = req.body && typeof req.body === 'object' ? req.body : {};
  const objective = normalizeText(body.objective || '');
  if (!objective) {
    return res.status(400).json({ ok: false, error: 'objective is required' });
  }
  state.objective = objective;
  return res.json({ ok: true, objective: state.objective });
});

app.post('/stop', async (req, res) => {
  const serverKey = currentServerKey();
  const bases = getBaseCandidates(serverKey);
  await stopBot();
  res.json({ ok: true, stopped: true, server: serverKey, baseCandidateCount: bases.length, bases });
});

// ===== RCON CLIENT =====
class RconClient {
  constructor(host, port, password) {
    this.host = host;
    this.port = port;
    this.password = password;
    this.socket = null;
    this.requestId = 1;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.socket = net.createConnection(this.port, this.host);
      this.socket.setTimeout(5000);
      this.socket.on('connect', () => resolve());
      this.socket.on('error', reject);
      this.socket.on('timeout', () => {
        this.socket?.destroy();
        reject(new Error('RCON connection timeout'));
      });
    });
  }

  async command(cmd) {
    if (!this.socket) throw new Error('Not connected');
    return new Promise((resolve, reject) => {
      try {
        const id = this.requestId++;
        const payload = Buffer.alloc(cmd.length + 10);
        payload.writeInt32LE(cmd.length + 10, 0);
        payload.writeInt32LE(id, 4);
        payload.writeInt32LE(3, 8);
        payload.write(Buffer.from(this.password + '\x00' + cmd + '\x00').toString(), 10);
        this.socket.write(payload);
        
        let buffer = Buffer.alloc(0);
        const handler = (data) => {
          buffer = Buffer.concat([buffer, data]);
          if (buffer.length >= 4) {
            const len = buffer.readInt32LE(0);
            if (buffer.length >= len + 4) {
              const response = buffer.toString('utf8', 12, 12 + len - 10);
              buffer = buffer.slice(len + 4);
              this.socket?.removeListener('data', handler);
              resolve(response);
            }
          }
        };
        this.socket.on('data', handler);
      } catch (e) {
        reject(e);
      }
    });
  }

  async disconnect() {
    if (this.socket) {
      this.socket.destroy();
      this.socket = null;
    }
  }
}
// ===== CIVILIZATIONS LAUNCHER =====
app.post('/civilizations/launch', async (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const host = normalizeText(body.host || 'solasai.aternos.me');
    const port = Number(body.port || 25565);
    const rconHost = normalizeText(body.rconHost || '127.0.0.1');
    const rconPort = Number(body.rconPort || 25575);
    const rconPassword = normalizeText(body.rconPassword || 'solasai-bot-pass');
    const count = Number(body.count || 50);
    const civsCount = 1;
    const botsPerCiv = count / civsCount;

    // Civilization coordinates: [x, z]
    const civCoords = [
      { x: 0, z: 0, name: 'SolasAI' }
    ];
    const civY = 200;

    // Jobs distribution
    const jobs = ['miner', 'builder', 'warrior', 'farmer'];

    // Unique bot names
    const uniqueNames = ['Alexios', 'Zephyr', 'Minerva', 'Thorne', 'Cassius', 'Lyra', 'Corvus', 'Selene', 'Orion', 'Iris', 'Hector', 'Astrid', 'Drake', 'Calista', 'Nero', 'Freya', 'Silas', 'Nova', 'Kai', 'Luna', 'Blaze', 'Sage', 'Atlas', 'Echo', 'forge', 'Storm', 'Cipher', 'Raven', 'Ember', 'Vale', 'Onyx', 'Sage2', 'Ace', 'Bolt', 'Crux', 'Dune', 'Flux', 'Grove', 'Haven', 'Ivor', 'Jade', 'Knox', 'Lux', 'Maxx', 'Nexus', 'Orbit', 'Phoenix', 'Quest', 'Ridge', 'Stone', 'Titan'];

    // Create bots configuration
    const bots = [];
    for (let i = 0; i < count; i++) {
      const civIdx = Math.floor(i / botsPerCiv);
      const civ = civCoords[civIdx];
      const job = jobs[i % jobs.length];
      const username = uniqueNames[i % uniqueNames.length];
      const objective = `You and your team is teleported, your goal is to build the biggest civilisation with your team, and conquer other civilisation. Strongest civilisation wins! Your team: ${civ.name}. Your role: ${job}. Base coordinate: ${civ.x} ${civY} ${civ.z}. Collaborate with nearby team members, share resources, self-assign useful tasks.`;

      bots.push({
        id: `civ-${civIdx}-bot-${i}`,
        username,
        role: job,
        objective,
        host,
        port,
        auth: 'offline',
        civIdx,
        civName: civ.name,
        civCoord: civ
      });
    }

    // Launch bots via swarm
    const swarmReq = {
      host,
      port,
      count,
      jobs: jobs.join(','),
      baseUsername: 'Solas',
      usernameMode: 'keyed',
      autoThink: true,
      launchCount: count,
      basePort: 8800,
      launch: true
    };

    const plan = buildSwarmPlan(swarmReq);
    state.lastSwarmPlan = plan;

    // Launch workers
    if (state.bot) {
      await stopBot();
    }
    await stopSwarmWorkers();
    const started = await launchSwarmWorkers(plan, {
      launchCount: count,
      basePort: 8800
    });

    // Schedule RCON commands to run after bots spawn
    setTimeout(async () => {
      try {
        const rcon = new RconClient(rconHost, rconPort, rconPassword);
        await rcon.connect();

        // Apply slow falling to all players
        const slowFallingCmd = 'effect give @a slow_falling 600 1 true';
        await rcon.command(slowFallingCmd);

        // Teleport bots to their civilization coordinates grouped by color/team
        for (let civIdx = 0; civIdx < civsCount; civIdx++) {
          const civ = civCoords[civIdx];
          const civPlayerNames = [];
          for (let i = civIdx * botsPerCiv; i < (civIdx + 1) * botsPerCiv; i++) {
            civPlayerNames.push(`Solas_${civ.name}_${(i % botsPerCiv) + 1}`);
          }

          // Teleport each bot in the civilization
          for (const playerName of civPlayerNames) {
            const teleportCmd = `execute as ${playerName} run tp @s ${civ.x} ${civY} ${civ.z}`;
            await rcon.command(teleportCmd);
            await sleep(50);
          }
        }

        await rcon.disconnect();
        console.log('[civilizations] RCON commands executed: slow falling + teleports');
      } catch (error) {
        console.error('[civilizations] RCON error:', error?.message || error);
      }
    }, 15000); // Wait 15 seconds for bots to spawn

    res.json({
      ok: true,
      accepted: true,
      civilizations: civsCount,
      totalBots: count,
      botsPerCiv,
      launched: started.length,
      workers: started,
      coordinates: civCoords,
      note: '100 bots launching in 4 civilizations with slow falling effect and teleport queued'
    });
  } catch (error) {
    return res.status(400).json({ ok: false, error: error instanceof Error ? error.message : 'civilization launch failed' });
  }
});

async function startAutostartBotIfConfigured() {
  if (process.env.SOLASAI_AUTOSTART !== '1') return;
  const host = normalizeText(process.env.SOLASAI_AUTOSTART_HOST || '');
  if (!host) {
    console.error('[bot-service] autostart skipped: SOLASAI_AUTOSTART_HOST missing');
    return;
  }

  const port = Number(process.env.SOLASAI_AUTOSTART_PORT || 25565);
  const username = normalizeText(process.env.SOLASAI_AUTOSTART_USERNAME || DEFAULT_BOT_USERNAME);
  const auth = normalizeText(process.env.SOLASAI_AUTOSTART_AUTH || DEFAULT_BOT_AUTH) || 'offline';
  const objective = normalizeText(process.env.SOLASAI_AUTOSTART_OBJECTIVE || state.objective || 'general1');
  const backendUrl = normalizeText(process.env.SOLASAI_AUTOSTART_BACKEND_URL || state.backendUrl || DEFAULT_BACKEND_URL);
  const sessionId = normalizeText(process.env.SOLASAI_AUTOSTART_SESSION_ID || `autostart-${Date.now()}`);
  const role = normalizeText(process.env.SOLASAI_AUTOSTART_ROLE || 'worker');
  const workerId = normalizeText(process.env.SOLASAI_AUTOSTART_WORKER_ID || '');

  try {
    state.objective = objective;
    state.backendUrl = backendUrl;
    state.sessionId = sessionId;
    createBot({ host, port, username, auth, role, workerId });
    console.log(`[bot-service] autostart requested for ${username}@${host}:${port}`);
  } catch (error) {
    console.error('[bot-service] autostart failed:', error?.message || error);
  }
}


app.listen(PORT, () => {
  try {
    const routes = [];
    const stack = app?._router?.stack || [];
    for (const layer of stack) {
      if (layer?.route?.path && layer?.route?.methods) {
        const methods = Object.keys(layer.route.methods)
          .filter((m) => layer.route.methods[m])
          .map((m) => m.toUpperCase());
        routes.push(`${methods.join(',')} ${layer.route.path}`);
      }
    }
    console.log(`[bot-service] routes registered: ${routes.length}`);
    if (routes.length) {
      console.log('[bot-service] route list:', routes.join(' | '));
    }
  } catch (error) {
    console.error('[bot-service] failed to print routes:', error?.message || error);
  }
  console.log(`[bot-service] running on http://127.0.0.1:${PORT}`);
  startAutostartBotIfConfigured().catch(() => {});
});

