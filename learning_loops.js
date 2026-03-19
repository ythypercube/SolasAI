import 'dotenv/config';

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8787').replace(/\/$/, '');
const API_KEY = process.env.API_KEY || process.env.TURBOWARP_API_KEY || '';
const LOOP_TARGET = (process.env.LOOP_TARGET || 'both').toLowerCase(); // both | turbowarp | minecraft

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postJson(path, body, timeoutMs = 8000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = { 'Content-Type': 'application/json' };
    if (API_KEY) headers['x-api-key'] = API_KEY;

    const response = await fetch(`${BACKEND_URL}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: controller.signal
    });
    const raw = await response.text();
    const parsed = raw ? safeJsonParse(raw, { raw }) : {};
    if (!response.ok) {
      throw new Error(`${path} ${response.status}: ${raw.slice(0, 240)}`);
    }
    return parsed;
  } finally {
    clearTimeout(timeout);
  }
}

function safeJsonParse(raw, fallback) {
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

const TURBOWARP_CASES = [
  {
    sessionId: 'loop-tw-1',
    message: 'make me a turbowarp script that moves sprite with arrows and increases score on touching coin',
    requiredKeywords: ['when green flag clicked', 'if', 'key', 'score']
  },
  {
    sessionId: 'loop-tw-2',
    message: 'how do i make lag lower in turbowarp with clones',
    requiredKeywords: ['clone', 'delete this clone', 'broadcast', 'limit']
  },
  {
    sessionId: 'loop-tw-3',
    message: 'make an advanced minecraft pvp training plan for cps aim strafing',
    requiredKeywords: ['strafe', 'timing', 'practice', 'aim']
  }
];

function scoreTurboWarpReply(reply, requiredKeywords) {
  const lower = String(reply || '').toLowerCase();
  let hit = 0;
  for (const keyword of requiredKeywords) {
    if (lower.includes(keyword.toLowerCase())) hit += 1;
  }
  const keywordScore = requiredKeywords.length > 0 ? hit / requiredKeywords.length : 1;
  const lengthScore = lower.length >= 40 && lower.length <= 900 ? 1 : 0.5;
  const refusalPenalty = /can't help with that|cannot help|won't help/.test(lower) ? 0.2 : 1;
  return clamp((keywordScore * 0.75 + lengthScore * 0.25) * refusalPenalty, 0, 1);
}

function buildTurboWarpImprovement(reply, requiredKeywords) {
  const lower = String(reply || '').toLowerCase();
  const missing = requiredKeywords.filter((k) => !lower.includes(k.toLowerCase()));
  if (missing.length === 0) {
    return 'Keep clear step-by-step instructions and practical examples.';
  }
  return `Include these missing details next time: ${missing.join(', ')}.`;
}

async function runTurboWarpLoop() {
  let idx = 0;
  while (true) {
    let waitMs = 12000;
    const scenario = TURBOWARP_CASES[idx % TURBOWARP_CASES.length];
    idx += 1;
    try {
      const chatRes = await postJson('/chat-plain', {
        sessionId: scenario.sessionId,
        message: scenario.message
      });
      const reply = String(chatRes.reply || '');
      const score = scoreTurboWarpReply(reply, scenario.requiredKeywords);
      const rating = score >= 0.65 ? 'good' : 'bad';
      const improvement = buildTurboWarpImprovement(reply, scenario.requiredKeywords);

      await postJson('/feedback', {
        sessionId: scenario.sessionId,
        message: scenario.message,
        reply,
        rating,
        improvement
      });

      console.log(`[TurboWarp loop] case=${scenario.sessionId} score=${score.toFixed(2)} rating=${rating}`);
    } catch (error) {
      const message = String(error?.message || error || 'unknown error');
      console.error('[TurboWarp loop] error:', message);
      if (message.includes('429')) {
        waitMs = 25000;
      }
    }

    await sleep(waitMs);
  }
}

const MC_CASES = [
  {
    id: 'low_hp_eat_priority',
    objective: 'pvp fight and survive',
    state: {
      health: 9,
      food: 20,
      utilityFoodSlot: 8,
      swordSlot: 0,
      axeSlot: 1,
      nearestEnemyDistance: 2.2,
      nearestEnemyName: 'Enemy',
      hasMeleeWeapon: true,
      onGround: true,
      focusedDistance: -1,
      focusedEntity: ''
    }
  },
  {
    id: 'resource_enemy_clear',
    objective: 'gather iron and redstone resources quickly',
    state: {
      health: 18,
      food: 20,
      utilityFoodSlot: 8,
      swordSlot: 0,
      pickaxeSlot: 1,
      nearestEnemyDistance: 5.5,
      nearestEnemyName: 'Raider',
      hasMeleeWeapon: true,
      focusedEntity: 'minecraft:iron_ore',
      focusedDistance: 2.3,
      onGround: true
    }
  },
  {
    id: 'resource_harvest_focus',
    objective: 'collect diamonds and mine efficiently',
    state: {
      health: 19,
      food: 20,
      utilityFoodSlot: 8,
      swordSlot: 0,
      pickaxeSlot: 1,
      nearestEnemyDistance: -1,
      hasMeleeWeapon: true,
      focusedEntity: 'minecraft:deepslate_diamond_ore',
      focusedDistance: 2.0,
      onGround: true
    }
  }
];

function evaluateMinecraftCase(caseId, action) {
  if (!action || typeof action !== 'object') {
    return { pass: false, reason: 'No action returned.' };
  }

  if (caseId === 'low_hp_eat_priority') {
    const pass = action.use === true && action.attack === false && Number(action.hotbarSlot) === 8;
    return { pass, reason: pass ? 'Eat priority ok.' : 'Expected food use priority at low HP.' };
  }

  if (caseId === 'resource_enemy_clear') {
    const pass = action.attack === true && Number(action.hotbarSlot) >= 0;
    return { pass, reason: pass ? 'Resource defense engage ok.' : 'Expected combat clear when enemy is nearby during resource mode.' };
  }

  if (caseId === 'resource_harvest_focus') {
    const pass = action.attack === true && Number(action.hotbarSlot) === 1;
    return { pass, reason: pass ? 'Harvest focus ok.' : 'Expected immediate mining with pickaxe in resource mode.' };
  }

  return { pass: true, reason: 'No rule.' };
}

function buildMcAdjustments(failures) {
  const adj = {
    eatCriticalHpDelta: 0,
    eatRecoverHpDelta: 0,
    eatLockTicksDelta: 0,
    resourceAggressionRadiusDelta: 0
  };

  for (const failure of failures) {
    if (failure.id === 'low_hp_eat_priority') {
      adj.eatCriticalHpDelta += 0.5;
      adj.eatRecoverHpDelta += 0.5;
      adj.eatLockTicksDelta += 1;
    }
    if (failure.id === 'resource_enemy_clear') {
      adj.resourceAggressionRadiusDelta += 0.5;
    }
    if (failure.id === 'resource_harvest_focus') {
      adj.resourceAggressionRadiusDelta += 0.25;
    }
  }

  return adj;
}

async function runMinecraftLoop() {
  const sessionId = 'mc-learning-loop';
  let caseIndex = 0;

  while (true) {
    let waitMs = 15000;
    const failures = [];
    try {
      const testCase = MC_CASES[caseIndex % MC_CASES.length];
      caseIndex += 1;

      const mcRes = await postJson('/mc-agent', {
        sessionId,
        objective: testCase.objective,
        state: testCase.state
      });

      const verdict = evaluateMinecraftCase(testCase.id, mcRes.action || {});
      if (!verdict.pass) {
        failures.push({ id: testCase.id, reason: verdict.reason });
      }

      const rating = failures.length === 0 ? 'good' : 'bad';
      const adjustments = buildMcAdjustments(failures);
      const improvement = failures.length === 0
        ? 'Hold current policy; tiny lock reduction only if over-conservative.'
        : failures.map((f) => `${f.id}: ${f.reason}`).join(' | ');

      await postJson('/mc-feedback', {
        sessionId,
        rating,
        adjustments,
        improvement
      });

      console.log(`[Minecraft loop] case=${testCase.id} rating=${rating} failures=${failures.length}`);
    } catch (error) {
      const message = String(error?.message || error || 'unknown error');
      console.error('[Minecraft loop] error:', message);
      if (message.includes('429')) {
        waitMs = 30000;
      }
    }

    await sleep(waitMs);
  }
}

async function main() {
  console.log(`[Learning loops] backend=${BACKEND_URL} target=${LOOP_TARGET}`);

  if (LOOP_TARGET === 'turbowarp') {
    await runTurboWarpLoop();
    return;
  }
  if (LOOP_TARGET === 'minecraft') {
    await runMinecraftLoop();
    return;
  }

  await Promise.all([
    runTurboWarpLoop(),
    runMinecraftLoop()
  ]);
}

main().catch((error) => {
  console.error('[Learning loops] fatal:', error?.message || error);
  process.exit(1);
});
