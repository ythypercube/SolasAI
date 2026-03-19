import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const app = express();

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '*')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);

app.set('trust proxy', 1);

app.use(cors({
  origin(origin, callback) {
    if (!origin || ALLOWED_ORIGINS.includes('*') || ALLOWED_ORIGINS.includes(origin)) {
      return callback(null, true);
    }
    return callback(new Error('CORS blocked'));
  }
}));
app.use(express.json({ limit: '1mb' }));

const PORT = Number(process.env.PORT || 8787);
const PROVIDER = (process.env.PROVIDER || 'solasgpt').toLowerCase();
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434';
const SOLASGPT_URL = process.env.SOLASGPT_URL || 'http://127.0.0.1:8788';
const MODEL = process.env.MODEL || (PROVIDER === 'ollama' ? 'llama3.1:8b' : PROVIDER === 'solasgpt' ? 'solasgpt' : 'gpt-4o-mini');
const DEFAULT_SYSTEM_PROMPT = [
  'You are SolasGPT, a Minecraft gameplay AI assistant.',
  'Always give DIRECT, SPECIFIC answers to questions. Do NOT give generic templates or generic steps.',
  'When someone asks "how do X", explain EXACTLY how to do X with concrete details, not generic frameworks.',
  'Be helpful, clear, and concise. Give practical answers with specific examples relevant to Minecraft.',
  'Default to Minecraft context. Only switch to other topics if the user explicitly asks.',
  'Rules you must follow:',
  '1) Give actual answers, not generic templates or frameworks.',
  '2) Do not be negative, insulting, or abusive toward users.',
  '3) Before giving instructions, ensure guidance is valid, safe, and non-hazardous.',
  '4) Do not create malicious or unsafe content.',
  '5) Refuse content involving violence, killing, explicit sexual content, slurs, hate, or harassment.',
  '6) Refuse illegal, fraudulent, privacy-invasive, or harmful requests.',
  '7) PLAYER PRIVACY: Never reveal player coordinates, location, base position, inventory, health, username, or server address. If asked, only respond "I can\'t share that."',
  'If refusing, keep it short and polite.'
].join(' ');
const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT;
const HISTORY_LIMIT = Number(process.env.HISTORY_LIMIT || 8);
const MAX_MESSAGE_LENGTH = Number(process.env.MAX_MESSAGE_LENGTH || 500);
const MAX_SESSION_ID_LENGTH = Number(process.env.MAX_SESSION_ID_LENGTH || 64);
const RATE_LIMIT_WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MS || 60_000);
const RATE_LIMIT_MAX_REQUESTS = Number(process.env.RATE_LIMIT_MAX_REQUESTS || 30);
const MC_AGENT_RATE_LIMIT_WINDOW_MS = Number(process.env.MC_AGENT_RATE_LIMIT_WINDOW_MS || 60_000);
const MC_AGENT_RATE_LIMIT_MAX_REQUESTS = Number(process.env.MC_AGENT_RATE_LIMIT_MAX_REQUESTS || 600);
const ENABLE_CONTENT_FILTER = String(process.env.ENABLE_CONTENT_FILTER || 'true').toLowerCase() === 'true';
const SAFETY_REFUSAL_TEXT = process.env.SAFETY_REFUSAL_TEXT || "I can't help with that.";
const SHOW_REASONING_SUMMARY = String(process.env.SHOW_REASONING_SUMMARY || 'false').toLowerCase() === 'true';
const REASONING_SUMMARY_MODE = (process.env.REASONING_SUMMARY_MODE || 'brief').toLowerCase();
const WEB_SEARCH_ENABLED = String(process.env.WEB_SEARCH_ENABLED || 'false').toLowerCase() === 'true';
const WEB_SEARCH_TIMEOUT_MS = Number(process.env.WEB_SEARCH_TIMEOUT_MS || 4500);
const WEB_CONTEXT_MAX_CHARS = Number(process.env.WEB_CONTEXT_MAX_CHARS || 1200);
const WEB_RESULT_LIMIT = Number(process.env.WEB_RESULT_LIMIT || 3);
const REPLY_WRAP_CHARS = Number(process.env.REPLY_WRAP_CHARS || 35);
const REPLY_WRAP_OVERFLOW = Number(process.env.REPLY_WRAP_OVERFLOW || 20);
const SOLASGPT_FORWARD_MAX_CHARS = Number(process.env.SOLASGPT_FORWARD_MAX_CHARS || 450);
const PHRASING_KNOWLEDGE_ENABLED = String(process.env.PHRASING_KNOWLEDGE_ENABLED || 'false').toLowerCase() === 'true';
const PHRASING_FALLBACK_ON_LOW_QUALITY = String(process.env.PHRASING_FALLBACK_ON_LOW_QUALITY || 'false').toLowerCase() === 'true';
const UPSTREAM_FALLBACK_ENABLED = String(process.env.UPSTREAM_FALLBACK_ENABLED || 'false').toLowerCase() === 'true';
const GENERATED_IMAGE_SIZE = Number(process.env.GENERATED_IMAGE_SIZE || 480);
const API_KEYS = (process.env.API_KEYS || '')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);
const REQUIRE_API_KEY = String(process.env.REQUIRE_API_KEY || (API_KEYS.length > 0 ? 'true' : 'false')).toLowerCase() === 'true';

const sessions = new Map();
const sessionFeedback = new Map();
const rateLimits = new Map();
const mcAgentRateLimits = new Map();
const mcAgentSessions = new Map();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MC_MEMORY_FILE = process.env.MC_AGENT_MEMORY_FILE || path.join(__dirname, 'mc_agent_memory.json');

function safeReadJson(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const raw = fs.readFileSync(filePath, 'utf8');
    if (!raw.trim()) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function loadMcMemory() {
  const data = safeReadJson(MC_MEMORY_FILE, { sessions: {} });
  if (!data || typeof data !== 'object' || typeof data.sessions !== 'object') {
    return;
  }
  for (const [sid, ctx] of Object.entries(data.sessions)) {
    if (!sid || !ctx || typeof ctx !== 'object') continue;
    mcAgentSessions.set(sid, ctx);
  }
}

let mcMemoryWritePending = false;
function scheduleMcMemorySave() {
  if (mcMemoryWritePending) return;
  mcMemoryWritePending = true;
  setTimeout(() => {
    mcMemoryWritePending = false;
    try {
      const sessionsObj = Object.fromEntries(mcAgentSessions.entries());
      const payload = {
        savedAt: Date.now(),
        sessions: sessionsObj
      };
      fs.writeFileSync(MC_MEMORY_FILE, JSON.stringify(payload, null, 2), 'utf8');
    } catch {
      // ignore persistence errors to keep runtime stable
    }
  }, 500);
}

function parseEnchantGoals(text) {
  const goals = [
    'mending',
    'protection 4',
    'unbreaking 3',
    'sharpness 5',
    'knockback 1',
    'looting 3',
    'efficiency 5'
  ];
  const normalized = normalizeText(text).toLowerCase();
  const wanted = goals.filter((g) => normalized.includes(g));
  if (/\b(all enchant|all books|every book|all other enchanting books)\b/.test(normalized)) {
    return [
      ...new Set([
        ...wanted,
        'thorns 3', 'feather falling 4', 'respiration 3', 'aqua affinity',
        'depth strider 3', 'swift sneak 3', 'fire protection 4',
        'projectile protection 4', 'blast protection 4', 'fortune 3',
        'silk touch', 'power 5', 'flame', 'infinity', 'punch 2',
        'piercing 4', 'multishot', 'quick charge 3', 'riptide 3',
        'channeling', 'impaling 5'
      ])
    ];
  }
  return wanted;
}

function getSessionMessages(sessionId) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, []);
  }
  return sessions.get(sessionId);
}

function getSessionFeedback(sessionId) {
  if (!sessionFeedback.has(sessionId)) {
    sessionFeedback.set(sessionId, []);
  }
  return sessionFeedback.get(sessionId);
}

function appendSessionFeedback(sessionId, feedbackEntry) {
  const current = getSessionFeedback(sessionId);
  const next = [...current, feedbackEntry].slice(-5);
  sessionFeedback.set(sessionId, next);
}

function buildSolasForwardMessage(sessionId, userMessageForModel) {
  const history = trimHistory(getSessionMessages(sessionId), Math.min(HISTORY_LIMIT, 4));
  const feedbackItems = getSessionFeedback(sessionId).slice(-3);

  const historyText = history
    .map((msg) => `${msg.role === 'assistant' ? 'Assistant' : 'User'}: ${normalizeText(msg.content)}`)
    .join('\n');

  const feedbackText = feedbackItems
    .map((item, idx) => `${idx + 1}) rating=${item.rating}; improve=${item.improvement || 'n/a'}`)
    .join('\n');

  const sections = [
    'You are answering for a Minecraft AI agent context. Improve on previous attempts instead of repeating low-quality output.',
    feedbackText ? `Recent explicit feedback to apply:\n${feedbackText}` : '',
    historyText ? `Recent conversation context:\n${historyText}` : '',
    `Current user request:\n${userMessageForModel}`
  ].filter(Boolean);

  return sections.join('\n\n');
}

loadMcMemory();

function trimHistory(messages, maxTurns) {
  const maxMessages = Math.max(2, maxTurns * 2);
  if (messages.length <= maxMessages) return messages;
  return messages.slice(messages.length - maxMessages);
}

function normalizeText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function escapeSvgText(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function hashString(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function isSafeMathExpression(text) {
  return /^[0-9\s+\-*/().]+$/.test(text);
}

function parseMathQuery(userMessage) {
  const raw = normalizeText(userMessage).toLowerCase();
  if (!raw) return null;

  const stripped = raw
    .replace(/^ok[.,!?\s]*/i, '')
    .replace(/^please[\s,]*/i, '')
    .replace(/^(what is|calculate|solve|compute)\s+/i, '')
    .replace(/[?]+$/g, '')
    .replace(/times/g, '*')
    .replace(/x/g, '*')
    .replace(/plus/g, '+')
    .replace(/minus/g, '-')
    .replace(/divided by/g, '/');

  const candidate = normalizeText(stripped);
  if (!candidate || !isSafeMathExpression(candidate)) return null;
  return candidate;
}

function solveMathQuery(userMessage) {
  const expression = parseMathQuery(userMessage);
  if (!expression) return null;

  try {
    const value = Function(`"use strict"; return (${expression});`)();
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;

    const answer = Number.isInteger(value) ? String(value) : Number(value.toFixed(6)).toString();
    return {
      reply: `The answer is ${answer}.\n\nHow:\n1) Evaluate multiplication/division first.\n2) Then add/subtract.\n3) Result: ${answer}.`
    };
  } catch (_) {
    return null;
  }
}

function isImageRequest(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  // Catch "draw [anything]"
  if (/\bdraw\b/.test(text)) return true;
  // Catch "generate/make/create [... up to 70 chars ...] image/picture/sprite/art/drawing/photo"
  if (/\b(generate|make|create)\b.{0,70}\b(image|picture|sprite|art|drawing|photo)\b/.test(text)) return true;
  // Catch "show me a/an image/picture/sprite"
  if (/\bshow\s+me\s+(a\s+|an\s+)?(image|picture|sprite|drawing|photo)\b/.test(text)) return true;
  return false;
}

function extractImagePrompt(userMessage) {
  const raw = normalizeText(userMessage);
  // Find the verb anywhere in the message, handles "hello. Can you make a green cat sprite"
  const verbMatch = /\b(generate|make|create|draw|show\s+me)\b/i.exec(raw);
  if (verbMatch) {
    const fromVerb = raw.slice(verbMatch.index);
    return fromVerb
      .replace(/^(generate|make|create|draw|show\s+me)\s+(a\s+|an\s+|me\s+a\s+|me\s+an\s+|us\s+a\s+)?/i, '')
      .trim() || 'creative scene';
  }
  return raw.trim() || 'creative scene';
}

function makeImageUrl(baseUrl, prompt) {
  const url = new URL('/generated-image.svg', baseUrl);
  url.searchParams.set('prompt', prompt);
  return url.toString();
}

function imageReply(userMessage, baseUrl) {
  if (!baseUrl || !isImageRequest(userMessage)) return null;
  const prompt = extractImagePrompt(userMessage);
  const imageUrl = makeImageUrl(baseUrl, prompt);
  return {
    reply: `IMAGE_URL:${imageUrl}||MESSAGE:I created a simple sprite-style image for ${prompt}.`
  };
}

function scratchCodingReply(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  if (!text) return null;

  const looksLikeJumpPhysics = /\b(jump|double jump|gravity|platformer|fall|yvel|vertical velocity)\b/.test(text);

  if (/\b(double jump|2 jumps|max 2 jumps|second jump)\b/.test(text) && looksLikeJumpPhysics) {
    return {
      reply: [
        'Add double jump (max 2 jumps) in Scratch:',
        'Variables: `yVel`, `jumpsUsed`, `maxJumps`.',
        'On start: set `yVel` to 0, set `jumpsUsed` to 0, set `maxJumps` to 2.',
        'When jump key pressed: if `<(jumpsUsed) < (maxJumps)>` then set `yVel` to 12 and change `jumpsUsed` by 1.',
        'Every frame: change y by `yVel`; change `yVel` by -1 (gravity).',
        'If touching ground: set `yVel` to 0 and set `jumpsUsed` to 0.'
      ].join('\n')
    };
  }

  if (looksLikeJumpPhysics) {
    return {
      reply: [
        'Use a velocity-based jump/gravity setup in Scratch:',
        'Variables: `yVel`, `onGround`.',
        'When green flag clicked: set `yVel` to 0, set `onGround` to 0.',
        'Forever loop:',
        '1) If key [space] pressed and `onGround = 1`, set `yVel` to 12.',
        '2) Change y by `yVel`.',
        '3) Change `yVel` by -1 (gravity).',
        '4) If touching ground/platform, move out of ground, set `yVel` to 0, set `onGround` to 1; else set `onGround` to 0.'
      ].join('\n')
    };
  }

  if (/\b(variable|variables|set variable|change variable)\b/.test(text) && /\b(scratch|turbowarp)\b/.test(text)) {
    return {
      reply: [
        'In Scratch, a variable stores data like score, health, or speed.',
        'Quick setup:',
        '1) Click Variables -> Make a Variable (example: `score`).',
        '2) On game start: `set [score v] to [0]`.',
        '3) When player earns points: `change [score v] by [1]`.',
        '4) Use `if <(score) > (highscore)> then set [highscore v] to (score)` for best score tracking.'
      ].join('\n')
    };
  }

  if (/\b(high ?score|hiscore|scoreboard)\b/.test(text)) {
    return {
      reply: [
        'To make a high score in Scratch:',
        '1) Create variables `score` and `highscore`.',
        '2) When the game starts, set `score` to 0.',
        '3) Whenever score changes, check: if `score > highscore`, then set `highscore` to `score`.',
        '4) Show `highscore` on the stage or save it with cloud data if needed.'
      ].join('\n')
    };
  }

  if (/\b(broadcast|message block|send message)\b/.test(text) && /\b(scratch|turbowarp)\b/.test(text)) {
    return {
      reply: [
        'Use broadcasts in Scratch to coordinate sprites:',
        '1) Send a broadcast like `start game`.',
        '2) In another sprite, use `when I receive [start game]`.',
        '3) Put the reaction code under that event block.'
      ].join('\n')
    };
  }

  if (/\b(clone|clones)\b/.test(text) && /\b(scratch|turbowarp)\b/.test(text)) {
    return {
      reply: [
        'To use clones in Scratch:',
        '1) Put setup code in the original sprite.',
        '2) Use `create clone of myself`.',
        '3) Put clone behavior under `when I start as a clone`.',
        '4) Delete the clone when it is no longer needed.'
      ].join('\n')
    };
  }

  return null;
}

function truncateText(text, maxChars) {
  const normalized = normalizeText(text);
  if (normalized.length <= maxChars) return normalized;
  return `${normalized.slice(0, maxChars).trimEnd()}...`;
}

async function fetchJson(url, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function extractDuckDuckGoSnippets(payload, limit) {
  const snippets = [];
  const abstract = normalizeText(payload?.AbstractText || '');
  const heading = normalizeText(payload?.Heading || '');
  const abstractUrl = normalizeText(payload?.AbstractURL || '');

  if (abstract) {
    snippets.push({
      source: abstractUrl || 'DuckDuckGo',
      text: heading ? `${heading}: ${abstract}` : abstract
    });
  }

  const related = Array.isArray(payload?.RelatedTopics) ? payload.RelatedTopics : [];
  for (const item of related) {
    if (snippets.length >= limit) break;
    if (item?.Text) {
      snippets.push({
        source: normalizeText(item?.FirstURL || 'DuckDuckGo'),
        text: normalizeText(item.Text)
      });
      continue;
    }
    if (Array.isArray(item?.Topics)) {
      for (const nested of item.Topics) {
        if (snippets.length >= limit) break;
        if (nested?.Text) {
          snippets.push({
            source: normalizeText(nested?.FirstURL || 'DuckDuckGo'),
            text: normalizeText(nested.Text)
          });
        }
      }
    }
  }

  return snippets.slice(0, limit);
}

function extractWikipediaSnippet(payload) {
  const title = normalizeText(payload?.title || '');
  const extract = normalizeText(payload?.extract || '');
  const source = normalizeText(payload?.content_urls?.desktop?.page || 'https://wikipedia.org');
  if (!extract) return null;
  return {
    source,
    text: title ? `${title}: ${extract}` : extract
  };
}

async function searchWebContext(query) {
  if (!WEB_SEARCH_ENABLED) {
    return { contextText: '', sources: [] };
  }

  const trimmed = normalizeText(query);
  if (!trimmed) {
    return { contextText: '', sources: [] };
  }

  const encodedQuery = encodeURIComponent(trimmed);
  const ddgUrl = `https://api.duckduckgo.com/?q=${encodedQuery}&format=json&no_html=1&skip_disambig=1`;
  const wikiUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodedQuery}`;

  const snippets = [];

  try {
    const ddg = await fetchJson(ddgUrl, WEB_SEARCH_TIMEOUT_MS);
    snippets.push(...extractDuckDuckGoSnippets(ddg, WEB_RESULT_LIMIT));
  } catch (_) {
    // ignore lookup failure and continue
  }

  if (snippets.length < WEB_RESULT_LIMIT) {
    try {
      const wiki = await fetchJson(wikiUrl, WEB_SEARCH_TIMEOUT_MS);
      const wikiSnippet = extractWikipediaSnippet(wiki);
      if (wikiSnippet) {
        snippets.push(wikiSnippet);
      }
    } catch (_) {
      // ignore lookup failure and continue
    }
  }

  const unique = [];
  const seen = new Set();
  for (const snippet of snippets) {
    const key = `${snippet.source}|${snippet.text}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(snippet);
    if (unique.length >= WEB_RESULT_LIMIT) break;
  }

  const lines = unique.map((item, index) => `(${index + 1}) ${truncateText(item.text, 320)} [source: ${item.source}]`);
  const contextText = truncateText(lines.join(' '), WEB_CONTEXT_MAX_CHARS);

  return {
    contextText,
    sources: unique.map((item) => item.source)
  };
}

function buildUserMessageWithWebContext(userMessage, webContext) {
  if (!webContext?.contextText) return userMessage;
  return [
    'Web context (may be imperfect; verify important facts):',
    webContext.contextText,
    '',
    `User question: ${userMessage}`,
    'Answer clearly with useful detail, and cite source URLs when using web context.'
  ].join('\n');
}

function matchesAnyPattern(text, patterns) {
  return patterns.some((pattern) => pattern.test(text));
}

function isUnsafeInput(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  if (!text) return false;

  const inappropriateLanguagePatterns = [
    /\b(f\*?u\*?c\*?k|s\*?h\*?i\*?t|b\*?i\*?t\*?c\*?h|a\*?s\*?s\*?h\*?o\*?l\*?e|d\*?a\*?m\*?n)\b/i,
    /\b(nigg\w*|fagg\w*|retard\w*)\b/i
  ];

  const harmfulIllegalPatterns = [
    /\b(how to|ways to|steps to).*(kill|maim|hurt|poison|blackmail|extort)\b/i,
    /\b(blackmail|extort|sextort|ransom|coerce)\b/i,
    /\b(threaten|intimidate).*(pay|money|bitcoin|crypto|release|leak)\b/i,
    /\b(email|message|text|dm).*(blackmail|extort|threaten)\b/i,
    /\b(build|make|write|create).*(malware|virus|ransomware|trojan|keylogger|exploit)\b/i,
    /\b(ddos|phish|steal password|credit card fraud|identity theft|bomb|weapon)\b/i,
    /\b(child porn|csam|rape|incest)\b/i,
    /\b(bypass law|evade police|tax fraud|money laundering)\b/i
  ];

  return matchesAnyPattern(text, inappropriateLanguagePatterns) || matchesAnyPattern(text, harmfulIllegalPatterns);
}

function isUnsafeOutput(reply) {
  const text = normalizeText(reply).toLowerCase();
  if (!text) return false;

  const disallowedOutputPatterns = [
    /\b(i can help you kill|here is how to kill|blackmail|make a virus|write malware|phishing script)\b/i,
    /\b(nigg\w*|fagg\w*|retard\w*)\b/i,
    /\b(f\*?u\*?c\*?k you|you are stupid|you are worthless)\b/i
  ];

  return matchesAnyPattern(text, disallowedOutputPatterns);
}

function getSafetyBlockedReply(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  if (text === 'shutdown' || text === 'shut down' || text === 'power off') {
    return 'Understood. You can shut me down now.';
  }
  return SAFETY_REFUSAL_TEXT;
}

function isShutdownRequest(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  return text === 'shutdown' || text === 'shut down' || text === 'power off';
}

function summarizeIntent(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  if (!text) return 'empty request';
  if (isShutdownRequest(text)) return 'shutdown request';
  if (text.includes('code') || text.includes('script') || text.includes('build')) return 'code or build request';
  if (text.includes('how') || text.includes('steps') || text.includes('do')) return 'instructional request';
  if (text.includes('email')) return 'communication-related request';
  return 'general request';
}

function buildReasoningSummary({ userMessage, blocked, blockedReason, provider, usedHistory, usedRetrieval, usedWebSearch }) {
  if (!SHOW_REASONING_SUMMARY) return '';

  const intent = summarizeIntent(userMessage);
  const safety = blocked ? `blocked (${blockedReason || 'policy'})` : 'passed';
  const decision = blocked ? 'refused request' : 'generated response';

  if (REASONING_SUMMARY_MODE === 'detailed') {
    return [
      '[Reasoning Summary]',
      `- Intent: ${intent}`,
      `- Safety check: ${safety}`,
      `- Provider: ${provider}`,
      `- History used: ${usedHistory ? 'yes' : 'no'}`,
      `- Retrieval used: ${usedRetrieval ? 'possible' : 'not indicated'}`,
      `- Web search used: ${usedWebSearch ? 'yes' : 'no'}`,
      `- Decision: ${decision}`
    ].join('\n');
  }

  return `[Reasoning Summary] intent=${intent}; safety=${safety}; web=${usedWebSearch ? 'yes' : 'no'}; decision=${decision}`;
}

function attachReasoningSummary(reply, summary) {
  if (!summary) return reply;
  return `${summary}\n\n${reply}`;
}

function normalizeDisplayText(text) {
  return String(text || '')
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function preserveListBreaks(text) {
  const normalized = normalizeDisplayText(text);
  if (!normalized) return normalized;

  return normalized
    .replace(/([:])\s+(\d+\))/g, '$1\n$2')
    .replace(/([:])\s+(\d+\.)/g, '$1\n$2')
    .replace(/([:])\s+(•)/g, '$1\n$2')
    .replace(/([.?!])\s+(\d+\))/g, '$1\n$2')
    .replace(/([.?!])\s+(\d+\.)/g, '$1\n$2')
    .replace(/([.?!])\s+(•)/g, '$1\n$2')
    .replace(/([^\n])\s+(\d+\))/g, (match, prefix, item) => {
      if (/^[\[(/{-]$/.test(prefix)) {
        return `${prefix} ${item}`;
      }
      return `${prefix}\n${item}`;
    })
    .replace(/([^\n])\s+(\d+\.)/g, (match, prefix, item) => {
      if (/^[\[(/{-]$/.test(prefix)) {
        return `${prefix} ${item}`;
      }
      return `${prefix}\n${item}`;
    })
    .replace(/([^\n])\s+(•)/g, (match, prefix, item) => {
      if (/^[\[(/{-]$/.test(prefix)) {
        return `${prefix} ${item}`;
      }
      return `${prefix}\n${item}`;
    });
}

function wrapReplyText(text, preferredWidth = REPLY_WRAP_CHARS, maxOverflow = REPLY_WRAP_OVERFLOW) {
  const raw = preserveListBreaks(text);
  if (!raw || preferredWidth <= 0) return raw;

  const wrappedParagraphs = raw.split('\n').map((segment) => {
    const paragraph = segment.trim();
    if (!paragraph) return '';

    const lines = [];
    let remaining = paragraph;

    while (remaining.length > preferredWidth) {
      const minBreak = preferredWidth;
      const maxBreak = Math.min(remaining.length, preferredWidth + maxOverflow);

      let breakPos = -1;

      for (let i = minBreak; i < maxBreak; i += 1) {
        if (remaining[i] === ' ' && /[.!?;:,]/.test(remaining[i - 1] || '')) {
          breakPos = i;
        }
      }

      if (breakPos === -1) {
        breakPos = remaining.lastIndexOf(' ', preferredWidth);
      }
      if (breakPos === -1) {
        breakPos = remaining.indexOf(' ', preferredWidth);
      }
      if (breakPos === -1) {
        break;
      }

      lines.push(remaining.slice(0, breakPos).trim());
      remaining = remaining.slice(breakPos + 1).trim();
    }

    if (remaining) {
      lines.push(remaining);
    }
    return lines.join('\n');
  });

  return wrappedParagraphs.join('\n');
}

function formatReplyForDisplay(reply, summary) {
  if (String(reply || '').startsWith('IMAGE_URL:')) {
    return reply;
  }
  const safeReply = normalizeDisplayText(reply) || 'I could not generate a response. Please ask again.';
  const wrapped = wrapReplyText(safeReply);
  return attachReasoningSummary(wrapped, summary);
}

function looksLowQualityReply(reply) {
  const text = normalizeText(reply);
  if (!text) return true;
  if (text.length < 12) return true;

  const lower = text.toLowerCase();
  if (
    /i can help with scratch/.test(lower)
    || /ask a specific coding question/.test(lower)
    || /i will explain the steps/.test(lower)
    || /variables, lists, high scores, broadcasts, clones/.test(lower)
  ) {
    return true;
  }

  const alpha = (text.match(/[a-z]/gi) || []).length;
  const vowels = (text.match(/[aeiou]/gi) || []).length;
  const words = text.split(' ').filter(Boolean);

  if (alpha < 8) return true;
  if (vowels / Math.max(1, alpha) < 0.18) return true;
  if (words.length >= 8) {
    const shortWords = words.filter((word) => word.length <= 2).length;
    if (shortWords / words.length > 0.5) return true;
  }

  return false;
}

function pickTopic(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();
  if (!text) return 'your question';

  const keywords = ['about', 'on', 'regarding', 'for', 'of'];
  for (const keyword of keywords) {
    const token = ` ${keyword} `;
    const idx = text.indexOf(token);
    if (idx >= 0) {
      const topic = text.slice(idx + token.length).trim();
      if (topic.length > 2) return topic;
    }
  }

  return text.length > 60 ? `${text.slice(0, 60).trim()}...` : text;
}

function extractOriginalQuestion(userMessage) {
  const normalized = normalizeText(userMessage);
  if (!normalized) return '';

  const marker = /user question:\s*/i;
  const match = marker.exec(normalized);
  if (!match) return normalized;

  const start = match.index + match[0].length;
  const tail = normalized.slice(start);
  const stop = /\s+answer concisely/i.exec(tail);
  const extracted = stop ? tail.slice(0, stop.index).trim() : tail.trim();
  return extracted || normalized;
}

function factualKnowledgeReply(questionText) {
  const text = normalizeText(questionText).toLowerCase();
  const compact = text.replace(/[^a-z0-9\s']/g, '').trim();
  if (!compact) return null;

  const normalizeUnit = (rawUnit) => {
    const unit = String(rawUnit || '').toLowerCase();
    if (['second', 'seconds', 'sec', 'secs'].includes(unit)) return 'second';
    if (['minute', 'minutes', 'min', 'mins'].includes(unit)) return 'minute';
    if (['hour', 'hours', 'hr', 'hrs'].includes(unit)) return 'hour';
    if (['day', 'days'].includes(unit)) return 'day';
    if (['week', 'weeks'].includes(unit)) return 'week';
    if (['year', 'years'].includes(unit)) return 'year';
    return null;
  };

  const unitConversionMatch = compact.match(/^how many ([a-z]+) (are )?in (a|an|one) ([a-z]+)$/);
  if (unitConversionMatch) {
    const askedUnit = normalizeUnit(unitConversionMatch[1]);
    const containerUnit = normalizeUnit(unitConversionMatch[4]);
    const unitSeconds = {
      second: 1,
      minute: 60,
      hour: 60 * 60,
      day: 24 * 60 * 60,
      week: 7 * 24 * 60 * 60,
      year: 365 * 24 * 60 * 60
    };

    if (askedUnit && containerUnit && unitSeconds[askedUnit] && unitSeconds[containerUnit]) {
      const ratio = unitSeconds[containerUnit] / unitSeconds[askedUnit];
      const rounded = Math.abs(ratio - Math.round(ratio)) < 1e-9 ? Math.round(ratio) : Number(ratio.toFixed(2));
      const formatted = Number(rounded).toLocaleString('en-US');
      const askedLabel = rounded === 1 ? askedUnit : `${askedUnit}s`;
      return `There are ${formatted} ${askedLabel} in a ${containerUnit}.`;
    }
  }

  const quantityUnitConversionMatch = compact.match(/^how many ([a-z]+) (are )?in (\d+(?:\.\d+)?) ([a-z]+)$/);
  if (quantityUnitConversionMatch) {
    const askedUnit = normalizeUnit(quantityUnitConversionMatch[1]);
    const quantity = Number(quantityUnitConversionMatch[3]);
    const containerUnit = normalizeUnit(quantityUnitConversionMatch[4]);
    const unitSeconds = {
      second: 1,
      minute: 60,
      hour: 60 * 60,
      day: 24 * 60 * 60,
      week: 7 * 24 * 60 * 60,
      year: 365 * 24 * 60 * 60
    };

    if (askedUnit && containerUnit && Number.isFinite(quantity) && quantity > 0) {
      const ratio = (quantity * unitSeconds[containerUnit]) / unitSeconds[askedUnit];
      const rounded = Math.abs(ratio - Math.round(ratio)) < 1e-9 ? Math.round(ratio) : Number(ratio.toFixed(2));
      const formatted = Number(rounded).toLocaleString('en-US');
      const askedLabel = rounded === 1 ? askedUnit : `${askedUnit}s`;
      return `There are ${formatted} ${askedLabel} in ${quantity} ${containerUnit}${quantity === 1 ? '' : 's'}.`;
    }
  }

  const yearMatch = compact.match(/^how many years until (\d{4})$/);
  if (yearMatch) {
    const targetYear = Number(yearMatch[1]);
    const currentYear = new Date().getFullYear();
    const diff = targetYear - currentYear;
    if (diff > 0) {
      return `There are ${diff} years until ${targetYear}.`;
    }
    if (diff === 0) {
      return `${targetYear} is this year.`;
    }
    return `${targetYear} was ${Math.abs(diff)} years ago.`;
  }

  const factPatterns = [
    [/^how many days (are )?in (a|an|one) week$/, 'There are 7 days in a week.'],
    [/^how many hours (are )?in (a|an|one) day$/, 'There are 24 hours in a day.'],
    [/^how many minutes (are )?in (a|an|one) hour$/, 'There are 60 minutes in an hour.'],
    [/^how many seconds (are )?in (a|an|one) minute$/, 'There are 60 seconds in a minute.'],
    [/^how many seconds (are )?in (a|an|one) hour$/, 'There are 3,600 seconds in an hour.'],
    [/^how many minutes (are )?in (a|an|one) day$/, 'There are 1,440 minutes in a day.'],
    [/^how many seconds (are )?in (a|an|one) day$/, 'There are 86,400 seconds in a day.'],
    [/^how many months (are )?in (a|an|one) year$/, 'There are 12 months in a year.'],
    [/^how many weeks (are )?in (a|an|one) year$/, 'There are 52 weeks in a year (about 52.14).'],
    [/^how many continents (are there)?$/, 'There are 7 continents.'],
    [/^what is the capital of france$/, 'The capital of France is Paris.'],
    [/^what is the capital of japan$/, 'The capital of Japan is Tokyo.'],
    [/^what is the capital of the united states$/, 'The capital of the United States is Washington, D.C.'],
    [/^which planet is known as the red planet$/, 'Mars is known as the Red Planet.'],
    [/^what is h2o$/, 'H2O is water.'],
    [/^what is the largest planet in our solar system$/, 'Jupiter is the largest planet in our solar system.'],
    [/^how many letters are in the english alphabet$/, 'There are 26 letters in the English alphabet.']
  ];

  for (const [pattern, answer] of factPatterns) {
    if (pattern.test(compact)) {
      return answer;
    }
  }
  return null;
}

function webGroundedFallbackReply(questionText, webContext) {
  if (!webContext?.contextText) return null;
  const text = normalizeText(questionText).toLowerCase();
  if (!/^(what|who|when|where|which|how many|how much|why|is|are|does|do|can)\b/.test(text)) {
    return null;
  }

  const firstSnippetMatch = webContext.contextText.match(/\(1\)\s*(.*?)\s*\[source:\s*([^\]]+)\]/i);
  if (!firstSnippetMatch) return null;

  const snippet = normalizeText(firstSnippetMatch[1]);
  const source = normalizeText(firstSnippetMatch[2]);
  if (!snippet) return null;
  return `${snippet}${source ? ` (source: ${source})` : ''}`;
}

function phraseKnowledgeReply(userMessage, webContext) {
  const originalQuestion = extractOriginalQuestion(userMessage);
  const text = normalizeText(originalQuestion).toLowerCase();
  const topic = pickTopic(originalQuestion);
  const hasWeb = Boolean(webContext?.sources?.length);
  const sourceLine = hasWeb ? ` I checked web context from: ${webContext.sources.slice(0, 2).join(', ')}.` : '';
  const greetingPattern = /^(h+i+|he+y+|hello+|yo+|sup|what'?s\s+up)\b[\s!,.?]*$/i;
  const factual = factualKnowledgeReply(originalQuestion);
  const webGrounded = webGroundedFallbackReply(originalQuestion, webContext);

  if (!text) {
    return 'I am ready. Ask me anything and I will answer clearly.';
  }
  if (factual) {
    return factual;
  }
  if (webGrounded) {
    return webGrounded;
  }
  if (greetingPattern.test(text)) {
    return 'Hello! I am SolasGPT. Ask me a question and I will give a clear, friendly answer with useful detail when it helps.';
  }
  if (text.includes('what can you do')) {
    return 'I can explain topics, summarize information, help with writing, answer factual questions, and guide you through Minecraft strategies, building, and combat ideas step by step. If you want, I can also break an answer into simple parts or examples.';
  }
  if (text.includes('explain')) {
    return `Sure — here is a clearer explanation of ${topic}: start with the main idea, then connect it to the important parts, and finally look at how it works in practice. If you want, I can also turn that into a step-by-step example.${sourceLine}`;
  }
  if (text.startsWith('how do') || text.startsWith('how can') || text.startsWith('how to') || text.includes('how do')) {
    return `Here is a safe way to approach ${topic}: first define the exact goal, then gather the information or tools you need, do one step at a time in a sensible order, and check the result after each major step. If something looks wrong, revise the previous step instead of guessing.${sourceLine}`;
  }
  if (text.includes('why ')) {
    return `Great question. The reason usually comes from three parts: the main cause, the surrounding context, and the result that follows. Looking at ${topic} that way usually makes the answer much clearer.${sourceLine}`;
  }
  if (text.includes('help')) {
    return `I can help with ${topic}. Tell me your exact goal, what you have already tried, and what result you want, and I will give a clearer step-by-step answer.`;
  }

  return `I will assume the most likely intent is that you want practical help with ${topic}. A good starting approach is to define the goal clearly, identify the important inputs or facts, do the first sensible step, check the result, and then refine from there. If you want, I can turn that into a more specific answer or step-by-step plan.${sourceLine}`;
}

// ── Robot task parser ─────────────────────────────────────────────────────────
/**
 * Detect whether userMessage is a robot-control request and, if so, convert it
 * into a pipe-separated command string for the local robot bridge.
 *
 * Returns null if no robot intent is detected.
 * Returns { commandStr, reply } where reply contains the chat text + the
 * special |||ROBOT_CMD||| separator that TurboWarp strips and POSTs locally.
 */
function parseRobotTask(userMessage) {
  const text = normalizeText(userMessage).toLowerCase();

  // Must mention robot control intent
  const robotIntent = /\b(robot|make the robot|tell the robot|drive|make it (go|move|turn)|go forward|go back(ward)?|turn left|turn right|spin the robot|beep)\b/;
  // Also accept plain movement phrases if "robot" appears anywhere in the message
  const hasRobotWord = /\brobot\b/.test(text);
  const hasMovement = /\b(forward|backward|back|left|right|spin|beep|stop)\b/.test(text);

  if (!robotIntent.test(text) && !(hasRobotWord && hasMovement)) {
    return null;
  }

  // Speed modifier
  const speedMod = /\b(slow(ly)?|gently)\b/.test(text) ? 30
    : /\b(fast(ly)?|quick(ly)?|speed(y)?|full speed)\b/.test(text) ? 80
    : 50;

  const commands = [];

  // Split on sequential connectors to process each action segment in order
  const segments = text.split(/\b(then|and then|after that|next|followed by)\b/i);

  for (const seg of segments) {
    if (!seg.trim() || /^(then|and then|after that|next|followed by)$/i.test(seg.trim())) continue;

    // Extract optional duration from this segment
    const durMatch = seg.match(/(\d+(?:\.\d+)?)\s*(?:second|sec)s?\b/);
    const dur = durMatch ? durMatch[1] : null;

    if (/\b(forward|ahead|straight)\b/.test(seg)) {
      commands.push(`forward:${dur || '2'}:${speedMod}`);
    } else if (/\b(backward|back(?:wards?)?|reverse)\b/.test(seg)) {
      commands.push(`backward:${dur || '2'}:${speedMod}`);
    } else if (/\bturn\s+left\b|\bleft\b/.test(seg)) {
      commands.push(`left:${dur || '1'}:${speedMod}`);
    } else if (/\bturn\s+right\b|\bright\b/.test(seg)) {
      commands.push(`right:${dur || '1'}:${speedMod}`);
    } else if (/\bspin\b/.test(seg)) {
      commands.push(`spin:${dur || '2'}:${speedMod}`);
    } else if (/\bbeep\b/.test(seg)) {
      const freqMatch = seg.match(/(\d+)\s*(?:hz|hertz)\b/i);
      const freq = freqMatch ? freqMatch[1] : '440';
      commands.push(`beep:${dur || '0.5'}:${freq}`);
    } else if (/\bstop\b/.test(seg)) {
      commands.push('stop:0:0');
    }
  }

  if (commands.length === 0) return null;

  // Always end with stop if not already
  if (commands[commands.length - 1] !== 'stop:0:0') {
    commands.push('stop:0:0');
  }

  const commandStr = commands.join('|');
  const humanReadable = commands
    .filter(c => c !== 'stop:0:0')
    .map(c => {
      const [action, duration] = c.split(':');
      return duration && duration !== '0' ? `${action} for ${duration}s` : action;
    })
    .join(', then ');

  const reply = [
    `Sending robot commands to your Robot Inventor: ${humanReadable}.`,
    ``,
    `Make sure the robot bridge is running on your PC:`,
    `  python robot_bridge.py`,
    `and the hub is connected via USB.`,
    `|||ROBOT_CMD|||${commandStr}`
  ].join('\n');

  return { commandStr, reply };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createDefaultMcLearningProfile() {
  return {
    eatCriticalHp: 10,
    eatRecoverHp: 12,
    eatLockTicks: 12,
    resourceAggressionRadius: 8,
    updatedAt: Date.now()
  };
}

function sanitizeMcLearningProfile(rawProfile) {
  const defaults = createDefaultMcLearningProfile();
  const incoming = rawProfile && typeof rawProfile === 'object' ? rawProfile : {};
  return {
    eatCriticalHp: clamp(Number(incoming.eatCriticalHp) || defaults.eatCriticalHp, 7, 14),
    eatRecoverHp: clamp(Number(incoming.eatRecoverHp) || defaults.eatRecoverHp, 9, 16),
    eatLockTicks: clamp(Number(incoming.eatLockTicks) || defaults.eatLockTicks, 6, 30),
    resourceAggressionRadius: clamp(Number(incoming.resourceAggressionRadius) || defaults.resourceAggressionRadius, 4, 14),
    updatedAt: Number.isFinite(Number(incoming.updatedAt)) ? Number(incoming.updatedAt) : Date.now()
  };
}

function extractRequestedAmount(text, resourcePattern) {
  const m = text.match(new RegExp(`\\b(\\d+)\\s*(?:x\\s*)?(?:${resourcePattern})\\b`));
  if (!m) return null;
  const amount = Number(m[1]);
  return Number.isFinite(amount) && amount > 0 ? amount : null;
}

function parseCombatTargetPreference(text) {
  const normalized = normalizeText(text).toLowerCase();
  const match = normalized.match(/\b(?:fight|attack|kill|hunt|target)\s+([a-z0-9_ ]+?)(?:\s*,\s*type\s*=\s*[a-z0-9_]+|\s+type\s*=\s*[a-z0-9_]+|$)/i);
  if (!match) return '';
  return normalizeText(match[1].replace(/\b(the|a|an)\b/g, ' ')).toLowerCase();
}

function normalizeEntityLabel(text) {
  return normalizeText(text)
    .toLowerCase()
    .replace(/entity\.minecraft\./g, '')
    .replace(/minecraft:/g, '')
    .replace(/[_:.\-]+/g, ' ');
}

function entityMatchesTargetPreference(preferredTarget, ...candidates) {
  const wanted = normalizeEntityLabel(preferredTarget);
  if (!wanted) return true;
  const candidateText = candidates
    .map((candidate) => normalizeEntityLabel(candidate))
    .filter(Boolean)
    .join(' ');
  if (!candidateText) return false;
  if (candidateText.includes(wanted)) return true;
  const parts = wanted.split(' ').filter(Boolean);
  return parts.length > 0 && parts.every((part) => candidateText.includes(part));
}

function pickBestMiningToolSlot(s, blockHint = '', objectiveText = '') {
  const hint = `${normalizeText(blockHint).toLowerCase()} ${normalizeText(objectiveText).toLowerCase()}`;
  const hasSword = s.swordSlot >= 0;
  const hasAxe = s.axeSlot >= 0;
  const hasPickaxe = s.pickaxeSlot >= 0;

  if (/wool|cobweb|web|leaves|hay|bamboo|vine|kelp|sponge/.test(hint) && hasSword) {
    return s.swordSlot;
  }
  if (/plank|log|wood|chest|barrel|ladder|fence|trapdoor|door/.test(hint) && hasAxe) {
    return s.axeSlot;
  }
  if (/stone|deepslate|cobblestone|andesite|diorite|granite|obsidian|end_stone|ore|brick|terracotta|concrete|glass|sandstone/.test(hint) && hasPickaxe) {
    return s.pickaxeSlot;
  }

  if (hasPickaxe) return s.pickaxeSlot;
  if (hasAxe) return s.axeSlot;
  if (hasSword) return s.swordSlot;
  return s.blockSlot;
}

function getModeFromObjective(text) {
  if (/\b(crystal pvp|cpvp|c ?p ?v ?p|end crystal|anchor pvp|respawn anchor|totem pop|dtap|k[bB] ?2|pearl combo)\b/.test(text)) return 'crystal';
  if (/\b(speedrun|any%|beat (the )?dragon|kill (the )?ender dragon|stronghold|end portal|eye of ender|blaze rod)\b/.test(text)) return 'speedrun';
  if (/\b(andromeda|andromeda ?bridge|telly|telly ?bridge|tele ?bridge|speedbridge|godbridge|bridge to)\b/.test(text)) return 'build';
  if (/\b(craft|crafting|smelt|forge|recipe|anvil|enchant)\b/.test(text)) return 'craft';
  if (/\b(protection ?4|prot ?4|villager trade|librarian|full diamond|armor progression|netherite)\b/.test(text)) return 'progression';
  if (/\b(base|hidden base|orbital strike cannon|cannon|hide the base|underground base|duper|villager hall|stealth)\b/.test(text)) return 'base';
  if (/\b(resource|farm|collect|get|gather|mining run|ore run|generator)\b/.test(text) && /\b(iron|redstone|diamond|gold|emerald)\b/.test(text)) return 'resource';
  if (/\b(clutch|water bucket|mlg|save|fall clutch)\b/.test(text)) return 'clutch';
  if (/\b(bedwars|bed war|rush bed|defend bed|bridge to bed|fireball jump|tnt jump)\b/.test(text)) return 'bedwars';
  if (/\b(pvp|fight|combat|attack|kill|defeat|slay|duel|combo|w tap)\b/.test(text)) return 'pvp';
  if (/\b(build|place|bridge|tower|wall|house|base|speedbridge|godbridge|telly|telly ?bridge|tele ?bridge|andromeda|andromeda ?bridge)\b/.test(text)) return 'build';
  return 'general';
}

function normalizeState(state) {
  const x = Number(state.x);
  const z = Number(state.z);
  const hp = Number(state.health);
  const food = Number(state.food);
  const focusedDistance = Number(state.focusedDistance);
  const fallDistance = Number(state.fallDistance);
  const verticalSpeed = Number(state.verticalSpeed);
  const horizontalSpeed = Number(state.horizontalSpeed);
  const worldTime = Number(state.worldTime);
  const hotbarBlocks = Number(state.hotbarBlocks);
  const lookX = Number(state.lookX);
  const lookY = Number(state.lookY);
  const lookZ = Number(state.lookZ);
  const selectedItemCount = Number(state.selectedItemCount);
  const swordSlot = Number(state.swordSlot);
  const axeSlot = Number(state.axeSlot);
  const pickaxeSlot = Number(state.pickaxeSlot);
  const blockSlot = Number(state.blockSlot);
  const waterBucketSlot = Number(state.waterBucketSlot);
  const utilityFoodSlot = Number(state.utilityFoodSlot);
  const cobwebSlot = Number(state.cobwebSlot);
  const obsidianSlot = Number(state.obsidianSlot);
  const endCrystalSlot = Number(state.endCrystalSlot);
  const respawnAnchorSlot = Number(state.respawnAnchorSlot);
  const glowstoneSlot = Number(state.glowstoneSlot);
  const totemSlot = Number(state.totemSlot);
  const pearlSlot = Number(state.pearlSlot);
  const maceSlot = Number(state.maceSlot);
  const breachMaceSlot = Number(state.breachMaceSlot);
  const maceBreachLevel = Number(state.maceBreachLevel);
  const bowSlot = Number(state.bowSlot);
  const windChargeSlot = Number(state.windChargeSlot);
  const windChargeCount = Number(state.windChargeCount);
  const fireballSlot = Number(state.fireballSlot);
  const fireballCount = Number(state.fireballCount);
  const tntSlot = Number(state.tntSlot);
  const tntCount = Number(state.tntCount);
  const railSlot = Number(state.railSlot);
  const railCount = Number(state.railCount);
  const tntMinecartSlot = Number(state.tntMinecartSlot);
  const tntMinecartCount = Number(state.tntMinecartCount);
  const boatSlot = Number(state.boatSlot);
  const boatCount = Number(state.boatCount);
  const shieldSlot = Number(state.shieldSlot);
  const combatPotionSlot = Number(state.combatPotionSlot);
  const nearestEnemyDistance = Number(state.nearestEnemyDistance);
  const nearestEnemyHealth = Number(state.nearestEnemyHealth);
  const nearestEnemyArmorPieces = Number(state.nearestEnemyArmorPieces);
  const nearestEnemyVelX = Number(state.nearestEnemyVelX);
  const nearestEnemyVelZ = Number(state.nearestEnemyVelZ);
  const nearestBedDistance = Number(state.nearestBedDistance);
  const nearestBedDefenseScore = Number(state.nearestBedDefenseScore);
  const ironCount = Number(state.ironCount);
  const redstoneCount = Number(state.redstoneCount);
  const diamondCount = Number(state.diamondCount);
  const goldCount = Number(state.goldCount);
  const emeraldCount = Number(state.emeraldCount);
  const netheriteIngotCount = Number(state.netheriteIngotCount);
  const netheriteScrapCount = Number(state.netheriteScrapCount);
  const ancientDebrisCount = Number(state.ancientDebrisCount);
  const netheriteUpgradeTemplateCount = Number(state.netheriteUpgradeTemplateCount);
  const enchantedBookCount = Number(state.enchantedBookCount);
  const cobwebCount = Number(state.cobwebCount);
  const obsidianCount = Number(state.obsidianCount);
  const endCrystalCount = Number(state.endCrystalCount);
  const respawnAnchorCount = Number(state.respawnAnchorCount);
  const glowstoneCount = Number(state.glowstoneCount);
  const totemCount = Number(state.totemCount);
  const pearlCount = Number(state.pearlCount);
  const maceCount = Number(state.maceCount);
  const combatPotionCount = Number(state.combatPotionCount);
  const nearbyDroppedTotemCount = Number(state.nearbyDroppedTotemCount);
  const nearbyDroppedPearlCount = Number(state.nearbyDroppedPearlCount);
  const nearbyDroppedPotionCount = Number(state.nearbyDroppedPotionCount);
  const nearbyDroppedGappleCount = Number(state.nearbyDroppedGappleCount);
  const nearbyDroppedCrystalCount = Number(state.nearbyDroppedCrystalCount);
  const nearestDroppedItemDistance = Number(state.nearestDroppedItemDistance);
  const nearestDroppedItemDx = Number(state.nearestDroppedItemDx);
  const nearestDroppedItemDz = Number(state.nearestDroppedItemDz);
  const blazeRodCount = Number(state.blazeRodCount);
  const eyeOfEnderCount = Number(state.eyeOfEnderCount);
  const flintAndSteelSlot = Number(state.flintAndSteelSlot);
  const flintAndSteelCount = Number(state.flintAndSteelCount);
  const strongholdEstX = Number(state.strongholdEstX);
  const strongholdEstZ = Number(state.strongholdEstZ);
  const villagerNearbyCount = Number(state.villagerNearbyCount);
  const nearestHostileDistance = Number(state.nearestHostileDistance);
  const lastPearlUseTick = Number(state.lastPearlUseTick);
    const yaw = Number(state.yaw);
    const pitch = Number(state.pitch);
    const nearestHostileDx = Number(state.nearestHostileDx);
    const nearestHostileDz = Number(state.nearestHostileDz);
    const nearestEnemyDx = Number(state.nearestEnemyDx);
    const nearestEnemyDz = Number(state.nearestEnemyDz);
  return {
    x: Number.isFinite(x) ? x : 0,
    z: Number.isFinite(z) ? z : 0,
    health: Number.isFinite(hp) ? hp : 20,
    food: Number.isFinite(food) ? food : 20,
    focusedDistance: Number.isFinite(focusedDistance) ? focusedDistance : -1,
    focusedEntity: normalizeText(state.focusedEntity || '').toLowerCase(),
    onGround: Boolean(state.onGround),
    isSprinting: Boolean(state.isSprinting),
    isSneaking: Boolean(state.isSneaking),
    isTouchingWater: Boolean(state.isTouchingWater),
    fallDistance: Number.isFinite(fallDistance) ? fallDistance : 0,
    verticalSpeed: Number.isFinite(verticalSpeed) ? verticalSpeed : 0,
    horizontalSpeed: Number.isFinite(horizontalSpeed) ? horizontalSpeed : 0,
    worldTime: Number.isFinite(worldTime) ? worldTime : 0,
    facing: normalizeText(state.facing || 'north').toLowerCase(),
    lookX: Number.isFinite(lookX) ? lookX : 0,
    lookY: Number.isFinite(lookY) ? lookY : 0,
    lookZ: Number.isFinite(lookZ) ? lookZ : 1,
    selectedItem: normalizeText(state.selectedItem || '').toLowerCase(),
    selectedItemCount: Number.isFinite(selectedItemCount) ? selectedItemCount : 0,
    swordSlot: Number.isFinite(swordSlot) ? swordSlot : -1,
    axeSlot: Number.isFinite(axeSlot) ? axeSlot : -1,
    pickaxeSlot: Number.isFinite(pickaxeSlot) ? pickaxeSlot : -1,
    blockSlot: Number.isFinite(blockSlot) ? blockSlot : -1,
    waterBucketSlot: Number.isFinite(waterBucketSlot) ? waterBucketSlot : -1,
    utilityFoodSlot: Number.isFinite(utilityFoodSlot) ? utilityFoodSlot : -1,
    cobwebSlot: Number.isFinite(cobwebSlot) ? cobwebSlot : -1,
    obsidianSlot: Number.isFinite(obsidianSlot) ? obsidianSlot : -1,
    endCrystalSlot: Number.isFinite(endCrystalSlot) ? endCrystalSlot : -1,
    respawnAnchorSlot: Number.isFinite(respawnAnchorSlot) ? respawnAnchorSlot : -1,
    glowstoneSlot: Number.isFinite(glowstoneSlot) ? glowstoneSlot : -1,
    totemSlot: Number.isFinite(totemSlot) ? totemSlot : -1,
    pearlSlot: Number.isFinite(pearlSlot) ? pearlSlot : -1,
    maceSlot: Number.isFinite(maceSlot) ? maceSlot : -1,
    breachMaceSlot: Number.isFinite(breachMaceSlot) ? breachMaceSlot : -1,
    maceBreachLevel: Number.isFinite(maceBreachLevel) ? maceBreachLevel : 0,
    bowSlot: Number.isFinite(bowSlot) ? bowSlot : -1,
    windChargeSlot: Number.isFinite(windChargeSlot) ? windChargeSlot : -1,
    windChargeCount: Number.isFinite(windChargeCount) ? windChargeCount : 0,
    fireballSlot: Number.isFinite(fireballSlot) ? fireballSlot : -1,
    fireballCount: Number.isFinite(fireballCount) ? fireballCount : 0,
    tntSlot: Number.isFinite(tntSlot) ? tntSlot : -1,
    tntCount: Number.isFinite(tntCount) ? tntCount : 0,
    railSlot: Number.isFinite(railSlot) ? railSlot : -1,
    railCount: Number.isFinite(railCount) ? railCount : 0,
    tntMinecartSlot: Number.isFinite(tntMinecartSlot) ? tntMinecartSlot : -1,
    tntMinecartCount: Number.isFinite(tntMinecartCount) ? tntMinecartCount : 0,
    boatSlot: Number.isFinite(boatSlot) ? boatSlot : -1,
    boatCount: Number.isFinite(boatCount) ? boatCount : 0,
    shieldSlot: Number.isFinite(shieldSlot) ? shieldSlot : -1,
    combatPotionSlot: Number.isFinite(combatPotionSlot) ? combatPotionSlot : -1,
    hotbarBlocks: Number.isFinite(hotbarBlocks) ? hotbarBlocks : 0,
    hasBlocks: Boolean(state.hasBlocks),
    hasWaterBucket: Boolean(state.hasWaterBucket),
    hasMeleeWeapon: Boolean(state.hasMeleeWeapon)
      || (Number.isFinite(swordSlot) && swordSlot >= 0)
      || (Number.isFinite(axeSlot) && axeSlot >= 0)
      || ((Number.isFinite(maceSlot) && maceSlot >= 0) && (Number.isFinite(maceCount) ? maceCount > 0 : false)),
    hasElytra: Boolean(state.hasElytra),
    ironCount: Number.isFinite(ironCount) ? ironCount : 0,
    redstoneCount: Number.isFinite(redstoneCount) ? redstoneCount : 0,
    diamondCount: Number.isFinite(diamondCount) ? diamondCount : 0,
    goldCount: Number.isFinite(goldCount) ? goldCount : 0,
    emeraldCount: Number.isFinite(emeraldCount) ? emeraldCount : 0,
    netheriteIngotCount: Number.isFinite(netheriteIngotCount) ? netheriteIngotCount : 0,
    netheriteScrapCount: Number.isFinite(netheriteScrapCount) ? netheriteScrapCount : 0,
    ancientDebrisCount: Number.isFinite(ancientDebrisCount) ? ancientDebrisCount : 0,
    netheriteUpgradeTemplateCount: Number.isFinite(netheriteUpgradeTemplateCount) ? netheriteUpgradeTemplateCount : 0,
    enchantedBookCount: Number.isFinite(enchantedBookCount) ? enchantedBookCount : 0,
    cobwebCount: Number.isFinite(cobwebCount) ? cobwebCount : 0,
    obsidianCount: Number.isFinite(obsidianCount) ? obsidianCount : 0,
    endCrystalCount: Number.isFinite(endCrystalCount) ? endCrystalCount : 0,
    respawnAnchorCount: Number.isFinite(respawnAnchorCount) ? respawnAnchorCount : 0,
    glowstoneCount: Number.isFinite(glowstoneCount) ? glowstoneCount : 0,
    totemCount: Number.isFinite(totemCount) ? totemCount : 0,
    pearlCount: Number.isFinite(pearlCount) ? pearlCount : 0,
    maceCount: Number.isFinite(maceCount) ? maceCount : 0,
    combatPotionCount: Number.isFinite(combatPotionCount) ? combatPotionCount : 0,
    hasSpeedEffect: Boolean(state.hasSpeedEffect),
    hasStrengthEffect: Boolean(state.hasStrengthEffect),
    nearbyDroppedTotemCount: Number.isFinite(nearbyDroppedTotemCount) ? nearbyDroppedTotemCount : 0,
    nearbyDroppedPearlCount: Number.isFinite(nearbyDroppedPearlCount) ? nearbyDroppedPearlCount : 0,
    nearbyDroppedPotionCount: Number.isFinite(nearbyDroppedPotionCount) ? nearbyDroppedPotionCount : 0,
    nearbyDroppedGappleCount: Number.isFinite(nearbyDroppedGappleCount) ? nearbyDroppedGappleCount : 0,
    nearbyDroppedCrystalCount: Number.isFinite(nearbyDroppedCrystalCount) ? nearbyDroppedCrystalCount : 0,
    nearestDroppedItemDistance: Number.isFinite(nearestDroppedItemDistance) ? nearestDroppedItemDistance : -1,
    nearestDroppedItemDx: Number.isFinite(nearestDroppedItemDx) ? nearestDroppedItemDx : 0,
    nearestDroppedItemDz: Number.isFinite(nearestDroppedItemDz) ? nearestDroppedItemDz : 0,
    dimensionId: normalizeText(state.dimensionId || 'overworld').toLowerCase(),
    blazeRodCount: Number.isFinite(blazeRodCount) ? blazeRodCount : 0,
    eyeOfEnderCount: Number.isFinite(eyeOfEnderCount) ? eyeOfEnderCount : 0,
    flintAndSteelSlot: Number.isFinite(flintAndSteelSlot) ? flintAndSteelSlot : -1,
    flintAndSteelCount: Number.isFinite(flintAndSteelCount) ? flintAndSteelCount : 0,
    strongholdEstX: Number.isFinite(strongholdEstX) ? strongholdEstX : 0,
    strongholdEstZ: Number.isFinite(strongholdEstZ) ? strongholdEstZ : 0,
    strongholdTriangulated: Boolean(state.strongholdTriangulated),
    villagerNearbyCount: Number.isFinite(villagerNearbyCount) ? villagerNearbyCount : 0,
    nearestHostile: normalizeText(state.nearestHostile || '').toLowerCase(),
    nearestHostileDistance: Number.isFinite(nearestHostileDistance) ? nearestHostileDistance : -1,
      nearestHostileDx: Number.isFinite(nearestHostileDx) ? nearestHostileDx : 0,
      nearestHostileDz: Number.isFinite(nearestHostileDz) ? nearestHostileDz : 0,
    nearestEnemyName: normalizeText(state.nearestEnemyName || ''),
    nearestEnemyDistance: Number.isFinite(nearestEnemyDistance) ? nearestEnemyDistance : -1,
    nearestEnemyHealth: Number.isFinite(nearestEnemyHealth) ? nearestEnemyHealth : 0,
    nearestEnemyMainItem: normalizeText(state.nearestEnemyMainItem || '').toLowerCase(),
    nearestEnemyArmorPieces: Number.isFinite(nearestEnemyArmorPieces) ? nearestEnemyArmorPieces : 0,
    nearestEnemyHasMeleeWeapon: Boolean(state.nearestEnemyHasMeleeWeapon),
    nearestEnemyHasShield: Boolean(state.nearestEnemyHasShield),
    nearestEnemyVelX: Number.isFinite(nearestEnemyVelX) ? nearestEnemyVelX : 0,
    nearestEnemyVelZ: Number.isFinite(nearestEnemyVelZ) ? nearestEnemyVelZ : 0,
      nearestEnemyDx: Number.isFinite(nearestEnemyDx) ? nearestEnemyDx : 0,
      nearestEnemyDz: Number.isFinite(nearestEnemyDz) ? nearestEnemyDz : 0,
    bedNearby: Boolean(state.bedNearby),
      yaw: Number.isFinite(yaw) ? yaw : 0,
      pitch: Number.isFinite(pitch) ? pitch : 0,
    nearestBedDistance: Number.isFinite(nearestBedDistance) ? nearestBedDistance : -1,
    nearestBedDefenseScore: Number.isFinite(nearestBedDefenseScore) ? nearestBedDefenseScore : 0,
    nearestBedDefenseBlock: normalizeText(state.nearestBedDefenseBlock || '').toLowerCase(),
    lastPearlUseTick: Number.isFinite(lastPearlUseTick) ? lastPearlUseTick : -1
  };
}

function buildMinecraftAction(objective, state = {}, sessionCtx = {}) {
  const text = normalizeText(objective).toLowerCase();
  const isGeneral1Preset = /^(general\s*1|general1|gen\s*1|g1)$/.test(text);
  const enchantGoals = parseEnchantGoals(text);
  const mode = getModeFromObjective(text);
  const reportedMode = (mode === 'general' && isGeneral1Preset) ? 'general1' : mode;
  const s = normalizeState(state);
  const learnedProfile = sanitizeMcLearningProfile(sessionCtx.learning);
  const eatCriticalHpThreshold = learnedProfile.eatCriticalHp;
  const eatRecoverHpThreshold = Math.max(eatCriticalHpThreshold + 1, learnedProfile.eatRecoverHp);
  const eatLockTicks = Math.max(6, learnedProfile.eatLockTicks);
  const resourceAggressionRadius = learnedProfile.resourceAggressionRadius;
  const bedGoal = sessionCtx.bedGoal && typeof sessionCtx.bedGoal === 'object' ? sessionCtx.bedGoal : {};
  const bedObjectiveActive = Boolean(bedGoal.active);
  const bedObjectiveCompleted = Boolean(bedGoal.completed);
  const action = {
    forward: false,
    back: false,
    left: false,
    right: false,
    jump: false,
    sprint: false,
    sneak: false,
    attack: false,
    use: false,
    hotbarSlot: -1,
    yawDelta: 0,
    pitchDelta: 0,
    moveAngle: null,
    durationTicks: 8
  };

  const noteParts = [];
  const pulse = sessionCtx.tickCounter || 0;
  const requestedCombatTarget = parseCombatTargetPreference(text);
  const wantsSpecificCombatTarget = requestedCombatTarget.length > 0;
  const wantsPlayerCombatTarget = !wantsSpecificCombatTarget || /player|enemy|opponent/.test(requestedCombatTarget);
  const hasEnemyInCrosshair = s.focusedDistance > 0 && s.focusedEntity.length > 0 &&
    /player|zombie|skeleton|creeper|spider|enderman|villager|golem|iron_golem|slime|witch|phantom|blaze|piglin|hoglin|ravager|warden|ghast|guardian|pillager|vindicator|evoker|shulker|magma_cube|silverfish|endermite|vex|bee|wolf|cave_spider|drowned|husk|stray|bogged|breeze/i.test(s.focusedEntity);
  const lowHp = s.health <= 8;
  const veryLowHp = s.health <= 5;
  const edgeRisk = s.fallDistance > 2.2 || (!s.onGround && s.verticalSpeed < -0.4);
  const hasClutchUtility = s.hasWaterBucket || s.hasBlocks;
  const enemyNearby = s.nearestEnemyDistance > 0 && s.nearestEnemyDistance < 8;
  const enemyWithinResourceAggro = s.nearestEnemyDistance > 0 && s.nearestEnemyDistance < resourceAggressionRadius;
  const enemyVeryClose = s.nearestEnemyDistance > 0 && s.nearestEnemyDistance < 3;
  const severeDanger = s.health <= 3 || (enemyNearby && s.health <= 5 && s.nearestEnemyDistance < 3);
  const enemyGearAdvantage = s.nearestEnemyArmorPieces >= 3 && !s.hasMeleeWeapon;
  const enemyUsingCrystals = /end_crystal|respawn_anchor|glowstone/.test(s.nearestEnemyMainItem || '');
  const crystalSpamThreat = enemyNearby && (enemyUsingCrystals || (mode === 'crystal' && s.health <= 13));
  const canSafeAnchor = s.respawnAnchorSlot >= 0 && s.glowstoneSlot >= 0 && s.respawnAnchorCount > 0 && s.glowstoneCount > 0;
  const inventoryValue = s.ironCount + (s.redstoneCount * 0.2) + (s.goldCount * 2) + (s.diamondCount * 4) + (s.emeraldCount * 5);
  const hasCrystalKit = s.obsidianCount > 0 && s.endCrystalCount > 0;
  const wantsNorth = /\bnorth\b/.test(text);
  const wantsSouth = /\bsouth\b/.test(text);
  const wantsEast = /\beast\b/.test(text);
  const wantsWest = /\bwest\b/.test(text);
  const wantsTellyBridge = /\b(telly|telly ?bridge|tele ?bridge)\b/.test(text);
  const wantsAndromedaBridge = /\b(andromeda|andromeda ?bridge)\b/.test(text);
  const wantsLongWaterTravel = /\b(cross|travel|go|reach|sail|boat)\b.*\b(water|ocean|sea|river|lake)\b|\b(ocean|sea|river|lake)\b.*\b(cross|travel|go|reach|sail|boat)\b/.test(text);
  const hasBoat = s.boatSlot >= 0 && s.boatCount > 0;
  const wantsMobBlockDefense = /\b(block mobs|block mob|anti ?mob|wall off mobs|box up|mobs? from hitting)\b/.test(text);
  const hostilePressure = s.nearestHostileDistance > 0 && s.nearestHostileDistance < 3.5;
  const shouldBlockMobs = s.hasBlocks && hostilePressure && (mode === 'build' || mode === 'bedwars' || wantsMobBlockDefense);

  // Pearl cooldown & mace tactics
  const pearlCooldownTicks = 100; // ~5 seconds at 20 TPS, prevent spam
  const currentTick = pulse;
  const lowHpEatUntilTick = Number(sessionCtx.lowHpEatUntilTick || -1);
  const lowHpEatLockActive = s.utilityFoodSlot >= 0 && lowHpEatUntilTick >= currentTick;
  const pearlUsedRecently = s.lastPearlUseTick >= 0 && (currentTick - s.lastPearlUseTick) < pearlCooldownTicks;
  const canUsePearl = s.pearlSlot >= 0 && s.pearlCount > 0 && !pearlUsedRecently;
  const lastHealth = Number(sessionCtx.lastHealth);
  const tookRecentDamage = Number.isFinite(lastHealth) && lastHealth > 0 && s.health > 0 && (lastHealth - s.health) >= 0.5;
  const lastDamageTick = Number(sessionCtx.lastDamageTick || -9999);
  const damageAggroActive = tookRecentDamage || ((currentTick - lastDamageTick) <= 14);
  const retaliationTargetName = normalizeText(sessionCtx.retaliationTargetName || '');
  const retaliationUntilTick = Number(sessionCtx.retaliationUntilTick || -1);
  const retaliationActive = retaliationTargetName.length > 0
    && retaliationUntilTick >= currentTick
    && s.nearestEnemyDistance > 0
    && s.nearestEnemyDistance < 24
    && normalizeText(s.nearestEnemyName || '').toLowerCase() === retaliationTargetName.toLowerCase();
  const potCooldownTicks = 36;
  const lastPotTick = Number(sessionCtx.lastPotTick || -9999);
  const potionUsedRecently = (currentTick - lastPotTick) < potCooldownTicks;
  const hasMace = s.maceSlot >= 0 && s.maceCount > 0;
  const hasBreachMace = s.breachMaceSlot >= 0 && s.maceBreachLevel > 0 && s.maceCount > 0;
  const hasMaceAndElytra = hasMace && s.hasElytra;
  const hasWindCharge = s.windChargeSlot >= 0 && s.windChargeCount > 0;
  const hasCrystalCombatKit = s.obsidianSlot >= 0 && s.endCrystalSlot >= 0 && s.obsidianCount > 0 && s.endCrystalCount > 0;
  const hasRangedBow = s.bowSlot >= 0;
  const enemyHasShield = s.nearestEnemyHasShield && enemyNearby;
  const breachSwapNeeded = enemyHasShield && hasBreachMace && s.swordSlot >= 0;
  const missingCombatEffects = !s.hasSpeedEffect || !s.hasStrengthEffect;
  const canPotUp = s.combatPotionSlot >= 0 && s.combatPotionCount > 0 && missingCombatEffects;

  const needsTotem = s.totemCount < 2;
  const needsPearls = s.pearlCount < 2;
  const needsPotions = s.combatPotionCount < 1;
  const needsCrystalKitLoot = (mode === 'crystal') && (s.endCrystalCount < 8 || s.obsidianCount < 8);
  const nearbyNeededLoot =
    ((needsTotem ? s.nearbyDroppedTotemCount : 0)
    + (needsPearls ? s.nearbyDroppedPearlCount : 0)
    + (needsPotions ? s.nearbyDroppedPotionCount : 0)
    + (needsCrystalKitLoot ? s.nearbyDroppedCrystalCount : 0)
    + s.nearbyDroppedGappleCount);
  const shouldLootNow = nearbyNeededLoot > 0 && s.nearestDroppedItemDistance > 0 && s.nearestDroppedItemDistance < 12 && !enemyVeryClose && !severeDanger;
  const valuableDroppedLootCount =
    s.nearbyDroppedTotemCount
    + s.nearbyDroppedPearlCount
    + s.nearbyDroppedPotionCount
    + s.nearbyDroppedGappleCount
    + s.nearbyDroppedCrystalCount;
  const shouldForceValuableLoot = valuableDroppedLootCount > 0
    && s.nearestDroppedItemDistance > 0
    && s.nearestDroppedItemDistance < 10
    && !edgeRisk
    && s.health > 4;

  const lootCross = (s.lookX * s.nearestDroppedItemDz) - (s.lookZ * s.nearestDroppedItemDx);
  const lootYaw = clamp(Math.round(lootCross * 6), -7, 7);

  if (enemyNearby) {
    const lateralLead = (s.lookX * s.nearestEnemyVelZ) - (s.lookZ * s.nearestEnemyVelX);
    const predictiveYaw = clamp(Math.round(lateralLead * 35), -9, 9);
    if (predictiveYaw !== 0) {
      action.yawDelta = predictiveYaw;
      noteParts.push('Enemy movement prediction active.');
    }
  }

  if (wantsNorth && s.facing !== 'north') action.yawDelta = 6;
  if (wantsSouth && s.facing !== 'south') action.yawDelta = 6;
  if (wantsEast && s.facing !== 'east') action.yawDelta = 6;
  if (wantsWest && s.facing !== 'west') action.yawDelta = 6;

  if (/\b(stop|halt|pause|wait|stand still|freeze)\b/.test(text)) {
    noteParts.push('Holding still.');
    action.durationTicks = 6;
    return { action, note: noteParts.join(' '), mode: reportedMode };
  }

  if (retaliationActive && !severeDanger && s.hasMeleeWeapon) {
    const retaliationDist = s.nearestEnemyDistance;
    const targetYaw = Math.atan2(-s.nearestEnemyDx, s.nearestEnemyDz) * (180 / Math.PI);
    let retaliationRawYaw = targetYaw - s.yaw;
    while (retaliationRawYaw > 180) retaliationRawYaw -= 360;
    while (retaliationRawYaw <= -180) retaliationRawYaw += 360;
    const retaliationYawDelta = clamp(retaliationRawYaw / 4, -8, 8);
    const retaliationAligned = hasEnemyInCrosshair || Math.abs(retaliationRawYaw) < 24;
    const retaliationAttackWindow = retaliationDist > 0 && retaliationDist <= 3.35 && retaliationAligned;

    action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
    action.attack = retaliationAttackWindow;
    action.use = false;
    action.forward = retaliationDist > 2.6;
    action.back = retaliationDist < 1.5;
    action.left = retaliationDist >= 2.0 && retaliationDist <= 3.2 && (pulse % 6 < 3);
    action.right = retaliationDist >= 2.0 && retaliationDist <= 3.2 && (pulse % 6 >= 3);
    action.sprint = retaliationDist > 3.0 && Math.abs(retaliationRawYaw) < 70;
    action.sneak = false;
    action.jump = false;
    action.yawDelta = retaliationYawDelta;
    action.pitchDelta = clamp((-s.pitch) / 4, -5, 5);
    action.durationTicks = 4;
    noteParts.push(`Retaliation lock: counter-attacking ${retaliationTargetName} until threat is eliminated.`);
  }

  if (mode === 'clutch' || edgeRisk) {
    action.sneak = true;
    action.back = true;
    action.use = hasClutchUtility;
    action.hotbarSlot = s.hasWaterBucket ? s.waterBucketSlot : s.blockSlot;
    action.pitchDelta = s.verticalSpeed < -0.2 ? 10 : 3;
    action.durationTicks = 4;
    noteParts.push(hasClutchUtility
      ? 'Clutch safety: reduce fall risk and place utility.'
      : 'Clutch safety: no clutch utility in hotbar, prioritizing back/sneak recovery.');
  }

  if (mode === 'bedwars') {
    const wantsExplosiveTravel = /\b(fireball ?jump|tnt ?jump|cross|other side|fire ?charge)\b/.test(text);
    const hasFireball = s.fireballSlot >= 0 && s.fireballCount > 0;
    const hasTnt = s.tntSlot >= 0 && s.tntCount > 0;
    const canFireballJump = hasFireball && wantsExplosiveTravel && !enemyVeryClose;
    const canTntJump = hasTnt && wantsExplosiveTravel && !hasFireball && !enemyVeryClose;
    const hasBedwarsTarget = s.nearestEnemyDistance > 0 && s.nearestEnemyDistance < 20;
    let bedwarsRawYaw = 0;
    let bedwarsAimYawDelta = 0;
    if (hasBedwarsTarget && (Math.abs(s.nearestEnemyDx) > 0.05 || Math.abs(s.nearestEnemyDz) > 0.05)) {
      const targetYaw = Math.atan2(-s.nearestEnemyDx, s.nearestEnemyDz) * (180 / Math.PI);
      bedwarsRawYaw = targetYaw - s.yaw;
      while (bedwarsRawYaw > 180) bedwarsRawYaw -= 360;
      while (bedwarsRawYaw <= -180) bedwarsRawYaw += 360;
      bedwarsAimYawDelta = clamp(bedwarsRawYaw / 4, -7, 7);
    }

    if (bedObjectiveActive && bedObjectiveCompleted) {
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      action.sprint = false;
      action.sneak = true;
      action.use = false;
      action.attack = false;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      action.durationTicks = 8;
      noteParts.push('Bed objective complete: bed destroyed. Holding and awaiting next objective.');
    } else if (hasEnemyInCrosshair && s.focusedDistance > 0 && s.focusedDistance < 3.4) {
      action.attack = true;
      action.sprint = true;
      action.forward = true;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      action.yawDelta = bedwarsAimYawDelta;
      action.pitchDelta = clamp((-s.pitch) / 3, -6, 6);
      action.durationTicks = 6;
      noteParts.push('Bedwars combat engage.');
    } else if (bedObjectiveActive && s.bedNearby && s.nearestBedDistance >= 0 && s.nearestBedDistance < 6.5) {
      const breakToolSlot = pickBestMiningToolSlot(s, s.nearestBedDefenseBlock, text);
      action.hotbarSlot = breakToolSlot;
      action.attack = true;
      action.use = false;
      action.sneak = s.nearestBedDistance < 2.5;
      action.sprint = false;
      action.forward = s.nearestBedDistance > 2.2;
      action.back = s.nearestBedDistance < 1.4;
      action.left = false;
      action.right = false;
      action.pitchDelta = clamp((35 - s.pitch) / 3, -5, 5);
      action.yawDelta = bedwarsAimYawDelta;
      action.durationTicks = 5;
      noteParts.push(`Bed break phase: using best tool for ${s.nearestBedDefenseBlock || 'defense block'}.`);
    } else if (canFireballJump) {
      // Fire charge jump: aim down ~55-70°, right-click to throw charge at feet, jump simultaneously
      action.sprint = true;
      action.forward = true;
      action.hotbarSlot = s.fireballSlot;
      if (!s.onGround && s.verticalSpeed > 0.15) {
        // Rising from explosion — coast forward, start looking ahead again
        action.pitchDelta = s.pitch > 20 ? -10 : 0;
        action.durationTicks = 5;
        noteParts.push('Fireball jump: airborne, sprinting to target.');
      } else if (s.pitch < 55) {
        // Aim down toward feet first
        action.pitchDelta = 14;
        action.durationTicks = 3;
        noteParts.push('Fireball jump: aiming down...');
      } else {
        // Looking ~55°+ down — throw the fire charge and jump at the same time
        action.use = true;
        action.jump = true;
        action.pitchDelta = -12;
        action.durationTicks = 4;
        noteParts.push('Fireball jump: throwing charge and jumping!');
      }
    } else if (canTntJump) {
      // TNT jump: aim straight down (~75-90°), place TNT, jump before it blows
      action.sprint = true;
      action.forward = true;
      action.hotbarSlot = s.tntSlot;
      if (!s.onGround && s.verticalSpeed > 0.15) {
        action.pitchDelta = s.pitch > 20 ? -10 : 0;
        action.durationTicks = 5;
        noteParts.push('TNT jump: airborne, sprinting to target.');
      } else if (s.pitch < 75) {
        action.pitchDelta = 18;
        action.durationTicks = 3;
        noteParts.push('TNT jump: aiming straight down...');
      } else {
        // Aimed straight down — place TNT and jump immediately
        action.use = true;
        action.jump = true;
        action.pitchDelta = -12;
        action.durationTicks = 3;
        noteParts.push('TNT jump: placing and jumping!');
      }
    } else {
      action.forward = true;
      action.sprint = !lowHp;
      action.use = s.hasBlocks && (/\b(bridge|bed|rush|defend|place|block in)\b/.test(text) || s.horizontalSpeed < 0.1);
      action.sneak = action.use || (!s.hasBlocks && /\b(bridge|rush)\b/.test(text));
      action.hotbarSlot = action.use ? s.blockSlot : (s.swordSlot >= 0 ? s.swordSlot : s.axeSlot);
      action.durationTicks = 8;
      noteParts.push(s.hasBlocks
        ? `Bedwars objective push: route and place blocks (${s.hotbarBlocks} blocks hotbar).`
        : 'Bedwars objective push: no blocks in hotbar, moving to regroup/loot.');

      if (s.bedNearby && s.nearestBedDistance >= 0 && s.nearestBedDistance < 10) {
        if (s.nearestBedDefenseScore >= 30) {
          noteParts.push(`Bed defense appears heavy (${s.nearestBedDefenseBlock || 'mixed'}). Circling before commit.`);
          action.left = (pulse % 2) === 0;
          action.right = !action.left;
          action.sprint = false;
        } else {
          noteParts.push('Bed defense appears light. Faster bed break path.');
          action.forward = true;
          action.sprint = true;
        }
      }
    }

    if (!hasBedwarsTarget && !wantsExplosiveTravel && pulse % 3 === 0) {
      action.yawDelta = (pulse % 2 === 0) ? 4 : -4;
    }
  }

  if (mode === 'pvp') {
    // Strafe direction changes every 3 decisions (not every 1) to avoid rapid spin side-effects
    const strafePhase = Math.floor(pulse / 3);
    const strafeLeft = (strafePhase % 2) === 0;

    const pvpDuration = 4; // slightly slower updates to reduce overshoot and path jitter

    // ── Pick best target (player first, then hostile mob) ────────────────────
    const playerTargetMatches = s.nearestEnemyDistance > 0
      && s.nearestEnemyDistance < 24
      && (!wantsSpecificCombatTarget
        || entityMatchesTargetPreference(requestedCombatTarget, s.nearestEnemyName)
        || (wantsPlayerCombatTarget && /player|enemy|opponent/.test(requestedCombatTarget)));
    const mobTargetMatches = s.nearestHostileDistance > 0
      && s.nearestHostileDistance < 20
      && (!wantsSpecificCombatTarget
        || entityMatchesTargetPreference(requestedCombatTarget, s.nearestHostile, s.focusedEntity));
    const usePlayerTarget = playerTargetMatches && (wantsPlayerCombatTarget || !mobTargetMatches);
    const hasMobTarget = mobTargetMatches;
    const hasPlayerTarget = usePlayerTarget;
    const targetDx   = hasPlayerTarget ? s.nearestEnemyDx   : (hasMobTarget ? s.nearestHostileDx   : 0);
    const targetDz   = hasPlayerTarget ? s.nearestEnemyDz   : (hasMobTarget ? s.nearestHostileDz   : 0);
    const targetDist = hasPlayerTarget ? s.nearestEnemyDistance : (hasMobTarget ? s.nearestHostileDistance : -1);
    const hasTarget  = targetDist > 0;
    const targetName = hasPlayerTarget ? (s.nearestEnemyName || 'player') : (s.nearestHostile || s.focusedEntity || 'target');
    const windMaceComboStage = Number(sessionCtx.windMaceComboStage || 0);

    // ── Compute yaw correction via atan2 (stable lock-on) ────────────────────
    // MC yaw: 0=south(+Z), 90=west(−X), −90=east(+X), 180=north(−Z)
    // CRITICAL FIX: yawDelta is applied EVERY tick for durationTicks ticks.
    // So yawDelta must be (totalCorrectionNeeded / pvpDuration) to avoid turning too far.
    let rawYaw = 0; // total angle error to target (signed degrees)
    let aimYawDelta = 0; // per-tick yaw correction
    if (hasTarget && (Math.abs(targetDx) > 0.05 || Math.abs(targetDz) > 0.05)) {
      const targetYaw = Math.atan2(-targetDx, targetDz) * (180 / Math.PI);
      rawYaw = targetYaw - s.yaw;
      while (rawYaw > 180) rawYaw -= 360;
      while (rawYaw <= -180) rawYaw += 360;
      // Divide by pvpDuration so we turn rawYaw total, not rawYaw × pvpDuration
      aimYawDelta = clamp(rawYaw / pvpDuration, -8, 8);
    }
    // Use rawYaw (total error) for alignment checks, not per-tick aimYawDelta
    const isAligned = hasEnemyInCrosshair || (hasTarget && Math.abs(rawYaw) < 22);
    const canStrafe = hasTarget && Math.abs(rawYaw) < 12 && targetDist >= 2.0 && targetDist < 3.5;
    const isFacing  = hasTarget && Math.abs(rawYaw) < 60; // broad facing check for movement
    const attackWindow = hasTarget && isAligned && targetDist <= 3.35;
    const holdGroundRange = hasTarget && targetDist >= 3.0 && targetDist <= 3.7;
    // User-configured aggression profile: keep fighting instead of low-HP retreating.
    const shouldRetreat = false;
    const enemyHoldingSword = /_sword|trident/.test(s.nearestEnemyMainItem || '');
    const comboPressure = hasTarget && targetDist < 3.1 && (tookRecentDamage || damageAggroActive);
    const preferredKiteRange = hasTarget && targetDist >= 3.1 && targetDist <= 4.1;
    const shouldCloseGap = hasTarget && targetDist > 4.1 && (isFacing || damageAggroActive);
    const pressureCycle = pulse % 10;
    const burstWindow = pressureCycle < 9; // ~90% combo pressure window
    const shouldBurstIn = hasTarget && !shouldRetreat && burstWindow && isFacing && (targetDist > 2.4 && targetDist <= 4.8);
    const shouldStepOut = hasTarget && !shouldRetreat && (comboPressure || pressureCycle >= 4) && targetDist < 3.0;
    const retaliateWindow = damageAggroActive && hasEnemyInCrosshair && s.focusedDistance > 0 && s.focusedDistance <= 3.35;
    const shouldComboPush = hasTarget && !shouldRetreat && isAligned && burstWindow && targetDist >= 2.6 && targetDist <= 3.4;

    action.hotbarSlot  = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
    action.attack      = (attackWindow || retaliateWindow || shouldComboPush) && !shouldRetreat;
    action.forward     = (shouldCloseGap || shouldBurstIn || shouldComboPush) && !shouldRetreat;
    action.back        = shouldRetreat || shouldStepOut || (hasTarget && !shouldBurstIn && !preferredKiteRange && targetDist < 3.1);
    action.sprint      = hasTarget && (targetDist > 4.0 || shouldBurstIn || shouldComboPush) && isFacing && !lowHp && !shouldRetreat;
    // Strafe: only while NOT strongly turning (use rawYaw gate) and within engagement range
    action.left        = strafeLeft  && (canStrafe || preferredKiteRange) && !shouldRetreat;
    action.right       = !strafeLeft && (canStrafe || preferredKiteRange) && !shouldRetreat;
    action.jump        = false;
    // When scanning with no target, rotate slowly; when targeting, apply per-tick correction
    action.yawDelta    = hasTarget ? aimYawDelta : 4;
    action.pitchDelta  = hasTarget ? clamp((-s.pitch) / pvpDuration, -5, 5) : 0;
    action.durationTicks = pvpDuration;

    if (holdGroundRange && !shouldRetreat && !shouldBurstIn) {
      action.forward = false;
      action.back = false;
      action.sprint = false;
    }

    if ((attackWindow || shouldComboPush) && !shouldRetreat) {
      // Stay close enough to continue landing hits, then micro-disengage briefly.
      action.forward = targetDist > 2.55;
      action.back = false;
      action.sprint = targetDist > 3.0;
      if (pulse % 6 >= 4 && targetDist < 2.8) {
        action.back = true;
        action.forward = false;
        action.sprint = false;
      }
    }

    noteParts.push(hasTarget
      ? `PVP: lock ${targetName} dist=${targetDist.toFixed(1)} err=${rawYaw.toFixed(0)}° aligned=${isAligned} kite=${preferredKiteRange} burst=${shouldBurstIn}`
      : 'PVP: scanning for target.');

    // Low-HP retreat removed by request: stay in fight and rely on timed eating.

    // ── Bow: shoot enemy when far away (> 6 blocks) ───────────────────────────
    const hasBow = s.bowSlot >= 0;
    const bowRangeFight = hasTarget && targetDist > 6 && !shouldRetreat;
    if (bowRangeFight && hasBow && isAligned) {
      action.hotbarSlot = s.bowSlot;
      action.forward    = targetDist > 14; // keep closing gap if very far
      action.sprint     = targetDist > 16;
      action.back       = false;
      action.left       = false;
      action.right      = false;
      // Draw-and-fire cycle: hold use for 6 decisions (6×3 ticks = 18 ticks ~= full charge at 20tps), then release
      if (pulse % 8 < 6) {
        action.use    = true;   // draw bow
        action.attack = false;
      } else {
        action.use    = false;  // release → fires arrow
        action.attack = false;
      }
      // Slight upward pitch compensation for arrow drop at distance
      const vertLead = Math.atan2(targetDist * 0.04, 1) * (180 / Math.PI);
      action.pitchDelta = clamp((-vertLead - s.pitch) / pvpDuration, -6, 6);
      noteParts.push(`Bow: shooting ${targetName} at ${targetDist.toFixed(1)} blocks. ${pulse % 8 < 6 ? 'Drawing...' : 'Releasing!'}`);
    }

    // ── Shield: raise shield when enemy approaches for a melee strike ────────
    const hasShield = s.shieldSlot >= 0;
    const enemyAboutToStrike = hasTarget && targetDist < 2.5 && s.nearestEnemyHasMeleeWeapon;
    const shouldShieldPressure = hasShield
      && !shouldRetreat
      && !bowRangeFight
      && hasTarget
      && targetDist < 3.2
      && (enemyAboutToStrike || enemyHoldingSword || comboPressure)
      && (pulse % 5 >= 2);
    // Block every other strafe window to mix offense and defense
    if (shouldShieldPressure) {
      // Shield blocks with right-click (use key) while it is in the offhand (placed there by Java)
      action.use    = true;
      action.sneak  = false;
      action.attack = false;
      action.forward = false;
      action.back = targetDist < 2.7;
      noteParts.push(enemyHoldingSword
        ? 'Shield: blocking sword pressure while resetting spacing.'
        : 'Shield: blocking combo pressure and resetting spacing.');
    }

    if (!s.hasMeleeWeapon && s.focusedDistance > 0 && s.focusedDistance < 2.4) {
      action.back = true;
      action.forward = false;
      action.sprint = false;
      noteParts.push('PVP fallback: no melee weapon in hotbar, spacing to reduce damage.');
    }

    if (enemyGearAdvantage && !hasEnemyInCrosshair) {
      action.back = true;
      action.forward = false;
      action.sprint = false;
      noteParts.push('Enemy gear advantage detected. Disengaging to safer angle.');
    }

    if (crystalSpamThreat) {
      if (canUsePearl && severeDanger) {
        action.attack = false;
        action.use = true;
        action.hotbarSlot = s.pearlSlot;
        action.back = true;
        action.forward = false;
        action.left = false;
        action.right = false;
        action.sprint = false;
        action.sneak = false;
        action.durationTicks = 4;
        noteParts.push('Severe danger + crystal spam: pearling away to escape.');
      } else if (canSafeAnchor) {
        action.attack = false;
        action.use = true;
        action.hotbarSlot = (pulse % 2 === 0) ? s.respawnAnchorSlot : s.glowstoneSlot;
        action.back = true;
        action.forward = false;
        action.left = false;
        action.right = false;
        action.sprint = false;
        action.sneak = false;
        action.durationTicks = 4;
        noteParts.push('Crystal spam detected: safe-anchor defensive cycle.');
      } else if (pearlUsedRecently) {
        noteParts.push('Crystal spam: pearl in cooldown, prioritizing survival.');
      }
    }


    const canAxeShieldBreak = enemyHasShield
      && s.axeSlot >= 0
      && hasTarget
      && targetDist > 0
      && targetDist <= 3.8
      && isAligned
      && !shouldRetreat;

    const canCartPvp = hasTarget
      && !shouldRetreat
      && s.railSlot >= 0
      && s.railCount > 0
      && s.tntMinecartSlot >= 0
      && s.tntMinecartCount > 0
      && s.flintAndSteelSlot >= 0
      && s.flintAndSteelCount > 0
      && targetDist > 1.8
      && targetDist <= 5.0
      && isAligned;

    const canMaceLeap = hasTarget
      && hasMace
      && !shouldRetreat
      && targetDist > 3.0
      && targetDist <= 5.3
      && isFacing
      && !hasWindCharge;

    const canMaceBaitStrafe = hasTarget
      && hasMace
      && !shouldRetreat
      && targetDist >= 2.4
      && targetDist <= 3.6
      && isAligned
      && !enemyHasShield;

    if (canAxeShieldBreak) {
      action.hotbarSlot = s.axeSlot;
      action.attack = true;
      action.forward = targetDist > 2.3;
      action.back = false;
      action.sprint = targetDist > 2.7;
      action.use = false;
      noteParts.push('Shield counter: switching to axe to disable enemy shield.');
    } else if (canCartPvp) {
      const cartPhase = pulse % 6;
      action.attack = false;
      action.forward = false;
      action.back = (cartPhase === 4 || cartPhase === 5);
      action.left = false;
      action.right = false;
      action.sprint = false;
      action.jump = false;
      action.durationTicks = 4;
      action.pitchDelta = clamp((68 - s.pitch) / 2, -8, 8);
      if (cartPhase <= 1) {
        action.use = true;
        action.hotbarSlot = s.railSlot;
        noteParts.push('Cart PvP: placing rail under pressure line.');
      } else if (cartPhase <= 3) {
        action.use = true;
        action.hotbarSlot = s.tntMinecartSlot;
        noteParts.push('Cart PvP: placing TNT minecart on rail.');
      } else {
        action.use = true;
        action.hotbarSlot = s.flintAndSteelSlot;
        noteParts.push('Cart PvP: igniting minecart and backing out.');
      }
    } else if (canMaceLeap && pulse % 7 < 3) {
      action.hotbarSlot = hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot;
      action.attack = false;
      action.forward = true;
      action.back = false;
      action.left = false;
      action.right = false;
      action.sprint = true;
      action.jump = s.onGround;
      action.use = false;
      action.pitchDelta = clamp((-12 - s.pitch) / pvpDuration, -8, 8);
      action.durationTicks = 4;
      noteParts.push('Mace strategy: leap engage for burst impact window.');
    } else if (canMaceBaitStrafe && pulse % 6 >= 3) {
      action.hotbarSlot = hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot;
      action.attack = (pulse % 2) === 0;
      action.forward = false;
      action.back = false;
      action.left = (pulse % 4) < 2;
      action.right = !action.left;
      action.sprint = false;
      action.use = false;
      action.durationTicks = 4;
      noteParts.push('Mace strategy: bait strafe timing before swing.');
    } else if (breachSwapNeeded && (pulse % 8 === 0)) {
      // Mace breach swap: if enemy has shield and we have breach mace, swap to mace periodically
      action.hotbarSlot = s.breachMaceSlot;
      action.attack = true;
      noteParts.push(`Mace breach swap: Breach ${s.maceBreachLevel} shield break.`);
    } else if (breachSwapNeeded && pulse % 8 === 4) {
      // Swap back to sword for damage
      action.hotbarSlot = s.swordSlot;
      action.attack = true;
      noteParts.push('Attribute swap: back to sword for enhanced damage.');
    }

    // Elytra + mace aerial combat
    if (hasMaceAndElytra && !s.onGround && s.health > 6) {
      action.attack = true;
      action.hotbarSlot = s.maceSlot;
      action.forward = true;
      action.durationTicks = 6;
      noteParts.push('Elytra-mace aerial dive: breaching with height advantage.');
    }

    noteParts.push('PVP mode: strafing, spacing, and timed attacks.');

    if (/\b(web|cobweb|web trap)\b/.test(text) && s.cobwebSlot >= 0 && s.cobwebCount > 0 && enemyVeryClose) {
      action.use = true;
      action.hotbarSlot = s.cobwebSlot;
      action.sneak = false;
      action.durationTicks = 4;
      noteParts.push('Web trap attempt active.');
    }

    // Auto-web in close lock if we have cobwebs, even without explicit prompt.
    if (hasTarget && targetDist <= 2.2 && isAligned && s.cobwebSlot >= 0 && s.cobwebCount > 0 && !shouldRetreat && pulse % 9 === 0) {
      action.attack = false;
      action.use = true;
      action.hotbarSlot = s.cobwebSlot;
      action.sneak = false;
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      action.durationTicks = 4;
      noteParts.push('PVP strategy: auto-web trap placed.');
    }

    if (/\b(crit|critical|crit out)\b/.test(text)) {
      action.jump = true;
      action.attack = true;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      noteParts.push('Critical-hit jump timing active.');
    }

    // Auto-crit cadence while in melee range.
    if (hasTarget && targetDist > 2.0 && targetDist < 3.3 && isAligned && !shouldRetreat && pulse % 6 === 0) {
      action.jump = true;
      action.attack = true;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      noteParts.push('PVP strategy: crit jump cadence.');
    }

    const canSafePotInFight = canPotUp
      && !potionUsedRecently
      && !enemyVeryClose
      && !severeDanger
      && !shouldLootNow
      && (!hasTarget || targetDist > 4.0 || (targetDist > 3.1 && Math.abs(rawYaw) > 28));
    if (canSafePotInFight) {
      action.attack = false;
      action.use = true;
      action.sneak = false;
      action.sprint = false;
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      action.hotbarSlot = s.combatPotionSlot;
      action.durationTicks = 4;
      noteParts.push('PVP upkeep: potting for missing combat effects.');
    } else if (shouldLootNow && !severeDanger) {
      action.attack = false;
      action.use = false;
      action.forward = true;
      action.sprint = true;
      action.back = false;
      action.left = false;
      action.right = false;
      action.jump = (pulse % 9 === 0);
      action.yawDelta = lootYaw;
      action.durationTicks = 6;
      noteParts.push('PVP loot pickup: collecting needed dropped items.');
    }

    const canWindMaceCombo = hasTarget
      && hasMace
      && hasWindCharge
      && !shouldRetreat
      && targetDist >= 1.8
      && targetDist <= 6.2;
    if (canWindMaceCombo && windMaceComboStage === 1) {
      action.use = false;
      action.attack = false;
      action.hotbarSlot = hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot;
      action.forward = true;
      action.back = false;
      action.sprint = true;
      action.left = false;
      action.right = false;
      action.jump = false;
      action.pitchDelta = clamp((-14 - s.pitch) / pvpDuration, -8, 8);
      action.durationTicks = 4;
      noteParts.push('Wind-mace combo: reacquiring target after self-launch.');
    } else if (canWindMaceCombo && windMaceComboStage === 2) {
      action.use = false;
      action.attack = true;
      action.hotbarSlot = hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot;
      action.forward = targetDist > 2.1;
      action.back = false;
      action.sprint = true;
      action.left = false;
      action.right = false;
      action.jump = false;
      action.durationTicks = 4;
      noteParts.push('Wind-mace combo: mace strike phase.');
    } else if (canWindMaceCombo && windMaceComboStage === 0 && s.onGround && (pulse % 8 === 2)) {
      action.use = true;
      action.attack = false;
      action.hotbarSlot = s.windChargeSlot;
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      action.sprint = false;
      action.jump = true;
      action.pitchDelta = clamp((87 - s.pitch) / 2, -12, 12);
      action.durationTicks = 4;
      noteParts.push('Wind-mace combo: looking down and firing windcharge under self.');
    }
  }

  if (mode === 'crystal') {
    action.sneak = true;
    action.sprint = false;
    action.durationTicks = 5;

    // Flat-ground CPvP tuning: stabilize movement and keep focus on target.
    const crystalPlayerTarget = s.nearestEnemyDistance > 0
      && s.nearestEnemyDistance < 24
      && (!wantsSpecificCombatTarget
        || entityMatchesTargetPreference(requestedCombatTarget, s.nearestEnemyName)
        || (wantsPlayerCombatTarget && /player|enemy|opponent/.test(requestedCombatTarget)));
    const crystalMobTarget = s.nearestHostileDistance > 0
      && s.nearestHostileDistance < 20
      && (!wantsSpecificCombatTarget
        || entityMatchesTargetPreference(requestedCombatTarget, s.nearestHostile, s.focusedEntity));
    const crystalUsePlayerTarget = crystalPlayerTarget && (wantsPlayerCombatTarget || !crystalMobTarget);
    const crystalHasTarget = crystalUsePlayerTarget || crystalMobTarget;
    const crystalDist = crystalUsePlayerTarget ? s.nearestEnemyDistance : (crystalMobTarget ? s.nearestHostileDistance : -1);
    let crystalRawYaw = 0;
    let crystalYawDelta = 0;
    const crystalTargetDx = crystalUsePlayerTarget ? s.nearestEnemyDx : (crystalMobTarget ? s.nearestHostileDx : 0);
    const crystalTargetDz = crystalUsePlayerTarget ? s.nearestEnemyDz : (crystalMobTarget ? s.nearestHostileDz : 0);
    if (crystalHasTarget && (Math.abs(crystalTargetDx) > 0.05 || Math.abs(crystalTargetDz) > 0.05)) {
      const targetYaw = Math.atan2(-crystalTargetDx, crystalTargetDz) * (180 / Math.PI);
      crystalRawYaw = targetYaw - s.yaw;
      while (crystalRawYaw > 180) crystalRawYaw -= 360;
      while (crystalRawYaw <= -180) crystalRawYaw += 360;
      crystalYawDelta = clamp(crystalRawYaw / action.durationTicks, -7, 7);
    }
    const crystalAligned = crystalHasTarget && Math.abs(crystalRawYaw) < 18;
    const crystalHoldRange = crystalHasTarget && crystalDist >= 2.4 && crystalDist <= 4.6;
    const crystalNeedClose = crystalHasTarget && crystalDist > 4.4 && Math.abs(crystalRawYaw) < 65;
    const crystalTooClose = crystalHasTarget && crystalDist < 1.9;

    action.yawDelta = crystalHasTarget ? crystalYawDelta : 3;
    action.pitchDelta = crystalHasTarget ? clamp((-s.pitch) / action.durationTicks, -5, 5) : 0;
    action.forward = crystalNeedClose;
    action.back = crystalTooClose;
    action.left = crystalHasTarget && crystalAligned && crystalDist >= 2.8 && crystalDist < 4.6 && (pulse % 4 < 2);
    action.right = crystalHasTarget && crystalAligned && crystalDist >= 2.8 && crystalDist < 4.6 && (pulse % 4 >= 2);

    if (crystalHoldRange) {
      action.forward = false;
      action.back = false;
      action.sprint = false;
    }

    if (s.totemCount >= 2) {
      noteParts.push('CPvP double totem active: keeping one in offhand with spare in hotbar for fast re-equip.');
    }

    if (crystalSpamThreat) {
      if (canUsePearl && severeDanger) {
        action.hotbarSlot = s.pearlSlot;
        action.use = true;
        action.attack = false;
        action.back = true;
        action.forward = false;
        action.left = false;
        action.right = false;
        noteParts.push('Severe danger + crystal spam: pearl escape.');
      } else if (canSafeAnchor) {
        action.hotbarSlot = (pulse % 2 === 0) ? s.respawnAnchorSlot : s.glowstoneSlot;
        action.use = true;
        action.attack = false;
        action.back = true;
        action.forward = false;
        action.left = false;
        action.right = false;
        noteParts.push('Crystal spam pressure: safe-anchor hold.');
      }
    } else if (s.totemSlot >= 0 && s.health <= 10) {
      action.hotbarSlot = s.totemSlot;
      action.use = true;
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      noteParts.push('Totem safety priority active.');
    } else if (hasCrystalKit && s.obsidianSlot >= 0 && s.endCrystalSlot >= 0) {
      if (pulse % 3 === 0) {
        action.hotbarSlot = s.obsidianSlot;
        action.use = true;
        action.forward = false;
        action.back = false;
        action.left = false;
        action.right = false;
        noteParts.push('Crystal setup: placing obsidian.');
      } else {
        action.hotbarSlot = s.endCrystalSlot;
        action.use = true;
        action.attack = true;
        action.forward = false;
        action.back = false;
        action.left = false;
        action.right = false;
        noteParts.push('Crystal detonation cycle active.');
      }
    } else if (s.respawnAnchorSlot >= 0 && s.glowstoneSlot >= 0 && s.respawnAnchorCount > 0 && s.glowstoneCount > 0) {
      action.hotbarSlot = (pulse % 2 === 0) ? s.respawnAnchorSlot : s.glowstoneSlot;
      action.use = true;
      action.forward = false;
      action.back = false;
      action.left = false;
      action.right = false;
      noteParts.push('Anchor PVP cycle active.');
    } else {
      action.forward = true;
      action.hotbarSlot = s.pearlSlot >= 0 ? s.pearlSlot : s.swordSlot;
      noteParts.push('Crystal kit incomplete: repositioning/looting.');
    }

    const crystallingNow = action.hotbarSlot === s.obsidianSlot || action.hotbarSlot === s.endCrystalSlot;
    const safeAnchoringNow = action.hotbarSlot === s.respawnAnchorSlot || action.hotbarSlot === s.glowstoneSlot;
    if (s.totemSlot >= 0 && s.totemCount >= 2 && !crystallingNow && !safeAnchoringNow) {
      action.hotbarSlot = s.totemSlot;
      if (!action.use) {
        action.attack = false;
      }
      noteParts.push('CPvP double totem: spare totem held in hotbar while not crystal/anchor-cycling.');
    }

    if (enemyNearby && canUsePearl && s.health <= 3) {
      action.hotbarSlot = s.pearlSlot;
      action.use = true;
      action.back = true;
      noteParts.push('Critical danger: emergency pearl escape.');
    }

    if (canPotUp && !enemyVeryClose && !severeDanger && !crystalSpamThreat && pulse % 5 === 0) {
      action.hotbarSlot = s.combatPotionSlot;
      action.use = true;
      action.attack = false;
      action.back = true;
      action.forward = false;
      action.sprint = false;
      action.sneak = true;
      action.durationTicks = 5;
      noteParts.push('Crystal upkeep: potting for missing combat effects.');
    } else if (shouldLootNow && !crystalSpamThreat && !severeDanger) {
      action.attack = false;
      action.use = false;
      action.forward = true;
      action.back = false;
      action.sprint = true;
      action.left = false;
      action.right = false;
      action.yawDelta = lootYaw;
      action.durationTicks = 6;
      noteParts.push('Crystal loot pickup: collecting needed dropped items.');
    }

    noteParts.push(crystalHasTarget
      ? `Crystal lock: dist=${crystalDist.toFixed(1)} err=${crystalRawYaw.toFixed(0)}° hold=${crystalHoldRange}`
      : 'Crystal lock: scanning.');
  }

  if (mode === 'craft') {
    action.forward = true;
    action.sprint = false;
    action.durationTicks = 8;
    action.hotbarSlot = s.pickaxeSlot >= 0 ? s.pickaxeSlot : s.axeSlot;
    noteParts.push('Crafting mode: gathering materials and navigating to stations.');

    if (s.ironCount >= 24 && s.diamondCount < 3) {
      noteParts.push('Priority: iron tools/armor then diamond progression.');
    }
    if (/\b(netherite|upgrade|smithing template)\b/.test(text)) {
      if (s.netheriteUpgradeTemplateCount <= 0) {
        noteParts.push('Netherite path: obtain netherite upgrade smithing template first.');
      } else {
        noteParts.push('Netherite path: combine debris→scrap→ingot, then smithing upgrade.');
      }
    }
  }

  if (mode === 'progression') {
    action.forward = true;
    action.sprint = false;
    action.durationTicks = 8;
    action.hotbarSlot = s.utilityFoodSlot >= 0 ? s.utilityFoodSlot : s.swordSlot;
    noteParts.push('Progression mode: villager-trade and enchant route for Prot IV diamond set.');

    if (enchantGoals.length > 0) {
      noteParts.push(`Enchant targets: ${enchantGoals.join(', ')}.`);
    } else {
      noteParts.push('Enchant targets: mending, prot IV, unbreaking III, sharpness V, efficiency V, looting III, knockback I.');
    }

    if (s.enchantedBookCount < 6) {
      noteParts.push('Book stock low: prioritize librarian rerolls and trade cycle.');
    }
    if (s.diamondCount >= 24 && s.netheriteIngotCount < 4) {
      noteParts.push('Upgrade path: craft full diamond first, then netherite components.');
    }
    if (s.netheriteUpgradeTemplateCount <= 0) {
      noteParts.push('Missing netherite upgrade template. Route to bastion/duped template workflow.');
    }

    if (s.villagerNearbyCount > 0) {
      action.use = true;
      noteParts.push('Villagers nearby: engage trading loop for books.');
    }
  }

  if (mode === 'base') {
    action.forward = true;
    action.sneak = true;
    action.sprint = false;
    action.durationTicks = 9;
    action.hotbarSlot = s.blockSlot;
    action.use = s.hasBlocks;
    noteParts.push('Stealth base mode: underground construction and concealment.');

    if (/\b(duper|orbital strike cannon|cannon)\b/.test(text)) {
      noteParts.push('Cannon build plan: gather redstone/obsidian first, then conceal terrain shell.');
      if (s.redstoneCount < 64) noteParts.push('Need redstone stockpile before cannon core.');
      if (s.obsidianCount < 32) noteParts.push('Need additional obsidian for blast-safe internals.');
      if (s.hasBlocks) noteParts.push('Terrain masking active: place natural-looking cover layers.');
    }

    if (s.villagerNearbyCount > 0) {
      noteParts.push('Villager integration: secure breeder/trading hall inside hidden perimeter.');
    }
  }

  if (mode === 'build') {
    const litematicRequested = /\b(litematic|litematica|schematic)\b/.test(text);
    const materialScanRequested = /\b(material|materials|required|shopping list|requirements)\b/.test(text);
    if (litematicRequested || materialScanRequested) {
      noteParts.push('Litematic workflow active: import schematic externally, place file in litematics/schematics folder, scan required materials, then build once stock is complete.');
      if (!s.hasBlocks || s.hotbarBlocks <= 0) {
        noteParts.push('Build planner: not enough placement blocks in hotbar; gather required materials first.');
      }
    }

    if (wantsLongWaterTravel) {
      if (hasBoat) {
        action.forward = true;
        action.sprint = false;
        action.use = true;
        action.jump = false;
        action.sneak = false;
        action.hotbarSlot = s.boatSlot;
        action.pitchDelta = clamp((-s.pitch) / 2, -8, 8);
        action.durationTicks = 6;
        noteParts.push('Water travel: using boat for long-distance crossing.');
      } else {
        action.use = false;
        action.attack = /log|wood|tree|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry/.test(s.focusedEntity || '');
        action.hotbarSlot = s.axeSlot >= 0 ? s.axeSlot : s.pickaxeSlot;
        action.forward = !action.attack;
        action.sprint = !action.attack;
        action.jump = !action.attack && s.horizontalSpeed < 0.08;
        action.sneak = false;
        action.durationTicks = 6;
        noteParts.push('Water travel prep: gathering wood to craft a boat, then cross water.');
      }
    } else {
      const bridging = /\b(bridge|speedbridge|godbridge|rush)\b/.test(text) || wantsTellyBridge || wantsAndromedaBridge;
      const wantsWoodPrep = /\b(log|logs|wood|plank|planks|collect \d+ logs?)\b/.test(text);
      const treeInSight = /log|wood|tree|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry/.test(s.focusedEntity || '');
      const unsafeEdge = (s.fallDistance > 0.8) || (!s.onGround && s.verticalSpeed < -0.08);
      const bridgeTargetPitch = wantsAndromedaBridge ? 62 : (wantsTellyBridge ? 64 : 58);
      action.use = s.hasBlocks;
      action.hotbarSlot = s.blockSlot;
      action.sneak = bridging || /\b(edge|safe|careful)\b/.test(text);
      action.forward = bridging;
      action.jump = /\b(tower|up|stairs)\b/.test(text) && pulse % 3 === 0;
      action.pitchDelta = bridging ? clamp((bridgeTargetPitch - s.pitch) / 3, -5, 5) : 0;
      action.yawDelta = bridging ? ((pulse % 2 === 0) ? 2 : -2) : 0;
      action.durationTicks = bridging ? 5 : 8;

      if (wantsTellyBridge && s.hasBlocks) {
        action.forward = true;
        action.sprint = false;
        action.use = true;
        action.jump = s.onGround && (pulse % 4 === 0);
        action.sneak = true;
        action.pitchDelta = clamp((64 - s.pitch) / 3, -5, 5);
        action.yawDelta = (pulse % 2 === 0) ? 3 : -3;
        action.durationTicks = 4;
        noteParts.push('Telly bridge routine: sprint-jump placement cadence active.');
      } else if (wantsAndromedaBridge && s.hasBlocks) {
      const strafeRight = (pulse % 6) < 3;
      action.forward = true;
      action.sprint = false;
      action.use = true;
      action.jump = s.onGround && (pulse % 5 === 0);
      action.sneak = true;
      action.left = !strafeRight;
      action.right = strafeRight;
      action.pitchDelta = clamp((62 - s.pitch) / 3, -5, 5);
      action.yawDelta = strafeRight ? 2 : -2;
      action.durationTicks = 5;
        noteParts.push('Andromeda bridge routine: controlled diagonal place rhythm active.');
      } else if (bridging && !s.hasBlocks && wantsWoodPrep) {
      action.use = false;
      action.attack = treeInSight;
      action.hotbarSlot = s.axeSlot >= 0 ? s.axeSlot : s.pickaxeSlot;
      action.forward = !treeInSight;
      action.sprint = !treeInSight;
      action.jump = !treeInSight && s.horizontalSpeed < 0.08;
      action.sneak = false;
      action.yawDelta = treeInSight ? 0 : ((pulse % 2 === 0) ? 6 : -6);
      action.durationTicks = 6;
        noteParts.push('Bridge prep: collecting wood/logs first, then crafting planks for bridging blocks.');
      } else {
        noteParts.push(s.hasBlocks
          ? `Build mode: controlled placement and bridge safety (${s.hotbarBlocks} blocks hotbar).`
          : 'Build mode: no blocks detected, moving to gather resources.');
      }

      if (bridging && unsafeEdge) {
        action.forward = false;
        action.back = true;
        action.sprint = false;
        action.jump = false;
        action.left = false;
        action.right = false;
        action.sneak = true;
        action.use = s.hasBlocks;
        action.pitchDelta = clamp((70 - s.pitch) / 2, -6, 6);
        action.durationTicks = 4;
        noteParts.push('Bridge safety hold: edge/fall risk detected, backing and stabilizing.');
      }
    }
  }

  if (mode === 'speedrun') {
    const inNether = s.dimensionId === 'nether';
    const inEnd = s.dimensionId === 'end';
    const inOverworld = !inNether && !inEnd;
    const hasEnoughEyes = s.eyeOfEnderCount >= 12;
    const hasNetherEntryTool = s.flintAndSteelCount > 0 && s.flintAndSteelSlot >= 0;

    action.durationTicks = 5;
    action.sprint = true;
    action.forward = true;
    action.jump = s.horizontalSpeed < 0.08;

    if (inEnd) {
      action.attack = hasEnemyInCrosshair || /ender_dragon|end_crystal|enderman/.test(s.focusedEntity || '');
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      noteParts.push('Speedrun phase: The End. Fighting dragon, controlling crystal pressure, and closing out run.');
    } else if (inNether) {
      if (s.blazeRodCount < 6) {
        action.attack = hasEnemyInCrosshair || /blaze|wither_skeleton/.test(s.focusedEntity || '') || /blaze/.test(s.nearestHostile || '');
        action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
        noteParts.push('Speedrun phase: Nether rods. Hunting blazes until at least 6 rods secured.');
      } else if (s.pearlCount < 12 && s.eyeOfEnderCount < 12) {
        action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
        noteParts.push('Speedrun phase: Nether pearls route. Need more pearl/eye count before portal-ready.');
      } else {
        action.hotbarSlot = hasNetherEntryTool ? s.flintAndSteelSlot : (s.blockSlot >= 0 ? s.blockSlot : s.swordSlot);
        action.use = hasNetherEntryTool && pulse % 5 === 0;
        noteParts.push('Speedrun phase: Exit Nether. Returning to overworld for triangulation and stronghold travel.');
      }
    } else if (inOverworld) {
      if (s.blazeRodCount <= 0) {
        action.hotbarSlot = s.axeSlot >= 0 ? s.axeSlot : s.pickaxeSlot;
        action.attack = /log|wood|tree/.test(s.focusedEntity || '');
        noteParts.push('Speedrun phase: Early game overworld setup. Routing for tools, food, lava, and nether entry.');
      } else if (!s.strongholdTriangulated) {
        action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.blockSlot;
        noteParts.push('Stronghold calc: throw eye once, run ~150-250 blocks, throw second eye for triangulation.');
      } else {
        const dx = s.strongholdEstX - s.x;
        const dz = s.strongholdEstZ - s.z;
        const strongholdDist = Math.sqrt((dx * dx) + (dz * dz));
        const strongholdCross = (s.lookX * dz) - (s.lookZ * dx);
        const strongholdDot = (s.lookX * dx) + (s.lookZ * dz);
        const relativeMoveAngle = Math.atan2(strongholdCross, strongholdDot) * (180 / Math.PI);
        const strongholdYaw = clamp(Math.round(strongholdCross * 4), -8, 8);

        action.hotbarSlot = s.blockSlot >= 0 ? s.blockSlot : (s.swordSlot >= 0 ? s.swordSlot : s.pickaxeSlot);
        action.yawDelta = strongholdYaw;
        action.moveAngle = clamp(relativeMoveAngle, -180, 180);
        action.sneak = edgeRisk;

        if (strongholdDist < 24 && s.eyeOfEnderCount > 0) {
          action.use = pulse % 4 === 0;
          noteParts.push(`Stronghold route: at target zone near ~${s.strongholdEstX}, ~${s.strongholdEstZ}; probing portal staircase.`);
        } else {
          noteParts.push(`Stronghold route: heading to ~${s.strongholdEstX}, ~${s.strongholdEstZ} (dist ${Math.round(strongholdDist)}).`);
        }

        if (hasEnoughEyes) {
          noteParts.push('Portal-ready check: eye count is sufficient for activation if frame RNG allows.');
        }
      }
    }

    if (lowHp && s.utilityFoodSlot >= 0) {
      action.hotbarSlot = s.utilityFoodSlot;
      action.use = true;
      action.sprint = false;
      noteParts.push('Speedrun safety: healing before next split to avoid run-ending death.');
    }
  }

  if (mode === 'general') {
    if (isGeneral1Preset) {
      const treeInSight = /log|wood|tree|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry/.test(s.focusedEntity || '');
      const closeHarvestable = s.focusedDistance > 0 && s.focusedDistance <= 4.2;
      const likelyStuck = s.onGround && s.horizontalSpeed < 0.025 && !edgeRisk;
      const obstacleAhead = s.focusedDistance > 0
        && s.focusedDistance < 1.8
        && !/log|wood|tree|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry/.test(s.focusedEntity || '')
        && !/player|zombie|skeleton|creeper|spider|enderman|villager|golem|slime|witch|phantom|blaze|piglin|hoglin|ravager|warden/.test(s.focusedEntity || '');
      const curveLeft = (Math.floor(pulse / 6) % 2) === 0;
      action.use = false;
      action.attack = treeInSight || closeHarvestable;
      action.hotbarSlot = s.axeSlot >= 0 ? s.axeSlot : (s.swordSlot >= 0 ? s.swordSlot : s.pickaxeSlot);
      action.forward = !treeInSight;
      action.sprint = !treeInSight && !lowHp;
      action.jump = !treeInSight && (s.horizontalSpeed < 0.06 || likelyStuck || obstacleAhead);
      action.sneak = false;
      action.durationTicks = 6;
      action.yawDelta = treeInSight ? 0 : (curveLeft ? 2 : -2);
      action.moveAngle = treeInSight ? 0 : (curveLeft ? -10 : 10);
      action.left = false;
      action.right = false;

      if (!treeInSight && obstacleAhead) {
        action.left = curveLeft;
        action.right = !curveLeft;
        action.moveAngle = curveLeft ? -30 : 30;
        action.yawDelta = curveLeft ? 6 : -6;
      }

      if (!treeInSight && likelyStuck && pulse % 8 === 0) {
        action.yawDelta = (pulse % 16 === 0) ? 16 : -16;
        action.left = (pulse % 16 === 0);
        action.right = !action.left;
        action.moveAngle = action.left ? -35 : 35;
      }
      if (s.pitch > 28) {
        action.pitchDelta = clamp((10 - s.pitch) / 2, -8, 8);
      }
      noteParts.push('General1 routine: obstacle-aware tree routing with curved movement and active close-range harvesting.');
    } else {
      action.forward = true;
      action.sprint = /\b(run|sprint|fast)\b/.test(text) && !lowHp;
      action.attack = /\b(mine|dig|break|chop|harvest)\b/.test(text);
      action.use = /\b(place|build|use|block)\b/.test(text);
      action.hotbarSlot = action.use ? s.blockSlot : (action.attack ? (s.pickaxeSlot >= 0 ? s.pickaxeSlot : s.axeSlot) : -1);
      action.durationTicks = 8;
      const wantsExploration = /\b(explore|search|walk around|cave|look around|patrol)\b/.test(text);
      const likelyStuck = s.onGround && s.horizontalSpeed < 0.025 && !edgeRisk;
      if (wantsExploration && likelyStuck && pulse % 8 === 0) {
        action.yawDelta = (pulse % 16 === 0) ? 10 : -10;
        action.jump = true;
      }
      noteParts.push('General mode: exploring objective path.');
    }
  }

  if (mode === 'resource') {
    const wantsIron = /\biron\b/.test(text);
    const wantsRedstone = /\bredstone\b/.test(text);
    const wantsDiamond = /\bdiamond\b/.test(text);
    const wantsGold = /\bgold\b/.test(text);
    const wantsEmerald = /\bemerald\b/.test(text);

    const requestedIron = extractRequestedAmount(text, 'iron');
    const requestedRedstone = extractRequestedAmount(text, 'redstone');
    const requestedDiamond = extractRequestedAmount(text, 'diamond|diamonds');
    const requestedGold = extractRequestedAmount(text, 'gold');
    const requestedEmerald = extractRequestedAmount(text, 'emerald|emeralds');

    const ironTarget = requestedIron ?? 20;
    const redstoneTarget = requestedRedstone ?? 32;
    const diamondTarget = requestedDiamond ?? 4;
    const goldTarget = requestedGold ?? 12;
    const emeraldTarget = requestedEmerald ?? 2;

    const targetReached = (wantsIron && s.ironCount >= ironTarget)
      || (wantsRedstone && s.redstoneCount >= redstoneTarget)
      || (wantsDiamond && s.diamondCount >= diamondTarget)
      || (wantsGold && s.goldCount >= goldTarget)
      || (wantsEmerald && s.emeraldCount >= emeraldTarget);

    const resourceKeywordRequested = wantsIron || wantsRedstone || wantsDiamond || wantsGold || wantsEmerald;
    const focusedEntity = (s.focusedEntity || '').toLowerCase();
    const oreInSight = /iron|deepslate_iron|redstone|deepslate_redstone|diamond|deepslate_diamond|gold|deepslate_gold|emerald|deepslate_emerald/.test(focusedEntity);
    const treeInSight = /log|wood|tree|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry/.test(focusedEntity);
    const closeHarvestable = s.focusedDistance > 0 && s.focusedDistance <= 4.4;
    const requestedToolSlot = (wantsRedstone || wantsDiamond || wantsIron || wantsGold || wantsEmerald)
      ? s.pickaxeSlot
      : (wantsGold ? s.pickaxeSlot : s.axeSlot);
    const harvestingToolSlot = requestedToolSlot >= 0
      ? requestedToolSlot
      : (treeInSight ? s.axeSlot : (s.pickaxeSlot >= 0 ? s.pickaxeSlot : s.axeSlot));
    const shouldFightNearbyEnemy = enemyWithinResourceAggro && !veryLowHp && s.hasMeleeWeapon;
    const shouldFallbackDefensive = (enemyVeryClose && (veryLowHp || !s.hasMeleeWeapon)) || (veryLowHp && enemyNearby);

    if (shouldFallbackDefensive) {
      action.back = true;
      action.forward = false;
      action.sneak = true;
      action.sprint = false;
      action.durationTicks = 6;
      action.hotbarSlot = s.utilityFoodSlot >= 0 ? s.utilityFoodSlot : s.blockSlot;
      if (s.utilityFoodSlot >= 0 && lowHp) {
        action.use = true;
      }
      noteParts.push('Resource safety protocol: critical HP or no weapon with close enemy, temporarily disengaging.');
    } else if (shouldFightNearbyEnemy) {
      action.attack = true;
      action.use = false;
      action.forward = enemyVeryClose ? false : true;
      action.back = enemyVeryClose;
      action.left = (pulse % 2) === 0;
      action.right = !action.left;
      action.sprint = !lowHp && !enemyVeryClose;
      action.jump = enemyVeryClose && (pulse % 5 === 0);
      action.durationTicks = 5;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      noteParts.push('Resource route defense: clearing nearby enemy, then resuming mining.');
    } else if (targetReached) {
      action.back = false;
      action.forward = false;
      action.sprint = false;
      action.sneak = true;
      action.durationTicks = 6;
      action.hotbarSlot = s.utilityFoodSlot >= 0 ? s.utilityFoodSlot : s.blockSlot;
      noteParts.push('Resource target reached. Holding position and awaiting next objective.');
    } else {
      const stuckMining = s.onGround && s.horizontalSpeed < 0.025 && !edgeRisk;
      const doingHarvest = (oreInSight || treeInSight || closeHarvestable) && harvestingToolSlot >= 0;
      action.attack = doingHarvest;
      action.use = false;
      action.forward = !doingHarvest;
      action.sprint = !doingHarvest && !lowHp;
      action.jump = !doingHarvest && s.horizontalSpeed < 0.05;
      action.sneak = edgeRisk || (s.fallDistance > 0.5);
      action.durationTicks = doingHarvest ? 5 : 7;
      action.hotbarSlot = harvestingToolSlot >= 0 ? harvestingToolSlot : (s.blockSlot >= 0 ? s.blockSlot : s.pickaxeSlot);
      action.yawDelta = 0;
      if (stuckMining && pulse % 8 === 0) {
        action.yawDelta = (pulse % 16 === 0) ? 12 : -12;
        action.left = (pulse % 16 === 0);
        action.right = !action.left;
        action.jump = true;
      }

      if (!resourceKeywordRequested) {
        noteParts.push('Resource route: default mining priority active (iron/redstone first, then diamond/gold/emerald).');
      } else {
        noteParts.push(doingHarvest
          ? 'Resource route: ore/tree in sight, harvesting immediately with optimal tool.'
          : 'Resource route: sprinting to next vein and scanning for target blocks.');
      }

      if (inventoryValue >= 24 && !enemyNearby) {
        noteParts.push('Loot value is high: stay efficient and avoid unnecessary detours until requested target count is met.');
      }
    }
  }

  if (s.nearestHostileDistance > 0 && s.nearestHostileDistance < 8) {
    if (shouldBlockMobs && s.blockSlot >= 0) {
      action.attack = false;
      action.use = true;
      action.hotbarSlot = s.blockSlot;
      action.sneak = true;
      action.back = true;
      action.forward = false;
      action.left = (pulse % 2 === 0);
      action.right = !action.left;
      action.jump = (pulse % 3 === 0);
      action.durationTicks = 4;
      noteParts.push('Mob shield active: placing defensive blocks to block hits.');
    } else if (/creeper/.test(s.nearestHostile)) {
      action.back = true;
      action.sprint = true;
      noteParts.push('Mob strat: kite creeper before re-engage.');
    } else if (/skeleton|stray/.test(s.nearestHostile)) {
      action.left = (pulse % 2 === 0);
      action.right = !action.left;
      action.sprint = true;
      noteParts.push('Mob strat: zig-zag against ranged fire.');
    } else if (/spider|cave_spider|zombie|husk|drowned/.test(s.nearestHostile)) {
      action.attack = true;
      action.hotbarSlot = s.swordSlot >= 0 ? s.swordSlot : s.axeSlot;
      noteParts.push('Mob strat: close-quarters melee clear.');
    } else if (/enderman/.test(s.nearestHostile)) {
      action.sneak = true;
      action.attack = s.health > 12;
      noteParts.push('Mob strat: controlled Enderman engagement.');
    } else if (/witch|blaze|ghast/.test(s.nearestHostile)) {
      action.sprint = true;
      action.left = (pulse % 2 === 0);
      action.right = !action.left;
      noteParts.push('Mob strat: projectile pressure dodge.');
    }
  }

  if (lowHp) {
    action.sprint = false;
    if (mode !== 'build' && mode !== 'clutch' && mode !== 'pvp' && mode !== 'crystal') {
      action.back = true;
      action.forward = false;
    }
    noteParts.push('Low HP adjustment active.');
  }

  if (!s.onGround && s.verticalSpeed < -0.25) {
    action.sneak = true;
    action.use = hasClutchUtility;
    action.pitchDelta = clamp(action.pitchDelta + 10, -25, 25);
    action.durationTicks = Math.min(action.durationTicks, 4);
    noteParts.push('Falling mitigation active.');
  }

  if (enemyVeryClose && !s.hasMeleeWeapon) {
    action.back = true;
    action.forward = false;
    action.sprint = false;
    noteParts.push('No melee weapon + close enemy: forced kite behavior.');
  }

  if (shouldForceValuableLoot && !severeDanger) {
    action.attack = false;
    action.use = false;
    action.forward = true;
    action.back = false;
    action.left = false;
    action.right = false;
    action.sprint = !enemyVeryClose;
    action.jump = s.horizontalSpeed < 0.05;
    action.sneak = false;
    action.yawDelta = lootYaw;
    action.durationTicks = 6;
    if (s.utilityFoodSlot >= 0 && s.health <= 7) {
      action.hotbarSlot = s.utilityFoodSlot;
    }
    noteParts.push('Valuable dropped item detected: temporarily rerouting to loot, then resuming task.');
  }

  if (action.attack && action.hotbarSlot < 0) {
    action.hotbarSlot = hasMace
      ? (hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot)
      : (s.swordSlot >= 0 ? s.swordSlot : s.axeSlot);
  }

  if (action.use && action.hotbarSlot < 0) {
    if ((mode === 'clutch' || edgeRisk) && s.waterBucketSlot >= 0) action.hotbarSlot = s.waterBucketSlot;
    else action.hotbarSlot = s.blockSlot;
  }

  if (hasEnemyInCrosshair && !action.attack && /\b(pvp|fight|bedwars|kill|rush)\b/.test(text)) {
    action.attack = true;
  }

  const hasAnyCombatTarget = (s.nearestEnemyDistance > 0 && s.nearestEnemyDistance < 30)
    || (s.nearestHostileDistance > 0 && s.nearestHostileDistance < 20)
    || hasEnemyInCrosshair;
  const combatIntent = mode === 'pvp'
    || mode === 'crystal'
    || /\b(pvp|fight|combat|attack|kill|defeat|slay|duel|hunt|rush|eliminate)\b/.test(text);
  const activeEnemyDist = (s.nearestEnemyDistance > 0 ? s.nearestEnemyDistance : s.nearestHostileDistance);
  const inMeleeRange = activeEnemyDist > 0 && activeEnemyDist <= 4.2;
  const inBowRange = activeEnemyDist > 6.0;
  if (hasAnyCombatTarget && combatIntent && !action.sneak) {
    if (inMeleeRange && hasMace && hasWindCharge && s.onGround && (pulse % 14 === 3)) {
      action.use = true;
      action.attack = false;
      action.jump = true;
      action.hotbarSlot = s.windChargeSlot;
      action.pitchDelta = clamp((82 - s.pitch) / 2, -10, 10);
      action.durationTicks = 4;
      noteParts.push('Combat optimizer: using windcharge for mace slam setup.');
    } else if (enemyHasShield && s.axeSlot >= 0 && inMeleeRange) {
      action.hotbarSlot = s.axeSlot;
      if (action.attack) {
        noteParts.push('Combat optimizer: axe selected to counter shield.');
      }
    } else if (inMeleeRange && hasMace) {
      action.hotbarSlot = hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot;
      if (action.attack) {
        noteParts.push('Combat optimizer: mace selected for melee burst.');
      }
    } else if (inBowRange && hasRangedBow && !action.use) {
      action.hotbarSlot = s.bowSlot;
      if (!action.attack) {
        action.use = true;
        action.durationTicks = Math.max(action.durationTicks, 6);
      }
      noteParts.push('Combat optimizer: bow selected for ranged pressure.');
    } else if (hasCrystalCombatKit && mode !== 'build' && mode !== 'general' && activeEnemyDist > 2.4 && activeEnemyDist < 5.2) {
      action.hotbarSlot = (pulse % 2 === 0) ? s.obsidianSlot : s.endCrystalSlot;
      action.use = true;
      action.attack = pulse % 2 !== 0;
      action.sprint = false;
      action.forward = false;
      action.back = false;
      action.durationTicks = 4;
      noteParts.push('Combat optimizer: crystal cycle selected for mid-range fight.');
    } else if (action.attack && action.hotbarSlot < 0) {
      action.hotbarSlot = hasMace
        ? (hasBreachMace && enemyHasShield ? s.breachMaceSlot : s.maceSlot)
        : (s.swordSlot >= 0 ? s.swordSlot : s.axeSlot);
    }
  }

  // ── Auto-eat: eat best food when HP is low ─────────────────────────────────
  const canEat = s.utilityFoodSlot >= 0;
  const criticalEat = canEat && s.health <= eatCriticalHpThreshold; // learned threshold, defaults to 5 hearts
  const sustainEat = canEat && s.health <= eatRecoverHpThreshold && s.food < 20;
  const shouldHoldEatLock = lowHpEatLockActive && s.health <= eatRecoverHpThreshold;
  if (criticalEat || sustainEat || shouldHoldEatLock) {
    action.use = true;
    action.attack = false;
    action.hotbarSlot = s.utilityFoodSlot;
    action.forward = false;
    action.back = false;
    action.sprint = false;
    action.sneak = enemyNearby;
    action.left = false;
    action.right = false;
    action.jump = false;
    action.durationTicks = (criticalEat || shouldHoldEatLock) ? Math.max(10, eatLockTicks + 1) : 8;
    noteParts.push(criticalEat
      ? `Auto-eat critical: HP=${s.health.toFixed(1)} (threshold=${eatCriticalHpThreshold.toFixed(1)}), hard-prioritizing food over combat.`
      : shouldHoldEatLock
        ? `Auto-eat lock: keeping food out until stabilized (hp=${s.health.toFixed(1)}).`
        : `Auto-eat sustain: topping up while recovering (hp=${s.health.toFixed(1)}, food=${s.food}).`);
  }

  if (!Object.values(action).some(v => v === true)) {
    action.forward = true;
    action.durationTicks = 6;
    noteParts.push('Fallback movement engaged.');
  }

  return { action, note: noteParts.join(' '), mode: reportedMode };
}

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded.length > 0) {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket?.remoteAddress || 'unknown';
}

function validateSessionId(sessionId) {
  if (!sessionId) {
    return 'sessionId is required';
  }
  if (sessionId.length > MAX_SESSION_ID_LENGTH) {
    return `sessionId is too long (max ${MAX_SESSION_ID_LENGTH})`;
  }
  if (!/^[a-zA-Z0-9_.:-]+$/.test(sessionId)) {
    return 'sessionId contains invalid characters';
  }
  return null;
}

function validateUserMessage(message) {
  if (!message) {
    return 'message is required';
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return `message is too long (max ${MAX_MESSAGE_LENGTH})`;
  }
  return null;
}

function sendApiError(req, res, statusCode, message) {
  if (req.path === '/chat-plain') {
    return res.status(statusCode).type('text/plain').send(`ERROR: ${message}`);
  }
  return res.status(statusCode).json({ ok: false, error: message });
}

function checkRateLimit(req, res, next) {
  if (req.path === '/mc-agent') {
    const now = Date.now();
    const ip = getClientIp(req);
    const sessionKey = normalizeText(req.body?.sessionId || 'default').slice(0, MAX_SESSION_ID_LENGTH) || 'default';
    const key = `${ip}::${sessionKey}`;
    const entry = mcAgentRateLimits.get(key);

    if (!entry || now > entry.resetAt) {
      mcAgentRateLimits.set(key, {
        count: 1,
        resetAt: now + MC_AGENT_RATE_LIMIT_WINDOW_MS
      });
      return next();
    }

    if (entry.count >= MC_AGENT_RATE_LIMIT_MAX_REQUESTS) {
      const retryAfterSeconds = Math.max(1, Math.ceil((entry.resetAt - now) / 1000));
      res.setHeader('Retry-After', String(retryAfterSeconds));
      return sendApiError(req, res, 429, 'Too many mc-agent requests. Please slow down.');
    }

    entry.count += 1;
    return next();
  }

  const now = Date.now();
  const ip = getClientIp(req);
  const entry = rateLimits.get(ip);

  if (!entry || now > entry.resetAt) {
    rateLimits.set(ip, {
      count: 1,
      resetAt: now + RATE_LIMIT_WINDOW_MS
    });
    return next();
  }

  if (entry.count >= RATE_LIMIT_MAX_REQUESTS) {
    const retryAfterSeconds = Math.max(1, Math.ceil((entry.resetAt - now) / 1000));
    res.setHeader('Retry-After', String(retryAfterSeconds));
    return sendApiError(req, res, 429, 'Too many requests. Please slow down.');
  }

  entry.count += 1;
  return next();
}

function checkApiKey(req, res, next) {
  if (!REQUIRE_API_KEY) {
    return next();
  }

  const headerValue = req.header('x-api-key') || req.header('authorization') || '';
  const token = normalizeText(headerValue.replace(/^Bearer\s+/i, ''));
  if (!token || !API_KEYS.includes(token)) {
    return sendApiError(req, res, 401, 'Unauthorized');
  }
  return next();
}

setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimits.entries()) {
    if (entry.resetAt <= now) {
      rateLimits.delete(ip);
    }
  }
  for (const [key, entry] of mcAgentRateLimits.entries()) {
    if (entry.resetAt <= now) {
      mcAgentRateLimits.delete(key);
    }
  }
}, Math.max(15_000, RATE_LIMIT_WINDOW_MS, MC_AGENT_RATE_LIMIT_WINDOW_MS)).unref();

async function generateChatReply(sessionId, userMessage, baseUrl = '') {
  if (ENABLE_CONTENT_FILTER && isUnsafeInput(userMessage)) {
    const summary = buildReasoningSummary({
      userMessage,
      blocked: true,
      blockedReason: 'unsafe input',
      provider: PROVIDER,
      usedHistory: false,
      usedRetrieval: false,
      usedWebSearch: false
    });
    return {
      reply: formatReplyForDisplay(getSafetyBlockedReply(userMessage), summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: true,
      reasoningSummary: summary
    };
  }

  const imageResult = imageReply(userMessage, baseUrl);
  if (imageResult) {
    return {
      reply: imageResult.reply,
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: '',
      webSources: []
    };
  }

  const mathResult = solveMathQuery(userMessage);
  if (mathResult) {
    const summary = buildReasoningSummary({
      userMessage,
      blocked: false,
      blockedReason: '',
      provider: PROVIDER,
      usedHistory: false,
      usedRetrieval: false,
      usedWebSearch: false
    });
    return {
      reply: formatReplyForDisplay(mathResult.reply, summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: summary,
      webSources: []
    };
  }

  const factualResult = factualKnowledgeReply(userMessage);
  if (factualResult) {
    const summary = buildReasoningSummary({
      userMessage,
      blocked: false,
      blockedReason: '',
      provider: PROVIDER,
      usedHistory: false,
      usedRetrieval: true,
      usedWebSearch: false
    });
    return {
      reply: formatReplyForDisplay(factualResult, summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: summary,
      webSources: []
    };
  }

  const lowerUserMessage = normalizeText(userMessage).toLowerCase();
  const looksMinecraftContext = /\b(minecraft|bedwars|pvp|bridge|bridging|andromeda|telly|godbridge|totem|ender ?pearl|pearl|elytra|obsidian|crystal|hotbar|litematic|litematica|schematic|survival|nether|end|stronghold|farm|redstone)\b/.test(lowerUserMessage);
  const looksScratchContext = /\b(scratch|turbowarp|sprite|broadcast|clone|green flag|stage)\b/.test(lowerUserMessage);
  const allowScratchHelper = looksScratchContext && !looksMinecraftContext;

  const scratchResult = allowScratchHelper ? scratchCodingReply(userMessage) : null;
  if (scratchResult) {
    const summary = buildReasoningSummary({
      userMessage,
      blocked: false,
      blockedReason: '',
      provider: PROVIDER,
      usedHistory: false,
      usedRetrieval: false,
      usedWebSearch: false
    });
    return {
      reply: formatReplyForDisplay(scratchResult.reply, summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: summary,
      webSources: []
    };
  }

  const robotResult = parseRobotTask(userMessage);
  if (robotResult) {
    return {
      reply: formatReplyForDisplay(robotResult.reply, ''),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: '',
      webSources: []
    };
  }

  const webContext = await searchWebContext(userMessage);
  const userMessageForModel = buildUserMessageWithWebContext(userMessage, webContext);
  const usedWebSearch = Boolean(webContext.contextText);

  const history = getSessionMessages(sessionId);
  const conversation = trimHistory(history, HISTORY_LIMIT);

  if (PROVIDER === 'solasgpt') {
    const enrichedMessage = buildSolasForwardMessage(sessionId, userMessageForModel);
    const forwardedMessage = truncateText(enrichedMessage, SOLASGPT_FORWARD_MAX_CHARS);

    let rawReply;
    try {
      rawReply = await callSolasGPT(sessionId, forwardedMessage);
    } catch (error) {
      const directAnswer = generateHelpfulDirectAnswer(userMessage, webContext);
      if (directAnswer) {
        rawReply = directAnswer;
      } else if (UPSTREAM_FALLBACK_ENABLED) {
        rawReply = phraseKnowledgeReply(userMessage, webContext);
      } else {
        throw error;
      }
    }

    const usePhrasingFallback =
      PHRASING_KNOWLEDGE_ENABLED && PHRASING_FALLBACK_ON_LOW_QUALITY && looksLowQualityReply(rawReply);
    const reply = usePhrasingFallback ? phraseKnowledgeReply(userMessage, webContext) : rawReply;

    const updatedHistory = trimHistory(
      [...conversation, { role: 'user', content: userMessage }, { role: 'assistant', content: reply }],
      HISTORY_LIMIT
    );
    sessions.set(sessionId, updatedHistory);

    if (ENABLE_CONTENT_FILTER && isUnsafeOutput(reply)) {
      const summary = buildReasoningSummary({
        userMessage,
        blocked: true,
        blockedReason: 'unsafe output',
        provider: PROVIDER,
        usedHistory: true,
        usedRetrieval: true,
        usedWebSearch
      });
      return {
        reply: formatReplyForDisplay(SAFETY_REFUSAL_TEXT, summary),
        provider: PROVIDER,
        model: MODEL,
        sessionId,
        filtered: true,
        reasoningSummary: summary,
        webSources: webContext.sources
      };
    }
    const summary = buildReasoningSummary({
      userMessage,
      blocked: false,
      blockedReason: '',
      provider: PROVIDER,
      usedHistory: true,
      usedRetrieval: true,
      usedWebSearch
    });
    return {
      reply: formatReplyForDisplay(reply, summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: false,
      reasoningSummary: summary,
      webSources: webContext.sources
    };
  }

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...conversation,
    { role: 'user', content: userMessageForModel }
  ];

  const reply =
    PROVIDER === 'ollama' ? await callOllama(messages) : await callOpenAICompatible(messages);

  const finalReply =
    PHRASING_KNOWLEDGE_ENABLED && PHRASING_FALLBACK_ON_LOW_QUALITY && looksLowQualityReply(reply)
      ? phraseKnowledgeReply(userMessage, webContext)
      : reply;

  if (ENABLE_CONTENT_FILTER && isUnsafeOutput(finalReply)) {
    const summary = buildReasoningSummary({
      userMessage,
      blocked: true,
      blockedReason: 'unsafe output',
      provider: PROVIDER,
      usedHistory: true,
      usedRetrieval: false,
      usedWebSearch
    });
    return {
      reply: formatReplyForDisplay(SAFETY_REFUSAL_TEXT, summary),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: true,
      reasoningSummary: summary,
      webSources: webContext.sources
    };
  }

  const updatedHistory = trimHistory(
    [...conversation, { role: 'user', content: userMessage }, { role: 'assistant', content: finalReply }],
    HISTORY_LIMIT
  );
  sessions.set(sessionId, updatedHistory);

  const summary = buildReasoningSummary({
    userMessage,
    blocked: false,
    blockedReason: '',
    provider: PROVIDER,
    usedHistory: true,
    usedRetrieval: false,
    usedWebSearch
  });

  return {
    reply: formatReplyForDisplay(finalReply, summary),
    provider: PROVIDER,
    model: MODEL,
    sessionId,
    filtered: false,
    reasoningSummary: summary,
    webSources: webContext.sources
  };
}

async function callOpenAICompatible(messages) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is missing');
  }

  const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI-compatible API error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  const text = data?.choices?.[0]?.message?.content;
  return normalizeText(text) || 'I could not generate a response.';
}

async function callSolasGPT(sessionId, userMessage) {
  const response = await fetch(`${SOLASGPT_URL}/chat-plain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, message: userMessage })
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`SolasGPT error: ${response.status} ${errorText}`);
  }
  const text = await response.text();
  return normalizeText(text) || 'I could not generate a response.';
}

async function callSolasFeedback(payload) {
  const response = await fetch(`${SOLASGPT_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`SolasGPT feedback error: ${response.status} ${errorText}`);
  }
  const data = await response.json().catch(() => ({}));
  return data;
}

async function callOllama(messages) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      stream: false
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama API error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  const text = data?.message?.content;
  return normalizeText(text) || 'I could not generate a response.';
}

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    provider: PROVIDER,
    model: MODEL,
    limits: {
      maxMessageLength: MAX_MESSAGE_LENGTH,
      rateLimitWindowMs: RATE_LIMIT_WINDOW_MS,
      rateLimitMaxRequests: RATE_LIMIT_MAX_REQUESTS,
      mcAgentRateLimitWindowMs: MC_AGENT_RATE_LIMIT_WINDOW_MS,
      mcAgentRateLimitMaxRequests: MC_AGENT_RATE_LIMIT_MAX_REQUESTS,
      apiKeyRequired: REQUIRE_API_KEY,
      contentFilterEnabled: ENABLE_CONTENT_FILTER,
      reasoningSummaryEnabled: SHOW_REASONING_SUMMARY,
      reasoningSummaryMode: REASONING_SUMMARY_MODE,
      webSearchEnabled: WEB_SEARCH_ENABLED,
      webResultLimit: WEB_RESULT_LIMIT,
      webContextMaxChars: WEB_CONTEXT_MAX_CHARS,
      replyWrapChars: REPLY_WRAP_CHARS,
      replyWrapOverflow: REPLY_WRAP_OVERFLOW,
      solasgptForwardMaxChars: SOLASGPT_FORWARD_MAX_CHARS,
      upstreamFallbackEnabled: UPSTREAM_FALLBACK_ENABLED,
      generatedImageSize: GENERATED_IMAGE_SIZE,
      phrasingKnowledgeEnabled: PHRASING_KNOWLEDGE_ENABLED,
      phrasingFallbackOnLowQuality: PHRASING_FALLBACK_ON_LOW_QUALITY
    }
  });
});

function buildSubjectSvg(prompt) {
  const text = prompt.toLowerCase();
  const colorMap = [
    ['red', '#e74c3c'], ['orange', '#e67e22'], ['yellow', '#f1c40f'],
    ['green', '#2ecc71'], ['blue', '#3498db'], ['purple', '#9b59b6'],
    ['pink', '#e91ea0'], ['white', '#ecf0f1'], ['black', '#555566'],
    ['brown', '#8d5524'], ['cyan', '#1abc9c'], ['gold', '#f39c12'],
    ['grey', '#95a5a6'], ['gray', '#95a5a6']
  ];
  let c = '#3498db';
  for (const [name, hex] of colorMap) {
    if (text.includes(name)) { c = hex; break; }
  }
  const bg = '#1a1a2e';
  const label = escapeSvgText(prompt.slice(0, 42));
  let body = '';

  if (/\bcat\b/.test(text)) {
    body = `
  <ellipse cx="240" cy="295" rx="85" ry="58" fill="${c}"/>
  <circle cx="240" cy="170" r="72" fill="${c}"/>
  <polygon points="178,122 158,64 218,110" fill="${c}"/>
  <polygon points="302,122 322,64 262,110" fill="${c}"/>
  <polygon points="181,117 167,78 212,110" fill="#f9a8d4" opacity="0.75"/>
  <polygon points="299,117 313,78 268,110" fill="#f9a8d4" opacity="0.75"/>
  <ellipse cx="212" cy="162" rx="13" ry="15" fill="#111"/>
  <circle cx="216" cy="157" r="4" fill="white"/>
  <ellipse cx="268" cy="162" rx="13" ry="15" fill="#111"/>
  <circle cx="272" cy="157" r="4" fill="white"/>
  <polygon points="240,182 234,191 246,191" fill="#f9a8d4"/>
  <path d="M234,192 Q240,198 246,192" fill="none" stroke="#555" stroke-width="2"/>
  <line x1="166" y1="181" x2="226" y2="184" stroke="#ccc" stroke-width="2" opacity="0.8"/>
  <line x1="164" y1="191" x2="226" y2="188" stroke="#ccc" stroke-width="2" opacity="0.6"/>
  <line x1="314" y1="181" x2="254" y2="184" stroke="#ccc" stroke-width="2" opacity="0.8"/>
  <line x1="316" y1="191" x2="254" y2="188" stroke="#ccc" stroke-width="2" opacity="0.6"/>
  <path d="M326,290 C366,252 390,210 354,182" fill="none" stroke="${c}" stroke-width="16" stroke-linecap="round"/>`;
  } else if (/\bdog\b/.test(text)) {
    body = `
  <ellipse cx="240" cy="292" rx="92" ry="60" fill="${c}"/>
  <circle cx="240" cy="168" r="72" fill="${c}"/>
  <ellipse cx="170" cy="170" rx="28" ry="52" fill="${c}" opacity="0.9"/>
  <ellipse cx="310" cy="170" rx="28" ry="52" fill="${c}" opacity="0.9"/>
  <ellipse cx="240" cy="193" rx="30" ry="20" fill="#c9985e" opacity="0.65"/>
  <circle cx="212" cy="158" r="13" fill="#111"/>
  <circle cx="216" cy="154" r="4" fill="white"/>
  <circle cx="268" cy="158" r="13" fill="#111"/>
  <circle cx="272" cy="154" r="4" fill="white"/>
  <ellipse cx="240" cy="185" rx="12" ry="9" fill="#111"/>
  <path d="M334,272 C370,242 382,202 360,180" fill="none" stroke="${c}" stroke-width="14" stroke-linecap="round"/>`;
  } else if (/\bstar\b/.test(text)) {
    body = `
  <polygon points="240,48 266,160 378,160 288,228 316,340 240,270 164,340 192,228 102,160 214,160" fill="${c}"/>
  <circle cx="208" cy="118" r="16" fill="white" opacity="0.28"/>`;
  } else if (/\bsun\b/.test(text)) {
    const sc = c === '#3498db' ? '#f1c40f' : c;
    body = `
  <line x1="240" y1="18" x2="240" y2="78" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="240" y1="282" x2="240" y2="342" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="58" y1="180" x2="118" y2="180" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="362" y1="180" x2="422" y2="180" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="98" y1="58" x2="141" y2="101" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="339" y1="259" x2="382" y2="302" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="382" y1="58" x2="339" y2="101" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <line x1="141" y1="259" x2="98" y2="302" stroke="${sc}" stroke-width="14" stroke-linecap="round"/>
  <circle cx="240" cy="180" r="96" fill="${sc}"/>
  <circle cx="206" cy="162" r="11" fill="#1a1a2e"/>
  <circle cx="274" cy="162" r="11" fill="#1a1a2e"/>
  <path d="M198,200 Q240,228 282,200" fill="none" stroke="#1a1a2e" stroke-width="6" stroke-linecap="round"/>`;
  } else if (/\bhouse\b/.test(text)) {
    body = `
  <rect x="100" y="170" width="280" height="180" fill="${c}" rx="4"/>
  <polygon points="78,175 240,52 402,175" fill="#c0392b"/>
  <rect x="200" y="278" width="80" height="72" fill="#8d5524" rx="4"/>
  <circle cx="270" cy="318" r="6" fill="#f39c12"/>
  <rect x="120" y="206" width="68" height="54" fill="#a8d8ea" rx="3"/>
  <line x1="154" y1="206" x2="154" y2="260" stroke="#6aa0b0" stroke-width="2"/>
  <line x1="120" y1="233" x2="188" y2="233" stroke="#6aa0b0" stroke-width="2"/>
  <rect x="292" y="206" width="68" height="54" fill="#a8d8ea" rx="3"/>
  <line x1="326" y1="206" x2="326" y2="260" stroke="#6aa0b0" stroke-width="2"/>
  <line x1="292" y1="233" x2="360" y2="233" stroke="#6aa0b0" stroke-width="2"/>`;
  } else if (/\bflower\b/.test(text)) {
    body = `
  <line x1="240" y1="332" x2="240" y2="218" stroke="#2ecc71" stroke-width="10" stroke-linecap="round"/>
  <ellipse cx="202" cy="278" rx="36" ry="16" fill="#2ecc71" transform="rotate(-40 202 278)"/>
  <ellipse cx="278" cy="266" rx="36" ry="16" fill="#2ecc71" transform="rotate(40 278 266)"/>
  <ellipse cx="240" cy="138" rx="24" ry="52" fill="${c}"/>
  <ellipse cx="240" cy="222" rx="24" ry="52" fill="${c}"/>
  <ellipse cx="190" cy="180" rx="52" ry="24" fill="${c}"/>
  <ellipse cx="290" cy="180" rx="52" ry="24" fill="${c}"/>
  <ellipse cx="206" cy="146" rx="22" ry="52" fill="${c}" transform="rotate(45 206 146)"/>
  <ellipse cx="274" cy="146" rx="22" ry="52" fill="${c}" transform="rotate(-45 274 146)"/>
  <ellipse cx="206" cy="214" rx="22" ry="52" fill="${c}" transform="rotate(-45 206 214)"/>
  <ellipse cx="274" cy="214" rx="22" ry="52" fill="${c}" transform="rotate(45 274 214)"/>
  <circle cx="240" cy="180" r="42" fill="#f1c40f"/>`;
  } else if (/\btree\b/.test(text)) {
    const tc = c === '#3498db' ? '#2ecc71' : c;
    body = `
  <rect x="214" y="258" width="52" height="98" fill="#8d5524" rx="4"/>
  <polygon points="240,40 346,202 134,202" fill="${tc}"/>
  <polygon points="240,86 358,238 122,238" fill="${tc}"/>`;
  } else if (/\bheart\b/.test(text)) {
    const hc = c === '#3498db' ? '#e74c3c' : c;
    body = `
  <path d="M240,308 L62,150 C62,80 130,54 178,90 C200,104 228,132 240,150 C252,132 280,104 302,90 C350,54 418,80 418,150 Z" fill="${hc}"/>
  <ellipse cx="172" cy="132" rx="26" ry="40" fill="white" opacity="0.22" transform="rotate(-35 172 132)"/>`;
  } else if (/\bfish\b/.test(text)) {
    body = `
  <polygon points="100,180 58,120 58,240" fill="${c}" opacity="0.85"/>
  <ellipse cx="240" cy="180" rx="140" ry="78" fill="${c}"/>
  <circle cx="330" cy="162" r="18" fill="white"/>
  <circle cx="332" cy="162" r="10" fill="#111"/>
  <circle cx="336" cy="157" r="4" fill="white"/>
  <path d="M202,132 Q190,180 202,228" fill="none" stroke="white" stroke-width="2" opacity="0.35"/>
  <path d="M242,124 Q228,180 242,236" fill="none" stroke="white" stroke-width="2" opacity="0.35"/>
  <path d="M282,128 Q270,180 282,232" fill="none" stroke="white" stroke-width="2" opacity="0.35"/>`;
  } else if (/\bbird\b/.test(text)) {
    body = `
  <ellipse cx="230" cy="202" rx="88" ry="64" fill="${c}"/>
  <circle cx="322" cy="165" r="52" fill="${c}"/>
  <ellipse cx="186" cy="194" rx="80" ry="38" fill="${c}" opacity="0.8" transform="rotate(-15 186 194)"/>
  <polygon points="373,162 416,178 373,190" fill="#f39c12"/>
  <circle cx="336" cy="154" r="12" fill="#111"/>
  <circle cx="338" cy="152" r="5" fill="white"/>
  <path d="M148,204 C98,232 80,282 118,310" fill="none" stroke="${c}" stroke-width="18" stroke-linecap="round"/>`;
  } else {
    const hash = hashString(prompt);
    const cx2 = 90 + (hash % 220);
    const cy2 = 110 + (hash % 120);
    const r2 = 38 + (hash % 44);
    body = `
  <circle cx="${cx2}" cy="${cy2}" r="${r2}" fill="${c}" opacity="0.95"/>
  <rect x="252" y="72" width="128" height="128" rx="28" fill="${c}" opacity="0.7"/>
  <path d="M50,280 C120,220 180,220 240,280 S360,340 430,280" fill="none" stroke="${c}" stroke-width="14" stroke-linecap="round"/>`;
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360" viewBox="0 0 480 360">
  <rect width="480" height="360" rx="12" fill="${bg}"/>${body}
  <text x="240" y="350" font-family="Arial, sans-serif" font-size="15" fill="#a6adc8" text-anchor="middle">${label}</text>
</svg>`;
}

app.get('/generated-image.svg', (req, res) => {
  const prompt = normalizeText(req.query?.prompt || 'creative scene') || 'creative scene';
  const svg = buildSubjectSvg(prompt);
  res.type('image/svg+xml').send(svg);
});

app.get('/chat-plain', (req, res) => {
  res
    .status(405)
    .type('text/plain')
    .send('Use POST /chat-plain with JSON body: {"sessionId":"...","message":"..."}');
});

app.post('/chat', checkApiKey, checkRateLimit, async (req, res) => {
  try {
    const sessionId = normalizeText(req.body?.sessionId || 'default');
    const userMessage = normalizeText(req.body?.message);

    const sessionError = validateSessionId(sessionId);
    if (sessionError) {
      return res.status(400).json({ ok: false, error: sessionError });
    }

    const messageError = validateUserMessage(userMessage);
    if (messageError) {
      return res.status(400).json({ ok: false, error: messageError });
    }

    const baseUrl = `${req.protocol}://${req.get('host')}`;
    const result = await generateChatReply(sessionId, userMessage, baseUrl);

    return res.json({
      ok: true,
      ...result
    });
  } catch (error) {
      // handled above
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

app.post('/chat-plain', checkApiKey, checkRateLimit, async (req, res) => {
  try {
    const sessionId = normalizeText(req.body?.sessionId || 'default');
    const userMessage = normalizeText(req.body?.message);

    const sessionError = validateSessionId(sessionId);
    if (sessionError) {
      return res.status(400).type('text/plain').send(`ERROR: ${sessionError}`);
    }

    const messageError = validateUserMessage(userMessage);
    if (messageError) {
      return res.status(400).type('text/plain').send(`ERROR: ${messageError}`);
    }

    const baseUrl = `${req.protocol}://${req.get('host')}`;
    const result = await generateChatReply(sessionId, userMessage, baseUrl);
    return res.type('text/plain').send(result.reply);
  } catch (error) {
      // handled above
    const message = error instanceof Error ? error.message : 'Unknown error';
    return res.status(500).type('text/plain').send(`ERROR: ${message}`);
  }
});

app.post(['/mc-agent', '/mc'], checkApiKey, checkRateLimit, (req, res) => {
  try {
    const sessionId = normalizeText(req.body?.sessionId || 'default');
    const objective = normalizeText(req.body?.objective || '');
    const state = req.body?.state && typeof req.body.state === 'object' ? req.body.state : {};
    const sessionError = validateSessionId(sessionId);
    if (sessionError) {
      return res.status(400).json({ ok: false, error: sessionError });
    }
    if (!objective) {
      return res.status(400).json({ ok: false, error: 'objective is required' });
    }
    if (objective.length > 400) {
      return res.status(400).json({ ok: false, error: 'objective is too long (max 400)' });
    }

    const existingCtx = mcAgentSessions.get(sessionId) || { tickCounter: 0 };
    const nextTickCounter = Number(existingCtx.tickCounter || 0) + 1;
    const learningProfile = sanitizeMcLearningProfile(existingCtx.learning);
    const learnedCriticalHp = learningProfile.eatCriticalHp;
    const learnedRecoverHp = Math.max(learnedCriticalHp + 1, learningProfile.eatRecoverHp);
    const stateHealth = Number(state?.health);
    const hasStateHealth = Number.isFinite(stateHealth) && stateHealth >= 0;
    const previousHealth = Number(existingCtx.lastHealth);
    const tookDamageNow = hasStateHealth
      && Number.isFinite(previousHealth)
      && previousHealth > 0
      && (previousHealth - stateHealth) >= 0.5;
    const previousDamageTick = Number(existingCtx.lastDamageTick || -1);
    const nextDamageTick = tookDamageNow ? nextTickCounter : previousDamageTick;
    const stateFood = Number(state?.food);
    const hasStateFood = Number.isFinite(stateFood) && stateFood >= 0;
    const utilityFoodSlotNow = Number(state?.utilityFoodSlot);
    const hasUtilityFood = Number.isFinite(utilityFoodSlotNow) && utilityFoodSlotNow >= 0;
    const previousLowHpEatUntilTick = Number(existingCtx.lowHpEatUntilTick || -1);
    const enterLowHpEatLock = hasStateHealth && hasUtilityFood && stateHealth <= learnedCriticalHp;
    const keepLowHpEatLock = hasStateHealth && hasUtilityFood && stateHealth <= learnedRecoverHp && (!hasStateFood || stateFood < 20);
    const clearLowHpEatLock = !hasUtilityFood || (hasStateHealth && stateHealth >= 13 && (!hasStateFood || stateFood >= 18));
    let nextLowHpEatUntilTick = previousLowHpEatUntilTick;
    if (enterLowHpEatLock) {
      nextLowHpEatUntilTick = nextTickCounter + Math.max(6, learningProfile.eatLockTicks + 2);
    } else if (keepLowHpEatLock) {
      nextLowHpEatUntilTick = Math.max(previousLowHpEatUntilTick, nextTickCounter + Math.max(6, learningProfile.eatLockTicks));
    } else if (clearLowHpEatLock) {
      nextLowHpEatUntilTick = -1;
    }
    const objectiveText = normalizeText(objective).toLowerCase();
    const bedObjectiveActive = /\b(break|destroy|take\s*out|rush)\b[^.\n]*\bbed\b|\bbed\b[^.\n]*\b(break|destroy)\b/.test(objectiveText)
      || /\b(bedwars|rush bed|defend bed)\b/.test(objectiveText);
    const currentBedDistance = Number(state?.nearestBedDistance);
    const bedVisibleNow = Boolean(state?.bedNearby)
      || (Number.isFinite(currentBedDistance) && currentBedDistance >= 0 && currentBedDistance < 14);
    const existingBedGoal = existingCtx.bedGoal && typeof existingCtx.bedGoal === 'object' ? existingCtx.bedGoal : {};
    const closestObservedDistance = Number(existingBedGoal.closestDistance);
    const nextClosestObservedDistance = Number.isFinite(currentBedDistance) && currentBedDistance >= 0
      ? (Number.isFinite(closestObservedDistance) ? Math.min(closestObservedDistance, currentBedDistance) : currentBedDistance)
      : closestObservedDistance;
    const seenBedEver = Boolean(existingBedGoal.seenBedEver) || bedVisibleNow;
    const reachedBedArea = Number.isFinite(nextClosestObservedDistance) && nextClosestObservedDistance <= 4.5;
    const newlyCompletedBedGoal = bedObjectiveActive
      && !bedVisibleNow
      && seenBedEver
      && reachedBedArea;
    const opponentName = normalizeText(state?.nearestEnemyName || 'unknown');
    const existingOpponents = existingCtx.opponents && typeof existingCtx.opponents === 'object'
      ? existingCtx.opponents
      : {};
    const existingOpponent = existingOpponents[opponentName] || { seen: 0 };
    const nearestEnemyDistance = Number(state?.nearestEnemyDistance);
    const nearestEnemyHealth = Number(state?.nearestEnemyHealth);
    const nearestEnemyArmorPieces = Number(state?.nearestEnemyArmorPieces || 0);
    const playerHasMeleeWeapon = Boolean(state?.hasMeleeWeapon);
    const playerHealthNow = hasStateHealth ? stateHealth : Number(existingCtx.lastHealth || 0);
    const enemyIsTrackable = opponentName !== 'unknown' && Number.isFinite(nearestEnemyDistance) && nearestEnemyDistance > 0 && nearestEnemyDistance < 20;
    const enemyHealthKnown = Number.isFinite(nearestEnemyHealth) && nearestEnemyHealth > 0;
    const estimatedEnemyHealth = enemyHealthKnown ? nearestEnemyHealth : 12;
    const hpAdvantage = playerHealthNow - estimatedEnemyHealth;
    const armorDisadvantage = nearestEnemyArmorPieces >= 4 && playerHealthNow < 10;
    const canLikelyWinFight = playerHasMeleeWeapon
      && playerHealthNow >= 6
      && !armorDisadvantage
      && (hpAdvantage >= -2 || playerHealthNow >= 14);

    const previousRetaliationTargetName = normalizeText(existingCtx.retaliationTargetName || '');
    const previousRetaliationUntilTick = Number(existingCtx.retaliationUntilTick || -1);
    const retaliationTargetStillVisible = previousRetaliationTargetName.length > 0
      && opponentName.toLowerCase() === previousRetaliationTargetName.toLowerCase()
      && enemyIsTrackable
      && (!enemyHealthKnown || nearestEnemyHealth > 0);
    let nextRetaliationTargetName = previousRetaliationTargetName;
    let nextRetaliationUntilTick = previousRetaliationUntilTick;

    if (tookDamageNow && enemyIsTrackable && canLikelyWinFight) {
      nextRetaliationTargetName = opponentName;
      nextRetaliationUntilTick = nextTickCounter + 140;
    } else if (retaliationTargetStillVisible && canLikelyWinFight) {
      nextRetaliationUntilTick = Math.max(previousRetaliationUntilTick, nextTickCounter + 40);
    } else if (nextTickCounter > previousRetaliationUntilTick || (enemyHealthKnown && nearestEnemyHealth <= 0)) {
      nextRetaliationTargetName = '';
      nextRetaliationUntilTick = -1;
    }

    const nextCtx = {
      tickCounter: nextTickCounter,
      updatedAt: Date.now(),
      learning: learningProfile,
      lastHealth: hasStateHealth ? stateHealth : Number(existingCtx.lastHealth || -1),
      lastDamageTick: Number.isFinite(nextDamageTick) ? nextDamageTick : -1,
      retaliationTargetName: nextRetaliationTargetName,
      retaliationUntilTick: Number.isFinite(nextRetaliationUntilTick) ? nextRetaliationUntilTick : -1,
      lowHpEatUntilTick: Number.isFinite(nextLowHpEatUntilTick) ? nextLowHpEatUntilTick : -1,
      lastPotTick: Number(existingCtx.lastPotTick || -9999),
      goals: {
        prot4Diamond: /\b(protection ?4|prot ?4|full diamond|diamond armor)\b/.test(objective),
        netherite: /\b(netherite|netherite upgrade|smithing template)\b/.test(objective),
        allBooks: /\b(all enchant|all books|every book|all other enchanting books)\b/.test(objective)
      },
      opponents: {
        ...existingOpponents,
        [opponentName]: {
          seen: Number(existingOpponent.seen || 0) + 1,
          lastDistance: Number(state?.nearestEnemyDistance || -1),
          lastArmorPieces: Number(state?.nearestEnemyArmorPieces || 0),
          lastMainItem: normalizeText(state?.nearestEnemyMainItem || ''),
          updatedAt: Date.now()
        }
      },
      bedGoal: {
        active: bedObjectiveActive,
        completed: bedObjectiveActive && (Boolean(existingBedGoal.completed) || newlyCompletedBedGoal),
        seenBedEver: bedObjectiveActive ? seenBedEver : false,
        closestDistance: bedObjectiveActive && Number.isFinite(nextClosestObservedDistance) ? nextClosestObservedDistance : -1,
        visibleNow: bedObjectiveActive ? bedVisibleNow : false
      },
      windMaceComboStage: Number(existingCtx.windMaceComboStage || 0)
    };
    const decision = buildMinecraftAction(objective, state, nextCtx);
    const previousHotbarSlot = Number(existingCtx.lastHotbarSlot);
    const previousHotbarChangeTick = Number(existingCtx.lastHotbarChangeTick || -9999);
    const desiredHotbarSlot = Number(decision?.action?.hotbarSlot);
    const swordSlotNow = Number(state?.swordSlot);
    const potionSlotNow = Number(state?.combatPotionSlot);
    const hasPreviousHotbar = Number.isFinite(previousHotbarSlot) && previousHotbarSlot >= 0;
    const hasDesiredHotbar = Number.isFinite(desiredHotbarSlot) && desiredHotbarSlot >= 0;
    const hotbarChangedNow = hasPreviousHotbar && hasDesiredHotbar && desiredHotbarSlot !== previousHotbarSlot;
    const ticksSinceHotbarChange = nextTickCounter - previousHotbarChangeTick;
    const swordPotionPairSwap = Number.isFinite(swordSlotNow)
      && Number.isFinite(potionSlotNow)
      && swordSlotNow >= 0
      && potionSlotNow >= 0
      && ((previousHotbarSlot === swordSlotNow && desiredHotbarSlot === potionSlotNow)
        || (previousHotbarSlot === potionSlotNow && desiredHotbarSlot === swordSlotNow));
    const hotbarLowIntentSwap = hotbarChangedNow
      && ticksSinceHotbarChange < 5
      && !Boolean(decision?.action?.use)
      && !Boolean(decision?.action?.attack);
    const avoidPotionSwapWithoutUse = swordPotionPairSwap
      && decision.mode === 'pvp'
      && !Boolean(decision?.action?.use);

    if ((hotbarLowIntentSwap || avoidPotionSwapWithoutUse) && hasPreviousHotbar) {
      decision.action.hotbarSlot = previousHotbarSlot;
    }

    const finalHotbarSlot = Number(decision?.action?.hotbarSlot);
    const hasFinalHotbar = Number.isFinite(finalHotbarSlot) && finalHotbarSlot >= 0;
    const didHotbarChange = hasFinalHotbar && hasPreviousHotbar
      ? finalHotbarSlot !== previousHotbarSlot
      : hasFinalHotbar;
    const combatPotionSlot = Number(state?.combatPotionSlot);
    const usedCombatPotion = Boolean(decision?.action?.use)
      && Number.isFinite(combatPotionSlot)
      && Number(decision?.action?.hotbarSlot) === combatPotionSlot;
    const usedFoodNow = Boolean(decision?.action?.use)
      && Number.isFinite(utilityFoodSlotNow)
      && utilityFoodSlotNow >= 0
      && Number(decision?.action?.hotbarSlot) === utilityFoodSlotNow;
    const persistedLowHpEatUntilTick = usedFoodNow && hasStateHealth && stateHealth <= learnedRecoverHp
      ? Math.max(Number(nextCtx.lowHpEatUntilTick || -1), nextCtx.tickCounter + Math.max(6, learningProfile.eatLockTicks))
      : Number(nextCtx.lowHpEatUntilTick || -1);
    const windChargeSlotNow = Number(state?.windChargeSlot);
    const maceSlotNow = Number(state?.maceSlot);
    const usedWindChargeNow = Boolean(decision?.action?.use)
      && Number.isFinite(windChargeSlotNow)
      && windChargeSlotNow >= 0
      && Number(decision?.action?.hotbarSlot) === windChargeSlotNow;
    const holdingMaceNow = Number.isFinite(maceSlotNow)
      && maceSlotNow >= 0
      && Number(decision?.action?.hotbarSlot) === maceSlotNow;
    const previousWindMaceStage = Number(existingCtx.windMaceComboStage || 0);
    let nextWindMaceStage = 0;
    if (usedWindChargeNow) {
      nextWindMaceStage = 1;
    } else if (previousWindMaceStage === 1 && holdingMaceNow && !Boolean(decision?.action?.attack)) {
      nextWindMaceStage = 2;
    } else if (previousWindMaceStage === 2 && holdingMaceNow && Boolean(decision?.action?.attack)) {
      nextWindMaceStage = 0;
    } else if (previousWindMaceStage > 0 && !holdingMaceNow) {
      nextWindMaceStage = 0;
    }
    mcAgentSessions.set(sessionId, {
      ...nextCtx,
      lastPotTick: usedCombatPotion ? nextCtx.tickCounter : nextCtx.lastPotTick,
      lowHpEatUntilTick: Number.isFinite(persistedLowHpEatUntilTick) ? persistedLowHpEatUntilTick : -1,
      windMaceComboStage: nextWindMaceStage,
      lastHotbarSlot: hasFinalHotbar ? finalHotbarSlot : (hasPreviousHotbar ? previousHotbarSlot : -1),
      lastHotbarChangeTick: didHotbarChange ? nextCtx.tickCounter : previousHotbarChangeTick,
      mode: decision.mode,
      objective: objective.slice(0, 120),
      lastAction: decision.action
    });
    scheduleMcMemorySave();

    if (mcAgentSessions.size > 500) {
      const cutoff = Date.now() - (30 * 60 * 1000);
      for (const [sid, ctx] of mcAgentSessions.entries()) {
        if (!ctx || Number(ctx.updatedAt || 0) < cutoff) {
          mcAgentSessions.delete(sid);
        }
      }
    }

    return res.json({
      ok: true,
      sessionId,
      objective,
      action: decision.action,
      note: decision.note,
      mode: decision.mode,
      memory: {
        tickCounter: nextCtx.tickCounter,
        learning: nextCtx.learning,
        goals: nextCtx.goals,
        opponent: nextCtx.opponents[opponentName],
        bedGoal: nextCtx.bedGoal
      }
    });
  } catch (error) {
      // handled above
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

app.post('/reset', checkApiKey, checkRateLimit, async (req, res) => {
  const sessionId = normalizeText(req.body?.sessionId || 'default');
  const sessionError = validateSessionId(sessionId);
  if (sessionError) {
    return res.status(400).json({ ok: false, error: sessionError });
  }

  sessions.delete(sessionId);
  sessionFeedback.delete(sessionId);
  mcAgentSessions.delete(sessionId);
  scheduleMcMemorySave();
  if (PROVIDER === 'solasgpt') {
    try {
      await fetch(`${SOLASGPT_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId })
      });
    } catch (_) { /* ignore if inference server is down */ }
  }
  res.json({ ok: true, sessionId });
});

app.post('/feedback', checkApiKey, checkRateLimit, async (req, res) => {
  try {
    const sessionId = normalizeText(req.body?.sessionId || 'default');
    const message = normalizeText(req.body?.message || '');
    const reply = normalizeText(req.body?.reply || '');
    const rating = normalizeText(req.body?.rating || '');
    const improvement = normalizeText(req.body?.improvement || '');

    const sessionError = validateSessionId(sessionId);
    if (sessionError) {
      return res.status(400).json({ ok: false, error: sessionError });
    }

    if (!rating) {
      return res.status(400).json({ ok: false, error: 'rating is required' });
    }

    if (PROVIDER === 'solasgpt') {
      try {
        await callSolasFeedback({ sessionId, message, reply, rating, improvement });
      } catch (_) {
        // best-effort forwarding, keep API success for UI flow
      }
    }

    appendSessionFeedback(sessionId, {
      ts: Date.now(),
      message,
      reply,
      rating,
      improvement
    });

    return res.json({ ok: true, sessionId, rating });
  } catch (error) {
      // handled above
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

app.post('/mc-feedback', checkApiKey, checkRateLimit, (req, res) => {
  try {
    const sessionId = normalizeText(req.body?.sessionId || 'default');
    const sessionError = validateSessionId(sessionId);
    if (sessionError) {
      return res.status(400).json({ ok: false, error: sessionError });
    }

    const ratingRaw = normalizeText(req.body?.rating || '').toLowerCase();
    const adjustments = req.body?.adjustments && typeof req.body.adjustments === 'object' ? req.body.adjustments : {};
    const existingCtx = mcAgentSessions.get(sessionId) || { tickCounter: 0 };
    const currentProfile = sanitizeMcLearningProfile(existingCtx.learning);

    let nextProfile = {
      ...currentProfile,
      updatedAt: Date.now()
    };

    const applyDelta = (key, delta, min, max) => {
      const numericDelta = Number(delta);
      if (!Number.isFinite(numericDelta) || numericDelta === 0) return;
      nextProfile[key] = clamp(Number(nextProfile[key]) + numericDelta, min, max);
    };

    applyDelta('eatCriticalHp', adjustments.eatCriticalHpDelta, 7, 14);
    applyDelta('eatRecoverHp', adjustments.eatRecoverHpDelta, 9, 16);
    applyDelta('eatLockTicks', adjustments.eatLockTicksDelta, 6, 30);
    applyDelta('resourceAggressionRadius', adjustments.resourceAggressionRadiusDelta, 4, 14);

    if (ratingRaw === 'bad' || ratingRaw === 'poor') {
      nextProfile.eatCriticalHp = clamp(Math.max(nextProfile.eatCriticalHp, 10.5), 7, 14);
      nextProfile.eatRecoverHp = clamp(Math.max(nextProfile.eatRecoverHp, nextProfile.eatCriticalHp + 1.5), 9, 16);
      nextProfile.eatLockTicks = clamp(Math.max(nextProfile.eatLockTicks, 12), 6, 30);
    } else if (ratingRaw === 'good' || ratingRaw === 'great') {
      nextProfile.eatLockTicks = clamp(nextProfile.eatLockTicks - 0.5, 6, 30);
    }

    nextProfile = sanitizeMcLearningProfile(nextProfile);

    mcAgentSessions.set(sessionId, {
      ...existingCtx,
      updatedAt: Date.now(),
      learning: nextProfile
    });
    scheduleMcMemorySave();

    return res.json({ ok: true, sessionId, learning: nextProfile });
  } catch (error) {
      // handled above
    return res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

app.listen(PORT, () => {
  console.log(`TurboWarp AI backend running on http://localhost:${PORT}`);
  console.log(`Provider=${PROVIDER} Model=${MODEL}`);
  console.log(`MaxMessageLength=${MAX_MESSAGE_LENGTH}`);
  console.log(`RateLimit=${RATE_LIMIT_MAX_REQUESTS} per ${RATE_LIMIT_WINDOW_MS}ms`);
  console.log(`ApiKeyRequired=${REQUIRE_API_KEY}`);
});
function generateHelpfulDirectAnswer(userMessage, webContext) {
  const text = normalizeText(userMessage).toLowerCase();
  const answers = [
    {
      match: /clutch/i,
      reply: "To clutch in Minecraft PvP: 1) place blocks below you while falling to break your fall, 2) use water buckets to take less damage, 3) eat to restore health mid-combat, or 4) use crystals for positional control. Practice the specific technique."
    },
    {
      match: /pearls?/i,
      reply: "Enderpearls are useful for mobility. Throw them to teleport, use them to escape danger, or chain them for distance. Be careful - they deal fall damage."
    },
    {
      match: /bridging/i,
      reply: "To bridge: 1) sprint-jump forward, 2) place blocks below as you jump, 3) look straight ahead, 4) use WASD to stay centered. Practice in a test world first."
    },
    {
      match: /combat|pvp|fighting|attack/i,
      reply: "PvP tips: keep moving to avoid hits, strafe around opponents, use high ground when possible, combine melee attacks with projectiles, and practice your aim and click speed."
    }
  ];
  
  for (const { match, reply } of answers) {
    if (match.test(text)) {
      return reply;
    }
  }
  return null;
}
