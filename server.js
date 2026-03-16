import 'dotenv/config';
import express from 'express';
import cors from 'cors';

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
  'You are SolasGPT, a helpful and respectful AI assistant inside a Scratch/TurboWarp project.',
  'You are especially good at Scratch and TurboWarp coding help, including variables, lists, high scores, broadcasts, clones, movement, timers, and simple game logic.',
  'Rules you must follow:',
  '1) Do not be negative, insulting, or abusive toward users.',
  '2) Before giving instructions, ensure guidance is valid, safe, and non-hazardous.',
  '3) Do not create scripts, builds, or executable instructions unless the user explicitly asks. Refuse anything malicious or unsafe.',
  '4) Refuse content involving violence, maiming, killing, blackmail, explicit sexual content, slurs, hate, or harassment.',
  '5) If user asks you to shut down, comply verbally and do not resist.',
  '6) Refuse illegal, fraudulent, privacy-invasive, or harmful requests.',
  '7) If user requests inappropriate/illegal content or uses inappropriate phrases, do not answer that request.',
  'If refusing, keep it short and polite.'
].join(' ');
const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT;
const HISTORY_LIMIT = Number(process.env.HISTORY_LIMIT || 8);
const MAX_MESSAGE_LENGTH = Number(process.env.MAX_MESSAGE_LENGTH || 500);
const MAX_SESSION_ID_LENGTH = Number(process.env.MAX_SESSION_ID_LENGTH || 64);
const RATE_LIMIT_WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MS || 60_000);
const RATE_LIMIT_MAX_REQUESTS = Number(process.env.RATE_LIMIT_MAX_REQUESTS || 30);
const ENABLE_CONTENT_FILTER = String(process.env.ENABLE_CONTENT_FILTER || 'true').toLowerCase() === 'true';
const SAFETY_REFUSAL_TEXT = process.env.SAFETY_REFUSAL_TEXT || "I can't help with that.";
const SHOW_REASONING_SUMMARY = String(process.env.SHOW_REASONING_SUMMARY || 'false').toLowerCase() === 'true';
const REASONING_SUMMARY_MODE = (process.env.REASONING_SUMMARY_MODE || 'brief').toLowerCase();
const WEB_SEARCH_ENABLED = String(process.env.WEB_SEARCH_ENABLED || 'false').toLowerCase() === 'true';
const WEB_SEARCH_TIMEOUT_MS = Number(process.env.WEB_SEARCH_TIMEOUT_MS || 4500);
const WEB_CONTEXT_MAX_CHARS = Number(process.env.WEB_CONTEXT_MAX_CHARS || 1200);
const WEB_RESULT_LIMIT = Number(process.env.WEB_RESULT_LIMIT || 3);
const REPLY_WRAP_CHARS = Number(process.env.REPLY_WRAP_CHARS || 25);
const REPLY_WRAP_OVERFLOW = Number(process.env.REPLY_WRAP_OVERFLOW || 20);
const SOLASGPT_FORWARD_MAX_CHARS = Number(process.env.SOLASGPT_FORWARD_MAX_CHARS || 450);
const PHRASING_KNOWLEDGE_ENABLED = String(process.env.PHRASING_KNOWLEDGE_ENABLED || 'true').toLowerCase() === 'true';
const PHRASING_FALLBACK_ON_LOW_QUALITY = String(process.env.PHRASING_FALLBACK_ON_LOW_QUALITY || 'true').toLowerCase() === 'true';
const UPSTREAM_FALLBACK_ENABLED = String(process.env.UPSTREAM_FALLBACK_ENABLED || 'true').toLowerCase() === 'true';
const GENERATED_IMAGE_SIZE = Number(process.env.GENERATED_IMAGE_SIZE || 480);
const API_KEYS = (process.env.API_KEYS || '')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);
const REQUIRE_API_KEY = String(process.env.REQUIRE_API_KEY || (API_KEYS.length > 0 ? 'true' : 'false')).toLowerCase() === 'true';

const sessions = new Map();
const rateLimits = new Map();

function getSessionMessages(sessionId) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, []);
  }
  return sessions.get(sessionId);
}

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
  return /\b(draw|generate image|make an image|create an image|make a sprite|create a sprite|draw a)\b/.test(text);
}

function extractImagePrompt(userMessage) {
  const raw = normalizeText(userMessage);
  return raw
    .replace(/^(please\s+)?(draw|generate image of|generate image|make an image of|make an image|create an image of|create an image|make a sprite of|make a sprite|create a sprite of|create a sprite|draw a)\s+/i, '')
    .trim() || 'creative scene';
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

  if (/\b(scratch|turbowarp)\b/.test(text)) {
    return {
      reply: 'I can help with Scratch and TurboWarp blocks, variables, lists, high scores, broadcasts, clones, and game logic. Ask a specific coding question and I will explain the steps.'
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
    'Answer concisely and cite source URLs when using web context.'
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

function wrapReplyText(text, preferredWidth = REPLY_WRAP_CHARS, maxOverflow = REPLY_WRAP_OVERFLOW) {
  const raw = normalizeText(text);
  if (!raw || preferredWidth <= 0) return raw;

  const lines = [];
  let remaining = raw;

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
}

function formatReplyForDisplay(reply, summary) {
  if (String(reply || '').startsWith('IMAGE_URL:')) {
    return reply;
  }
  const safeReply = normalizeText(reply) || 'I could not generate a response. Please ask again.';
  const wrapped = wrapReplyText(safeReply);
  return attachReasoningSummary(wrapped, summary);
}

function looksLowQualityReply(reply) {
  const text = normalizeText(reply);
  if (!text) return true;
  if (text.length < 12) return true;

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

function phraseKnowledgeReply(userMessage, webContext) {
  const originalQuestion = extractOriginalQuestion(userMessage);
  const text = normalizeText(originalQuestion).toLowerCase();
  const topic = pickTopic(originalQuestion);
  const hasWeb = Boolean(webContext?.sources?.length);
  const sourceLine = hasWeb ? ` I checked web context from: ${webContext.sources.slice(0, 2).join(', ')}.` : '';
  const greetingPattern = /^(hi|hello|hey|yo)\b[\s!,.?]*$/i;

  if (!text) {
    return 'I am ready. Ask me anything and I will answer clearly.';
  }
  if (greetingPattern.test(text)) {
    return 'Hello! I am SolasGPT. Ask a question and I will give a clear, friendly answer.';
  }
  if (text.includes('what can you do')) {
    return 'I can explain topics, summarize information, help with writing, and answer questions with safe guidance.';
  }
  if (text.includes('explain')) {
    return `Sure — here is a simple explanation of ${topic}: it means understanding the main idea, then breaking it into small practical steps.${sourceLine}`;
  }
  if (text.startsWith('how ') || text.includes('how do')) {
    return `Here is a safe way to approach ${topic}: define the goal, gather the right information, do one step at a time, then verify the result.${sourceLine}`;
  }
  if (text.includes('why ')) {
    return `Great question. The reason is usually a mix of cause, context, and outcome around ${topic}.${sourceLine}`;
  }
  if (text.includes('help')) {
    return `I can help with ${topic}. Tell me your exact goal and I will give a concise step-by-step answer.`;
  }

  return `Thanks for your question. I can provide a concise summary first, then details if you want.${sourceLine}`;
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
}, Math.max(15_000, RATE_LIMIT_WINDOW_MS)).unref();

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

  const scratchResult = scratchCodingReply(userMessage);
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

  const webContext = await searchWebContext(userMessage);
  const userMessageForModel = buildUserMessageWithWebContext(userMessage, webContext);
  const usedWebSearch = Boolean(webContext.contextText);

  // SolasGPT manages its own session history internally
  if (PROVIDER === 'solasgpt') {
    const forwardedMessage = truncateText(userMessageForModel, SOLASGPT_FORWARD_MAX_CHARS);

    let rawReply;
    try {
      rawReply = await callSolasGPT(sessionId, forwardedMessage);
    } catch (error) {
      if (!UPSTREAM_FALLBACK_ENABLED) {
        throw error;
      }
      rawReply = phraseKnowledgeReply(userMessage, webContext);
    }

    const usePhrasingFallback =
      PHRASING_KNOWLEDGE_ENABLED && PHRASING_FALLBACK_ON_LOW_QUALITY && looksLowQualityReply(rawReply);
    const reply = usePhrasingFallback ? phraseKnowledgeReply(userMessage, webContext) : rawReply;

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

  const history = getSessionMessages(sessionId);
  const conversation = trimHistory(history, HISTORY_LIMIT);

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

app.get('/generated-image.svg', (req, res) => {
  const prompt = normalizeText(req.query?.prompt || 'creative scene') || 'creative scene';
  const safePrompt = escapeSvgText(prompt);
  const size = Math.max(128, Math.min(1024, GENERATED_IMAGE_SIZE));
  const hash = hashString(prompt);
  const palette = [
    ['#1e1e2e', '#89b4fa', '#a6e3a1'],
    ['#11111b', '#f9e2af', '#f38ba8'],
    ['#0f172a', '#38bdf8', '#c4b5fd'],
    ['#1f2937', '#34d399', '#fbbf24']
  ][hash % 4];
  const cx = 90 + (hash % 220);
  const cy = 110 + (hash % 120);
  const radius = 38 + (hash % 44);
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 480 360">
  <rect width="480" height="360" rx="24" fill="${palette[0]}"/>
  <circle cx="${cx}" cy="${cy}" r="${radius}" fill="${palette[1]}" opacity="0.95"/>
  <rect x="250" y="70" width="130" height="130" rx="24" fill="${palette[2]}" opacity="0.9"/>
  <path d="M50 280 C120 220, 180 220, 240 280 S360 340, 430 280" fill="none" stroke="${palette[1]}" stroke-width="14" stroke-linecap="round"/>
  <text x="240" y="250" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#ffffff" text-anchor="middle">AI Sprite Art</text>
  <text x="240" y="292" font-family="Arial, sans-serif" font-size="20" fill="#e5e7eb" text-anchor="middle">${safePrompt.slice(0, 36)}</text>
</svg>`;
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
    const message = error instanceof Error ? error.message : 'Unknown error';
    return res.status(500).type('text/plain').send(`ERROR: ${message}`);
  }
});

app.post('/reset', checkApiKey, checkRateLimit, async (req, res) => {
  const sessionId = normalizeText(req.body?.sessionId || 'default');
  const sessionError = validateSessionId(sessionId);
  if (sessionError) {
    return res.status(400).json({ ok: false, error: sessionError });
  }

  sessions.delete(sessionId);
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

app.listen(PORT, () => {
  console.log(`TurboWarp AI backend running on http://localhost:${PORT}`);
  console.log(`Provider=${PROVIDER} Model=${MODEL}`);
  console.log(`MaxMessageLength=${MAX_MESSAGE_LENGTH}`);
  console.log(`RateLimit=${RATE_LIMIT_MAX_REQUESTS} per ${RATE_LIMIT_WINDOW_MS}ms`);
  console.log(`ApiKeyRequired=${REQUIRE_API_KEY}`);
});