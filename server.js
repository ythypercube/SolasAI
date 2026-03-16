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

async function generateChatReply(sessionId, userMessage) {
  if (ENABLE_CONTENT_FILTER && isUnsafeInput(userMessage)) {
    return {
      reply: getSafetyBlockedReply(userMessage),
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: true
    };
  }

  // SolasGPT manages its own session history internally
  if (PROVIDER === 'solasgpt') {
    const reply = await callSolasGPT(sessionId, userMessage);
    if (ENABLE_CONTENT_FILTER && isUnsafeOutput(reply)) {
      return {
        reply: SAFETY_REFUSAL_TEXT,
        provider: PROVIDER,
        model: MODEL,
        sessionId,
        filtered: true
      };
    }
    return { reply, provider: PROVIDER, model: MODEL, sessionId, filtered: false };
  }

  const history = getSessionMessages(sessionId);
  const conversation = trimHistory(history, HISTORY_LIMIT);

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...conversation,
    { role: 'user', content: userMessage }
  ];

  const reply =
    PROVIDER === 'ollama' ? await callOllama(messages) : await callOpenAICompatible(messages);

  if (ENABLE_CONTENT_FILTER && isUnsafeOutput(reply)) {
    return {
      reply: SAFETY_REFUSAL_TEXT,
      provider: PROVIDER,
      model: MODEL,
      sessionId,
      filtered: true
    };
  }

  const updatedHistory = trimHistory(
    [...conversation, { role: 'user', content: userMessage }, { role: 'assistant', content: reply }],
    HISTORY_LIMIT
  );
  sessions.set(sessionId, updatedHistory);

  return {
    reply,
    provider: PROVIDER,
    model: MODEL,
    sessionId,
    filtered: false
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
      contentFilterEnabled: ENABLE_CONTENT_FILTER
    }
  });
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

    const result = await generateChatReply(sessionId, userMessage);

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

    const result = await generateChatReply(sessionId, userMessage);
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