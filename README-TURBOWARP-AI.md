# TurboWarp Real AI Chat Setup

## Completion status (this machine)

- [x] Python model server running on `127.0.0.1:8788`
- [x] Node bridge running on `127.0.0.1:8787`
- [x] Health checks pass (`/health`)
- [x] Chat endpoint works (`/chat-plain`)
- [x] Reset endpoint works (`/reset`)
- [ ] Manual TurboWarp editor clicks (load `.sb3`, add Fetch extension, place blocks)

This project runs a real AI chat flow with:

- TurboWarp project UI (`.sb3`)
- Node bridge API (`turbowarp-ai-backend`)
- Python model API (`model/inference_server.py`)

You **do not** need to upload to Scratch first. TurboWarp can load a local `.sb3` directly.

## 1) Start services (SolasGPT default)

Terminal 1:

```bash
cd /mnt/data/SolasAI/model
curl https://YOUR-RENDER-BACKEND.onrender.com/health
```

### Render Blueprint (faster)

This repo now includes `render.yaml` in `turbowarp-ai-backend/`.

- In Render dashboard: **Blueprints** -> **New Blueprint Instance**.
- Pick this repo and set **Root Directory** to `turbowarp-ai-backend`.
- Render auto-loads `render.yaml`, including `SOLASGPT_URL=https://solasai-database.onrender.com`.
- Set secret `API_KEYS` before first deploy.

## 11) Host the model ser
/mnt/data/SolasAI/.venv/bin/python /mnt/data/SolasAI/model/inference_server.py --port 8788
```

Terminal 2:

```bash
cd /mnt/data/SolasAI/turbowarp-ai-backend
npm install
node server.js
```

Health checks:

```bash
curl http://127.0.0.1:8788/health
curl http://127.0.0.1:8787/health
```

Expected backend response includes:

```json
{"ok":true,"provider":"solasgpt","model":"solasgpt"}
```

## 2) Open your project in TurboWarp

1. Open `https://turbowarp.org/editor` (or TurboWarp Desktop).
2. Click `File -> Load from your computer`.
3. Choose your `.sb3` file (for example `SolasAI.sb3`).
4. Add extension: **Fetch**.

## 3) Variables and lists

Create variables (for all sprites):

- `sessionId`
- `userPrompt`
- `assistantReply`
- `requestBody`
- `responseText`
- `statusText`

Create list:

- `chatLog`

## 4) Init blocks

```
when green flag clicked
set [sessionId v] to (join [user-] (pick random (1000) to (9999)))
delete all of [chatLog v]
set [statusText v] to [Ready]
```

## 5) Send message blocks (recommended plain-text mode)

```
when this sprite clicked
ask [Type your message:] and wait
set [userPrompt v] to (answer)
if <(length of (userPrompt)) = [0]> then
   stop [this script v]
end

add (join [You: ] (userPrompt)) to [chatLog v]
set [statusText v] to [Thinking...]

set [requestBody v] to (join [{"sessionId":"} (join (sessionId) (join [","message":"] (join (userPrompt) ["}])))

set [responseText v] to (fetch POST [https://YOUR-RENDER-BACKEND.onrender.com/chat-plain] with headers [Content-Type: application/json, x-api-key: YOUR_KEY] and body (requestBody))
set [assistantReply v] to (responseText)
add (join [AI: ] (assistantReply)) to [chatLog v]
set [statusText v] to [Ready]
```

## 6) Reset button blocks

```
when this sprite clicked
set [requestBody v] to (join [{"sessionId":"} (join (sessionId) ["}]))
set [responseText v] to (fetch POST [https://YOUR-RENDER-BACKEND.onrender.com/reset] with headers [Content-Type: application/json, x-api-key: YOUR_KEY] and body (requestBody))
delete all of [chatLog v]
set [statusText v] to [Ready]
```

## 7) Optional JSON mode

Use this only if you want metadata (`provider`, `model`, `sessionId`):

Endpoint:

```text
POST https://YOUR-RENDER-BACKEND.onrender.com/chat
```

For simplicity, `/chat-plain` is recommended.

## 8) Provider options

Default `.env` is SolasGPT.

### A) SolasGPT (your local trained model)

In `turbowarp-ai-backend/.env`:

```env
PROVIDER=solasgpt
MODEL=solasgpt
SOLASGPT_URL=http://127.0.0.1:8788
```

### B) Ollama

```env
PROVIDER=ollama
MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### C) OpenAI-compatible

```env
PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=YOUR_KEY
MODEL=gpt-4o-mini
```

Restart `node server.js` after `.env` changes.

## 9) Abuse protection and limits

Set these in `turbowarp-ai-backend/.env` (or Render env vars):

```env
# lock down browser origins in production
ALLOWED_ORIGINS=https://turbowarp.org,https://your-site.example

# require API key for /chat, /chat-plain, /reset
REQUIRE_API_KEY=true
API_KEYS=replace-with-long-random-key

# input validation
MAX_MESSAGE_LENGTH=500
MAX_SESSION_ID_LENGTH=64

# per-IP anti-spam
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=30
```

`/health` now reports active limits.

## 10) Deploy backend on Render

1. Push `turbowarp-ai-backend` to GitHub.
2. In Render, create a **Web Service** from that repo.
3. Build command: `npm install`
4. Start command: `node server.js`
5. Add env vars from section 9 plus your provider vars (`PROVIDER`, `SOLASGPT_URL`, etc.).
6. After deploy, test:

```bash
curl https://YOUR-RENDER-BACKEND.onrender.com/health
```

### Render Blueprint (faster)

This repo now includes `render.yaml` in `turbowarp-ai-backend/`.

- In Render dashboard: **Blueprints** -> **New Blueprint Instance**.
- Pick this repo and set **Root Directory** to `turbowarp-ai-backend`.
- Render auto-loads `render.yaml`, including `SOLASGPT_URL=https://solasai-database.onrender.com`.
- Set secret `API_KEYS` before first deploy.

## 11) Host the model server

If `PROVIDER=solasgpt`, Render backend must reach your Python model URL in `SOLASGPT_URL`.

- Easiest: host model server on another public VM/GPU host and set `SOLASGPT_URL=https://...`
- Or use another provider (`openai` / `ollama`) if you do not want to expose your model host.

## 12) Scratch upload note

- You can upload the `.sb3` to Scratch for sharing.
- Real AI API calls in this setup are intended for TurboWarp.
- Scratch website runtime may block or limit this architecture.