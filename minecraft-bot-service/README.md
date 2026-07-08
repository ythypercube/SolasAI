# Minecraft Bot Service

Independent bot process that joins a server and follows objectives via the existing `/mc-agent` backend.

## Install

- `cd /mnt/data/SolasAI/minecraft-bot-service`
- `npm install`

## Run

- `npm start`

Default API port: `8789`

Web UI:

- Open `http://127.0.0.1:8789`
- Set your Render backend URL in **Backend URL** (example: `https://solasai-backend.onrender.com/mc-agent`)
- Fill server + objective and click **Start Bot**

## API

### `POST /start`

Body example:

```json
{
  "host": "example.org",
  "port": 25565,
  "username": "SolasAIBot",
  "auth": "offline",
  "objective": "general1",
  "backendUrl": "https://solasai-backend.onrender.com/mc-agent"
}
```

### `POST /objective`

```json
{ "objective": "collect wood and craft" }
```

### `POST /stop`

Stops and disconnects the bot.

### `GET /status`

Returns current bot/service state.

## Notes

- For premium servers, use valid bot credentials and supported auth flow.
- This service is separate from the Fabric client and can run while your client is offline.
