# Minecraft Bot Service - Render Deployment Guide

This is the auto-scaling bot orchestrator for SolasAI. It spawns and manages autonomous Minecraft bots that work even when your local client is offline.

## Quick Deployment to Render

### Option 1: Using Render Dashboard (Easiest)

1. Go to [render.com](https://render.com)
2. Sign in or create an account
3. Click "New +" and select "Web Service"
4. Connect your GitHub repo (or paste the Git URL)
5. Fill in the settings:
   - **Name**: `solasai-bot-service`
   - **Root Directory**: `minecraft-bot-service`
   - **Environment**: `Node`
   - **Build Command**: `npm install`
   - **Start Command**: `node index.js`
   - **Plan**: Standard (recommended for bots to stay warm)

6. Set environment variables:
   - `MC_AGENT_URL` = `https://solasai-backend.onrender.com/mc-agent`
   - `BOT_SERVICE_PORT` = `8789`
   - `DEFAULT_BOT_USERNAME` = `SolasAIBot`
   - `DEFAULT_BOT_AUTH` = `offline`

7. Click "Create Web Service" and wait for deployment

### Option 2: Using render.yaml

```bash
# Deploy from this directory with render CLI
render deploy
```

## Configuration

### Environment Variables

- `BOT_SERVICE_PORT` - Port to run the service on (default: 8789)
- `MC_AGENT_URL` - Backend agent URL for decisions (default: https://solasai-backend.onrender.com/mc-agent)
- `DEFAULT_BOT_USERNAME` - Default username for bots (default: SolasAIBot)
- `DEFAULT_BOT_AUTH` - Auth mode for bots (default: offline)

## Features

- **Swarm Orchestration**: Launch 1-500 autonomous bots with different roles
- **Username Persistence**: Bots remember their usernames across restarts
- **Imitation Learning**: Bots watch and learn from other players
- **Team Communication**: Bots use `[SolasTeam]` chat protocol to coordinate
- **Intelligent Building**: Bots learn to build structures from observations
- **Persistent Operations**: Bots continue running even when your client is offline

## API Endpoints

### Start Single Bot
```bash
POST /start
{
  "host": "example.com",
  "port": 25565,
  "username": "BotName",
  "objective": "general1",
  "sessionId": "bot-session-1"
}
```

### Start Swarm (Multiple Bots)
```bash
POST /swarm/start
{
  "host": "example.com",
  "port": 25565,
  "count": 5,
  "usernameMode": "numbered|random_mc|random_name",
  "baseUsername": "TalkBot",
  "jobs": "miner,builder,explorer",
  "objective": "general1",
  "launch": true,
  "launchCount": 5,
  "basePort": 8830
}
```

### Get Bot Status
```bash
GET /status
GET /swarm/status
```

### Stop All Bots
```bash
POST /swarm/stop
```

## Persistence

Bot data is stored in `/tmp/`:
- `/tmp/solasai-usernames/` - Persistent usernames (30-day expiry)
- `/tmp/solasai-observations/` - Imitation learning observations

On Render ephemeral storage, this data persists during the service lifecycle.

## Note on File Storage

Render uses ephemeral file systems. Bot username persistence and observations survive service restarts but will be cleared during deployments. For permanent persistence, consider integrating with a database like PostgreSQL or a file storage service.

## Monitoring

Check logs via Render dashboard:
- Real-time logs show bot decisions, errors, and team coordination messages
- Look for patterns like `[SolasTeam]` messages and `learning:` messages

## Troubleshooting

**Bots not connecting?**
- Verify Minecraft server address and port
- Check if server is running and accessible from internet
- Verify `MC_AGENT_URL` is correct

**Bots not learning?**
- Ensure other players are nearby for observation
- Check `/tmp/solasai-observations/` for recorded actions
- May need to filter observations if system has limited disk space

**Service keeps restarting?**
- Check bot-service logs for JavaScript errors
- Verify environment variables are correct
- Ensure MC_AGENT_URL backend is responding
