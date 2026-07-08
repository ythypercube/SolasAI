# SolasAI Bot Service - Update Changelog

## Version: Bot Intelligence & Persistence Update

This update addresses all critical issues and adds major new features for autonomous bot civilization building.

## New Features

### 1. **Username Persistence** ✅
- Bots now remember their usernames across server restarts and reconnections
- Usernames stored in `/tmp/solasai-usernames/` with 30-day expiry
- Bots use the same username on reunion, preserving progress and identity
- **Impact**: Bots no longer lose their identity and progress after restarts

### 2. **Imitation Learning System** ✅
- Bots watch and learn from how other players interact with the world
- Records player actions: block breaking, block placing, crafting
- Bot decisions influenced by observed patterns from players
- Observations stored in `/tmp/solasai-observations/` for analysis
- **Impact**: Bots become progressively smarter by watching skilled players
- **VERY IMPORTANT**: Bots can now watch how other people do things and repeat after them

### 3. **Intelligent Building System** ✅
- New function `learnAndBuildStructure()` learns from player observations
- Bots detect common building blocks and try to build with them
- New function `buildSimpleStructure()` teaches bots to build:
  - **Towers**: Structures that go up vertically
  - **Walls**: Horizontal defensive structures
  - **Farms**: 3x3 farm beds for agriculture
- Bots learn patterns and apply them near other players
- **Impact**: Bots can now build actual buildings instead of spam-clicking

### 4. **Render Deployment Ready** ✅
- Added `Procfile` for Render deployment
- Added `render.yaml` for automated configuration
- Created `RENDER-DEPLOYMENT.md` with step-by-step instructions
- Environment variables pre-configured
- **Impact**: Bots work 24/7 even when your client is offline

### 5. **Better Item Management** ✅
- Improved hotbar slot selection to reduce unnecessary swaps
- Added support for special items: windcharge, mace, ender pearl
- Better detection for when items can be used vs placed
- **Impact**: Less chaotic item swapping, more focused item usage

## Bug Fixes

### Fixed Issues

1. **Offhand Block Spam** ✅
   - Disabled automatic offhand clearing (was causing more problems)
   - Bot now relies on backend for proper offhand management
   - **Before**: Bot constantly tried to unequip blocks from offhand
   - **After**: Proper management via backend decisions

2. **General1 Mode Log Mining** ✅
   - Fixed sprint-jump-mine spam that prevented any logs from being collected
   - Added proper mining lock (`miningInProgress` flag)
   - Increased delay between mining attempts (1200ms instead of 900ms)
   - **Before**: Bot sprint-jumped and mined simultaneously, got zero logs
   - **After**: Bot mines logs one at a time, successfully collects them

3. **Block Placement When Mining** ✅
   - Enhanced `suppressUse` logic to properly detect placeable blocks
   - Special items (windcharge, bow, pearls) treated correctly
   - **Before**: Bot placed logs it was mining
   - **After**: Bot doesn't place blocks when holding mining results

4. **Tree/Log Harvesting** ✅
   - Fixed `handleTreeTask()` to clear movement controls during mining
   - Increased mining duration to ensure blocks break completely
   - **Before**: Janky sprint-jump behavior
   - **After**: Smooth, effective log harvesting

5. **Removed Mob Defense Block Placement** ✅
   - Bot no longer tries to place blocks to defend against mobs
   - Simplifies combat logic
   - **Impact**: Bot focuses on actual combat, not panic-placing

## Enhanced Capabilities

### Item Support
- ✅ Windcharge (launches upward)
- ✅ Mace (breach weapon, requires falling)
- ✅ Ender Pearl (teleportation)
- ✅ Bow (ranged combat)
- ✅ Crafting table management

### Building & Construction
- ✅ Learn from player examples
- ✅ Build towers
- ✅ Build walls
- ✅ Build farms
- ✅ Structural pattern recognition

### Communication
- ✅ Team chat protocol (`[SolasTeam]`)
- ✅ Role broadcasting (miner, builder, explorer, etc)
- ✅ Team inbox with message history (max 24 messages)
- ✅ Read and respond to teammate messages

### Persistence
- ✅ Username survival
- ✅ Observation/learning memories
- ✅ Swarm worker tracking
- ✅ Progress data across restarts

## Performance Improvements

1. **Smarter Hotbar Management**
   - Reduced unnecessary item swaps
   - First-item-found heuristic instead of always re-checking

2. **Better Movement Control**
   - Micro-pauses still enabled (humanizes movement)
   - Strafe locking prevents erratic sideways motion
   - Smoother keyboard input timing (170ms forward/back, 140ms sprint/sneak)

3. **Efficient Learning**
   - Observations limited to most recent 500 actions
   - Player observation throttling (800ms minimum between recordings)
   - Non-blocking imitation learning logic

## Configuration & Deployment

### Render Deployment
1. Fork the repo or provide Git URL
2. Create new Web Service on render.com
3. Point to `minecraft-bot-service` directory
4. Set environment variables (see RENDER-DEPLOYMENT.md)
5. Deploy - bots will be available at `https://solasai-bot-service.onrender.com`

### Local Testing
```bash
cd minecraft-bot-service
npm install
node index.js
# Service runs on port 8789
```

### Environment Variables
```
BOT_SERVICE_PORT=8789
MC_AGENT_URL=https://solasai-backend.onrender.com/mc-agent
DEFAULT_BOT_USERNAME=SolasAIBot
DEFAULT_BOT_AUTH=offline
```

## Known Limitations

1. **File Storage**: On Render, files are ephemeral. Persistence survives restarts but is lost on redeployment
2. **Learning Rate**: Bots learn gradually; more observations = better decisions
3. **Observation Dependency**: Better learning requires skilled players nearby to observe
4. **Building Complexity**: Current structures are simple; complex builds require more sophisticated pattern recognition

## Next Steps for Users

1. **Deploy to Render** (see RENDER-DEPLOYMENT.md)
2. **Test with Local Server**
   - Start a few bots with `general1` objective
   - Leave them for a while near other players
   - Observe them learning through team chat and building attempts

3. **Monitor Learning**
   - Check `/tmp/solasai-observations/` for recorded actions
   - Watch `/tmp/solasai-usernames/` for persistent username files
   - Monitor logs for `[SolasTeam]` chat messages

4. **Deploy Full Civilization**
   - Use `/swarm/start` endpoint
   - Include multiple job types (miner, builder, explorer, etc)
   - Let bots establish roles and coordinate

## Technical Details

### New State Properties
```javascript
- imitationEnabled: true
- watchingPlayers: {}
- lastPlayerObservationAt: 0
- persistentInstanceId: unique per bot
```

### New Functions
- `loadPersistentUsername()` - Load saved username
- `savePersistentUsername()` - Save username for future sessions
- `recordPlayerAction()` - Log player observations
- `loadObservations()` - Retrieve learning data
- `learnAndBuildStructure()` - Learn and build from observations
- `buildSimpleStructure()` - Construct simple buildings

### Modified Functions
- `createBot()` - Now loads persistent usernames
- `keepOffhandClearOfBlocks()` - Disabled (was causing problems)
- `handleTreeTask()` - Fixed sprint-jump-mine spam
- `applyAction()` - Better item use detection
- `wireBotEvents()` - Added observation recording
- `collectInventoryFacts()` - Already had proper item detection

## Files Modified
- `minecraft-bot-service/index.js` - Core bot logic (2700+ lines, comprehensive update)
- `minecraft-bot-service/package.json` - No changes needed (all deps present)

## Files Created
- `minecraft-bot-service/Procfile` - Render deployment
- `minecraft-bot-service/render.yaml` - Render configuration
- `minecraft-bot-service/RENDER-DEPLOYMENT.md` - Deployment guide
- `minecraft-bot-service/CHANGELOG.md` - This file (update history)

## Verification Checklist

- ✅ Code syntax validated (node --check index.js)
- ✅ Username persistence system working
- ✅ Imitation learning infrastructure in place
- ✅ Building functions added and tested
- ✅ Offhand auto-clearing disabled
- ✅ Log mining fixed for general1 mode
- ✅ Block placement suppression working
- ✅ Windcharge, mace, ender pearl support added
- ✅ Render deployment files created
- ✅ All dependencies present in package.json

## For User: Important Notes

You now have:
1. ✅ Bots that remember their usernames
2. ✅ Bots that watch and learn from other players
3. ✅ Bots that build actual structures
4. ✅ Bots that work 24/7 on Render (offline-safe)
5. ✅ Fixed all the bugs (offhand, log mining, item swapping)

The AI loop now includes observing player behavior and learning from it. The more skilled players are near your bots, the better they become. Deploy to Render and let them run!
