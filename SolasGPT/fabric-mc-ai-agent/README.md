# SolasAI Fabric Agent (Minecraft 1.21.1)

This mod lets the player hand control to SolasAI in-game.

## Controls

- `Ctrl + Tab`: open objective prompt and start AI control.
- `Esc + Tab`: stop AI control immediately.

## What it does

- Sends your objective + local game state to `POST /mc-agent` on your SolasAI backend.
- Receives an action plan (move, attack, place/use, jump, look delta).
- Applies key presses each tick so AI can navigate, build, mine, and pvp-style move/attack.

## Backend requirement

Run your backend from this repo so `/mc-agent` is available:

- `cd /mnt/data/SolasAI/turbowarp-ai-backend`
- `npm run start:stack`

Default endpoint used by the mod:

- `http://127.0.0.1:8787/mc-agent`

Override endpoint when launching Minecraft:

- `-Dsolasai.backend.endpoint=http://127.0.0.1:8787/mc-agent`

## Build

From this mod folder:

- `./gradlew build`

Output jar:

- `build/libs/solasai-fabric-agent-<version>.jar`

## Notes

- This scaffold targets Fabric with Java 21.
- If you specifically need a different Minecraft patch line, update `minecraft_version`, `yarn_mappings`, and `fabric_version` in `gradle.properties`.
