# Litematic Build Service

External service for SolasAI build workflows.

## Features
- Download/import `.litematic` (or `.zip` containing `.litematic`) from URL
- Save file into your Litematica schematics folder
- Parse required block/material counts from schematic regions
- Generate gather/build plan from current inventory

## Quick start
```bash
cd /mnt/data/SolasAI/litematic-build-service
npm install
npm start
```

Service runs on `http://127.0.0.1:8790` by default.

## Endpoints
- `GET /health`
- `POST /import-litematic` body: `{ "url": "...", "name": "optional" }`
- `POST /scan-litematic` body: `{ "fileName": "house.litematic" }`
- `POST /build-plan` body: `{ "fileName": "house.litematic", "inventory": { "minecraft:stone": 128 } }`

## Notes
- Place Litematica mod in your Minecraft `mods` folder on the client.
- Set `LITEMATICS_DIR` to your real schematics folder if needed.
- This service plans and parses; placement/execution remains in your Minecraft agent runtime.
