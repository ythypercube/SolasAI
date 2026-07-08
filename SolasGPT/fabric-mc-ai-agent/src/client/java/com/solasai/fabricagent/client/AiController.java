package com.solasai.fabricagent.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.ChatScreen;
import net.minecraft.client.gui.screen.DeathScreen;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.GameOptions;
import net.minecraft.block.Block;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.passive.GolemEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.entity.projectile.ProjectileEntity;
import net.minecraft.item.BlockItem;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.screen.slot.SlotActionType;
import net.minecraft.screen.slot.Slot;
import net.minecraft.text.Text;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.screen.PlayerScreenHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AiController {
    private static final AiController INSTANCE = new AiController();
    private static final Logger LOGGER = LoggerFactory.getLogger("SolasAIAgent");

    private final AiServiceClient aiServiceClient = new AiServiceClient();

    private boolean active;
    private String objective = "";
    private boolean requestInFlight;
    private long localTickCounter = 0;
    private long nextOffhandCheckTick = 0;
    private long nextCombatDebugHudTick = 0;
    private long nextHotbarOrganizeTick = 0;
    private long nextArmorCheckTick = 0;
    private long offhandEmptySinceTick = -1;
    private int offhandPendingHotbarSlot = -1; // temp hotbar slot during 2-step offhand equip
    private boolean offhandPendingNeedsSelect = false;
    private long lastPearlUseTick = -1;
    private long nextDecisionTick;
    private long nextFallRescueScanTick = 0;
    private long nextCraftActionTick = 0;
    private long nextTrashDropTick = 0;
    private long evadeStrafeUntilTick = 0;
    private long evadeCounterUntilTick = 0;
    private boolean evadeStrafeLeft = false;
    private long lastChatForceCloseTick = -1;
    private int forcedConsumeTicksRemaining = 0;
    private int consecutiveBackendErrors = 0;
    private long lastErrorNoticeTick = -1;
    private AgentAction currentAction = AgentAction.idle();
    private int actionTicksRemaining;
    private String sessionId = "default";

    // Debug overlay
    private boolean debugOverlayVisible = false;
    private String lastNote = "";
    private String lastMode = "general";
    private String lastAnnouncedTask = "";
    private long lastTaskAnnounceTick = -1;

    // Eye-of-ender throw triangulation
    private int eyeThrowCount = 0;
    private double eyeThrow1X = 0, eyeThrow1Z = 0, eyeThrow2X = 0, eyeThrow2Z = 0;
    private float eyeThrow1Yaw = 0, eyeThrow2Yaw = 0;
    private int strongholdEstX = 0, strongholdEstZ = 0;
    private boolean strongholdTriangulated = false;
    private long lastEyeRecordTick = -1;
    private int respawnRetryTicks = 0;
    private long nextCreativeGrantTick = 0;
    private long lastCreativeGrantTick = -1;
    private String lastCreativeGrantSignature = "";
    private long nextIntentChatTick = 0;
    private long nextAttackProbeTick = 0;
    private long nextUseProbeTick = 0;

    private static final Pattern CREATIVE_ITEM_ID_PATTERN = Pattern.compile("\\b(minecraft:[a-z0-9_]+)\\b");
    private static final Pattern CREATIVE_COUNT_PATTERN = Pattern.compile("\\b(\\d{1,3})\\b");
    private static final Map<String, String> CREATIVE_ITEM_ALIASES = createCreativeItemAliases();
    private static final float FORCE_SHIELD_MIN_HEALTH = 8.0f;

    public static AiController getInstance() {
        return INSTANCE;
    }

    public boolean isActive() {
        return active;
    }

    public long getLastPearlUseTick() {
        return lastPearlUseTick;
    }

    public boolean isDebugOverlayVisible() {
        return debugOverlayVisible;
    }

    public void toggleDebugOverlay() {
        debugOverlayVisible = !debugOverlayVisible;
    }

    public String getLastNote() {
        return lastNote;
    }

    public String getLastMode() {
        return lastMode;
    }

    public String getObjective() {
        return objective;
    }

    public String getCurrentTask() {
        if (lastNote != null && !lastNote.isBlank()) {
            String note = lastNote.trim();
            String lower = note.toLowerCase();
            boolean generic = lower.contains("general mode: exploring objective path")
                    || lower.contains("holding still")
                    || lower.equals("(waiting...)");
            if (!generic) {
                return note;
            }
        }
        if (objective != null && !objective.isBlank()) {
            return "Working on objective: " + objective + " (mode=" + lastMode + ")";
        }
        return "Idle";
    }

    public int getStrongholdEstX() {
        return strongholdEstX;
    }

    public int getStrongholdEstZ() {
        return strongholdEstZ;
    }

    public boolean isStrongholdTriangulated() {
        return strongholdTriangulated;
    }

    /**
     * Called from GameStateSnapshot when an eye-of-ender entity is observed
     * travelling near the player.  After two distinct observations the stronghold
     * coordinates are triangulated from the two bearing lines.
     *
     * @param ex  world X of the eye entity
     * @param ez  world Z of the eye entity
     * @param travelYaw  yaw angle (degrees) of the eye's velocity direction
     */
    public void recordEyeThrow(double ex, double ez, float travelYaw) {
        if (lastEyeRecordTick >= 0 && (localTickCounter - lastEyeRecordTick) < 30) {
            return; // Debounce: ignore if same throw still tracked
        }
        lastEyeRecordTick = localTickCounter;

        if (eyeThrowCount == 0) {
            eyeThrow1X = ex; eyeThrow1Z = ez; eyeThrow1Yaw = travelYaw;
            eyeThrowCount = 1;
            strongholdTriangulated = false;
        } else {
            // Skip if too close to first throw origin (< 15 blocks)  
            double dist = Math.sqrt(Math.pow(ex - eyeThrow1X, 2) + Math.pow(ez - eyeThrow1Z, 2));
            if (eyeThrowCount == 1 && dist < 15) {
                // Update first throw with better position
                eyeThrow1X = ex; eyeThrow1Z = ez; eyeThrow1Yaw = travelYaw;
                return;
            }
            eyeThrow2X = ex; eyeThrow2Z = ez; eyeThrow2Yaw = travelYaw;
            eyeThrowCount = 2;
            triangulateStronghold();
        }
    }

    public void resetEyeThrows() {
        eyeThrowCount = 0;
        strongholdTriangulated = false;
        lastEyeRecordTick = -1;
    }

    private void triangulateStronghold() {
        double rad1 = Math.toRadians(eyeThrow1Yaw);
        double rad2 = Math.toRadians(eyeThrow2Yaw);
        double dx1 = -Math.sin(rad1);
        double dz1 =  Math.cos(rad1);
        double dx2 = -Math.sin(rad2);
        double dz2 =  Math.cos(rad2);
        // Solve: p1 + t*d1 = p2 + s*d2  (2D, ignoring Y)
        // det = dx1*dz2 - dz1*dx2
        double det = dx1 * dz2 - dz1 * dx2;
        if (Math.abs(det) < 1e-4) {
            strongholdTriangulated = false;
            return;
        }
        double t = ((eyeThrow2X - eyeThrow1X) * dz2 - (eyeThrow2Z - eyeThrow1Z) * dx2) / det;
        strongholdEstX = (int) Math.round(eyeThrow1X + t * dx1);
        strongholdEstZ = (int) Math.round(eyeThrow1Z + t * dz1);
        strongholdTriangulated = true;
    }

    public void start(MinecraftClient client, String objectiveText) {
        if (objectiveText == null || objectiveText.isBlank()) {
            return;
        }
        String trimmedObjective = objectiveText.trim();
        boolean objectiveTruncated = false;
        if (trimmedObjective.length() > 400) {
            trimmedObjective = trimmedObjective.substring(0, 400);
            objectiveTruncated = true;
        }
        this.objective = trimmedObjective;
        this.active = true;
        this.requestInFlight = false;
        this.currentAction = AgentAction.idle();
        this.actionTicksRemaining = 0;
        this.nextDecisionTick = 0;
        this.nextOffhandCheckTick = 0;
        this.nextCombatDebugHudTick = 0;
        this.nextHotbarOrganizeTick = 0;
        this.nextArmorCheckTick = 0;
        this.offhandEmptySinceTick = -1;
        this.offhandPendingHotbarSlot = -1;
        this.offhandPendingNeedsSelect = false;
        this.lastPearlUseTick = -1;
        this.nextCraftActionTick = 0;
        this.nextTrashDropTick = 0;
        this.evadeStrafeUntilTick = 0;
        this.evadeCounterUntilTick = 0;
        this.evadeStrafeLeft = false;
        this.forcedConsumeTicksRemaining = 0;
        this.consecutiveBackendErrors = 0;
        this.lastErrorNoticeTick = -1;
        this.lastNote = "";
        this.lastMode = "general";
        this.lastAnnouncedTask = "";
        this.lastTaskAnnounceTick = -1;
        this.respawnRetryTicks = 0;
        this.nextCreativeGrantTick = 0;
        this.lastCreativeGrantTick = -1;
        this.lastCreativeGrantSignature = "";
        this.nextIntentChatTick = 0;
        this.nextAttackProbeTick = 0;
        this.nextUseProbeTick = 0;
        this.sessionId = "mc-" + UUID.randomUUID().toString().replace("-", "").substring(0, 20);
        if (client.player != null) {
            client.player.sendMessage(Text.literal("SolasAI control enabled: " + this.objective), false);
            if (objectiveTruncated) {
                client.player.sendMessage(Text.literal("SolasAI note: objective trimmed to 400 chars for backend compatibility."), false);
            }
        }
    }

    public void stop(MinecraftClient client, String reason) {
        this.active = false;
        this.requestInFlight = false;
        this.currentAction = AgentAction.idle();
        this.actionTicksRemaining = 0;
        this.nextOffhandCheckTick = 0;
        this.nextCombatDebugHudTick = 0;
        this.nextHotbarOrganizeTick = 0;
        this.nextArmorCheckTick = 0;
        this.offhandEmptySinceTick = -1;
        this.offhandPendingHotbarSlot = -1;
        this.offhandPendingNeedsSelect = false;
        this.lastPearlUseTick = -1;
        this.nextCraftActionTick = 0;
        this.nextTrashDropTick = 0;
        this.evadeStrafeUntilTick = 0;
        this.evadeCounterUntilTick = 0;
        this.evadeStrafeLeft = false;
        this.forcedConsumeTicksRemaining = 0;
        this.consecutiveBackendErrors = 0;
        this.lastErrorNoticeTick = -1;
        this.lastNote = "";
        this.lastMode = "general";
        this.lastAnnouncedTask = "";
        this.lastTaskAnnounceTick = -1;
        this.respawnRetryTicks = 0;
        this.nextCreativeGrantTick = 0;
        this.lastCreativeGrantTick = -1;
        this.lastCreativeGrantSignature = "";
        this.nextIntentChatTick = 0;
        this.nextAttackProbeTick = 0;
        this.nextUseProbeTick = 0;
        resetKeys(client);
        if (client.player != null && reason != null && !reason.isBlank()) {
            client.player.sendMessage(Text.literal(reason), false);
        }
    }

    public void tick(MinecraftClient client) {
        localTickCounter++;
        handleAutoRespawn(client);
        if (!active || client.player == null || client.world == null) {
            return;
        }
        if (client.currentScreen instanceof DeathScreen || client.player.isDead()) {
            // Don't keep applying clutch/use actions while dead screen is up.
            return;
        }

        if (client.currentScreen instanceof ChatScreen) {
            client.setScreen(null);
            if (lastChatForceCloseTick < 0 || (localTickCounter - lastChatForceCloseTick) >= 80) {
                client.player.sendMessage(Text.literal("SolasAI: closed chat so item use/combat actions keep working."), false);
                lastChatForceCloseTick = localTickCounter;
            }
        }

        boolean craftIntentActive = shouldRunLocalCrafting(client.player);
        if (!craftIntentActive && shouldMaintainCombatLoadout()) {
            ensureCombatHotbarLoadout(client, client.player);
        }
        ensureBestArmorLoadout(client, client.player);
        applyLocalTrashCleanup(client, client.player);
        applyCurrentAction(client);
        applyLocalGeneralTreeAssist(client);
        if (craftIntentActive) {
            applyLocalCrafting(client, client.player);
        }
        applyCreativeItemAccess(client, client.player);
        applyLocalAimAssist(client);
        applyLocalAerialMacePursuit(client);
        applyLocalPathSafety(client);
        applyLocalProjectileEvasion(client);
        applyLocalEmergencyEat(client);
        applyLocalWaterRecovery(client);
        applyLocalFallRecovery(client);
        applyCombatDebugHud(client, client.player);
        maybeAnnounceTask(client);

        if (actionTicksRemaining > 0) {
            actionTicksRemaining--;
        }

        if (requestInFlight || localTickCounter < nextDecisionTick) {
            return;
        }

        requestInFlight = true;
        GameStateSnapshot snapshot = GameStateSnapshot.capture(client);
        aiServiceClient.requestDecision(sessionId, objective, snapshot)
                .whenComplete((decision, error) -> client.execute(() -> {
                    requestInFlight = false;
                    nextDecisionTick = localTickCounter + 2;

                    if (!active) {
                        return;
                    }

                    if (error != null) {
                        consecutiveBackendErrors++;
                        LOGGER.error("SolasAI backend request failed (attempt {}), sessionId={}.", consecutiveBackendErrors, sessionId, error);
                        currentAction = AgentAction.idle();
                        actionTicksRemaining = 3;
                        nextDecisionTick = localTickCounter + Math.min(40, 6 + (consecutiveBackendErrors * 4L));

                        if (client.player != null && (lastErrorNoticeTick < 0 || (localTickCounter - lastErrorNoticeTick) >= 40)) {
                            String reason = rootCauseMessage(error);
                            client.player.sendMessage(Text.literal("SolasAI backend error (retrying): " + reason), false);
                            lastErrorNoticeTick = localTickCounter;
                        }

                        if (consecutiveBackendErrors >= 8) {
                            stop(client, "SolasAI stopped after repeated backend errors. Check backend and try again.");
                        }
                        return;
                    }

                    consecutiveBackendErrors = 0;
                    if (decision != null) {
                        currentAction = decision.action() != null ? decision.action() : AgentAction.idle();
                        lastNote = decision.note() != null ? decision.note() : "";
                        lastMode = decision.mode() != null ? decision.mode() : "general";
                        maybeAnnounceTask(client);
                        String lowerNote = lastNote.toLowerCase();
                        if (lowerNote.contains("objective complete") || lowerNote.contains("bed destroyed")) {
                            stop(client, "SolasAI objective complete. Control disabled.");
                            return;
                        }
                    } else {
                        currentAction = AgentAction.idle();
                    }
                    actionTicksRemaining = Math.max(2, currentAction.durationTicks());
                }));
    }

    private void applyCreativeItemAccess(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null) {
            return;
        }
        if (localTickCounter < nextCreativeGrantTick) {
            return;
        }
        if (!player.getAbilities().creativeMode) {
            return;
        }

        CreativeGiveRequest request = parseCreativeGiveRequest(objective);
        if (request == null) {
            return;
        }

        int currentCount = countInventoryItem(player.getInventory(), request.itemId());
        if (currentCount >= request.count()) {
            nextCreativeGrantTick = localTickCounter + 40;
            return;
        }

        if (request.signature().equals(lastCreativeGrantSignature)
            && lastCreativeGrantTick >= 0
            && (localTickCounter - lastCreativeGrantTick) < 120) {
            return;
        }

        String command = "give @s " + request.itemId() + " " + request.count();
        try {
            if (player.networkHandler != null) {
                player.networkHandler.sendChatCommand(command);
            }
        } catch (Throwable commandError) {
            if (player.networkHandler != null) {
                player.networkHandler.sendChatMessage("/" + command);
            }
        }

        lastCreativeGrantSignature = request.signature();
        lastCreativeGrantTick = localTickCounter;
        nextCreativeGrantTick = localTickCounter + 60;
        player.sendMessage(Text.literal("SolasAI creative access: /" + command), false);
    }

    private CreativeGiveRequest parseCreativeGiveRequest(String objectiveText) {
        if (objectiveText == null || objectiveText.isBlank()) {
            return null;
        }
        String text = objectiveText.trim();
        String lower = text.toLowerCase();

        boolean creativeRequested = lower.contains("creative")
                || lower.contains("from creative")
                || lower.contains("/give")
                || lower.contains("give me")
                || lower.contains("get me")
                || lower.contains("spawn in")
                || lower.contains("command block")
                || lower.contains("barrier")
                || lower.contains("structure block")
                || lower.contains("debug stick")
                || lower.contains("bedrock");
        if (!creativeRequested) {
            return null;
        }

        String itemId = null;
        Matcher idMatcher = CREATIVE_ITEM_ID_PATTERN.matcher(lower);
        if (idMatcher.find()) {
            itemId = idMatcher.group(1);
        } else {
            for (Map.Entry<String, String> entry : CREATIVE_ITEM_ALIASES.entrySet()) {
                if (lower.contains(entry.getKey())) {
                    itemId = entry.getValue();
                    break;
                }
            }
        }
        if (itemId == null || itemId.isBlank()) {
            return null;
        }

        int count = lower.contains("stack") ? 64 : 1;
        Matcher countMatcher = CREATIVE_COUNT_PATTERN.matcher(lower);
        while (countMatcher.find()) {
            int parsed = safeParseInt(countMatcher.group(1), -1);
            if (parsed >= 1) {
                count = parsed;
                break;
            }
        }
        count = Math.max(1, Math.min(64, count));

        String signature = itemId + "#" + count;
        return new CreativeGiveRequest(itemId, count, signature);
    }

    private static int safeParseInt(String raw, int fallback) {
        try {
            return Integer.parseInt(raw);
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static Map<String, String> createCreativeItemAliases() {
        Map<String, String> aliases = new LinkedHashMap<>();
        aliases.put("enchanted golden apple", "minecraft:enchanted_golden_apple");
        aliases.put("golden apple", "minecraft:golden_apple");
        aliases.put("netherite sword", "minecraft:netherite_sword");
        aliases.put("netherite pickaxe", "minecraft:netherite_pickaxe");
        aliases.put("totem", "minecraft:totem_of_undying");
        aliases.put("end crystal", "minecraft:end_crystal");
        aliases.put("wind charge", "minecraft:wind_charge");
        aliases.put("command block", "minecraft:command_block");
        aliases.put("structure block", "minecraft:structure_block");
        aliases.put("debug stick", "minecraft:debug_stick");
        aliases.put("barrier", "minecraft:barrier");
        aliases.put("bedrock", "minecraft:bedrock");
        aliases.put("elytra", "minecraft:elytra");
        aliases.put("obsidian", "minecraft:obsidian");
        aliases.put("diamond", "minecraft:diamond");
        aliases.put("iron", "minecraft:iron_ingot");
        aliases.put("mace", "minecraft:mace");
        return aliases;
    }

    private record CreativeGiveRequest(String itemId, int count, String signature) {}

    private String rootCauseMessage(Throwable throwable) {
        if (throwable == null) {
            return "Unknown error";
        }
        Throwable cursor = throwable;
        while (cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        String msg = cursor.getMessage();
        if (msg == null || msg.isBlank()) {
            msg = cursor.getClass().getSimpleName();
        }
        if (msg.length() > 140) {
            msg = msg.substring(0, 140);
        }
        return msg;
    }

    private void applyCurrentAction(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) return;

        resetKeys(client);
        ensurePreferredOffhand(client, player);
        trackPearlUsage(player);

        GameOptions options = client.options;

        int slot = currentAction.hotbarSlot();
        if (slot >= 0 && slot < 9) {
            player.getInventory().setSelectedSlot(slot);
        } else if (isIdleAction(currentAction)) {
            int hotbarTotemSlot = findHotbarTotemSlot(player.getInventory());
            if (hotbarTotemSlot >= 0) {
                player.getInventory().setSelectedSlot(hotbarTotemSlot);
            }
        }

        boolean kbRecovery = isRecoveringFromKnockback(player);
        options.forwardKey.setPressed(!kbRecovery && currentAction.forward());
        options.backKey.setPressed(!kbRecovery && currentAction.back());
        options.leftKey.setPressed(!kbRecovery && currentAction.left());
        options.rightKey.setPressed(!kbRecovery && currentAction.right());

        boolean hasExplicitDirectionalInput = !kbRecovery && (
            currentAction.forward()
            || currentAction.back()
            || currentAction.left()
            || currentAction.right()
        );
        if (!kbRecovery && !hasExplicitDirectionalInput) {
            applyMoveAngle(options, currentAction.moveAngle());
        }

        options.jumpKey.setPressed(currentAction.jump());
        options.sprintKey.setPressed(currentAction.sprint());
        options.sneakKey.setPressed(currentAction.sneak());
        options.attackKey.setPressed(currentAction.attack());
        options.useKey.setPressed(currentAction.use());

        applyLocalMeleeCommit(client, player, options);

        applyLocalAttackEvasion(client, player, options);

        if (shouldHumanizeMovement()) {
            applyHumanizedInputNoise(options);
        }

        ItemStack heldMain = player.getMainHandStack();
        boolean heldConsumable = isLikelyConsumable(heldMain);
        if (currentAction.use() && heldConsumable && canConsumeNow(player, heldMain)) {
            forcedConsumeTicksRemaining = Math.max(forcedConsumeTicksRemaining, 40);
        }
        if (forcedConsumeTicksRemaining > 0) {
            if (heldConsumable && canConsumeNow(player, heldMain)) {
                options.useKey.setPressed(true);
                options.attackKey.setPressed(false);
            } else {
                forcedConsumeTicksRemaining = 0;
            }
        }

        if (currentAction.yawDelta() != 0f || currentAction.pitchDelta() != 0f) {
            float newYaw = player.getYaw() + currentAction.yawDelta();
            float newPitch = MathHelper.clamp(player.getPitch() + currentAction.pitchDelta(), -90f, 90f);
            player.setYaw(newYaw);
            player.setHeadYaw(newYaw);
            player.setBodyYaw(newYaw);
            player.setPitch(newPitch);
        }

        if (currentAction.use() && heldMain != null && !heldMain.isEmpty()) {
            String heldId = Registries.ITEM.getId(heldMain.getItem()).toString();
            String mode = lastMode == null ? "" : lastMode.toLowerCase();
            if (heldId.endsWith("wind_charge") && ("pvp".equals(mode) || "crystal".equals(mode))) {
                float forcedPitch = 87f;
                player.setPitch(forcedPitch);
            }
        }

        // Directly invoke the interaction manager so attack/use actually fire,
        // since setPressed() alone does not trigger Minecraft's interaction pipeline.
        if (currentAction.attack()) {
            if (client.crosshairTarget instanceof EntityHitResult entityHit
                    && entityHit.getEntity() != null
                    && player.getAttackCooldownProgress(0f) >= 1.0f) {
                client.interactionManager.attackEntity(player, entityHit.getEntity());
                player.swingHand(Hand.MAIN_HAND);
                maybeSendActionProbe(client, "LEFTCLICK entity");
            } else if (!isCombatMode() && client.crosshairTarget instanceof BlockHitResult bhr) {
                client.interactionManager.attackBlock(bhr.getBlockPos(), bhr.getSide());
                player.swingHand(Hand.MAIN_HAND);
                maybeSendActionProbe(client, "LEFTCLICK block");
            } else {
                maybeSendActionProbe(client, "LEFTCLICK attempted");
            }
        }
        if (currentAction.use()) {
            Hand useHand = getPreferredUseHand(player);
            boolean shouldForceItemUse = isLikelyConsumable(heldMain) || forcedConsumeTicksRemaining > 0;
            player.setCurrentHand(useHand);
            if (shouldForceItemUse) {
                client.interactionManager.interactItem(player, useHand);
            } else if (client.targetedEntity != null) {
                client.interactionManager.interactEntity(
                        player, client.targetedEntity, useHand);
            } else if (client.crosshairTarget instanceof BlockHitResult bhr) {
                client.interactionManager.interactBlock(player, useHand, bhr);
            } else {
                client.interactionManager.interactItem(player, useHand);
            }
            maybeSendActionProbe(client, "RIGHTCLICK " + (useHand == Hand.OFF_HAND ? "offhand" : "mainhand"));
        }

        if (forcedConsumeTicksRemaining > 0) {
            forcedConsumeTicksRemaining--;
        }

        player.setSprinting(currentAction.sprint());
        player.setSneaking(currentAction.sneak());
        player.setJumping(currentAction.jump());

        if (currentAction.jump() && player.isOnGround()) {
            player.jump();
        }
    }

    private void applyLocalMeleeCommit(MinecraftClient client, ClientPlayerEntity player, GameOptions options) {
        if (client == null || player == null || options == null || client.interactionManager == null) {
            return;
        }
        if (!isCombatIntentActive(client, player)) {
            return;
        }
        if (currentAction != null && currentAction.use()) {
            return;
        }

        PlayerEntity enemy = findNearestEnemyPlayer(client, player);
        if (enemy == null || enemy.isDead() || enemy.isSpectator()) {
            return;
        }

        double dist = player.distanceTo(enemy);
        if (dist > 10.0) {
            return;
        }

        if (isRecoveringFromKnockback(player)) {
            return;
        }

        if (dist > 2.6) {
            options.forwardKey.setPressed(true);
            options.sprintKey.setPressed(true);
            options.backKey.setPressed(false);
        }

        if (dist <= 3.8 && player.getAttackCooldownProgress(0f) >= 0.84f) {
            options.attackKey.setPressed(true);
            client.interactionManager.attackEntity(player, enemy);
            player.swingHand(Hand.MAIN_HAND);
        }
    }

    private Hand getPreferredUseHand(ClientPlayerEntity player) {
        if (player == null) {
            return Hand.MAIN_HAND;
        }
        ItemStack offhand = player.getOffHandStack();
        if (offhand != null && !offhand.isEmpty()) {
            String offhandId = Registries.ITEM.getId(offhand.getItem()).toString();
            if (offhandId.endsWith("shield")) {
                return Hand.OFF_HAND;
            }
        }
        return Hand.MAIN_HAND;
    }

    private boolean isRecoveringFromKnockback(ClientPlayerEntity player) {
        if (player == null) {
            return false;
        }
        Vec3d velocity = player.getVelocity();
        double horizontalSpeed = Math.sqrt((velocity.x * velocity.x) + (velocity.z * velocity.z));
        return player.hurtTime > 0 || horizontalSpeed > 0.22;
    }

    private void maybeSendActionProbe(MinecraftClient client, String message) {
        if (client == null || client.player == null || message == null || message.isBlank()) {
            return;
        }
        boolean isAttack = message.startsWith("LEFTCLICK");
        long nextAllowedTick = isAttack ? nextAttackProbeTick : nextUseProbeTick;
        if (localTickCounter < nextAllowedTick) {
            return;
        }
        client.player.sendMessage(Text.literal("SolasAI probe: " + message), false);
        if (isAttack) {
            nextAttackProbeTick = localTickCounter + 20;
        } else {
            nextUseProbeTick = localTickCounter + 20;
        }
    }

    private void applyLocalPathSafety(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !active) {
            return;
        }
        if (isCombatIntentActive(client, player)) {
            PlayerEntity nearbyEnemy = findNearestEnemyPlayer(client, player);
            if (nearbyEnemy != null && player.distanceTo(nearbyEnemy) <= 9.0) {
                return;
            }
        }

        GameOptions options = client.options;

        // Water safety: always swim upward so the bot doesn't drown while pathing.
        if (player.isTouchingWater()) {
            options.jumpKey.setPressed(true);
            player.setJumping(true);
        }

        boolean movingForward = options.forwardKey.isPressed();
        boolean movingBack = options.backKey.isPressed();
        if (!movingForward && !movingBack) {
            return;
        }

        double directionSign = movingBack ? -1.0 : 1.0;
        double yawRad = Math.toRadians(player.getYaw());
        double dirX = -Math.sin(yawRad) * directionSign;
        double dirZ = Math.cos(yawRad) * directionSign;

        boolean dropAhead = isDropAhead(client, player, dirX, dirZ, 1.2, 3);
        boolean deepTrapAhead = isDropAhead(client, player, dirX, dirZ, 1.2, 7);
        boolean obstacleAhead = isObstacleAhead(client, player, dirX, dirZ, 0.9);
        boolean wallAhead = isWallAhead(client, player, dirX, dirZ, 1.0);
        Entity trapTarget = deepTrapAhead ? findAimAssistTarget(client, player) : null;
        boolean enemyInTrap = trapTarget != null && isEntityInDeepTrap(client, trapTarget, 6);

        if (deepTrapAhead || dropAhead) {
            // Path safety override: don't walk into caves/holes/traps.
            options.forwardKey.setPressed(false);
            options.backKey.setPressed(false);
            options.sprintKey.setPressed(false);
            options.sneakKey.setPressed(true);
            player.setSprinting(false);
            player.setSneaking(true);

            boolean bridged = false;
            if (!deepTrapAhead && movingForward && player.isOnGround()) {
                bridged = tryBridgeAhead(client, player, dirX, dirZ);
            }

            if (enemyInTrap && player.squaredDistanceTo(trapTarget) <= (5.0 * 5.0) && hasClearMeleeLine(client, player)) {
                // Keep pressure from the edge instead of dropping in.
                options.attackKey.setPressed(true);
                options.leftKey.setPressed((localTickCounter % 8) < 4);
                options.rightKey.setPressed((localTickCounter % 8) >= 4);
            } else if (bridged) {
                options.useKey.setPressed(true);
                options.attackKey.setPressed(false);
                options.leftKey.setPressed(false);
                options.rightKey.setPressed(false);
                options.sneakKey.setPressed(true);
            } else {
                boolean leftSafer = isSideSafer(client, player, true);
                options.leftKey.setPressed(leftSafer);
                options.rightKey.setPressed(!leftSafer);

                float turn = leftSafer ? 8f : -8f;
                float newYaw = player.getYaw() + turn;
                player.setYaw(newYaw);
                player.setHeadYaw(newYaw);
                player.setBodyYaw(newYaw);
            }
            return;
        }

        // Obstacle handling: hop over single-block bumps while moving.
        if ((obstacleAhead || wallAhead) && player.isOnGround()) {
            options.jumpKey.setPressed(true);
            player.setJumping(true);
            player.jump();
            if (wallAhead) {
                boolean leftSafer = isSideSafer(client, player, true);
                options.leftKey.setPressed(leftSafer);
                options.rightKey.setPressed(!leftSafer);
            }
        }
    }

    private void applyLocalGeneralTreeAssist(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !active) {
            return;
        }

        String objectiveText = objective == null ? "" : objective.trim().toLowerCase();
        boolean generalTreeObjective = objectiveText.matches("^(general\\s*1|general1|gen\\s*1|g1)$")
                || objectiveText.contains("tree")
                || objectiveText.contains("wood")
                || objectiveText.contains("log");
        if (!generalTreeObjective) {
            return;
        }

        BlockPos nearestLog = findNearestTreeLog(client, player, 20);
        if (nearestLog == null) {
            return;
        }

        Vec3d targetCenter = new Vec3d(nearestLog.getX() + 0.5, nearestLog.getY() + 0.65, nearestLog.getZ() + 0.5);
        double dx = targetCenter.x - player.getX();
        double dz = targetCenter.z - player.getZ();
        double horizontalDist = Math.sqrt((dx * dx) + (dz * dz));

        float targetYaw = (float) (Math.toDegrees(Math.atan2(-dx, dz)));
        float yawDelta = MathHelper.wrapDegrees(targetYaw - player.getYaw());
        float adjustedYaw = player.getYaw() + MathHelper.clamp(yawDelta, -8f, 8f);

        double eyeY = player.getEyeY();
        double dy = targetCenter.y - eyeY;
        double pitchTo = -Math.toDegrees(Math.atan2(dy, Math.max(0.001, horizontalDist)));
        float adjustedPitch = MathHelper.clamp((float) pitchTo, -70f, 70f);

        player.setYaw(adjustedYaw);
        player.setHeadYaw(adjustedYaw);
        player.setBodyYaw(adjustedYaw);
        player.setPitch(adjustedPitch);

        GameOptions options = client.options;
        if (horizontalDist > 3.2) {
            boolean leftClear = isSideSafer(client, player, true);
            boolean rightClear = isSideSafer(client, player, false);
            if (leftClear != rightClear) {
                options.leftKey.setPressed(leftClear);
                options.rightKey.setPressed(rightClear);
            } else if ((localTickCounter % 14) < 7) {
                options.leftKey.setPressed(true);
                options.rightKey.setPressed(false);
            } else {
                options.leftKey.setPressed(false);
                options.rightKey.setPressed(true);
            }
            options.backKey.setPressed(false);
            options.forwardKey.setPressed(true);
            options.sprintKey.setPressed(true);
            if (player.horizontalCollision && player.isOnGround()) {
                options.jumpKey.setPressed(true);
                player.jump();
            }
            return;
        }

        options.forwardKey.setPressed(false);
        options.backKey.setPressed(false);
        options.leftKey.setPressed(false);
        options.rightKey.setPressed(false);
        options.sprintKey.setPressed(false);

        boolean harvested = false;
        if (client.crosshairTarget instanceof BlockHitResult bhr) {
            BlockPos hitPos = bhr.getBlockPos();
            Block block = client.world.getBlockState(hitPos).getBlock();
            String blockId = net.minecraft.registry.Registries.BLOCK.getId(block).toString();
            if (isTreeLikeBlockId(blockId)) {
                client.options.attackKey.setPressed(true);
                client.interactionManager.attackBlock(hitPos, bhr.getSide());
                player.swingHand(Hand.MAIN_HAND);
                harvested = true;
            }
        }

        if (!harvested && horizontalDist <= 4.3) {
            Block nearestBlock = client.world.getBlockState(nearestLog).getBlock();
            String nearestId = net.minecraft.registry.Registries.BLOCK.getId(nearestBlock).toString();
            if (isTreeLikeBlockId(nearestId)) {
                client.options.attackKey.setPressed(true);
                client.interactionManager.attackBlock(nearestLog, Direction.UP);
                player.swingHand(Hand.MAIN_HAND);
            }
        }
    }

    private BlockPos findNearestTreeLog(MinecraftClient client, ClientPlayerEntity player, int radius) {
        BlockPos origin = player.getBlockPos();
        BlockPos best = null;
        double bestDistSq = Double.MAX_VALUE;

        int minY = Math.max(client.world.getBottomY(), origin.getY() - 3);
        int maxY = Math.min(client.world.getTopYInclusive(), origin.getY() + 5);
        for (int y = minY; y <= maxY; y++) {
            for (int x = origin.getX() - radius; x <= origin.getX() + radius; x++) {
                for (int z = origin.getZ() - radius; z <= origin.getZ() + radius; z++) {
                    BlockPos pos = new BlockPos(x, y, z);
                    String blockId = net.minecraft.registry.Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock()).toString();
                    if (!isTreeLikeBlockId(blockId)) {
                        continue;
                    }
                    double distSq = pos.getSquaredDistance(origin);
                    if (distSq < bestDistSq) {
                        bestDistSq = distSq;
                        best = pos;
                    }
                }
            }
        }
        return best;
    }

    private boolean isTreeLikeBlockId(String blockId) {
        if (blockId == null || blockId.isBlank()) {
            return false;
        }
        return blockId.endsWith("_log")
                || blockId.endsWith("_wood")
                || blockId.contains("oak_log")
                || blockId.contains("spruce_log")
                || blockId.contains("birch_log")
                || blockId.contains("jungle_log")
                || blockId.contains("acacia_log")
                || blockId.contains("dark_oak_log")
                || blockId.contains("mangrove_log")
                || blockId.contains("cherry_log");
    }

    private void applyLocalProjectileEvasion(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !active) {
            return;
        }
        if (!("pvp".equalsIgnoreCase(lastMode) || "bedwars".equalsIgnoreCase(lastMode) || "crystal".equalsIgnoreCase(lastMode))) {
            return;
        }

        ProjectileEntity threat = null;
        double bestTime = Double.MAX_VALUE;

        for (Entity entity : client.world.getEntities()) {
            if (!(entity instanceof ProjectileEntity projectile) || projectile.isRemoved()) {
                continue;
            }
            if (projectile.getOwner() == player) {
                continue;
            }

            Vec3d projectilePos = new Vec3d(projectile.getX(), projectile.getY(), projectile.getZ());
            Vec3d rel = player.getEyePos().subtract(projectilePos);
            Vec3d vel = projectile.getVelocity();
            double speed = vel.length();
            if (speed < 0.2) continue;

            double dist = rel.length();
            if (dist > 20) continue;

            Vec3d velNorm = vel.normalize();
            Vec3d relNorm = rel.normalize();
            double toward = velNorm.dotProduct(relNorm);
            if (toward < 0.9) continue; // not headed straight enough at player

            double timeToImpact = dist / speed;
            if (timeToImpact > 0.9) continue;

            if (timeToImpact < bestTime) {
                bestTime = timeToImpact;
                threat = projectile;
            }
        }

        if (threat == null) {
            return;
        }

        GameOptions options = client.options;
        Vec3d threatVel = threat.getVelocity();
        double lateral = (-Math.sin(Math.toRadians(player.getYaw())) * threatVel.z)
                + (Math.cos(Math.toRadians(player.getYaw())) * threatVel.x);
        boolean strafeLeft = lateral >= 0;

        options.leftKey.setPressed(strafeLeft);
        options.rightKey.setPressed(!strafeLeft);
        options.sprintKey.setPressed(true);
        options.jumpKey.setPressed(player.isOnGround());
        player.setSprinting(true);
        player.setJumping(player.isOnGround());
        if (player.isOnGround()) {
            player.jump();
        }
    }

    private void applyLocalEmergencyEat(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || !active) {
            return;
        }
        if (player.isUsingItem()) {
            return;
        }

        int emergencySlot = findEmergencyFoodHotbarSlot(player);
        if (emergencySlot < 0) {
            return;
        }

        boolean lowHp = player.getHealth() <= 10f;
        boolean criticalHp = player.getHealth() <= 7f;
        boolean hungerLow = player.getHungerManager().getFoodLevel() < 19;
        if (!(criticalHp || (lowHp && hungerLow))) {
            return;
        }

        player.getInventory().setSelectedSlot(emergencySlot);
        GameOptions options = client.options;
        options.useKey.setPressed(true);
        options.attackKey.setPressed(false);
    }

    private int findEmergencyFoodHotbarSlot(ClientPlayerEntity player) {
        int bestSlot = -1;
        int bestScore = -1;
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = player.getInventory().getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();

            int score = -1;
            if (itemId.endsWith("enchanted_golden_apple")) score = 100;
            else if (itemId.endsWith("golden_apple")) score = 90;
            else if (itemId.endsWith("cooked_beef") || itemId.endsWith("cooked_porkchop")) score = 60;
            else if (itemId.endsWith("bread") || itemId.endsWith("baked_potato")) score = 45;
                else if (itemId.endsWith("apple")
                    || itemId.endsWith("carrot")
                    || itemId.endsWith("potato")
                    || itemId.endsWith("bread")
                    || itemId.endsWith("stew")
                    || itemId.endsWith("rabbit")
                    || itemId.endsWith("cod")
                    || itemId.endsWith("salmon")
                    || itemId.endsWith("chicken")
                    || itemId.endsWith("beef")
                    || itemId.endsWith("porkchop")
                    || itemId.endsWith("mutton")) score = 30;

            if (score > bestScore) {
                bestScore = score;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private boolean isLikelyConsumable(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return false;
        }
        String itemId = Registries.ITEM.getId(stack.getItem()).toString();
        if (itemId.contains("potion")) {
            return true;
        }
        return itemId.endsWith("enchanted_golden_apple")
                || itemId.endsWith("golden_apple")
                || itemId.endsWith("apple")
                || itemId.endsWith("bread")
                || itemId.endsWith("carrot")
                || itemId.endsWith("potato")
                || itemId.endsWith("beetroot")
                || itemId.endsWith("stew")
                || itemId.endsWith("rabbit")
                || itemId.endsWith("cod")
                || itemId.endsWith("salmon")
                || itemId.endsWith("chicken")
                || itemId.endsWith("beef")
                || itemId.endsWith("porkchop")
                || itemId.endsWith("mutton")
                || itemId.endsWith("cookie")
                || itemId.endsWith("melon_slice")
                || itemId.endsWith("chorus_fruit")
                || itemId.endsWith("honey_bottle");
    }

    private boolean canConsumeNow(ClientPlayerEntity player, ItemStack stack) {
        if (player == null || stack == null || stack.isEmpty()) {
            return false;
        }
        String itemId = Registries.ITEM.getId(stack.getItem()).toString();
        if (itemId.contains("potion") || itemId.endsWith("honey_bottle")) {
            return true;
        }
        return player.canConsume(false);
    }

    private void applyLocalWaterRecovery(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || !active) {
            return;
        }
        if (!player.isTouchingWater()) {
            return;
        }
        client.options.jumpKey.setPressed(true);
        client.options.forwardKey.setPressed(true);
        player.setJumping(true);
        if (player.getAir() < player.getMaxAir() - 20) {
            player.setPitch(-35f);
        }
        if (!player.isOnGround()) {
            return;
        }
        int emptyBucketSlot = findHotbarItemExact(player.getInventory(), "minecraft:bucket");
        if (emptyBucketSlot < 0) {
            return;
        }
        player.getInventory().setSelectedSlot(emptyBucketSlot);
        player.setPitch(80f);
        client.options.useKey.setPressed(true);
    }

    private void applyLocalFallRecovery(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !active) {
            return;
        }
        if (player.isOnGround() || player.getVelocity().y > -0.45) {
            return;
        }
        if (hasAnyClutchItem(player)) {
            return;
        }
        if (localTickCounter < nextFallRescueScanTick) {
            return;
        }
        nextFallRescueScanTick = localTickCounter + 2;

        BlockPos best = findNearestSafeLanding(client, player, 6, 18);
        if (best == null) {
            return;
        }

        double tx = best.getX() + 0.5;
        double tz = best.getZ() + 0.5;
        double dx = tx - player.getX();
        double dz = tz - player.getZ();
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        if (horizontal < 0.01) {
            return;
        }

        float targetYaw = (float) (Math.atan2(-dx, dz) * (180.0 / Math.PI));
        float yawError = MathHelper.wrapDegrees(targetYaw - player.getYaw());
        float yawStep = MathHelper.clamp(yawError, -18f, 18f);
        float newYaw = player.getYaw() + yawStep;
        player.setYaw(newYaw);
        player.setHeadYaw(newYaw);
        player.setBodyYaw(newYaw);

        GameOptions options = client.options;
        options.forwardKey.setPressed(true);
        options.sprintKey.setPressed(false);
        options.jumpKey.setPressed(false);
    }

    private boolean hasAnyClutchItem(ClientPlayerEntity player) {
        PlayerInventory inv = player.getInventory();
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = inv.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String id = Registries.ITEM.getId(stack.getItem()).toString();
            if (id.endsWith("water_bucket")
                    || id.endsWith("powder_snow_bucket")
                    || id.endsWith("cobweb")
                    || id.endsWith("hay_block")
                    || id.endsWith("slime_block")
                    || id.endsWith("honey_block")) {
                return true;
            }
        }
        return false;
    }

    private BlockPos findNearestSafeLanding(MinecraftClient client, ClientPlayerEntity player, int radius, int depth) {
        BlockPos best = null;
        double bestScore = Double.MAX_VALUE;
        int px = MathHelper.floor(player.getX());
        int py = MathHelper.floor(player.getY());
        int pz = MathHelper.floor(player.getZ());

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                for (int drop = 1; drop <= depth; drop++) {
                    int y = py - drop;
                    BlockPos pos = new BlockPos(px + dx, y, pz + dz);
                    String blockId = Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock()).toString();
                    boolean safe = blockId.endsWith(":water")
                            || blockId.endsWith("hay_block")
                            || blockId.endsWith("slime_block")
                            || blockId.endsWith("honey_block")
                            || blockId.endsWith("cobweb")
                            || blockId.contains("leaves");
                    if (!safe) continue;

                    double distSq = (dx * dx) + (dz * dz) + (drop * 0.2);
                    if (distSq < bestScore) {
                        bestScore = distSq;
                        best = pos;
                    }
                }
            }
        }
        return best;
    }

    private int findHotbarItemExact(PlayerInventory inventory, String itemIdExact) {
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String id = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemIdExact.equals(id)) return slot;
        }
        return -1;
    }

    private boolean isObstacleAhead(MinecraftClient client, ClientPlayerEntity player, double dirX, double dirZ, double distance) {
        BlockPos feetAhead = BlockPos.ofFloored(player.getX() + (dirX * distance), player.getY(), player.getZ() + (dirZ * distance));
        BlockPos headAhead = feetAhead.up();
        boolean feetBlocked = client.world.getBlockState(feetAhead).isSolidBlock(client.world, feetAhead);
        boolean headBlocked = client.world.getBlockState(headAhead).isSolidBlock(client.world, headAhead);
        return feetBlocked || headBlocked;
    }

    private boolean isWallAhead(MinecraftClient client, ClientPlayerEntity player, double dirX, double dirZ, double distance) {
        BlockPos feet = BlockPos.ofFloored(player.getX() + (dirX * distance), player.getY(), player.getZ() + (dirZ * distance));
        BlockPos head = feet.up();
        BlockPos twoHead = head.up();
        boolean feetSolid = client.world.getBlockState(feet).isSolidBlock(client.world, feet);
        boolean headSolid = client.world.getBlockState(head).isSolidBlock(client.world, head);
        boolean twoHeadSolid = client.world.getBlockState(twoHead).isSolidBlock(client.world, twoHead);
        return feetSolid && (headSolid || twoHeadSolid);
    }

    private boolean tryBridgeAhead(MinecraftClient client, ClientPlayerEntity player, double dirX, double dirZ) {
        int blockSlot = findHotbarPlaceBlockSlot(player.getInventory());
        if (blockSlot < 0) {
            return false;
        }
        player.getInventory().setSelectedSlot(blockSlot);

        double targetX = player.getX() + (dirX * 0.95);
        double targetZ = player.getZ() + (dirZ * 0.95);
        BlockPos supportPos = BlockPos.ofFloored(targetX, player.getY() - 1.0, targetZ);
        if (!client.world.getBlockState(supportPos).isSolidBlock(client.world, supportPos)) {
            return false;
        }

        Vec3d hitPos = new Vec3d(supportPos.getX() + 0.5, supportPos.getY() + 1.0, supportPos.getZ() + 0.5);
        BlockHitResult placeHit = new BlockHitResult(hitPos, Direction.UP, supportPos, false);
        player.setPitch(78f);
        client.interactionManager.interactBlock(player, Hand.MAIN_HAND, placeHit);
        return true;
    }

    private int findHotbarPlaceBlockSlot(PlayerInventory inventory) {
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            if (stack.getItem() instanceof BlockItem) {
                return slot;
            }
        }
        return -1;
    }

    private boolean hasClearMeleeLine(MinecraftClient client, ClientPlayerEntity player) {
        if (client.crosshairTarget == null) {
            return false;
        }
        if (client.crosshairTarget.getType() != HitResult.Type.ENTITY) {
            return false;
        }
        if (!(client.crosshairTarget instanceof EntityHitResult entityHit) || entityHit.getEntity() == null) {
            return false;
        }
        return player.distanceTo(entityHit.getEntity()) <= 4.2f;
    }

    private boolean isCombatMode() {
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        return "pvp".equals(mode) || "crystal".equals(mode) || "bedwars".equals(mode) || objectiveLooksCombat();
    }

    private boolean objectiveLooksCombat() {
        if (objective == null || objective.isBlank()) {
            return false;
        }
        String lower = objective.toLowerCase();
        return lower.contains("pvp")
                || lower.contains("crystal")
                || lower.contains("fight")
                || lower.contains("attack")
                || lower.contains("kill")
                || lower.contains("duel");
    }

    private boolean isCombatIntentActive(MinecraftClient client, ClientPlayerEntity player) {
        if (!active) {
            return false;
        }
        if (isCombatMode()) {
            return true;
        }
        return findNearestEnemyPlayer(client, player) != null;
    }

    private boolean shouldHumanizeMovement() {
        if (!active || currentAction == null) {
            return false;
        }
        if (isCombatMode()) {
            return false;
        }
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        if (!("pvp".equals(mode) || "general".equals(mode) || "bedwars".equals(mode))) {
            return false;
        }
        return currentAction.forward() || currentAction.left() || currentAction.right() || currentAction.sprint();
    }

    private void applyHumanizedInputNoise(GameOptions options) {
        if ((localTickCounter % 17) == 0) {
            options.sprintKey.setPressed(false);
        }
        if ((localTickCounter % 23) == 7 && !options.useKey.isPressed()) {
            options.jumpKey.setPressed(false);
        }
        if ((localTickCounter % 19) == 5 && !options.leftKey.isPressed() && !options.rightKey.isPressed()) {
            options.leftKey.setPressed(true);
        }
    }

    private boolean isDropAhead(MinecraftClient client, ClientPlayerEntity player, double dirX, double dirZ, double distance, int depthBlocks) {
        int baseY = MathHelper.floor(player.getY());
        int baseX = MathHelper.floor(player.getX() + (dirX * distance));
        int baseZ = MathHelper.floor(player.getZ() + (dirZ * distance));

        // If no solid support for multiple blocks below the next step, treat as dangerous drop.
        for (int depth = 1; depth <= depthBlocks; depth++) {
            BlockPos checkPos = new BlockPos(baseX, baseY - depth, baseZ);
            if (client.world.getBlockState(checkPos).isSolidBlock(client.world, checkPos)) {
                return false;
            }
        }
        return true;
    }

    private boolean isSideSafer(MinecraftClient client, ClientPlayerEntity player, boolean left) {
        double yawRad = Math.toRadians(player.getYaw());
        double fwdX = -Math.sin(yawRad);
        double fwdZ = Math.cos(yawRad);
        double sideX = left ? -fwdZ : fwdZ;
        double sideZ = left ? fwdX : -fwdX;
        return !isDropAhead(client, player, sideX, sideZ, 1.0, 2);
    }

    private boolean isEntityInDeepTrap(MinecraftClient client, Entity entity, int depthBlocks) {
        if (client.world == null || entity == null) {
            return false;
        }
        int x = MathHelper.floor(entity.getX());
        int z = MathHelper.floor(entity.getZ());
        int y = MathHelper.floor(entity.getY());

        boolean airColumn = true;
        for (int depth = 1; depth <= depthBlocks; depth++) {
            BlockPos below = new BlockPos(x, y - depth, z);
            if (client.world.getBlockState(below).isSolidBlock(client.world, below)) {
                airColumn = false;
                break;
            }
        }
        return airColumn;
    }

    private void applyLocalAerialMacePursuit(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !active || client.interactionManager == null) {
            return;
        }
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        if (!("pvp".equals(mode) || "crystal".equals(mode))) {
            return;
        }

        Entity target = findAimAssistTarget(client, player);
        if (!(target instanceof LivingEntity) || target == player) {
            return;
        }

        double dist = player.distanceTo(target);
        boolean targetAirborne = !target.isOnGround() || target.getY() > (player.getY() + 2.0);
        boolean playerAirborne = player.isGliding() || !player.isOnGround();
        boolean shouldChaseAir = targetAirborne && dist > 3.2;

        if (shouldChaseAir) {
            if (equipChestArmorFromInventory(client, player, findElytraInventorySlot(player.getInventory()))) {
                return;
            }
            client.options.forwardKey.setPressed(true);
            client.options.sprintKey.setPressed(true);
            if (player.isOnGround()) {
                client.options.jumpKey.setPressed(true);
                player.setJumping(true);
                player.jump();
            }
            return;
        }

        if (playerAirborne || dist <= 3.2) {
            if (equipChestArmorFromInventory(client, player, findBestChestplateInventorySlot(player.getInventory()))) {
                return;
            }
            int maceSlot = findBestMaceSlot(player.getInventory());
            if (maceSlot >= 0 && maceSlot < 9) {
                player.getInventory().setSelectedSlot(maceSlot);
            }
        }
    }

    private void applyLocalAimAssist(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null || !shouldUseAimAssist()) {
            return;
        }

        Entity target = findAimAssistTarget(client, player);
        if (target == null) {
            return;
        }

        Vec3d aimPoint = getAimAssistPoint(target);
        double dx = aimPoint.x - player.getX();
        double dz = aimPoint.z - player.getZ();
        double dy = aimPoint.y - player.getEyeY();
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        if (horizontal < 0.001) {
            return;
        }

        float targetYaw = (float) (Math.atan2(-dx, dz) * (180.0 / Math.PI));
        float targetPitch = (float) (-Math.atan2(dy, horizontal) * (180.0 / Math.PI));

        float yawError = MathHelper.wrapDegrees(targetYaw - player.getYaw());
        float pitchError = MathHelper.wrapDegrees(targetPitch - player.getPitch());

        float yawStep = MathHelper.clamp(yawError, -18f, 18f);
        float pitchStep = MathHelper.clamp(pitchError, -12f, 12f);

        if (Math.abs(yawError) < 0.75f) yawStep = yawError;
        if (Math.abs(pitchError) < 0.75f) pitchStep = pitchError;

        float newYaw = player.getYaw() + yawStep;
        float newPitch = MathHelper.clamp(player.getPitch() + pitchStep, -90f, 90f);
        player.setYaw(newYaw);
        player.setHeadYaw(newYaw);
        player.setBodyYaw(newYaw);
        player.setPitch(newPitch);
    }

    private boolean shouldUseAimAssist() {
        if (!active) {
            return false;
        }
        if (!isCombatMode()) {
            return false;
        }
        return currentAction != null && !currentAction.use();
    }

    private void applyCombatDebugHud(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null || !active) {
            return;
        }
        if (localTickCounter < nextCombatDebugHudTick) {
            return;
        }
        nextCombatDebugHudTick = localTickCounter + 10;

        boolean combatIntent = isCombatIntentActive(client, player);
        PlayerInventory inventory = player.getInventory();
        int preferredSlot = findPreferredOffhandInventorySlot(client, player, inventory);
        String preferredOffhand = preferredSlot >= 0
                ? Registries.ITEM.getId(inventory.getStack(preferredSlot).getItem()).toString()
                : "none";
        ItemStack offhand = player.getOffHandStack();
        String currentOffhand = (offhand == null || offhand.isEmpty())
                ? "none"
                : Registries.ITEM.getId(offhand.getItem()).toString();
        PlayerEntity enemy = findNearestEnemyPlayer(client, player);
        String distText = enemy == null ? "-" : String.valueOf(Math.round(player.distanceTo(enemy) * 10.0) / 10.0);
        boolean forceShield = shouldForceShieldOffhand(player, inventory);

        String msg = "SolasAI dbg combat=" + combatIntent
                + " shieldLock=" + forceShield
                + " off=" + compactItemId(currentOffhand)
                + " pref=" + compactItemId(preferredOffhand)
                + " dist=" + distText
                + " mode=" + (lastMode == null ? "" : lastMode);
        player.sendMessage(Text.literal(msg), true);
    }

    private String compactItemId(String itemId) {
        if (itemId == null || itemId.isBlank()) {
            return "none";
        }
        int idx = itemId.indexOf(':');
        if (idx >= 0 && idx + 1 < itemId.length()) {
            return itemId.substring(idx + 1);
        }
        return itemId;
    }

    private String getObjectiveCombatTarget() {
        if (objective == null || objective.isBlank()) {
            return "";
        }
        String lower = objective.toLowerCase();
        Matcher matcher = Pattern.compile("\\b(?:fight|attack|kill|hunt|target)\\s+([a-z0-9_ ]+?)(?:\\s*,\\s*type\\s*=\\s*[a-z0-9_]+|\\s+type\\s*=\\s*[a-z0-9_]+|$)").matcher(lower);
        if (!matcher.find()) {
            return "";
        }
        return matcher.group(1)
                .replaceAll("\\b(the|a|an)\\b", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private boolean entityMatchesObjectiveTarget(Entity entity, String requestedTarget) {
        if (entity == null) {
            return false;
        }
        if (requestedTarget == null || requestedTarget.isBlank()) {
            return true;
        }
        String normalizedRequested = requestedTarget.toLowerCase().replace('_', ' ').trim();
        String entityName = entity.getName().getString().toLowerCase().replace('_', ' ');
        String entityTypeId = Registries.ENTITY_TYPE.getId(entity.getType()).toString().toLowerCase().replace("minecraft:", "").replace('_', ' ');
        if (entityName.contains(normalizedRequested) || entityTypeId.contains(normalizedRequested)) {
            return true;
        }
        String[] requestedWords = normalizedRequested.split(" ");
        for (String word : requestedWords) {
            if (word.isBlank()) {
                continue;
            }
            if (!(entityName.contains(word) || entityTypeId.contains(word))) {
                return false;
            }
        }
        return requestedWords.length > 0;
    }

    private Entity findAimAssistTarget(MinecraftClient client, ClientPlayerEntity player) {
        String requestedTarget = getObjectiveCombatTarget();
        boolean wantsSpecificTarget = requestedTarget != null && !requestedTarget.isBlank();
        boolean wantsPlayerTarget = !wantsSpecificTarget
                || requestedTarget.contains("player")
                || requestedTarget.contains("enemy")
                || requestedTarget.contains("opponent");

        if (client.targetedEntity != null
                && client.targetedEntity != player
                && entityMatchesObjectiveTarget(client.targetedEntity, requestedTarget)
                && player.squaredDistanceTo(client.targetedEntity) <= (28 * 28)) {
            return client.targetedEntity;
        }

        if (wantsSpecificTarget) {
            Entity matched = null;
            double matchedDistSq = Double.MAX_VALUE;
            for (Entity entity : client.world.getEntities()) {
                if (entity == null || entity == player || entity.isSpectator()) {
                    continue;
                }
                if (!(entity instanceof PlayerEntity || entity instanceof HostileEntity || entity instanceof GolemEntity)) {
                    continue;
                }
                if (!entityMatchesObjectiveTarget(entity, requestedTarget)) {
                    continue;
                }
                double distSq = player.squaredDistanceTo(entity);
                if (distSq <= (28 * 28) && distSq < matchedDistSq) {
                    matched = entity;
                    matchedDistSq = distSq;
                }
            }
            if (matched != null) {
                return matched;
            }
        }

        Entity best = null;
        double bestDistSq = Double.MAX_VALUE;
        boolean preferPlayers = "bedwars".equalsIgnoreCase(lastMode) || "pvp".equalsIgnoreCase(lastMode) || "crystal".equalsIgnoreCase(lastMode);

        if (preferPlayers && wantsPlayerTarget) {
            for (PlayerEntity other : client.world.getPlayers()) {
                if (other == null || other == player || other.isSpectator() || player.isTeammate(other)) {
                    continue;
                }
                double distSq = player.squaredDistanceTo(other);
                if (distSq < bestDistSq && distSq <= (24 * 24)) {
                    best = other;
                    bestDistSq = distSq;
                }
            }
        }

        if (best == null && ("pvp".equalsIgnoreCase(lastMode) || "crystal".equalsIgnoreCase(lastMode))) {
            for (Entity entity : client.world.getEntities()) {
                if (entity == null || entity == player || entity.isSpectator()) {
                    continue;
                }
                if (!(entity instanceof HostileEntity) && !(entity instanceof GolemEntity)) {
                    continue;
                }
                double distSq = player.squaredDistanceTo(entity);
                if (distSq < bestDistSq && distSq <= (20 * 20)) {
                    best = entity;
                    bestDistSq = distSq;
                }
            }
        }

        return best;
    }

    private Vec3d getAimAssistPoint(Entity target) {
        double centerX = (target.getBoundingBox().minX + target.getBoundingBox().maxX) * 0.5;
        double centerZ = (target.getBoundingBox().minZ + target.getBoundingBox().maxZ) * 0.5;
        double centerY;
        if (target instanceof LivingEntity living) {
            centerY = living.getY() + (living.getHeight() * 0.6);
        } else {
            centerY = (target.getBoundingBox().minY + target.getBoundingBox().maxY) * 0.5;
        }
        return new Vec3d(centerX, centerY, centerZ);
    }

    private void ensurePreferredOffhand(MinecraftClient client, ClientPlayerEntity player) {
        if (localTickCounter < nextOffhandCheckTick) {
            return;
        }
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        int offhandCheckInterval = 1;
        nextOffhandCheckTick = localTickCounter + offhandCheckInterval;

        ItemStack offhand = player.getOffHandStack();
        boolean offhandHasBlock = offhand != null && !offhand.isEmpty() && offhand.getItem() instanceof BlockItem;
        PlayerInventory inventory = player.getInventory();
        int desiredOffhandSlot = findPreferredOffhandInventorySlot(client, player, inventory);
        String desiredOffhandId = desiredOffhandSlot >= 0
            ? Registries.ITEM.getId(inventory.getStack(desiredOffhandSlot).getItem()).toString()
            : "";
        String currentOffhandId = (offhand != null && !offhand.isEmpty())
            ? Registries.ITEM.getId(offhand.getItem()).toString()
            : "";
        if (offhand != null && !offhand.isEmpty() && !offhandHasBlock && desiredOffhandId.equals(currentOffhandId)) {
            offhandEmptySinceTick = -1;
            offhandPendingHotbarSlot = -1;
            offhandPendingNeedsSelect = false;
            return;
        }

        if (offhandEmptySinceTick < 0) {
            offhandEmptySinceTick = localTickCounter;
            return;
        }
        if ((localTickCounter - offhandEmptySinceTick) < 2) {
            return;
        }

        if (client.interactionManager == null) {
            return;
        }

        // Pending hotbar flow:
        // 1) Select the hotbar slot
        // 2) Swap selected hotbar item to offhand
        if (offhandPendingHotbarSlot >= 0 && offhandPendingHotbarSlot < 9) {
            ItemStack staged = inventory.getStack(offhandPendingHotbarSlot);
            if (!staged.isEmpty()) {
                if (offhandPendingNeedsSelect) {
                    inventory.setSelectedSlot(offhandPendingHotbarSlot);
                    offhandPendingNeedsSelect = false;
                    nextOffhandCheckTick = localTickCounter + 1;
                    return;
                }
                int screenSlot = 36 + offhandPendingHotbarSlot;
                client.interactionManager.clickSlot(
                        player.playerScreenHandler.syncId,
                        screenSlot,
                        PlayerInventory.OFF_HAND_SLOT,
                        SlotActionType.SWAP,
                        player
                );
            }
            offhandPendingHotbarSlot = -1;
            offhandPendingNeedsSelect = false;
            offhandEmptySinceTick = -1;
            return;
        }

        int sourceInventorySlot = desiredOffhandSlot;
        if (sourceInventorySlot < 0) {
            return;
        }

        if (sourceInventorySlot < 9) {
            // Item already in hotbar: first select it, then offhand swap on next check
            offhandPendingHotbarSlot = sourceInventorySlot;
            offhandPendingNeedsSelect = true;
            nextOffhandCheckTick = localTickCounter + 1;
        } else {
            // Item is in main inventory (slots 9-35) — 2-step:
            // Step 1: move it to a temp hotbar slot (slot 5), then select it and offhand swap
            final int tempHotbarSlot = 5;
            int screenSlot = toPlayerScreenSlot(sourceInventorySlot);
            if (screenSlot < 0) return;
            client.interactionManager.clickSlot(
                    player.playerScreenHandler.syncId,
                    screenSlot,
                    tempHotbarSlot,
                    SlotActionType.SWAP,
                    player
            );
            offhandPendingHotbarSlot = tempHotbarSlot;
            offhandPendingNeedsSelect = true;
            // Force next check to run soon to complete select+swap
            nextOffhandCheckTick = localTickCounter + 2;
        }
    }

    private void ensureCombatHotbarLoadout(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null || client.interactionManager == null) {
            return;
        }
        if (!shouldMaintainCombatLoadout()) {
            return;
        }
        if (localTickCounter < nextHotbarOrganizeTick) {
            return;
        }
        nextHotbarOrganizeTick = localTickCounter + 1;

        PlayerInventory inventory = player.getInventory();
        String preferredStyle = getPreferredCombatStyle(inventory);
        int desiredSword = findBestCombatWeaponSlot(inventory, true);
        int desiredAxe = findToolSlot(inventory, "axe");
        int desiredMace = findBestMaceSlot(inventory);
        int desiredWindCharge = findInventoryItemExact(inventory, "minecraft:wind_charge");
        int desiredPearl = findInventoryItemExact(inventory, "minecraft:ender_pearl");
        int desiredCobweb = findInventoryItemExact(inventory, "minecraft:cobweb");
        int desiredBow = findInventoryItemExact(inventory, "minecraft:bow");
        int desiredObsidian = findInventoryItemExact(inventory, "minecraft:obsidian");
        int desiredCrystal = findInventoryItemExact(inventory, "minecraft:end_crystal");
        int desiredAnchor = findInventoryItemExact(inventory, "minecraft:respawn_anchor");
        int desiredGlowstone = findInventoryItemExact(inventory, "minecraft:glowstone");
        int desiredFood = findBestFoodSlot(inventory);
        int desiredTotem = findTotemSlot(inventory);

        if ("mace".equals(preferredStyle)) {
            if (swapToHotbarSlot(client, player, inventory, desiredMace, 0)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredWindCharge, 1)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredPearl, 2)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredSword, 3)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredCobweb, 4)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredBow, 5)) return;
        } else if ("crystal".equals(preferredStyle)) {
            if (swapToHotbarSlot(client, player, inventory, desiredObsidian, 0)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredCrystal, 1)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredAnchor, 2)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredGlowstone, 3)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredSword, 4)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredPearl, 5)) return;
        } else {
            if (swapToHotbarSlot(client, player, inventory, desiredSword, 0)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredAxe, 1)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredPearl, 2)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredBow, 3)) return;
            if (swapToHotbarSlot(client, player, inventory, desiredCobweb, 4)) return;
        }

        if (swapToHotbarSlot(client, player, inventory, desiredFood, 7)) return;
        swapToHotbarSlot(client, player, inventory, desiredTotem, 8);
    }

    private void applyLocalTrashCleanup(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null || client.interactionManager == null) {
            return;
        }
        if (localTickCounter < nextTrashDropTick) {
            return;
        }
        if (player.getHealth() <= 10f) {
            nextTrashDropTick = localTickCounter + 30;
            return;
        }
        Entity nearbyThreat = findAimAssistTarget(client, player);
        if (nearbyThreat != null && player.squaredDistanceTo(nearbyThreat) < (8 * 8)) {
            nextTrashDropTick = localTickCounter + 20;
            return;
        }

        PlayerInventory inventory = player.getInventory();
        int emptySlots = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                emptySlots++;
            }
        }
        if (emptySlots > 1) {
            nextTrashDropTick = localTickCounter + 30;
            return;
        }

        int trashSlot = findDropTrashSlot(inventory);
        if (trashSlot < 0) {
            nextTrashDropTick = localTickCounter + 40;
            return;
        }

        int screenSlot = toPlayerScreenSlot(trashSlot);
        if (screenSlot < 0) {
            nextTrashDropTick = localTickCounter + 20;
            return;
        }
        client.interactionManager.clickSlot(
                player.playerScreenHandler.syncId,
                screenSlot,
                1,
                SlotActionType.THROW,
                player
        );
        nextTrashDropTick = localTickCounter + 8;
    }

    private int findDropTrashSlot(PlayerInventory inventory) {
        int bestSlot = -1;
        int bestCount = 0;

        for (int slot = 9; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (!isDropTrashItem(stack, itemId)) {
                continue;
            }
            if (stack.getCount() > bestCount) {
                bestCount = stack.getCount();
                bestSlot = slot;
            }
        }

        if (bestSlot >= 0) {
            return bestSlot;
        }

        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (!isDropTrashItem(stack, itemId)) {
                continue;
            }
            if (stack.getCount() > bestCount) {
                bestCount = stack.getCount();
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private boolean isDropTrashItem(ItemStack stack, String itemId) {
        if (stack == null || stack.isEmpty()) {
            return false;
        }
        if (stack.getItem() instanceof BlockItem) {
            return false;
        }

        String id = itemId == null ? "" : itemId;
        if (id.isEmpty()) {
            return false;
        }

        if (id.endsWith("_sword") || id.endsWith("_axe") || id.endsWith("_pickaxe") || id.endsWith("_shovel") || id.endsWith("_hoe")) {
            return false;
        }
        if (id.endsWith("_helmet") || id.endsWith("_chestplate") || id.endsWith("_leggings") || id.endsWith("_boots") || id.endsWith("shield") || id.endsWith("elytra")) {
            return false;
        }
        if (id.endsWith("totem_of_undying") || id.endsWith("ender_pearl") || id.endsWith("end_crystal") || id.endsWith("obsidian")
                || id.endsWith("respawn_anchor") || id.endsWith("glowstone") || id.endsWith("crossbow") || id.endsWith("bow") || id.endsWith("arrow")
                || id.endsWith("mace") || id.endsWith("trident") || id.endsWith("golden_apple") || id.endsWith("enchanted_golden_apple")
                || id.endsWith("potion") || id.endsWith("water_bucket") || id.endsWith("bucket") || id.endsWith("flint_and_steel")) {
            return false;
        }
        if (id.endsWith("_log") || id.endsWith("_wood") || id.endsWith("_planks") || id.endsWith("stick") || id.endsWith("crafting_table")) {
            return false;
        }
        if (id.endsWith("diamond") || id.endsWith("emerald") || id.endsWith("gold_ingot") || id.endsWith("iron_ingot") || id.endsWith("redstone") || id.endsWith("lapis_lazuli")) {
            return false;
        }

        return id.endsWith("rotten_flesh")
                || id.endsWith("poisonous_potato")
                || id.endsWith("spider_eye")
                || id.endsWith("seeds")
                || id.endsWith("beetroot_seeds")
                || id.endsWith("melon_seeds")
                || id.endsWith("pumpkin_seeds")
                || id.endsWith("feather")
                || id.endsWith("snowball")
                || id.endsWith("egg")
                || id.endsWith("kelp")
                || id.endsWith("seagrass")
                || id.endsWith("poppy")
                || id.endsWith("dandelion")
                || id.endsWith("blue_orchid")
                || id.endsWith("allium")
                || id.endsWith("azure_bluet")
                || id.endsWith("red_tulip")
                || id.endsWith("orange_tulip")
                || id.endsWith("white_tulip")
                || id.endsWith("pink_tulip")
                || id.endsWith("oxeye_daisy")
                || id.endsWith("cornflower")
                || id.endsWith("lily_of_the_valley")
                || id.endsWith("sunflower")
                || id.endsWith("lilac")
                || id.endsWith("rose_bush")
                || id.endsWith("peony");
    }

    private boolean shouldRunLocalCrafting(ClientPlayerEntity player) {
        if (!active || player == null) {
            return false;
        }
        String objectiveText = objective == null ? "" : objective.toLowerCase();
        boolean objectiveCraftHint = objectiveText.contains("craft")
                || objectiveText.contains("recipe")
                || objectiveText.contains("plank")
                || objectiveText.contains("stick")
                || objectiveText.contains("crafting table")
                || objectiveText.contains("workbench");
        return objectiveCraftHint || "craft".equalsIgnoreCase(lastMode);
    }

    private boolean isEnemyAimingAtPlayer(PlayerEntity enemy, ClientPlayerEntity player) {
        if (enemy == null || player == null) {
            return false;
        }
        Vec3d enemyLook = enemy.getRotationVec(1.0f);
        Vec3d toPlayer = player.getPos().add(0, player.getHeight() * 0.6, 0)
                .subtract(enemy.getPos().add(0, enemy.getHeight() * 0.6, 0));
        if (toPlayer.lengthSquared() < 0.0001) {
            return false;
        }
        Vec3d toPlayerNorm = toPlayer.normalize();
        double dot = enemyLook.dotProduct(toPlayerNorm);
        return dot > 0.92;
    }

    private boolean enemyHasLikelyMeleeWeapon(PlayerEntity enemy) {
        if (enemy == null) {
            return false;
        }
        ItemStack enemyMain = enemy.getMainHandStack();
        if (enemyMain == null || enemyMain.isEmpty()) {
            return false;
        }
        String enemyMainId = Registries.ITEM.getId(enemyMain.getItem()).toString();
        return enemyMainId.endsWith("_sword") || enemyMainId.endsWith("_axe") || enemyMainId.endsWith("mace") || enemyMainId.endsWith("trident");
    }

    private void applyLocalAttackEvasion(MinecraftClient client, ClientPlayerEntity player, GameOptions options) {
        if (!isCombatMode() || client == null || player == null || options == null) {
            return;
        }

        PlayerEntity enemy = findNearestEnemyPlayer(client, player);
        if (enemy != null) {
            double distance = player.distanceTo(enemy);
            boolean threatened = distance > 0.5 && distance < 3.3
                    && enemyHasLikelyMeleeWeapon(enemy)
                    && isEnemyAimingAtPlayer(enemy, player)
                    && enemy.getAttackCooldownProgress(0f) > 0.75f;

            if (threatened && localTickCounter >= evadeStrafeUntilTick) {
                evadeStrafeLeft = !evadeStrafeLeft;
                evadeStrafeUntilTick = localTickCounter + 4;
                evadeCounterUntilTick = evadeStrafeUntilTick + 5;
            }
        }

        if (localTickCounter < evadeStrafeUntilTick) {
            options.forwardKey.setPressed(false);
            options.backKey.setPressed(false);
            options.leftKey.setPressed(evadeStrafeLeft);
            options.rightKey.setPressed(!evadeStrafeLeft);
            options.attackKey.setPressed(false);
            options.useKey.setPressed(false);
            options.sprintKey.setPressed(true);
            return;
        }

        if (localTickCounter < evadeCounterUntilTick) {
            int swordSlot = findBestCombatWeaponSlot(player.getInventory(), true);
            if (swordSlot >= 0 && swordSlot < 9) {
                player.getInventory().setSelectedSlot(swordSlot);
            }
            options.forwardKey.setPressed(true);
            options.backKey.setPressed(false);
            options.leftKey.setPressed(false);
            options.rightKey.setPressed(false);
            options.attackKey.setPressed(true);
            options.useKey.setPressed(false);
            options.sprintKey.setPressed(true);
        }
    }

    private void applyLocalCrafting(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null || client.interactionManager == null) {
            return;
        }
        if (localTickCounter < nextCraftActionTick) {
            return;
        }
        if (!(player.playerScreenHandler instanceof PlayerScreenHandler handler)) {
            return;
        }
        if (!handler.getCursorStack().isEmpty()) {
            nextCraftActionTick = localTickCounter + 2;
            return;
        }

        String objectiveText = objective == null ? "" : objective.toLowerCase();
        boolean wantsCraftingTable = objectiveText.contains("crafting table") || objectiveText.contains("workbench");
        boolean wantsSticks = objectiveText.contains("stick");
        boolean wantsPlanks = objectiveText.contains("plank") || objectiveText.contains("wood");

        int logSlot = findInventorySlotBySuffix(player.getInventory(), "_log");
        if (logSlot < 0) {
            logSlot = findInventorySlotBySuffix(player.getInventory(), "_wood");
        }
        int plankSlot = findInventorySlotBySuffix(player.getInventory(), "_planks");
        int plankCount = countInventoryItemsBySuffix(player.getInventory(), "_planks");
        int stickCount = countInventoryItem(player.getInventory(), "minecraft:stick");
        int craftingTableCount = countInventoryItem(player.getInventory(), "minecraft:crafting_table");

        if (wantsCraftingTable && craftingTableCount <= 0 && plankCount >= 4) {
            if (craftCraftingTable(client, player, handler, plankSlot)) {
                nextCraftActionTick = localTickCounter + 3;
            }
            return;
        }

        if ((wantsSticks || wantsCraftingTable || "craft".equalsIgnoreCase(lastMode)) && stickCount < 4 && plankCount >= 2) {
            if (craftSticks(client, player, handler, plankSlot)) {
                nextCraftActionTick = localTickCounter + 3;
            }
            return;
        }

        if ((wantsPlanks || wantsSticks || wantsCraftingTable || "craft".equalsIgnoreCase(lastMode)) && plankCount < 8 && logSlot >= 0) {
            if (craftPlanks(client, player, handler, logSlot)) {
                nextCraftActionTick = localTickCounter + 3;
            }
            return;
        }

        nextCraftActionTick = localTickCounter + 10;
    }

    private boolean craftPlanks(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler, int logSlot) {
        if (logSlot < 0) {
            return false;
        }
        clearCraftingGrid(client, player, handler);
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, logSlot, 1)) {
            return false;
        }
        virtualMouseQuickMove(client, player, handler, 0);
        clearCraftingGrid(client, player, handler);
        return true;
    }

    private boolean craftSticks(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler, int plankSlot) {
        if (plankSlot < 0) {
            return false;
        }
        clearCraftingGrid(client, player, handler);
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 1)) {
            return false;
        }
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 3)) {
            clearCraftingGrid(client, player, handler);
            return false;
        }
        virtualMouseQuickMove(client, player, handler, 0);
        clearCraftingGrid(client, player, handler);
        return true;
    }

    private boolean craftCraftingTable(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler, int plankSlot) {
        if (plankSlot < 0) {
            return false;
        }
        clearCraftingGrid(client, player, handler);
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 1)) return false;
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 2)) return false;
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 3)) return false;
        if (!virtualMousePlaceOneInCraftSlot(client, player, handler, plankSlot, 4)) {
            clearCraftingGrid(client, player, handler);
            return false;
        }
        virtualMouseQuickMove(client, player, handler, 0);
        clearCraftingGrid(client, player, handler);
        return true;
    }

    private void clearCraftingGrid(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler) {
        for (int slotId = 1; slotId <= 4; slotId++) {
            if (!handler.getSlot(slotId).getStack().isEmpty()) {
                virtualMouseQuickMove(client, player, handler, slotId);
            }
        }
    }

    private boolean virtualMousePlaceOneInCraftSlot(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler, int inventorySlot, int craftSlotId) {
        int sourceSlotId = toPlayerScreenSlot(inventorySlot);
        if (sourceSlotId < 0 || craftSlotId < 1 || craftSlotId > 4) {
            return false;
        }
        if (!handler.getCursorStack().isEmpty()) {
            return false;
        }

        client.interactionManager.clickSlot(
                handler.syncId,
                sourceSlotId,
                0,
                SlotActionType.PICKUP,
                player
        );

        if (handler.getCursorStack().isEmpty()) {
            return false;
        }

        client.interactionManager.clickSlot(
                handler.syncId,
                craftSlotId,
                1,
                SlotActionType.PICKUP,
                player
        );

        client.interactionManager.clickSlot(
                handler.syncId,
                sourceSlotId,
                0,
                SlotActionType.PICKUP,
                player
        );

        return true;
    }

    private void virtualMouseQuickMove(MinecraftClient client, ClientPlayerEntity player, PlayerScreenHandler handler, int slotId) {
        client.interactionManager.clickSlot(
                handler.syncId,
                slotId,
                0,
                SlotActionType.QUICK_MOVE,
                player
        );
    }

    private int findInventorySlotBySuffix(PlayerInventory inventory, String suffix) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith(suffix)) {
                return slot;
            }
        }
        return -1;
    }

    private int countInventoryItemsBySuffix(PlayerInventory inventory, String suffix) {
        int total = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith(suffix)) {
                total += stack.getCount();
            }
        }
        return total;
    }

    private int countInventoryItem(PlayerInventory inventory, String exactItemId) {
        int total = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (exactItemId.equals(itemId)) {
                total += stack.getCount();
            }
        }
        return total;
    }

    private int findTotemSlot(PlayerInventory inventory) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith("totem_of_undying")) {
                return slot;
            }
        }
        return -1;
    }

    private int countTotems(PlayerInventory inventory) {
        int total = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith("totem_of_undying")) {
                total += stack.getCount();
            }
        }
        return total;
    }

    private boolean swapToHotbarSlot(MinecraftClient client, ClientPlayerEntity player, PlayerInventory inventory, int sourceInventorySlot, int targetHotbarSlot) {
        if (sourceInventorySlot < 0 || targetHotbarSlot < 0 || targetHotbarSlot > 8) {
            return false;
        }
        if (sourceInventorySlot == targetHotbarSlot) {
            return false;
        }
        int screenSlot = toPlayerScreenSlot(sourceInventorySlot);
        if (screenSlot < 0) {
            return false;
        }
        ItemStack target = inventory.getStack(targetHotbarSlot);
        ItemStack source = inventory.getStack(sourceInventorySlot);
        if (ItemStack.areItemsAndComponentsEqual(target, source)) {
            return false;
        }
        client.interactionManager.clickSlot(
                player.playerScreenHandler.syncId,
                screenSlot,
                targetHotbarSlot,
                SlotActionType.SWAP,
                player
        );
        return true;
    }

    private int findBestCombatWeaponSlot(PlayerInventory inventory, boolean preferSword) {
        int bestSlot = -1;
        int bestScore = Integer.MIN_VALUE;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            int score = Integer.MIN_VALUE;
            if (itemId.endsWith("_sword")) score = 300 + weaponTierScore(itemId);
            else if (itemId.endsWith("_axe")) score = (preferSword ? 180 : 280) + weaponTierScore(itemId);
            else if (itemId.equals("minecraft:trident")) score = 260;
            if (score > bestScore) {
                bestScore = score;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private int findBestMaceSlot(PlayerInventory inventory) {
        int bestSlot = -1;
        int bestBreach = -1;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (!itemId.endsWith("mace")) continue;
            int breach = getBreachLevel(stack);
            if (breach > bestBreach) {
                bestBreach = breach;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private int findElytraInventorySlot(PlayerInventory inventory) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith("elytra")) {
                return slot;
            }
        }
        return -1;
    }

    private int findBestChestplateInventorySlot(PlayerInventory inventory) {
        int bestSlot = -1;
        int bestScore = Integer.MIN_VALUE;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            int score = Integer.MIN_VALUE;
            if (itemId.endsWith("_chestplate")) {
                score = 200 + weaponTierScore(itemId);
            }
            if (score > bestScore) {
                bestScore = score;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private boolean equipChestArmorFromInventory(MinecraftClient client, ClientPlayerEntity player, int sourceInventorySlot) {
        if (sourceInventorySlot < 0 || client.interactionManager == null) {
            return false;
        }
        ItemStack source = player.getInventory().getStack(sourceInventorySlot);
        if (source == null || source.isEmpty()) {
            return false;
        }

        String sourceId = Registries.ITEM.getId(source.getItem()).toString();
        ItemStack equipped = player.getEquippedStack(EquipmentSlot.CHEST);
        String equippedId = equipped == null || equipped.isEmpty()
                ? ""
                : Registries.ITEM.getId(equipped.getItem()).toString();
        if (sourceId.equals(equippedId)) {
            return false;
        }

        int sourceScreenSlot = toPlayerScreenSlot(sourceInventorySlot);
        int chestArmorScreenSlot = findPlayerChestArmorScreenSlot(player);
        if (sourceScreenSlot < 0 || chestArmorScreenSlot < 0) {
            return false;
        }

        client.interactionManager.clickSlot(
            player.playerScreenHandler.syncId,
            sourceScreenSlot,
            0,
            SlotActionType.QUICK_MOVE,
            player
        );
        ItemStack recheckChest = player.getEquippedStack(EquipmentSlot.CHEST);
        String recheckId = recheckChest == null || recheckChest.isEmpty()
            ? ""
            : Registries.ITEM.getId(recheckChest.getItem()).toString();
        if (sourceId.equals(recheckId)) {
            return true;
        }

        client.interactionManager.clickSlot(
                player.playerScreenHandler.syncId,
                sourceScreenSlot,
                0,
                SlotActionType.PICKUP,
                player
        );
        client.interactionManager.clickSlot(
                player.playerScreenHandler.syncId,
                chestArmorScreenSlot,
                0,
                SlotActionType.PICKUP,
                player
        );
        client.interactionManager.clickSlot(
                player.playerScreenHandler.syncId,
                sourceScreenSlot,
                0,
                SlotActionType.PICKUP,
                player
        );
        return true;
    }

    private int findBestArmorInventorySlot(PlayerInventory inventory, String suffix) {
        int bestSlot = -1;
        int bestScore = Integer.MIN_VALUE;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (!itemId.endsWith("_" + suffix)) continue;
            int score = 200 + weaponTierScore(itemId);
            if (score > bestScore) {
                bestScore = score;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private int getEquippedArmorScore(ItemStack equipped) {
        if (equipped == null || equipped.isEmpty()) {
            return Integer.MIN_VALUE;
        }
        String equippedId = Registries.ITEM.getId(equipped.getItem()).toString();
        return 200 + weaponTierScore(equippedId);
    }

    private boolean equipArmorPieceFromInventory(MinecraftClient client, ClientPlayerEntity player, int sourceInventorySlot, int armorInventoryIndex) {
        if (sourceInventorySlot < 0 || client == null || client.interactionManager == null) {
            return false;
        }
        ItemStack source = player.getInventory().getStack(sourceInventorySlot);
        if (source == null || source.isEmpty()) {
            return false;
        }
        int sourceScreenSlot = toPlayerScreenSlot(sourceInventorySlot);
        int armorScreenSlot = findPlayerArmorScreenSlot(player, armorInventoryIndex);
        if (sourceScreenSlot < 0 || armorScreenSlot < 0) {
            return false;
        }

        client.interactionManager.clickSlot(
            player.playerScreenHandler.syncId,
            sourceScreenSlot,
            0,
            SlotActionType.QUICK_MOVE,
            player
        );
        ItemStack sourceAfter = player.getInventory().getStack(sourceInventorySlot);
        if (sourceAfter == null || sourceAfter.isEmpty()) {
            return true;
        }

        client.interactionManager.clickSlot(
            player.playerScreenHandler.syncId,
            sourceScreenSlot,
            0,
            SlotActionType.PICKUP,
            player
        );
        client.interactionManager.clickSlot(
            player.playerScreenHandler.syncId,
            armorScreenSlot,
            0,
            SlotActionType.PICKUP,
            player
        );
        client.interactionManager.clickSlot(
            player.playerScreenHandler.syncId,
            sourceScreenSlot,
            0,
            SlotActionType.PICKUP,
            player
        );
        return true;
    }

    private int findPlayerArmorScreenSlot(ClientPlayerEntity player, int armorInventoryIndex) {
        if (player == null || player.playerScreenHandler == null) {
            return -1;
        }
        int altArmorIndex = toAlternateArmorInventoryIndex(armorInventoryIndex);
        for (int slotId = 0; slotId < player.playerScreenHandler.slots.size(); slotId++) {
            Slot slot = player.playerScreenHandler.slots.get(slotId);
            if (slot != null && (slot.getIndex() == armorInventoryIndex || slot.getIndex() == altArmorIndex)) {
                return slotId;
            }
        }
        return -1;
    }

    private int toAlternateArmorInventoryIndex(int armorInventoryIndex) {
        if (armorInventoryIndex < 36 || armorInventoryIndex > 39) {
            return armorInventoryIndex;
        }
        // Some mappings expose armor slots as 0..3 instead of 36..39.
        return 39 - armorInventoryIndex;
    }

    private void ensureBestArmorLoadout(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || player == null || client.interactionManager == null) {
            return;
        }
        if (localTickCounter < nextArmorCheckTick) {
            return;
        }
        nextArmorCheckTick = localTickCounter + 1;

        PlayerInventory inventory = player.getInventory();

        int helmetSlot = findBestArmorInventorySlot(inventory, "helmet");
        int chestSlot = findBestChestplateInventorySlot(inventory);
        int leggingsSlot = findBestArmorInventorySlot(inventory, "leggings");
        int bootsSlot = findBestArmorInventorySlot(inventory, "boots");

        ItemStack equippedHelmet = player.getEquippedStack(EquipmentSlot.HEAD);
        if (helmetSlot >= 0 && (200 + weaponTierScore(Registries.ITEM.getId(inventory.getStack(helmetSlot).getItem()).toString())) > getEquippedArmorScore(equippedHelmet)) {
            if (equipArmorPieceFromInventory(client, player, helmetSlot, 39)) return;
        }

        ItemStack equippedChest = player.getEquippedStack(EquipmentSlot.CHEST);
        if (chestSlot >= 0 && (200 + weaponTierScore(Registries.ITEM.getId(inventory.getStack(chestSlot).getItem()).toString())) > getEquippedArmorScore(equippedChest)) {
            if (equipChestArmorFromInventory(client, player, chestSlot)) return;
        }

        ItemStack equippedLeggings = player.getEquippedStack(EquipmentSlot.LEGS);
        if (leggingsSlot >= 0 && (200 + weaponTierScore(Registries.ITEM.getId(inventory.getStack(leggingsSlot).getItem()).toString())) > getEquippedArmorScore(equippedLeggings)) {
            if (equipArmorPieceFromInventory(client, player, leggingsSlot, 37)) return;
        }

        ItemStack equippedBoots = player.getEquippedStack(EquipmentSlot.FEET);
        if (bootsSlot >= 0 && (200 + weaponTierScore(Registries.ITEM.getId(inventory.getStack(bootsSlot).getItem()).toString())) > getEquippedArmorScore(equippedBoots)) {
            equipArmorPieceFromInventory(client, player, bootsSlot, 36);
        }
    }

    private int findPlayerChestArmorScreenSlot(ClientPlayerEntity player) {
        if (player == null || player.playerScreenHandler == null) {
            return -1;
        }
        return findPlayerArmorScreenSlot(player, 38);
    }

    private int findToolSlot(PlayerInventory inventory, String suffix) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith("_" + suffix)) {
                return slot;
            }
        }
        return -1;
    }

    private int findFirstBlockSlot(PlayerInventory inventory) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            if (stack.getItem() instanceof BlockItem) {
                return slot;
            }
        }
        return -1;
    }

    private int findBestFoodSlot(PlayerInventory inventory) {
        int bestSlot = -1;
        int bestPriority = -1;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            int priority = foodPriority(itemId);
            if (priority > bestPriority) {
                bestPriority = priority;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    private int foodPriority(String itemId) {
        if (itemId.endsWith("enchanted_golden_apple")) return 10;
        if (itemId.endsWith("golden_apple")) return 9;
        if (itemId.endsWith("cooked_beef") || itemId.endsWith("cooked_porkchop")) return 7;
        if (itemId.endsWith("cooked_mutton") || itemId.endsWith("cooked_salmon") || itemId.endsWith("cooked_chicken") || itemId.endsWith("cooked_cod")) return 6;
        if (itemId.endsWith("bread") || itemId.endsWith("pumpkin_pie") || itemId.endsWith("baked_potato")) return 4;
        if (itemId.endsWith("apple") || itemId.endsWith("cooked_rabbit")) return 3;
        if (itemId.endsWith("carrot") || itemId.endsWith("beetroot")) return 2;
        return -1;
    }

    private int weaponTierScore(String itemId) {
        if (itemId.contains("netherite")) return 60;
        if (itemId.contains("diamond")) return 50;
        if (itemId.contains("iron")) return 40;
        if (itemId.contains("stone")) return 30;
        if (itemId.contains("golden")) return 20;
        if (itemId.contains("wooden")) return 10;
        return 0;
    }

    private int getBreachLevel(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return 0;
        }
        try {
            var enchantments = net.minecraft.enchantment.EnchantmentHelper.getEnchantments(stack);
            for (var entry : enchantments.getEnchantmentEntries()) {
                var registryEntry = entry.getKey();
                String enchantId = registryEntry.getKey().map(key -> key.getValue().toString()).orElse("");
                if (enchantId.endsWith(":breach") || enchantId.contains("breach")) {
                    return entry.getIntValue();
                }
            }
        } catch (Exception ignored) {
            return 0;
        }
        return 0;
    }

    private boolean isIdleAction(AgentAction action) {
        if (action == null) return true;
        return !action.forward()
                && !action.back()
                && !action.left()
                && !action.right()
                && !action.jump()
                && !action.sprint()
                && !action.sneak()
                && !action.attack()
                && !action.use()
                && action.yawDelta() == 0f
                && action.pitchDelta() == 0f;
    }

    private int findHotbarTotemSlot(PlayerInventory inventory) {
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.endsWith("totem_of_undying")) {
                return slot;
            }
        }
        return -1;
    }

    private boolean shouldMaintainCombatLoadout() {
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        return "pvp".equals(mode) || "crystal".equals(mode) || objectiveLooksCombat();
    }

    private String getPreferredCombatStyle(PlayerInventory inventory) {
        int maceSlot = findBestMaceSlot(inventory);
        int windChargeSlot = findInventoryItemExact(inventory, "minecraft:wind_charge");
        int crystalSlot = findInventoryItemExact(inventory, "minecraft:end_crystal");
        int obsidianSlot = findInventoryItemExact(inventory, "minecraft:obsidian");
        int anchorSlot = findInventoryItemExact(inventory, "minecraft:respawn_anchor");
        int glowstoneSlot = findInventoryItemExact(inventory, "minecraft:glowstone");

        if (maceSlot >= 0 && windChargeSlot >= 0) {
            return "mace";
        }
        if (crystalSlot >= 0 && obsidianSlot >= 0 && anchorSlot >= 0 && glowstoneSlot >= 0) {
            return "crystal";
        }
        return "sword";
    }

    private int findInventoryItemExact(PlayerInventory inventory, String itemIdExact) {
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemIdExact.equals(itemId)) {
                return slot;
            }
        }
        return -1;
    }

    private PlayerEntity findNearestEnemyPlayer(MinecraftClient client, ClientPlayerEntity player) {
        if (client == null || client.world == null || player == null) {
            return null;
        }
        PlayerEntity nearest = null;
        double nearestDistSq = Double.MAX_VALUE;
        for (PlayerEntity other : client.world.getPlayers()) {
            if (other == null || other == player || other.isSpectator() || player.isTeammate(other)) {
                continue;
            }
            double distSq = player.squaredDistanceTo(other);
            if (distSq < nearestDistSq && distSq <= (24 * 24)) {
                nearest = other;
                nearestDistSq = distSq;
            }
        }
        return nearest;
    }

    private boolean shouldPreferShieldOffhand(ClientPlayerEntity player, PlayerEntity enemy) {
        if (player == null || enemy == null) {
            return false;
        }
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        if (!("pvp".equals(mode) || "crystal".equals(mode) || "bedwars".equals(mode) || objectiveLooksCombat())) {
            return false;
        }
        String enemyMainItem = enemy.getMainHandStack().isEmpty()
                ? ""
                : Registries.ITEM.getId(enemy.getMainHandStack().getItem()).toString();
        double distance = player.distanceTo(enemy);
        double enemyDy = enemy.getY() - player.getY();
        double enemyVelY = enemy.getVelocity().y;
        boolean likelyMaceDive = enemyMainItem.endsWith("mace")
                || enemyMainItem.endsWith("wind_charge")
                || enemy.fallDistance > 2.5f;
        boolean closeMeleeThreat = (enemyMainItem.endsWith("_sword")
                || enemyMainItem.endsWith("_axe")
                || enemyMainItem.endsWith("trident")
                || enemyMainItem.endsWith("mace"))
                && distance < 5.6;
        boolean rangedThreat = (enemyMainItem.endsWith("bow") || enemyMainItem.endsWith("crossbow")) && distance < 11.0;
        boolean preferTotemAtCriticalHealth = player.getHealth() <= 8.0f;
        if (preferTotemAtCriticalHealth) {
            return false;
        }
        return (likelyMaceDive && enemyDy > 2.2 && enemyVelY < -0.5 && distance < 4.8)
                || closeMeleeThreat
                || rangedThreat;
    }

    private boolean shouldForceShieldOffhand(ClientPlayerEntity player, PlayerInventory inventory) {
        if (player == null || inventory == null) {
            return false;
        }
        if (!isCombatMode()) {
            return false;
        }
        if (player.getHealth() <= FORCE_SHIELD_MIN_HEALTH) {
            return false;
        }
        return findInventoryItemExact(inventory, "minecraft:shield") >= 0;
    }

    private boolean shouldPreferDoubleTotem(ClientPlayerEntity player, PlayerEntity enemy, PlayerInventory inventory) {
        if (player == null || enemy == null || inventory == null) {
            return false;
        }
        String mode = lastMode == null ? "" : lastMode.toLowerCase();
        if (!("pvp".equals(mode) || "crystal".equals(mode))) {
            return false;
        }
        if (countTotems(inventory) < 2) {
            return false;
        }
        String enemyMainItem = enemy.getMainHandStack().isEmpty()
                ? ""
                : Registries.ITEM.getId(enemy.getMainHandStack().getItem()).toString();
        boolean enemyCrystalPressure = enemyMainItem.endsWith("end_crystal")
                || enemyMainItem.endsWith("respawn_anchor")
                || enemyMainItem.endsWith("glowstone");
        return enemyCrystalPressure && player.distanceTo(enemy) < 6.0;
    }

    private int findPreferredOffhandInventorySlot(MinecraftClient client, ClientPlayerEntity player, PlayerInventory inventory) {
        int totemSlot = -1;
        int shieldSlot = -1;

        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack == null || stack.isEmpty()) {
                continue;
            }

            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (totemSlot < 0 && itemId.endsWith("totem_of_undying")) {
                totemSlot = slot;
                continue;
            }
            if (shieldSlot < 0 && itemId.endsWith("shield")) {
                shieldSlot = slot;
                continue;
            }
        }

        PlayerEntity enemy = findNearestEnemyPlayer(client, player);
        if (shouldForceShieldOffhand(player, inventory) && shieldSlot >= 0) {
            return shieldSlot;
        }
        boolean shouldDoubleTotem = shouldPreferDoubleTotem(player, enemy, inventory);
        boolean shouldShield = !shouldDoubleTotem && shouldPreferShieldOffhand(player, enemy);

        if (shouldShield && shieldSlot >= 0) return shieldSlot;

        if (totemSlot >= 0) return totemSlot;
        if (shieldSlot >= 0) return shieldSlot;
        return -1;
    }

    private int toPlayerScreenSlot(int inventorySlot) {
        if (inventorySlot >= 9 && inventorySlot <= 35) {
            return inventorySlot;
        }
        if (inventorySlot >= 0 && inventorySlot <= 8) {
            return 36 + inventorySlot;
        }
        return -1;
    }

    private void resetKeys(MinecraftClient client) {
        if (client == null) return;
        GameOptions options = client.options;
        options.forwardKey.setPressed(false);
        options.backKey.setPressed(false);
        options.leftKey.setPressed(false);
        options.rightKey.setPressed(false);
        options.jumpKey.setPressed(false);
        options.sprintKey.setPressed(false);
        options.sneakKey.setPressed(false);
        options.attackKey.setPressed(false);
        options.useKey.setPressed(false);

        if (client.player != null) {
            client.player.setSprinting(false);
            client.player.setSneaking(false);
            client.player.setJumping(false);
        }
    }

    private void handleAutoRespawn(MinecraftClient client) {
        if (!active || client == null) {
            return;
        }
        if (respawnRetryTicks > 0) {
            respawnRetryTicks--;
        }

        if (client.player == null) {
            return;
        }

        if ((client.currentScreen instanceof DeathScreen || client.player.isDead()) && respawnRetryTicks <= 0) {
            try {
                client.player.requestRespawn();
                client.setScreen(null);
                respawnRetryTicks = 30;
                client.player.sendMessage(Text.literal("SolasAI: auto-respawned, resuming task."), false);
            } catch (Exception ex) {
                LOGGER.error("SolasAI auto-respawn failed", ex);
                respawnRetryTicks = 30;
            }
        }
    }

    private void applyMoveAngle(GameOptions options, float moveAngle) {
        if (Float.isNaN(moveAngle)) {
            return;
        }

        float angle = moveAngle;
        while (angle > 180f) angle -= 360f;
        while (angle <= -180f) angle += 360f;

        boolean forward = false;
        boolean back = false;
        boolean left = false;
        boolean right = false;

        if (angle > -22.5f && angle <= 22.5f) {
            forward = true;
        } else if (angle > 22.5f && angle <= 67.5f) {
            forward = true;
            left = true;
        } else if (angle > 67.5f && angle <= 112.5f) {
            left = true;
        } else if (angle > 112.5f && angle <= 157.5f) {
            back = true;
            left = true;
        } else if (angle > 157.5f || angle <= -157.5f) {
            back = true;
        } else if (angle > -157.5f && angle <= -112.5f) {
            back = true;
            right = true;
        } else if (angle > -112.5f && angle <= -67.5f) {
            right = true;
        } else if (angle > -67.5f && angle <= -22.5f) {
            forward = true;
            right = true;
        }

        options.forwardKey.setPressed(forward);
        options.backKey.setPressed(back);
        options.leftKey.setPressed(left);
        options.rightKey.setPressed(right);
    }

    private void maybeAnnounceTask(MinecraftClient client) {
        if (client == null || client.player == null) {
            return;
        }
        String task = getCurrentTask();
        if (task == null || task.isBlank()) {
            return;
        }

        boolean changed = !task.equals(lastAnnouncedTask);
        boolean cooldownElapsed = lastTaskAnnounceTick < 0 || (localTickCounter - lastTaskAnnounceTick) >= 120;
        if (changed && cooldownElapsed) {
            client.player.sendMessage(Text.literal("SolasAI task: " + task), false);
            lastAnnouncedTask = task;
            lastTaskAnnounceTick = localTickCounter;
        }
        if (localTickCounter >= nextIntentChatTick) {
            PlayerEntity enemy = findNearestEnemyPlayer(client, client.player);
            String offhandId = compactItemId(client.player.getOffHandStack().isEmpty()
                    ? "none"
                    : Registries.ITEM.getId(client.player.getOffHandStack().getItem()).toString());
            String intent = isCombatIntentActive(client, client.player) ? "combat" : lastMode;
            String enemyDist = enemy == null ? "-" : String.valueOf(Math.round(client.player.distanceTo(enemy) * 10.0) / 10.0);
            client.player.sendMessage(Text.literal("SolasAI intent: " + intent + " offhand=" + offhandId + " enemyDist=" + enemyDist), false);
            nextIntentChatTick = localTickCounter + 40;
        }
    }

    private void trackPearlUsage(ClientPlayerEntity player) {
        if (currentAction.use() && currentAction.hotbarSlot() >= 0 && currentAction.hotbarSlot() < 9) {
            ItemStack stack = player.getInventory().getStack(currentAction.hotbarSlot());
            if (!stack.isEmpty()) {
                String itemId = Registries.ITEM.getId(stack.getItem()).toString();
                if (itemId.endsWith("ender_pearl")) {
                    lastPearlUseTick = localTickCounter;
                }
            }
        }
    }
}
