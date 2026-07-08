package com.solasai.fabricagent.client;

import com.google.gson.JsonObject;

import net.minecraft.block.BedBlock;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.effect.StatusEffects;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.passive.GolemEntity;
import net.minecraft.entity.passive.VillagerEntity;
import net.minecraft.enchantment.EnchantmentHelper;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Direction;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

import java.util.HashMap;
import java.util.Map;

public record GameStateSnapshot(
        double x,
        double y,
        double z,
        float yaw,
        float pitch,
        float health,
        int food,
        boolean onGround,
    boolean isSprinting,
    boolean isSneaking,
    boolean isTouchingWater,
    double verticalSpeed,
    double horizontalSpeed,
    float fallDistance,
    long worldTime,
        String facing,
        double lookX,
        double lookY,
        double lookZ,
        String selectedItem,
        int selectedItemCount,
        int swordSlot,
        int axeSlot,
        int pickaxeSlot,
        int blockSlot,
        int waterBucketSlot,
        int utilityFoodSlot,
        int cobwebSlot,
        int obsidianSlot,
        int endCrystalSlot,
        int respawnAnchorSlot,
        int glowstoneSlot,
        int totemSlot,
        int pearlSlot,
        int maceSlot,
        int breachMaceSlot,
        int maceBreachLevel,
        int bowSlot,
        int windChargeSlot,
        int windChargeCount,
        int shieldSlot,
        int hotbarBlocks,
        boolean hasBlocks,
        boolean hasWaterBucket,
        boolean hasMeleeWeapon,
        boolean hasElytra,
        int cobwebCount,
        int obsidianCount,
        int endCrystalCount,
        int respawnAnchorCount,
        int glowstoneCount,
        int totemCount,
        int pearlCount,
        int maceCount,
        int villagerNearbyCount,
        String nearestHostile,
        double nearestHostileDistance,
        double nearestHostileDx,
        double nearestHostileDz,
        int ironCount,
        int redstoneCount,
        int diamondCount,
        int goldCount,
        int emeraldCount,
        int netheriteIngotCount,
        int netheriteScrapCount,
        int ancientDebrisCount,
        int netheriteUpgradeTemplateCount,
        int enchantedBookCount,
        String nearestEnemyName,
        double nearestEnemyDistance,
        double nearestEnemyHealth,
        String nearestEnemyMainItem,
        int nearestEnemyArmorPieces,
        boolean nearestEnemyHasMeleeWeapon,
        boolean nearestEnemyHasShield,
        double nearestEnemyVelX,
        double nearestEnemyVelY,
        double nearestEnemyVelZ,
        double nearestEnemyDy,
        double nearestEnemyDx,
        double nearestEnemyDz,
        boolean bedNearby,
        double nearestBedDistance,
        int nearestBedDefenseScore,
        String nearestBedDefenseBlock,
        String focusedEntity,
        double focusedDistance,
        long lastPearlUseTick,
        int combatPotionSlot,
        int combatPotionCount,
        boolean hasSpeedEffect,
        boolean hasStrengthEffect,
        int nearbyDroppedTotemCount,
        int nearbyDroppedPearlCount,
        int nearbyDroppedPotionCount,
        int nearbyDroppedGappleCount,
    int railSlot,
    int railCount,
    int tntMinecartSlot,
    int tntMinecartCount,
        int nearbyDroppedCrystalCount,
        double nearestDroppedItemDistance,
        double nearestDroppedItemDx,
        double nearestDroppedItemDz,
        // Speedrun / dimension fields
        String dimensionId,
        int blazeRodCount,
        int eyeOfEnderCount,
        int flintAndSteelSlot,
        int flintAndSteelCount,
        int strongholdEstX,
        int strongholdEstZ,
        boolean strongholdTriangulated,
        // Explosive jump items (bedwars)
        int fireballSlot,
        int fireballCount,
        int tntSlot,
        int tntCount,
        // Boat travel
        int boatSlot,
        int boatCount
) {
    public static GameStateSnapshot capture(MinecraftClient client) {
        PlayerEntity player = client.player;
        if (player == null) {
            return new GameStateSnapshot(
                    // 1-7: position, rotation, health, food
                    0.0, 0.0, 0.0, 0f, 0f, 20f, 20,
                    // 8-15: movement state, velocity, fall distance, world time
                    true, false, false, false, 0.0, 0.0, 0f, 0L,
                    // 16-20: facing, look direction
                    "north", 0.0, 0.0, 0.0, "",
                    // 21-40: selected item count, weapon slots
                    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, 0,
                    // 37-40: inventory flags
                    false, false, false, false,
                    // 41-49: item counts
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    // 50-51: nearby entities
                    "", -1.0, 0.0, 0.0,
                    // 52-61: resource counts
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    // 62-70: enemy info
                    "", -1.0, 0.0, "", 0, false, false, 0.0, 0.0, 0.0, 0.0,
                    // 71-74: bed info
                    false, -1.0, 0, "",
                    // 75-77: focused entity, distance, pearl cooldown
                    "", -1.0, -1L,
                    // 78-89: potion/effects + dropped loot summary
                    -1, 0, false, false,
                    0, 0, 0, 0,
                    -1, 0, -1, 0,
                    0,
                    -1.0, 0.0, 0.0,
                    // 90-97: dimension + speedrun fields
                    "overworld", 0, 0, -1, 0, 0, 0, false,
                    // 98-101: explosive jump items
                    -1, 0, -1, 0,
                    // 102-103: boat travel
                    -1, 0
            );
        }

        String entityName = "";
        double distance = -1;
        if (client.crosshairTarget instanceof EntityHitResult entityHitResult) {
            Entity e = entityHitResult.getEntity();
            entityName = e.getType().getUntranslatedName();
            distance = player.distanceTo(e);
        }

        Direction horizontalFacing = player.getHorizontalFacing();
        Vec3d look = player.getRotationVec(1.0f);

        ItemStack selected = player.getMainHandStack();
        String selectedItem = selected.isEmpty() ? "" : Registries.ITEM.getId(selected.getItem()).toString();
        int selectedItemCount = selected.isEmpty() ? 0 : selected.getCount();

        int hotbarBlocks = 0;
        boolean hasBlocks = false;
        boolean hasWaterBucket = false;
        boolean hasMeleeWeapon = false;
        int swordSlot = -1;
        int axeSlot = -1;
        int pickaxeSlot = -1;
        int blockSlot = -1;
        int waterBucketSlot = -1;
        int utilityFoodSlot = -1;
        int cobwebSlot = -1;
        int obsidianSlot = -1;
        int endCrystalSlot = -1;
        int respawnAnchorSlot = -1;
        int glowstoneSlot = -1;
        int totemSlot = -1;
        int pearlSlot = -1;
        int breachMaceSlot = -1;
        int maceBreachLevel = 0;
        int bowSlot = -1;
        int windChargeSlot = -1;
        int windChargeCount = 0;
        int shieldSlot = -1;
        int fireballSlot = -1;
        int fireballCount = 0;
        int tntSlot = -1;
        int tntCount = 0;
        int boatSlot = -1;
        int boatCount = 0;
        int railSlot = -1;
        int railCount = 0;
        int tntMinecartSlot = -1;
        int tntMinecartCount = 0;

        int ironCount = 0;
        int redstoneCount = 0;
        int diamondCount = 0;
        int goldCount = 0;
        int emeraldCount = 0;
        int netheriteIngotCount = 0;
        int netheriteScrapCount = 0;
        int ancientDebrisCount = 0;
        int netheriteUpgradeTemplateCount = 0;
        int enchantedBookCount = 0;
        int cobwebCount = 0;
        int obsidianCount = 0;
        int endCrystalCount = 0;
        int respawnAnchorCount = 0;
        int glowstoneCount = 0;
        int totemCount = 0;
        int pearlCount = 0;
        int maceSlot = -1;
        int maceCount = 0;
        int bestHotbarMaceBreach = -1;
        boolean hasElytra = false;
        int combatPotionSlot = -1;
        int combatPotionCount = 0;
        boolean hasSpeedEffect = player.hasStatusEffect(StatusEffects.SPEED);
        boolean hasStrengthEffect = player.hasStatusEffect(StatusEffects.STRENGTH);

        int nearbyDroppedTotemCount = 0;
        int nearbyDroppedPearlCount = 0;
        int nearbyDroppedPotionCount = 0;
        int nearbyDroppedGappleCount = 0;
        int nearbyDroppedCrystalCount = 0;
        double nearestDroppedItemDistance = -1;
        double nearestDroppedItemDx = 0;
        double nearestDroppedItemDz = 0;

        // Speedrun-specific inventory
        int blazeRodCount = 0;
        int eyeOfEnderInvCount = 0;
        int flintAndSteelSlot = -1;
        int flintAndSteelCount = 0;

        // Dimension detection
        String dimensionId = "overworld";
        if (client.world != null) {
            if (client.world.getRegistryKey().equals(World.NETHER)) dimensionId = "nether";
            else if (client.world.getRegistryKey().equals(World.END)) dimensionId = "end";
        }

        int villagerNearbyCount = 0;
        String nearestHostile = "";
        double nearestHostileDistance = -1;
        double nearestHostileDx = 0, nearestHostileDz = 0;
        int bestFoodPriority = -1;

        for (int i = 0; i < 9; i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            if (itemId.contains("planks") || itemId.contains("wool") || itemId.contains("concrete") || itemId.contains("terracotta") || itemId.contains("stone") || itemId.contains("cobblestone") || itemId.contains("deepslate") || itemId.contains("brick") || itemId.contains("sandstone") || itemId.contains("netherrack") || itemId.contains("obsidian") || itemId.contains("log")) {
                hotbarBlocks += stack.getCount();
                hasBlocks = true;
                if (blockSlot < 0) blockSlot = i;
            }
            if (itemId.endsWith("water_bucket")) {
                hasWaterBucket = true;
                if (waterBucketSlot < 0) waterBucketSlot = i;
            }
            if (itemId.endsWith("_sword") || itemId.endsWith("_axe") || itemId.equals("minecraft:trident")) {
                hasMeleeWeapon = true;
            }
            if (itemId.endsWith("_sword") || itemId.equals("minecraft:trident")) {
                if (swordSlot < 0) swordSlot = i;
            }
            if (itemId.endsWith("_axe")) {
                if (axeSlot < 0) axeSlot = i;
            }
            if (itemId.endsWith("_pickaxe")) {
                if (pickaxeSlot < 0) pickaxeSlot = i;
            }
            {
                int fp = -1;
                if (itemId.endsWith("enchanted_golden_apple")) fp = 10;
                else if (itemId.endsWith("golden_apple")) fp = 9;
                else if (itemId.endsWith("cooked_beef") || itemId.endsWith("cooked_porkchop")) fp = 7;
                else if (itemId.endsWith("cooked_mutton") || itemId.endsWith("cooked_salmon") || itemId.endsWith("cooked_chicken") || itemId.endsWith("cooked_cod")) fp = 6;
                else if (itemId.endsWith("bread") || itemId.endsWith("pumpkin_pie") || itemId.endsWith("baked_potato")) fp = 4;
                else if (itemId.endsWith("apple") || itemId.endsWith("cooked_rabbit")) fp = 3;
                else if (itemId.endsWith("carrot") || itemId.endsWith("beetroot")) fp = 2;
                if (fp > bestFoodPriority) { bestFoodPriority = fp; utilityFoodSlot = i; }
            }
            if (itemId.endsWith("cobweb")) {
                if (cobwebSlot < 0) cobwebSlot = i;
            }
            if (itemId.endsWith("obsidian")) {
                if (obsidianSlot < 0) obsidianSlot = i;
            }
            if (itemId.endsWith("end_crystal")) {
                if (endCrystalSlot < 0) endCrystalSlot = i;
            }
            if (itemId.endsWith("respawn_anchor")) {
                if (respawnAnchorSlot < 0) respawnAnchorSlot = i;
            }
            if (itemId.endsWith("glowstone") || itemId.endsWith("glowstone_dust")) {
                if (glowstoneSlot < 0) glowstoneSlot = i;
            }
            if (itemId.endsWith("totem_of_undying")) {
                if (totemSlot < 0) totemSlot = i;
            }
            if (itemId.endsWith("ender_pearl")) {
                if (pearlSlot < 0) pearlSlot = i;
            }
            if (itemId.endsWith("mace")) {
                if (maceSlot < 0) maceSlot = i;
                int breachLevel = getBreachLevel(stack);
                if (breachLevel > bestHotbarMaceBreach) {
                    bestHotbarMaceBreach = breachLevel;
                    breachMaceSlot = i;
                    maceBreachLevel = breachLevel;
                }
            }
            if (combatPotionSlot < 0 && itemId.endsWith("potion")) {
                String potionName = stack.getName().getString().toLowerCase();
                if (potionName.contains("strength") || potionName.contains("swiftness") || potionName.contains("speed") || potionName.contains("regeneration") || potionName.contains("fire resistance")) {
                    combatPotionSlot = i;
                }
            }
            if (bowSlot < 0 && (itemId.endsWith("bow") && !itemId.endsWith("crossbow"))) {
                bowSlot = i;
            }
            if (windChargeSlot < 0 && itemId.endsWith("wind_charge")) {
                windChargeSlot = i;
            }
            if (shieldSlot < 0 && itemId.endsWith("shield")) {
                shieldSlot = i;
            }
            if (fireballSlot < 0 && itemId.endsWith("fire_charge")) {
                fireballSlot = i;
            }
            if (tntSlot < 0 && itemId.endsWith(":tnt")) {
                tntSlot = i;
            }
            if (railSlot < 0 && (itemId.endsWith(":rail") || itemId.endsWith("_rail"))) {
                railSlot = i;
            }
            if (tntMinecartSlot < 0 && itemId.endsWith("tnt_minecart")) {
                tntMinecartSlot = i;
            }
            if (boatSlot < 0 && itemId.endsWith("_boat")) {
                boatSlot = i;
            }
        }

        for (int i = 0; i < player.getInventory().size(); i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (stack.isEmpty()) continue;
            String itemId = Registries.ITEM.getId(stack.getItem()).toString();
            int count = stack.getCount();
            if (itemId.endsWith("iron_ingot")) ironCount += count;
            if (itemId.endsWith("redstone") || itemId.endsWith("redstone_dust")) redstoneCount += count;
            if (itemId.endsWith("diamond")) diamondCount += count;
            if (itemId.endsWith("gold_ingot")) goldCount += count;
            if (itemId.endsWith("emerald")) emeraldCount += count;
            if (itemId.endsWith("netherite_ingot")) netheriteIngotCount += count;
            if (itemId.endsWith("netherite_scrap")) netheriteScrapCount += count;
            if (itemId.endsWith("ancient_debris")) ancientDebrisCount += count;
            if (itemId.endsWith("netherite_upgrade_smithing_template")) netheriteUpgradeTemplateCount += count;
            if (itemId.endsWith("enchanted_book")) enchantedBookCount += count;
            if (itemId.endsWith("cobweb")) cobwebCount += count;
            if (itemId.endsWith("obsidian")) obsidianCount += count;
            if (itemId.endsWith("end_crystal")) endCrystalCount += count;
            if (itemId.endsWith("respawn_anchor")) respawnAnchorCount += count;
            if (itemId.endsWith("glowstone") || itemId.endsWith("glowstone_dust")) glowstoneCount += count;
            if (itemId.endsWith("totem_of_undying")) totemCount += count;
            if (itemId.endsWith("ender_pearl")) pearlCount += count;
            if (itemId.endsWith("mace")) maceCount += count;
            if (itemId.endsWith("wind_charge")) windChargeCount += count;
            if (itemId.endsWith("fire_charge")) fireballCount += count;
            if (itemId.endsWith(":tnt")) tntCount += count;
            if (itemId.endsWith(":rail") || itemId.endsWith("_rail")) railCount += count;
            if (itemId.endsWith("tnt_minecart")) tntMinecartCount += count;
            if (itemId.endsWith("_boat")) boatCount += count;
            if (itemId.endsWith("elytra")) hasElytra = true;
            if (itemId.endsWith("blaze_rod")) blazeRodCount += count;
            if (itemId.endsWith("ender_eye") || itemId.endsWith("eye_of_ender")) eyeOfEnderInvCount += count;
            if (itemId.endsWith("flint_and_steel")) {
                flintAndSteelCount += count;
                if (i < 9 && flintAndSteelSlot < 0) flintAndSteelSlot = i;
            }
            if (itemId.endsWith("potion")) {
                String potionName = stack.getName().getString().toLowerCase();
                if (potionName.contains("strength") || potionName.contains("swiftness") || potionName.contains("speed") || potionName.contains("regeneration") || potionName.contains("fire resistance")) {
                    combatPotionCount += count;
                }
            }
        }

        String nearestEnemyName = "";
        double nearestEnemyDistance = -1;
        double nearestEnemyHealth = 0;
        String nearestEnemyMainItem = "";
        int nearestEnemyArmorPieces = 0;
        boolean nearestEnemyHasMeleeWeapon = false;
        boolean nearestEnemyHasShield = false;
        double nearestEnemyVelX = 0;
        double nearestEnemyVelY = 0;
        double nearestEnemyVelZ = 0;
        double nearestEnemyDx = 0, nearestEnemyDy = 0, nearestEnemyDz = 0;

        if (client.world != null) {
            for (PlayerEntity other : client.world.getPlayers()) {
                if (other == null || other == player || other.isSpectator() || player.isTeammate(other)) continue;
                double dist = player.distanceTo(other);
                if (nearestEnemyDistance < 0 || dist < nearestEnemyDistance) {
                    nearestEnemyDistance = dist;
                    nearestEnemyName = other.getName().getString();
                    nearestEnemyHealth = other.getHealth();
                    nearestEnemyVelX = other.getVelocity().x;
                    nearestEnemyVelY = other.getVelocity().y;
                    nearestEnemyVelZ = other.getVelocity().z;
                    nearestEnemyDx = other.getX() - player.getX();
                    nearestEnemyDy = other.getY() - player.getY();
                    nearestEnemyDz = other.getZ() - player.getZ();

                    ItemStack enemyMain = other.getMainHandStack();
                    nearestEnemyMainItem = enemyMain.isEmpty() ? "" : Registries.ITEM.getId(enemyMain.getItem()).toString();
                    nearestEnemyHasMeleeWeapon = nearestEnemyMainItem.endsWith("_sword")
                            || nearestEnemyMainItem.endsWith("_axe")
                            || nearestEnemyMainItem.equals("minecraft:trident");

                    ItemStack enemyOffhand = other.getOffHandStack();
                    nearestEnemyHasShield = !enemyOffhand.isEmpty() && Registries.ITEM.getId(enemyOffhand.getItem()).toString().endsWith("shield");

                    int armorPieces = 0;
                    if (!other.getEquippedStack(EquipmentSlot.HEAD).isEmpty()) armorPieces++;
                    if (!other.getEquippedStack(EquipmentSlot.CHEST).isEmpty()) armorPieces++;
                    if (!other.getEquippedStack(EquipmentSlot.LEGS).isEmpty()) armorPieces++;
                    if (!other.getEquippedStack(EquipmentSlot.FEET).isEmpty()) armorPieces++;
                    nearestEnemyArmorPieces = armorPieces;
                }
            }

            for (Entity entity : client.world.getEntities()) {
                if (entity instanceof VillagerEntity villager && !villager.isSpectator()) {
                    if (player.distanceTo(villager) < 16) {
                        villagerNearbyCount++;
                    }
                }
                if (entity instanceof HostileEntity hostile && !hostile.isSpectator()) {
                    double dist = player.distanceTo(hostile);
                    if (nearestHostileDistance < 0 || dist < nearestHostileDistance) {
                        nearestHostileDistance = dist;
                                                nearestHostileDx = hostile.getX() - player.getX();
                                                nearestHostileDz = hostile.getZ() - player.getZ();
                        nearestHostile = hostile.getType().getUntranslatedName();
                    }
                }
                // Golems (iron golem, snow golem) are not HostileEntity but can be targeted
                if (entity instanceof GolemEntity golem && !golem.isSpectator()) {
                    double dist = player.distanceTo(golem);
                                            nearestHostileDx = golem.getX() - player.getX();
                                            nearestHostileDz = golem.getZ() - player.getZ();
                    if (nearestHostileDistance < 0 || dist < nearestHostileDistance) {
                        nearestHostileDistance = dist;
                        nearestHostile = golem.getType().getUntranslatedName();
                    }
                }

                if (entity instanceof ItemEntity itemEntity && !itemEntity.isRemoved()) {
                    ItemStack drop = itemEntity.getStack();
                    if (!drop.isEmpty()) {
                        String dropId = Registries.ITEM.getId(drop.getItem()).toString();
                        int dropCount = drop.getCount();
                        double dist = player.distanceTo(itemEntity);
                        if (dist < 12) {
                            boolean trackedLoot = false;
                            if (dropId.endsWith("totem_of_undying")) {
                                nearbyDroppedTotemCount += dropCount;
                                trackedLoot = true;
                            }
                            if (dropId.endsWith("ender_pearl")) {
                                nearbyDroppedPearlCount += dropCount;
                                trackedLoot = true;
                            }
                            if (dropId.endsWith("potion")) {
                                nearbyDroppedPotionCount += dropCount;
                                trackedLoot = true;
                            }
                            if (dropId.endsWith("golden_apple")) {
                                nearbyDroppedGappleCount += dropCount;
                                trackedLoot = true;
                            }
                            if (dropId.endsWith("end_crystal")) {
                                nearbyDroppedCrystalCount += dropCount;
                                trackedLoot = true;
                            }

                            if (trackedLoot && (nearestDroppedItemDistance < 0 || dist < nearestDroppedItemDistance)) {
                                nearestDroppedItemDistance = dist;
                                nearestDroppedItemDx = itemEntity.getX() - player.getX();
                                nearestDroppedItemDz = itemEntity.getZ() - player.getZ();
                            }
                        }
                    }
                }

                // Eye-of-ender entity tracking for stronghold triangulation
                String entityTypeId = Registries.ENTITY_TYPE.getId(entity.getType()).getPath();
                if ("eye_of_ender".equals(entityTypeId)) {
                    Vec3d vel = entity.getVelocity();
                    if (vel.horizontalLength() > 0.05) {
                        float travelYaw = (float) Math.toDegrees(Math.atan2(-vel.x, vel.z));
                        AiController.getInstance().recordEyeThrow(entity.getX(), entity.getZ(), travelYaw);
                    }
                }
            }
        }

        boolean bedNearby = false;
        double nearestBedDistance = -1;
        int nearestBedDefenseScore = 0;
        String nearestBedDefenseBlock = "";
        BlockPos nearestBedPos = findNearestBed(client, player, 14);
        if (nearestBedPos != null) {
            bedNearby = true;
            nearestBedDistance = Math.sqrt(player.squaredDistanceTo(
                    nearestBedPos.getX() + 0.5,
                    nearestBedPos.getY() + 0.5,
                    nearestBedPos.getZ() + 0.5
            ));

            DefenseInfo defense = analyzeBedDefense(client, nearestBedPos);
            nearestBedDefenseScore = defense.score;
            nearestBedDefenseBlock = defense.topBlock;
        }

        return new GameStateSnapshot(
                player.getX(),
                player.getY(),
                player.getZ(),
                player.getYaw(),
                player.getPitch(),
                player.getHealth(),
                player.getHungerManager().getFoodLevel(),
                player.isOnGround(),
                player.isSprinting(),
                player.isSneaking(),
                player.isTouchingWater(),
                player.getVelocity().y,
                Math.sqrt((player.getVelocity().x * player.getVelocity().x) + (player.getVelocity().z * player.getVelocity().z)),
                0f,
                System.currentTimeMillis() / 50,
                horizontalFacing.asString(),
                look.x,
                look.y,
                look.z,
                selectedItem,
                selectedItemCount,
                swordSlot,
                axeSlot,
                pickaxeSlot,
                blockSlot,
                waterBucketSlot,
                utilityFoodSlot,
                cobwebSlot,
                obsidianSlot,
                endCrystalSlot,
                respawnAnchorSlot,
                glowstoneSlot,
                totemSlot,
                pearlSlot,
                maceSlot,
                breachMaceSlot,
                maceBreachLevel,
                bowSlot,
                windChargeSlot,
                windChargeCount,
                shieldSlot,
                hotbarBlocks,
                hasBlocks,
                hasWaterBucket,
                hasMeleeWeapon,
                hasElytra,
                cobwebCount,
                obsidianCount,
                endCrystalCount,
                respawnAnchorCount,
                glowstoneCount,
                totemCount,
                pearlCount,
                maceCount,
                villagerNearbyCount,
                nearestHostile,
                nearestHostileDistance,
                nearestHostileDx,
                nearestHostileDz,
                ironCount,
                redstoneCount,
                diamondCount,
                goldCount,
                emeraldCount,
                netheriteIngotCount,
                netheriteScrapCount,
                ancientDebrisCount,
                netheriteUpgradeTemplateCount,
                enchantedBookCount,
                nearestEnemyName,
                nearestEnemyDistance,
                nearestEnemyHealth,
                nearestEnemyMainItem,
                nearestEnemyArmorPieces,
                nearestEnemyHasMeleeWeapon,
                nearestEnemyHasShield,
                nearestEnemyVelX,
                nearestEnemyVelY,
                nearestEnemyVelZ,
                nearestEnemyDy,
                nearestEnemyDx,
                nearestEnemyDz,
                bedNearby,
                nearestBedDistance,
                nearestBedDefenseScore,
                nearestBedDefenseBlock,
                entityName,
                distance,
                AiController.getInstance().getLastPearlUseTick(),
                combatPotionSlot,
                combatPotionCount,
                hasSpeedEffect,
                hasStrengthEffect,
                nearbyDroppedTotemCount,
                nearbyDroppedPearlCount,
                nearbyDroppedPotionCount,
                nearbyDroppedGappleCount,
                railSlot,
                railCount,
                tntMinecartSlot,
                tntMinecartCount,
                nearbyDroppedCrystalCount,
                nearestDroppedItemDistance,
                nearestDroppedItemDx,
                nearestDroppedItemDz,
                // speedrun / dimension
                dimensionId,
                blazeRodCount,
                eyeOfEnderInvCount,
                flintAndSteelSlot,
                flintAndSteelCount,
                AiController.getInstance().getStrongholdEstX(),
                AiController.getInstance().getStrongholdEstZ(),
                AiController.getInstance().isStrongholdTriangulated(),
                // explosive jump items
                fireballSlot,
                fireballCount,
                tntSlot,
                tntCount,
                // boat travel
                boatSlot,
                boatCount
        );
    }

    private static BlockPos findNearestBed(MinecraftClient client, PlayerEntity player, int radius) {
        if (client.world == null) return null;
        BlockPos center = player.getBlockPos();
        BlockPos nearest = null;
        double nearestDistSq = Double.MAX_VALUE;

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -4; dy <= 4; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    BlockPos pos = center.add(dx, dy, dz);
                    BlockState state = client.world.getBlockState(pos);
                    if (!(state.getBlock() instanceof BedBlock)) continue;

                    double distSq = player.squaredDistanceTo(pos.getX() + 0.5, pos.getY() + 0.5, pos.getZ() + 0.5);
                    if (distSq < nearestDistSq) {
                        nearestDistSq = distSq;
                        nearest = pos.toImmutable();
                    }
                }
            }
        }
        return nearest;
    }

    private static DefenseInfo analyzeBedDefense(MinecraftClient client, BlockPos bedPos) {
        if (client.world == null) return new DefenseInfo(0, "");
        int score = 0;
        Map<String, Integer> blockCounts = new HashMap<>();

        Box defenseBox = new Box(
                bedPos.getX() - 2, bedPos.getY() - 2, bedPos.getZ() - 2,
                bedPos.getX() + 3, bedPos.getY() + 3, bedPos.getZ() + 3
        );

        for (BlockPos p : BlockPos.iterate(
                (int) Math.floor(defenseBox.minX), (int) Math.floor(defenseBox.minY), (int) Math.floor(defenseBox.minZ),
                (int) Math.floor(defenseBox.maxX), (int) Math.floor(defenseBox.maxY), (int) Math.floor(defenseBox.maxZ)
        )) {
            BlockState state = client.world.getBlockState(p);
            Block block = state.getBlock();
            if (state.isAir() || block instanceof BedBlock) continue;
            score++;
            String id = Registries.BLOCK.getId(block).toString();
            blockCounts.merge(id, 1, Integer::sum);
        }

        String topBlock = "";
        int topCount = -1;
        for (Map.Entry<String, Integer> e : blockCounts.entrySet()) {
            if (e.getValue() > topCount) {
                topCount = e.getValue();
                topBlock = e.getKey();
            }
        }

        return new DefenseInfo(score, topBlock);
    }

    private static int getBreachLevel(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return 0;
        }
        try {
            var enchantments = EnchantmentHelper.getEnchantments(stack);
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

    private record DefenseInfo(int score, String topBlock) {}

    public JsonObject toJson() {
        JsonObject json = new JsonObject();
        json.addProperty("x", x);
        json.addProperty("y", y);
        json.addProperty("z", z);
        json.addProperty("yaw", yaw);
        json.addProperty("pitch", pitch);
        json.addProperty("health", health);
        json.addProperty("food", food);
        json.addProperty("onGround", onGround);
        json.addProperty("isSprinting", isSprinting);
        json.addProperty("isSneaking", isSneaking);
        json.addProperty("isTouchingWater", isTouchingWater);
        json.addProperty("verticalSpeed", verticalSpeed);
        json.addProperty("horizontalSpeed", horizontalSpeed);
        json.addProperty("fallDistance", fallDistance);
        json.addProperty("worldTime", worldTime);
        json.addProperty("facing", facing);
        json.addProperty("lookX", lookX);
        json.addProperty("lookY", lookY);
        json.addProperty("lookZ", lookZ);
        json.addProperty("selectedItem", selectedItem);
        json.addProperty("selectedItemCount", selectedItemCount);
        json.addProperty("swordSlot", swordSlot);
        json.addProperty("axeSlot", axeSlot);
        json.addProperty("pickaxeSlot", pickaxeSlot);
        json.addProperty("blockSlot", blockSlot);
        json.addProperty("waterBucketSlot", waterBucketSlot);
        json.addProperty("utilityFoodSlot", utilityFoodSlot);
        json.addProperty("cobwebSlot", cobwebSlot);
        json.addProperty("obsidianSlot", obsidianSlot);
        json.addProperty("endCrystalSlot", endCrystalSlot);
        json.addProperty("respawnAnchorSlot", respawnAnchorSlot);
        json.addProperty("glowstoneSlot", glowstoneSlot);
        json.addProperty("totemSlot", totemSlot);
        json.addProperty("pearlSlot", pearlSlot);
        json.addProperty("breachMaceSlot", breachMaceSlot);
        json.addProperty("maceBreachLevel", maceBreachLevel);
        json.addProperty("bowSlot", bowSlot);
        json.addProperty("windChargeSlot", windChargeSlot);
        json.addProperty("windChargeCount", windChargeCount);
        json.addProperty("shieldSlot", shieldSlot);
        json.addProperty("hotbarBlocks", hotbarBlocks);
        json.addProperty("hasBlocks", hasBlocks);
        json.addProperty("hasWaterBucket", hasWaterBucket);
        json.addProperty("hasMeleeWeapon", hasMeleeWeapon);
        json.addProperty("cobwebCount", cobwebCount);
        json.addProperty("obsidianCount", obsidianCount);
        json.addProperty("endCrystalCount", endCrystalCount);
        json.addProperty("respawnAnchorCount", respawnAnchorCount);
        json.addProperty("glowstoneCount", glowstoneCount);
        json.addProperty("totemCount", totemCount);
        json.addProperty("pearlCount", pearlCount);
        json.addProperty("villagerNearbyCount", villagerNearbyCount);
        json.addProperty("nearestHostile", nearestHostile);
        json.addProperty("nearestHostileDistance", nearestHostileDistance);
        json.addProperty("nearestHostileDx", nearestHostileDx);
        json.addProperty("nearestHostileDz", nearestHostileDz);
        json.addProperty("ironCount", ironCount);
        json.addProperty("redstoneCount", redstoneCount);
        json.addProperty("diamondCount", diamondCount);
        json.addProperty("goldCount", goldCount);
        json.addProperty("emeraldCount", emeraldCount);
        json.addProperty("netheriteIngotCount", netheriteIngotCount);
        json.addProperty("netheriteScrapCount", netheriteScrapCount);
        json.addProperty("ancientDebrisCount", ancientDebrisCount);
        json.addProperty("netheriteUpgradeTemplateCount", netheriteUpgradeTemplateCount);
        json.addProperty("enchantedBookCount", enchantedBookCount);
        json.addProperty("nearestEnemyName", nearestEnemyName);
        json.addProperty("nearestEnemyDistance", nearestEnemyDistance);
        json.addProperty("nearestEnemyHealth", nearestEnemyHealth);
        json.addProperty("nearestEnemyMainItem", nearestEnemyMainItem);
        json.addProperty("nearestEnemyArmorPieces", nearestEnemyArmorPieces);
        json.addProperty("nearestEnemyHasMeleeWeapon", nearestEnemyHasMeleeWeapon);
        json.addProperty("nearestEnemyVelX", nearestEnemyVelX);
        json.addProperty("nearestEnemyVelY", nearestEnemyVelY);
        json.addProperty("nearestEnemyVelZ", nearestEnemyVelZ);
        json.addProperty("nearestEnemyDy", nearestEnemyDy);
        json.addProperty("nearestEnemyDx", nearestEnemyDx);
        json.addProperty("nearestEnemyDz", nearestEnemyDz);
        json.addProperty("bedNearby", bedNearby);
        json.addProperty("nearestBedDistance", nearestBedDistance);
        json.addProperty("nearestBedDefenseScore", nearestBedDefenseScore);
        json.addProperty("nearestBedDefenseBlock", nearestBedDefenseBlock);
        json.addProperty("focusedEntity", focusedEntity);
        json.addProperty("focusedDistance", focusedDistance);
        json.addProperty("maceSlot", maceSlot);
        json.addProperty("maceCount", maceCount);
        json.addProperty("hasElytra", hasElytra);
        json.addProperty("nearestEnemyHasShield", nearestEnemyHasShield);
        json.addProperty("lastPearlUseTick", lastPearlUseTick);
        json.addProperty("combatPotionSlot", combatPotionSlot);
        json.addProperty("combatPotionCount", combatPotionCount);
        json.addProperty("hasSpeedEffect", hasSpeedEffect);
        json.addProperty("hasStrengthEffect", hasStrengthEffect);
        json.addProperty("nearbyDroppedTotemCount", nearbyDroppedTotemCount);
        json.addProperty("nearbyDroppedPearlCount", nearbyDroppedPearlCount);
        json.addProperty("nearbyDroppedPotionCount", nearbyDroppedPotionCount);
        json.addProperty("nearbyDroppedGappleCount", nearbyDroppedGappleCount);
        json.addProperty("nearbyDroppedCrystalCount", nearbyDroppedCrystalCount);
        json.addProperty("nearestDroppedItemDistance", nearestDroppedItemDistance);
        json.addProperty("nearestDroppedItemDx", nearestDroppedItemDx);
        json.addProperty("nearestDroppedItemDz", nearestDroppedItemDz);
        json.addProperty("dimensionId", dimensionId);
        json.addProperty("blazeRodCount", blazeRodCount);
        json.addProperty("eyeOfEnderCount", eyeOfEnderCount);
        json.addProperty("flintAndSteelSlot", flintAndSteelSlot);
        json.addProperty("flintAndSteelCount", flintAndSteelCount);
        json.addProperty("strongholdEstX", strongholdEstX);
        json.addProperty("strongholdEstZ", strongholdEstZ);
        json.addProperty("strongholdTriangulated", strongholdTriangulated);
        json.addProperty("fireballSlot", fireballSlot);
        json.addProperty("fireballCount", fireballCount);
        json.addProperty("tntSlot", tntSlot);
        json.addProperty("tntCount", tntCount);
        json.addProperty("boatSlot", boatSlot);
        json.addProperty("boatCount", boatCount);
        json.addProperty("railSlot", railSlot);
        json.addProperty("railCount", railCount);
        json.addProperty("tntMinecartSlot", tntMinecartSlot);
        json.addProperty("tntMinecartCount", tntMinecartCount);
        return json;
    }
}
