package com.solasai.fabricagent.client;

import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.argument;
import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.literal;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import org.lwjgl.glfw.GLFW;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.solasai.fabricagent.SolasAIFabricAgent;
import com.mojang.brigadier.arguments.StringArgumentType;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientReceiveMessageEvents;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.fabricmc.fabric.api.client.screen.v1.ScreenEvents;
import net.fabricmc.fabric.api.client.screen.v1.Screens;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.gui.screen.multiplayer.ConnectScreen;
import net.minecraft.client.gui.screen.multiplayer.MultiplayerScreen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.network.ServerAddress;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.text.Text;

public class SolasAIFabricAgentClient implements ClientModInitializer {
    private static final HttpClient BOT_SERVICE_HTTP = HttpClient.newBuilder().build();
    private static final String BOT_SERVICE_BASE = resolveBotServiceBase();
    private static final boolean CHAT_PROFILE_BUILD = isChatProfileBuild();
    private static final boolean AUTO_CHAT_REPLY_ENABLED = Boolean.parseBoolean(
            System.getProperty("solasai.chat.autoreply", CHAT_PROFILE_BUILD ? "true" : "false"));
    private boolean lastTabDown;
    private boolean lastCtrlDown;
    private boolean lastEscDown;
    private boolean lastF3Down;
    private int ctrlTapWindow;
    private int escTapWindow;
    private int f3TapWindow;
    private static boolean aiJoinEnabled = false;
    private static boolean aiJoinConnecting = false;
    private static boolean aiJoinStartedInWorld = false;
    private static String aiJoinObjective = "";
    private static String aiJoinServerAddress = "";
    private static String swarmBotCountDefault = "12";
    private static String swarmUsernameModeDefault = "numbered";
    private static String swarmBaseUsernameDefault = "Solas";
    private static String swarmJobsDefault = "miner,builder,farmer,guard,scout";
    private static String swarmAutoThinkDefault = "true";
    private static volatile boolean voiceEnabled = Boolean.parseBoolean(
            System.getProperty("solasai.voice.enabled", "true"));
    private static volatile boolean simpleVoiceChatMode = Boolean.parseBoolean(
            System.getProperty("solasai.voice.simplevc", "true"));
    private static volatile String voiceName = resolveVoiceName();

    @Override
    public void onInitializeClient() {
        ClientTickEvents.END_CLIENT_TICK.register(this::onClientTick);
        registerClientCommands();

        // HUD overlay: draw AI status when debug overlay is toggled on
        HudRenderCallback.EVENT.register((DrawContext context, RenderTickCounter tickCounter) -> {
            MinecraftClient mc = MinecraftClient.getInstance();
            if (mc == null || mc.player == null) return;
            AiController ai = AiController.getInstance();
            if (!ai.isActive() || !ai.isDebugOverlayVisible()) return;
            renderAiOverlay(context, mc, ai);
        });

        if (AUTO_CHAT_REPLY_ENABLED) {
            final AiServiceClient chatServiceClient = new AiServiceClient();
            final long[] lastChatReplyMs = {0L};
            
            // Only use CHAT event (more reliable); skip GAME to avoid duplicates
            ClientReceiveMessageEvents.CHAT.register((message, signedMessage, sender, params, receptionTimestamp) -> {
                String senderName = sender != null ? sender.name() : null;
                handleIncomingChatLine(chatServiceClient, lastChatReplyMs, message.getString(), senderName);
            });
        }

        SolasAIFabricAgent.LOGGER.info("SolasAI Fabric Agent client initialized. Ctrl+Tab start, Esc+Tab stop, F3+Tab overlay.");
    }

    private static boolean isChatProfileBuild() {
        try {
            return SolasAIFabricAgentClient.class.getClassLoader().getResource("solasai-chat-profile.flag") != null;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static String resolveBotServiceBase() {
        String fromProp = System.getProperty("solasai.botservice.base", "").trim();
        if (!fromProp.isBlank()) return fromProp.replaceAll("/$", "");

        String fromEnv = String.valueOf(System.getenv("SOLASAI_BOT_SERVICE_BASE")).trim();
        if (!fromEnv.equals("null") && !fromEnv.isBlank()) return fromEnv.replaceAll("/$", "");

        return "https://solasai-bot-service.onrender.com";
    }

    private static String resolveVoiceName() {
        String fromProp = System.getProperty("solasai.voice.name", "").trim();
        if (!fromProp.isBlank()) return fromProp;

        String fromEnv = String.valueOf(System.getenv("SOLASAI_VOICE")).trim();
        if (!fromEnv.equals("null") && !fromEnv.isBlank()) return fromEnv;

        return "en-US-BrianNeural";
    }

    private static String humanizePrompt(String promptText) {
        String trimmed = promptText == null ? "" : promptText.trim();
        if (trimmed.isBlank()) {
            return "";
        }
        return "Reply like a normal Minecraft player in 1-2 short sentences. "
                + "Do not say you are an AI, model, bot, or assistant. "
                + "No formal intro. Message: " + trimmed;
    }

    private static String humanizeReply(String rawReply) {
        String text = sanitizeOutgoingChat(rawReply);
        if (text.isBlank()) {
            return "";
        }

        text = text.replaceAll("(?i)^\\s*(hello|hi|hey)[!.\\s,-]*", "");
        text = text.replaceAll("(?i)\\b(i am|i'm)\\s+solas(ai|gpt)\\b[,.!\\s]*", "");
        text = text.replaceAll("(?i)\\bask me (anything|questions?)\\b[,.!\\s]*", "");
        text = text.replaceAll("(?i)\\b(as an ai|as a language model)\\b[,.!\\s]*", "");
        text = text.replaceAll("\\s+", " ").trim();

        if (text.isBlank()) {
            return "Yeah, what do you need?";
        }
        if (text.length() > 220) {
            text = text.substring(0, 220).trim();
        }

        String lower = text.toLowerCase(Locale.ROOT);
        if (lower.startsWith("i can help") || lower.startsWith("i am here to help")) {
            return "sure, what are you trying to do?";
        }
        return text;
    }

    private static String extractAddressedPrompt(String rawChat) {
        if (rawChat == null || rawChat.isBlank()) {
            return "";
        }
        String text = rawChat.trim();
        String lower = text.toLowerCase();
        int mentionIndex = lower.indexOf("solasai");
        if (mentionIndex < 0) {
            return text;
        }
        String tail = text.substring(mentionIndex + "solasai".length()).trim();
        tail = tail.replaceFirst("^[\\s:>,-]+", "").trim();
        return tail.isBlank() ? "Reply briefly in Minecraft chat." : tail;
    }

    /**
     * Returns true if the prompt is asking for sensitive player info
     * (location, inventory, health, username, server, etc.).
     */
    private static boolean isSensitiveQuery(String prompt) {
        if (prompt == null || prompt.isBlank()) return false;
        String p = prompt.toLowerCase();
        // Location / coordinates
        if (p.matches(".*\\bwhere(\\s+(are|r))?\\s+(you|u)\\b.*")) return true;
        if (p.matches(".*\\b(coords?|coordinates?|position|location|loc\\b|x\\s*y\\s*z|your\\s+base|your\\s+home|base\\s+loc|home\\s+loc)\\b.*")) return true;
        if (p.matches(".*\\bwhere\\s+(do\\s+you\\s+live|is\\s+your\\s+base|is\\s+your\\s+home|are\\s+you\\s+(at|living|hiding|staying))\\b.*")) return true;
        // Inventory / items / gear
        if (p.matches(".*\\b(inventory|inv\\b|your\\s+items?|what\\s+(items?|do\\s+you\\s+have|are\\s+you\\s+holding|\\s*gear|armor|armour|equipment)|hotbar|what.*in\\s+your\\s+(hand|slot|inventory))\\b.*")) return true;
        if (p.matches(".*\\b(your\\s+(gear|armor|armour|equipment|sword|axe|pickaxe|bow|items?|loot|stuff|things))\\b.*")) return true;
        if (p.matches(".*\\bwhat\\s+do\\s+you\\s+have\\b.*")) return true;
        if (p.matches(".*\\bshow\\s+(me\\s+)?(your\\s+)?(inv|inventory|items?|gear)\\b.*")) return true;
        // Health / food
        if (p.matches(".*\\b(your\\s+)?(health|hearts?|hp\\b|hunger|food\\s+level|starving)\\b.*")) return true;
        if (p.matches(".*\\bhow\\s+(much\\s+)?(health|hp|hearts?|food)\\b.*")) return true;
        // Username / identity
        if (p.matches(".*\\b(your\\s+)?(ign|username|user\\s+name|minecraft\\s+name|player\\s+name|account)\\b.*")) return true;
        if (p.matches(".*\\bwhat(\\s+is|'?s)?\\s+your\\s+(name|ign|username|account)\\b.*")) return true;
        if (p.matches(".*\\bwho\\s+are\\s+you\\b.*")) return true;
        // Server / IP
        if (p.matches(".*\\b(server\\s+(ip|address|name)|what\\s+server|which\\s+server|join\\s+what)\\b.*")) return true;
        return false;
    }

    private static void handleIncomingChatLine(AiServiceClient chatServiceClient, long[] lastChatReplyMs, String text, String senderName) {
        if (text == null || text.isBlank()) return;
        if (text.contains("SolasAI:")) return;
        if (!text.toLowerCase().contains("solasai")) return;

        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc == null || mc.player == null) return;
        String ownName = mc.player.getName().getString();
        String lower = text.toLowerCase();

        // Ignore direct self-chat lines and self death feed lines.
        if (senderName != null && senderName.equalsIgnoreCase(ownName)) return;
        if (lower.startsWith("<" + ownName.toLowerCase() + ">")) return;
        if (lower.startsWith(ownName.toLowerCase() + " ")
                && (lower.contains(" drowned")
                || lower.contains(" fell")
                || lower.contains(" was slain")
                || lower.contains(" was killed")
                || lower.contains(" blew up")
                || lower.contains(" burned")
                || lower.contains(" starved")
                || lower.contains(" tried to swim"))) {
            return;
        }

        String promptText = extractAddressedPrompt(text);
        if (promptText.isBlank()) return;

        long now = System.currentTimeMillis();
        if (now - lastChatReplyMs[0] < 4000L) return;
        lastChatReplyMs[0] = now;

        // Block sensitive queries locally — never send them to the backend or leak anything
        if (isSensitiveQuery(promptText)) {
            final String deflection = sanitizeOutgoingChat("SolasAI: I can't share that.");
            mc.execute(() -> {
                if (mc.player != null && mc.player.networkHandler != null) {
                    mc.player.networkHandler.sendChatMessage(deflection);
                }
            });
            return;
        }

        String safeName = "bot";
        String sessionId = "chat-" + safeName + "-" + now;

        chatServiceClient.requestChatReply(sessionId, humanizePrompt(promptText))
                .whenComplete((reply, err) -> {
                    if (reply == null || reply.isBlank()) return;
                    String refinedReply = humanizeReply(reply);
                    String full = sanitizeOutgoingChat(refinedReply);
                    if (full.length() > 256) full = full.substring(0, 256);
                    if (full.isBlank()) return;
                    final String localMessage = full;
                    final String spokenReply = refinedReply;
                    mc.execute(() -> {
                        if (mc.player != null && mc.player.networkHandler != null) {
                            mc.player.networkHandler.sendChatMessage(localMessage);
                            if (voiceEnabled) {
                                speakText(spokenReply);
                            }
                        }
                    });
                });
    }

    private static void speakText(String text) {
        String clean = sanitizeOutgoingChat(text);
        if (clean.isBlank()) {
            return;
        }
        if (simpleVoiceChatMode) {
            clean = clean.replaceAll("(?i)^\\s*solasai:\\s*", "");
        }
        if (clean.length() > 240) {
            clean = clean.substring(0, 240);
        }

        final String scriptPath = "/mnt/data/SolasAI/turbowarp-ai-backend/solasai_speak.py";
        final String spoken = clean;

        Thread t = new Thread(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                        "/usr/bin/python3",
                        scriptPath,
                        "--voice",
                        voiceName,
                        spoken
                );
                pb.redirectErrorStream(true);
                Process p = pb.start();
                p.waitFor(20, TimeUnit.SECONDS);
            } catch (Exception ignored) {
                // Don't interrupt gameplay for TTS failures.
            }
        }, "solasai-tts");
        t.setDaemon(true);
        t.start();
    }

    private static String sanitizeOutgoingChat(String raw) {
        if (raw == null || raw.isBlank()) {
            return "";
        }
        StringBuilder builder = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char ch = raw.charAt(i);
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                builder.append(' ');
                continue;
            }
            if (!Character.isISOControl(ch)
                    && !Character.isSurrogate(ch)
                    && Character.getType(ch) != Character.PRIVATE_USE
                    && ch != 0x7F) {
                builder.append(ch);
            }
        }
        String cleaned = builder.toString().replaceAll("\\s+", " ").trim();
        return cleaned;
    }

    private static String currentServerKeyForBases(MinecraftClient mc) {
        if (mc == null) {
            return "unknown:25565";
        }
        if (mc.getCurrentServerEntry() != null && mc.getCurrentServerEntry().address != null) {
            String address = mc.getCurrentServerEntry().address.trim().toLowerCase();
            if (!address.isBlank()) {
                return address.contains(":") ? address : address + ":25565";
            }
        }
        if (mc.isInSingleplayer()) {
            return "singleplayer:0";
        }
        return "unknown:25565";
    }

    private static void requestAndPrintBasesForCurrentServer() {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc == null || mc.player == null) {
            return;
        }

        String serverKey = currentServerKeyForBases(mc);
        String encodedServer = URLEncoder.encode(serverKey, StandardCharsets.UTF_8);
        String url = BOT_SERVICE_BASE + "/bases?server=" + encodedServer;
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .build();

        BOT_SERVICE_HTTP.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenAccept(response -> {
                    if (response.statusCode() < 200 || response.statusCode() >= 300) {
                        mc.execute(() -> {
                            if (mc.player != null) {
                                mc.player.sendMessage(Text.literal("SolasAI bases lookup failed: HTTP " + response.statusCode()), false);
                            }
                        });
                        return;
                    }

                    try {
                        JsonObject root = JsonParser.parseString(response.body()).getAsJsonObject();
                        JsonArray bases = root.has("bases") && root.get("bases").isJsonArray()
                                ? root.getAsJsonArray("bases")
                                : new JsonArray();

                        mc.execute(() -> {
                            if (mc.player == null) {
                                return;
                            }
                            mc.player.sendMessage(Text.literal("SolasAI base candidates for " + serverKey + ": " + bases.size()), false);
                            for (JsonElement el : bases) {
                                if (!el.isJsonObject()) continue;
                                JsonObject entry = el.getAsJsonObject();
                                String kind = entry.has("kind") ? entry.get("kind").getAsString() : "candidate";
                                int x = entry.has("x") ? entry.get("x").getAsInt() : 0;
                                int y = entry.has("y") ? entry.get("y").getAsInt() : 0;
                                int z = entry.has("z") ? entry.get("z").getAsInt() : 0;
                                double confidence = entry.has("confidence") ? entry.get("confidence").getAsDouble() : 0.0;
                                String details = entry.has("details") ? entry.get("details").getAsString() : "";
                                String line = String.format("- %s @ %d %d %d (conf=%.2f)%s",
                                        kind,
                                        x,
                                        y,
                                        z,
                                        confidence,
                                        details.isBlank() ? "" : " [" + details + "]");
                                mc.player.sendMessage(Text.literal(line), false);
                            }
                            if (bases.isEmpty()) {
                                mc.player.sendMessage(Text.literal("No base candidates saved yet for this server."), false);
                            }
                        });
                    } catch (Exception parseError) {
                        mc.execute(() -> {
                            if (mc.player != null) {
                                mc.player.sendMessage(Text.literal("SolasAI bases parse error: " + parseError.getMessage()), false);
                            }
                        });
                    }
                })
                .exceptionally(err -> {
                    mc.execute(() -> {
                        if (mc.player != null) {
                            mc.player.sendMessage(Text.literal("SolasAI bases lookup failed. Is bot service online at " + BOT_SERVICE_BASE + "?"), false);
                        }
                    });
                    return null;
                });
    }

    private static void sendAskPromptToChat(String prompt) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc == null || mc.player == null) {
            return;
        }

        String cleanedPrompt = prompt == null ? "" : prompt.trim();
        if (cleanedPrompt.isBlank()) {
            mc.player.sendMessage(Text.literal("SolasAI ask: prompt cannot be empty."), false);
            return;
        }

        String sessionId = "chat-bot-" + System.currentTimeMillis();
        AiServiceClient chatServiceClient = new AiServiceClient();

        chatServiceClient.requestChatReply(sessionId, humanizePrompt(cleanedPrompt))
                .whenComplete((reply, err) -> {
                    mc.execute(() -> {
                        if (mc.player == null) {
                            return;
                        }

                        if (err != null) {
                            mc.player.sendMessage(Text.literal("SolasAI ask failed: backend error."), false);
                            return;
                        }

                        if (reply == null || reply.isBlank()) {
                            mc.player.sendMessage(Text.literal("SolasAI ask failed: no response from backend."), false);
                            return;
                        }

                        String refinedReply = humanizeReply(reply);
                        String full = sanitizeOutgoingChat(refinedReply);
                        if (full.length() > 256) full = full.substring(0, 256);
                        if (full.isBlank()) {
                            mc.player.sendMessage(Text.literal("SolasAI ask failed: empty response."), false);
                            return;
                        }

                        if (mc.player.networkHandler != null) {
                            mc.player.networkHandler.sendChatMessage(full);
                            if (voiceEnabled) {
                                speakText(refinedReply);
                            }
                        }
                    });
                });
    }

    private void registerClientCommands() {
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            dispatcher.register(literal("solasai")
                    .then(literal("stronghold")
                            .executes(ctx -> {
                                AiController ai = AiController.getInstance();
                                if (ai.isStrongholdTriangulated()) {
                                    ctx.getSource().sendFeedback(Text.literal("Stronghold estimate: ~" + ai.getStrongholdEstX() + ", ~" + ai.getStrongholdEstZ()));
                                } else {
                                    ctx.getSource().sendFeedback(Text.literal("Stronghold not triangulated yet. Throw eye once, move far, throw eye again."));
                                }
                                return 1;
                            }))
                    .then(literal("task")
                            .executes(ctx -> {
                                AiController ai = AiController.getInstance();
                                ctx.getSource().sendFeedback(Text.literal("Current task: " + ai.getCurrentTask()));
                                return 1;
                            }))
                        .then(literal("tas")
                            .executes(ctx -> {
                            AiController ai = AiController.getInstance();
                            ctx.getSource().sendFeedback(Text.literal("Current task: " + ai.getCurrentTask()));
                            return 1;
                            }))
                    .then(literal("backend")
                            .executes(ctx -> {
                                ctx.getSource().sendFeedback(Text.literal("SolasAI backend: " + AiServiceClient.getBackendEndpoint()));
                                return 1;
                            })
                            .then(argument("url", StringArgumentType.greedyString())
                                    .executes(ctx -> {
                                        String url = StringArgumentType.getString(ctx, "url");
                                        AiServiceClient.setBackendEndpoint(url);
                                        ctx.getSource().sendFeedback(Text.literal("SolasAI backend updated: " + AiServiceClient.getBackendEndpoint()));
                                        return 1;
                                    })))
                    .then(literal("bases")
                            .executes(ctx -> {
                                requestAndPrintBasesForCurrentServer();
                                return 1;
                            }))
                    .then(literal("ask")
                            .then(argument("prompt", StringArgumentType.greedyString())
                                    .executes(ctx -> {
                                        String prompt = StringArgumentType.getString(ctx, "prompt");
                                        sendAskPromptToChat(prompt);
                                        return 1;
                                    })))
                    .then(literal("hear")
                            .then(argument("spoken", StringArgumentType.greedyString())
                                    .executes(ctx -> {
                                        String spoken = StringArgumentType.getString(ctx, "spoken");
                                        sendAskPromptToChat(spoken);
                                        return 1;
                                    })))
                    .then(literal("voice")
                            .executes(ctx -> {
                                voiceEnabled = !voiceEnabled;
                                ctx.getSource().sendFeedback(Text.literal("SolasAI voice: " + (voiceEnabled ? "ON" : "OFF")
                                        + " (" + voiceName + ")"));
                                return 1;
                            })
                            .then(literal("name")
                                    .then(argument("voice", StringArgumentType.greedyString())
                                            .executes(ctx -> {
                                                String next = StringArgumentType.getString(ctx, "voice").trim();
                                                if (!next.isBlank()) {
                                                    voiceName = next;
                                                }
                                                ctx.getSource().sendFeedback(Text.literal("SolasAI voice set: " + voiceName));
                                                return 1;
                                            })))
                            .then(literal("test")
                                    .executes(ctx -> {
                                        speakText("yo, comms check. can you hear me?");
                                        ctx.getSource().sendFeedback(Text.literal("SolasAI voice test played."));
                                        return 1;
                                    }))
                            .then(literal("simplevc")
                                    .executes(ctx -> {
                                        simpleVoiceChatMode = !simpleVoiceChatMode;
                                        ctx.getSource().sendFeedback(Text.literal("SolasAI Simple Voice Chat mode: " + (simpleVoiceChatMode ? "ON" : "OFF")));
                                        return 1;
                                    })
                                    .then(literal("on")
                                            .executes(ctx -> {
                                                simpleVoiceChatMode = true;
                                                ctx.getSource().sendFeedback(Text.literal("SolasAI Simple Voice Chat mode: ON"));
                                                return 1;
                                            }))
                                    .then(literal("off")
                                            .executes(ctx -> {
                                                simpleVoiceChatMode = false;
                                                ctx.getSource().sendFeedback(Text.literal("SolasAI Simple Voice Chat mode: OFF"));
                                                return 1;
                                            })))
                            .then(literal("on")
                                    .executes(ctx -> {
                                        voiceEnabled = true;
                                        ctx.getSource().sendFeedback(Text.literal("SolasAI voice: ON (" + voiceName + ")"));
                                        return 1;
                                    }))
                            .then(literal("off")
                                    .executes(ctx -> {
                                        voiceEnabled = false;
                                        ctx.getSource().sendFeedback(Text.literal("SolasAI voice: OFF"));
                                        return 1;
                                    }))));

            dispatcher.register(literal("askai")
                    .then(argument("prompt", StringArgumentType.greedyString())
                            .executes(ctx -> {
                                String prompt = StringArgumentType.getString(ctx, "prompt");
                                sendAskPromptToChat(prompt);
                                return 1;
                            })));
        });
    }

    private void renderAiOverlay(DrawContext context, MinecraftClient mc, AiController ai) {
        int x = 4;
        int y = 4;
        int lineH = 10;
        int bg = 0x99000000; // semi-transparent black

        String mode = ai.getLastMode();
        String obj = ai.getObjective();
        String note = ai.getLastNote();

        // Trim long strings so they fit without wrapping
        if (obj.length() > 72) obj = obj.substring(0, 69) + "...";
        if (note.length() > 90) note = note.substring(0, 87) + "...";

        String line0 = "\u00a7eSolasAI \u00a77[\u00a7aON\u00a77] doing: \u00a7b" + mode;
        String line1 = "\u00a77Going: \u00a7f" + obj;
        String line2 = "\u00a77Mind: \u00a7f" + (note.isEmpty() ? "(waiting...)" : note);

        context.fill(x - 2, y - 2, x + mc.textRenderer.getWidth(line0) + 2, y + lineH * 3 + 4, bg);
        context.drawText(mc.textRenderer, line0, x, y, 0xFFFFFF, false);
        context.drawText(mc.textRenderer, line1, x, y + lineH, 0xFFFFFF, false);
        context.drawText(mc.textRenderer, line2, x, y + lineH * 2, 0xFFFFFF, false);

        if (ai.isStrongholdTriangulated()) {
            String line3 = "\u00a7bStronghold estimate: \u00a7f~" + ai.getStrongholdEstX() + ", ~" + ai.getStrongholdEstZ();
            context.fill(x - 2, y + lineH * 3, x + mc.textRenderer.getWidth(line3) + 2, y + lineH * 4 + 2, bg);
            context.drawText(mc.textRenderer, line3, x, y + lineH * 3, 0xFFFFFF, false);
        }
    }

    private void onClientTick(MinecraftClient client) {
        if (client.getWindow() == null) {
            return;
        }

        long window = client.getWindow().getHandle();
        boolean tabDown = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_TAB) == GLFW.GLFW_PRESS;
        boolean ctrlDown = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_LEFT_CONTROL) == GLFW.GLFW_PRESS
                || GLFW.glfwGetKey(window, GLFW.GLFW_KEY_RIGHT_CONTROL) == GLFW.GLFW_PRESS;
        boolean escDown = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS;
        boolean f3Down = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_F3) == GLFW.GLFW_PRESS;

        boolean tabPressed = tabDown && !lastTabDown;
        boolean ctrlPressed = ctrlDown && !lastCtrlDown;
        boolean escPressed = escDown && !lastEscDown;
        boolean f3Pressed = f3Down && !lastF3Down;

        if (ctrlTapWindow > 0) ctrlTapWindow--;
        if (escTapWindow > 0) escTapWindow--;
        if (f3TapWindow > 0) f3TapWindow--;
        if (ctrlPressed) ctrlTapWindow = 8;
        if (escPressed) escTapWindow = 8;
        if (f3Pressed) f3TapWindow = 8;

        if (tabPressed && (ctrlDown || ctrlTapWindow > 0)) {
            if (client.player != null && client.currentScreen == null) {
                client.setScreen(new AiPromptScreen());
            }
        }

        if (escPressed && AiController.getInstance().isActive()) {
            AiController.getInstance().stop(client, "SolasAI control disabled.");
        }

        if (tabPressed && (escDown || escTapWindow > 0)) {
            if (AiController.getInstance().isActive()) {
                AiController.getInstance().stop(client, "SolasAI control disabled.");
            }
        }

        // F3 + Tab → toggle AI status overlay (no screen required, works any time)
        if (tabPressed && (f3Down || f3TapWindow > 0)) {
            AiController ai = AiController.getInstance();
            ai.toggleDebugOverlay();
            if (client.player != null) {
                String state = ai.isDebugOverlayVisible() ? "ON" : "OFF";
                client.player.sendMessage(
                    Text.literal("\u00a7eSolasAI overlay \u00a7f" + state), false);
            }
        }

        lastTabDown = tabDown;
        lastCtrlDown = ctrlDown;
        lastEscDown = escDown;
        lastF3Down = f3Down;

        if (aiJoinEnabled) {
            if (client.world != null && client.player != null) {
                aiJoinConnecting = false;
                if (!aiJoinStartedInWorld && !AiController.getInstance().isActive()) {
                    AiController.getInstance().start(client, aiJoinObjective);
                    aiJoinStartedInWorld = true;
                    client.player.sendMessage(Text.literal("SolasAI AI Join active: " + aiJoinObjective), false);
                }
            } else {
                aiJoinStartedInWorld = false;
            }
        }

        AiController.getInstance().tick(client);
    }

    public static boolean isAiJoinEnabled() {
        return aiJoinEnabled;
    }

    public static String getAiJoinObjective() {
        return aiJoinObjective;
    }

    public static String getAiJoinServerAddress() {
        return aiJoinServerAddress;
    }

    public static String getSwarmBotCountDefault() {
        return swarmBotCountDefault;
    }

    public static String getSwarmUsernameModeDefault() {
        return swarmUsernameModeDefault;
    }

    public static String getSwarmBaseUsernameDefault() {
        return swarmBaseUsernameDefault;
    }

    public static String getSwarmJobsDefault() {
        return swarmJobsDefault;
    }

    public static String getSwarmAutoThinkDefault() {
        return swarmAutoThinkDefault;
    }

    public static void enableAiJoin(MinecraftClient client, Screen parent, String objective, String serverAddress) {
        String normalizedObjective = objective == null ? "" : objective.trim();
        String normalizedAddress = serverAddress == null ? "" : serverAddress.trim();
        if (normalizedObjective.isBlank() || normalizedAddress.isBlank()) {
            if (client != null && client.player != null) {
                client.player.sendMessage(Text.literal("SolasAI AI Join: server and objective are required."), false);
            }
            return;
        }

        aiJoinObjective = normalizedObjective.length() > 400 ? normalizedObjective.substring(0, 400) : normalizedObjective;
        aiJoinServerAddress = normalizedAddress;
        aiJoinEnabled = true;
        aiJoinConnecting = true;
        aiJoinStartedInWorld = false;

        try {
            ServerAddress parsedAddress = ServerAddress.parse(aiJoinServerAddress);
            ServerInfo targetServer = new ServerInfo("SolasAI AI Join", aiJoinServerAddress, ServerInfo.ServerType.OTHER);
            ConnectScreen.connect(parent, client, parsedAddress, targetServer, false, null);
        } catch (Exception ex) {
            aiJoinEnabled = false;
            aiJoinConnecting = false;
            aiJoinStartedInWorld = false;
            if (client != null && client.player != null) {
                client.player.sendMessage(Text.literal("SolasAI AI Join failed: " + ex.getMessage()), false);
            }
        }
    }

    public static void startSwarmExperiment(
            MinecraftClient client,
            Screen parent,
            String serverAddress,
            String objective,
            int count,
            String usernameMode,
            String baseUsername,
            String jobsCsv,
            boolean autoThink
    ) {
        String normalizedAddress = serverAddress == null ? "" : serverAddress.trim();
        String normalizedObjective = objective == null ? "" : objective.trim();
        if (normalizedAddress.isBlank()) {
            if (client != null && client.player != null) {
                client.player.sendMessage(Text.literal("SolasAI swarm: server address is required."), false);
            }
            return;
        }

        int safeCount = Math.max(1, Math.min(500, count));
        String safeMode = (usernameMode == null ? "numbered" : usernameMode.trim().toLowerCase());
        if (safeMode.isBlank()) safeMode = "numbered";
        String safeBase = (baseUsername == null ? "Solas" : baseUsername.trim());
        if (safeBase.isBlank()) safeBase = "Solas";
        String safeJobs = jobsCsv == null ? "" : jobsCsv.trim();

        swarmBotCountDefault = Integer.toString(safeCount);
        swarmUsernameModeDefault = safeMode;
        swarmBaseUsernameDefault = safeBase;
        swarmJobsDefault = safeJobs;
        swarmAutoThinkDefault = Boolean.toString(autoThink);

        ServerAddress parsed;
        try {
            parsed = ServerAddress.parse(normalizedAddress);
        } catch (Exception ex) {
            if (client != null && client.player != null) {
                client.player.sendMessage(Text.literal("SolasAI swarm: invalid server address."), false);
            }
            return;
        }

        JsonObject payload = new JsonObject();
        payload.addProperty("host", parsed.getAddress());
        payload.addProperty("port", parsed.getPort());
        payload.addProperty("auth", "offline");
        payload.addProperty("count", safeCount);
        payload.addProperty("usernameMode", safeMode);
        payload.addProperty("baseUsername", safeBase);
        payload.addProperty("objective", normalizedObjective);
        payload.addProperty("jobs", safeJobs);
        payload.addProperty("autoThink", autoThink);
        payload.addProperty("chatReadEnabled", true);
        payload.addProperty("launch", true);
        payload.addProperty("launchCount", safeCount);
        payload.addProperty("basePort", 8800);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BOT_SERVICE_BASE + "/swarm/start"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload.toString()))
                .build();

        BOT_SERVICE_HTTP.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenAccept(response -> {
                    MinecraftClient mc = MinecraftClient.getInstance();
                    if (mc == null) return;
                    mc.execute(() -> {
                        if (mc.player == null) return;
                        if (response.statusCode() >= 200 && response.statusCode() < 300) {
                            mc.player.sendMessage(Text.literal("SolasAI swarm: request accepted for " + safeCount + " bots."), false);
                            if (parent != null) {
                                mc.setScreen(parent);
                            } else {
                                mc.setScreen(null);
                            }
                        } else {
                            mc.player.sendMessage(Text.literal("SolasAI swarm failed: HTTP " + response.statusCode()), false);
                        }
                    });
                })
                .exceptionally(error -> {
                    MinecraftClient mc = MinecraftClient.getInstance();
                    if (mc != null) {
                        mc.execute(() -> {
                            if (mc.player != null) {
                                mc.player.sendMessage(Text.literal("SolasAI swarm failed: bot-service offline?"), false);
                            }
                        });
                    }
                    return null;
                });
    }

    public static void disableAiJoin(MinecraftClient client, boolean leaveServer) {
        aiJoinEnabled = false;
        aiJoinConnecting = false;
        aiJoinStartedInWorld = false;
        if (AiController.getInstance().isActive()) {
            AiController.getInstance().stop(client, "SolasAI AI Join stopped.");
        }
        if (leaveServer && client != null && client.world != null) {
            client.disconnect(Text.literal("SolasAI AI Join stopped."));
            client.setScreen(new MultiplayerScreen(new TitleScreen()));
        }
    }
}

