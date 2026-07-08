package com.solasai.fabricagent.client;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class AiServiceClient {
    private static final String DEFAULT_ENDPOINT = "https://solasai-backend.onrender.com/mc-agent";
    private static final String LOCAL_FALLBACK_ENDPOINT = "http://127.0.0.1:8787/mc-agent";
    private static volatile String configuredEndpoint = resolveInitialEndpoint();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    public static String getBackendEndpoint() {
        return configuredEndpoint;
    }

    public static void setBackendEndpoint(String endpoint) {
        configuredEndpoint = normalizeEndpoint(endpoint);
    }

    private static String resolveInitialEndpoint() {
        String endpoint = System.getProperty("solasai.backend.endpoint", "").trim();
        if (endpoint.isEmpty()) {
            endpoint = System.getenv("SOLASAI_BACKEND_ENDPOINT");
            endpoint = endpoint == null ? "" : endpoint.trim();
        }
        if (endpoint.isEmpty()) {
            endpoint = DEFAULT_ENDPOINT;
        }
        return normalizeEndpoint(endpoint);
    }

    private static String normalizeEndpoint(String endpoint) {
        String value = endpoint == null ? "" : endpoint.trim();
        if (value.isEmpty()) {
            return DEFAULT_ENDPOINT;
        }
        if (value.endsWith("/")) {
            value = value.substring(0, value.length() - 1);
        }
        if (value.endsWith("/mc")) {
            value = value.substring(0, value.length() - 3) + "/mc-agent";
        }
        if (!value.endsWith("/mc-agent")) {
            value = value + "/mc-agent";
        }
        return value;
    }

    public CompletableFuture<BackendDecision> requestDecision(String sessionId, String objective, GameStateSnapshot snapshot) {
        String endpoint = configuredEndpoint;

        JsonObject body = new JsonObject();
        body.addProperty("sessionId", sessionId);
        body.addProperty("objective", objective);
        body.add("state", snapshot.toJson());

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(8))
                .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
                .thenApply(response -> {
                    if (response.statusCode() < 200 || response.statusCode() >= 300) {
                        String responseBody = response.body() == null ? "" : response.body();
                        String lowerBody = responseBody.toLowerCase();
                        if (response.statusCode() == 404 && (lowerBody.contains("cannot post /mc") || lowerBody.contains("cannot post /mc-agent"))) {
                            throw new RuntimeException(
                                    "AI backend missing /mc-agent route on deployed server. "
                                            + "Deploy latest turbowarp-ai-backend server.js to Render.");
                        }
                        if (responseBody.length() > 160) {
                            responseBody = responseBody.substring(0, 160);
                        }
                        throw new RuntimeException("AI backend HTTP " + response.statusCode() + ": " + responseBody);
                    }
                    return parseDecision(response.body());
                });
    }

    /** POST to /chat-plain: pass player chat to the AI and return a plain-text reply. */
    public CompletableFuture<String> requestChatReply(String sessionId, String message) {
        return requestChatReplyOnce(configuredEndpoint, sessionId, message)
                .thenCompose(primary -> {
                    if (primary != null && !primary.isBlank()) {
                        return CompletableFuture.completedFuture(primary);
                    }
                    if (normalizeEndpoint(configuredEndpoint).equals(normalizeEndpoint(LOCAL_FALLBACK_ENDPOINT))) {
                        return CompletableFuture.completedFuture(null);
                    }
                    return requestChatReplyOnce(LOCAL_FALLBACK_ENDPOINT, sessionId, message);
                });
    }

    private CompletableFuture<String> requestChatReplyOnce(String endpoint, String sessionId, String message) {
        String base = normalizeEndpoint(endpoint);
        if (base.endsWith("/mc-agent")) {
            base = base.substring(0, base.length() - "/mc-agent".length());
        }
        String chatEndpoint = base + "/chat-plain";

        JsonObject body = new JsonObject();
        body.addProperty("sessionId", sessionId);
        body.addProperty("message", message);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(chatEndpoint))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(10))
                .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
                .thenApply(response -> {
                    if (response.statusCode() < 200 || response.statusCode() >= 300) return null;
                    String text = response.body() == null ? "" : response.body().trim();
                    if (text.startsWith("ERROR:") || text.isEmpty()) return null;
                    return text;
                })
                .exceptionally(err -> null);
    }

    /** Legacy alias used by nothing currently – kept to avoid future breakage. */
    public CompletableFuture<AgentAction> requestAction(String sessionId, String objective, GameStateSnapshot snapshot) {
        return requestDecision(sessionId, objective, snapshot).thenApply(BackendDecision::action);
    }

    private BackendDecision parseDecision(String jsonText) {
        JsonObject root = JsonParser.parseString(jsonText).getAsJsonObject();
        if (!root.has("ok") || !root.get("ok").getAsBoolean()) {
            String error = root.has("error") ? root.get("error").getAsString() : "Unknown backend error";
            throw new RuntimeException(error);
        }
        JsonObject action = root.has("action") && root.get("action").isJsonObject()
                ? root.getAsJsonObject("action")
                : new JsonObject();
        String note = root.has("note") && root.get("note").isJsonPrimitive()
                ? root.get("note").getAsString() : "";
        String mode = root.has("mode") && root.get("mode").isJsonPrimitive()
                ? root.get("mode").getAsString() : "general";
        return new BackendDecision(AgentAction.fromJson(action), note, mode);
    }
}
