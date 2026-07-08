package com.solasai.fabricagent.client;

/** Holds the full response from one /mc-agent call. */
public record BackendDecision(AgentAction action, String note, String mode) {
    public static BackendDecision ofAction(AgentAction action) {
        return new BackendDecision(action, "", "general");
    }
}
