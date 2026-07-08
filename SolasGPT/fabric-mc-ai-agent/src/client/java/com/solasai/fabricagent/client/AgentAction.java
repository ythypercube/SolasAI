package com.solasai.fabricagent.client;

import com.google.gson.JsonObject;

public record AgentAction(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean jump,
        boolean sprint,
        boolean sneak,
        boolean attack,
        boolean use,
        int hotbarSlot,
        float yawDelta,
        float pitchDelta,
        float moveAngle,
        int durationTicks
) {
    public static AgentAction idle() {
        return new AgentAction(false, false, false, false, false, false, false, false, false, -1, 0f, 0f, Float.NaN, 6);
    }

    public static AgentAction fromJson(JsonObject action) {
        if (action == null) return idle();
        return new AgentAction(
                getBoolean(action, "forward"),
                getBoolean(action, "back"),
                getBoolean(action, "left"),
                getBoolean(action, "right"),
                getBoolean(action, "jump"),
                getBoolean(action, "sprint"),
                getBoolean(action, "sneak"),
                getBoolean(action, "attack"),
                getBoolean(action, "use"),
                getInt(action, "hotbarSlot", -1),
                getFloat(action, "yawDelta"),
                getFloat(action, "pitchDelta"),
                getFloat(action, "moveAngle", Float.NaN),
                Math.max(2, getInt(action, "durationTicks", 8))
        );
    }

    private static boolean getBoolean(JsonObject o, String key) {
        return o.has(key) && o.get(key).isJsonPrimitive() && o.get(key).getAsBoolean();
    }

    private static float getFloat(JsonObject o, String key, float fallback) {
        if (!o.has(key) || !o.get(key).isJsonPrimitive()) return fallback;
        return o.get(key).getAsFloat();
    }

    private static float getFloat(JsonObject o, String key) {
        return getFloat(o, key, 0f);
    }

    private static int getInt(JsonObject o, String key, int fallback) {
        if (!o.has(key) || !o.get(key).isJsonPrimitive()) return fallback;
        return o.get(key).getAsInt();
    }
}
