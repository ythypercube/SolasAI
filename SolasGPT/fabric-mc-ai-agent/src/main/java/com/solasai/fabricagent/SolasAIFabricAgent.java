package com.solasai.fabricagent;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.fabricmc.api.ModInitializer;

public class SolasAIFabricAgent implements ModInitializer {
    public static final String MOD_ID = "solasai_fabric_agent";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("SolasAI Fabric Agent initialized");
    }
}
