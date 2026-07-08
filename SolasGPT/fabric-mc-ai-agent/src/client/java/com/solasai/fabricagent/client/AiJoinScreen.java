package com.solasai.fabricagent.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.text.Text;

public class AiJoinScreen extends Screen {
    private final Screen parent;
    private TextFieldWidget serverField;
    private TextFieldWidget objectiveField;
    private TextFieldWidget botCountField;
    private TextFieldWidget usernameModeField;
    private TextFieldWidget baseUsernameField;
    private TextFieldWidget jobsField;
    private TextFieldWidget autoThinkField;

    public AiJoinScreen(Screen parent) {
        super(Text.literal("SolasAI AI Join"));
        this.parent = parent;
    }

    @Override
    protected void init() {
        int centerX = this.width / 2;
        int topY = this.height / 2 - 108;

        serverField = new TextFieldWidget(textRenderer, centerX - 170, topY, 340, 20, Text.literal("Server"));
        serverField.setMaxLength(120);
        serverField.setPlaceholder(Text.literal("Server address (example: hypixel.net:25565)"));
        String previousAddress = SolasAIFabricAgentClient.getAiJoinServerAddress();
        if (previousAddress != null && !previousAddress.isBlank()) {
            serverField.setText(previousAddress);
        }
        addDrawableChild(serverField);

        objectiveField = new TextFieldWidget(textRenderer, centerX - 170, topY + 24, 340, 20, Text.literal("Objective"));
        objectiveField.setMaxLength(400);
        objectiveField.setPlaceholder(Text.literal("What should the AI do? (can include jobs)"));
        String previousObjective = SolasAIFabricAgentClient.getAiJoinObjective();
        if (previousObjective != null && !previousObjective.isBlank()) {
            objectiveField.setText(previousObjective);
        }
        addDrawableChild(objectiveField);

        botCountField = new TextFieldWidget(textRenderer, centerX - 170, topY + 48, 108, 20, Text.literal("Bots"));
        botCountField.setMaxLength(4);
        botCountField.setText(SolasAIFabricAgentClient.getSwarmBotCountDefault());
        botCountField.setPlaceholder(Text.literal("1..500"));
        addDrawableChild(botCountField);

        usernameModeField = new TextFieldWidget(textRenderer, centerX - 56, topY + 48, 226, 20, Text.literal("Username mode"));
        usernameModeField.setMaxLength(36);
        usernameModeField.setText(SolasAIFabricAgentClient.getSwarmUsernameModeDefault());
        usernameModeField.setPlaceholder(Text.literal("numbered | random_mc | random_name"));
        addDrawableChild(usernameModeField);

        baseUsernameField = new TextFieldWidget(textRenderer, centerX - 170, topY + 72, 340, 20, Text.literal("Base username"));
        baseUsernameField.setMaxLength(20);
        baseUsernameField.setText(SolasAIFabricAgentClient.getSwarmBaseUsernameDefault());
        baseUsernameField.setPlaceholder(Text.literal("Example: Worker / Builder / Solas"));
        addDrawableChild(baseUsernameField);

        jobsField = new TextFieldWidget(textRenderer, centerX - 170, topY + 96, 340, 20, Text.literal("Jobs"));
        jobsField.setMaxLength(240);
        jobsField.setText(SolasAIFabricAgentClient.getSwarmJobsDefault());
        jobsField.setPlaceholder(Text.literal("Comma-separated jobs (miner,builder,farmer,guard,scout)"));
        addDrawableChild(jobsField);

        autoThinkField = new TextFieldWidget(textRenderer, centerX - 170, topY + 120, 340, 20, Text.literal("Auto think"));
        autoThinkField.setMaxLength(5);
        autoThinkField.setText(SolasAIFabricAgentClient.getSwarmAutoThinkDefault());
        autoThinkField.setPlaceholder(Text.literal("true / false"));
        addDrawableChild(autoThinkField);

        addDrawableChild(ButtonWidget.builder(Text.literal("Start / Queue"), button -> submit())
            .dimensions(centerX - 170, topY + 152, 165, 20)
                .build());
        addDrawableChild(ButtonWidget.builder(Text.literal("Cancel"), button -> close())
            .dimensions(centerX + 5, topY + 152, 165, 20)
                .build());

        setInitialFocus(serverField);
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        context.fill(0, 0, this.width, this.height, 0xB0000000);
        int headY = this.height / 2 - 132;
        context.drawCenteredTextWithShadow(textRenderer, Text.literal("SolasAI Swarm Join Experiment"), width / 2, headY, 0xFFFFFF);
        context.drawCenteredTextWithShadow(textRenderer, Text.literal("count=1 joins this client. count>1 sends swarm plan to bot-service."), width / 2, headY + 14, 0xB0B0B0);
        super.render(context, mouseX, mouseY, delta);
    }

    private void submit() {
        MinecraftClient client = MinecraftClient.getInstance();
        String address = serverField != null ? serverField.getText() : "";
        String objective = objectiveField != null ? objectiveField.getText() : "";
        String countRaw = botCountField != null ? botCountField.getText() : "1";
        String usernameMode = usernameModeField != null ? usernameModeField.getText() : "numbered";
        String baseUsername = baseUsernameField != null ? baseUsernameField.getText() : "Solas";
        String jobs = jobsField != null ? jobsField.getText() : "";
        String autoThinkRaw = autoThinkField != null ? autoThinkField.getText() : "true";

        int botCount = 1;
        try {
            botCount = Integer.parseInt(countRaw.trim());
        } catch (Exception ignored) {
            botCount = 1;
        }
        if (botCount < 1) botCount = 1;
        if (botCount > 500) botCount = 500;

        boolean autoThink = !"false".equalsIgnoreCase(autoThinkRaw.trim());

        if (botCount <= 1) {
            SolasAIFabricAgentClient.enableAiJoin(client, parent, objective, address);
        } else {
            SolasAIFabricAgentClient.startSwarmExperiment(
                    client,
                    parent,
                    address,
                    objective,
                    botCount,
                    usernameMode,
                    baseUsername,
                    jobs,
                    autoThink
            );
        }
    }

    @Override
    public void close() {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client != null) {
            client.setScreen(parent);
        }
    }
}
