package com.solasai.fabricagent.client;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.input.KeyInput;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class AiPromptScreen extends Screen {
    private TextFieldWidget promptField;

    public AiPromptScreen() {
        super(Text.literal("SolasAI Objective"));
    }

    @Override
    protected void init() {
        int centerX = this.width / 2;
        int centerY = this.height / 2;

        promptField = new TextFieldWidget(textRenderer, centerX - 150, centerY - 20, 300, 20, Text.literal("Objective"));
        promptField.setMaxLength(300);
        promptField.setPlaceholder(Text.literal("Tell AI what to do (build, pvp, mine, explore...)"));
        promptField.setFocused(true);
        addDrawableChild(promptField);

        addDrawableChild(ButtonWidget.builder(Text.literal("Start AI"), button -> submit())
                .dimensions(centerX - 150, centerY + 12, 145, 20)
                .build());

        addDrawableChild(ButtonWidget.builder(Text.literal("Cancel"), button -> close())
                .dimensions(centerX + 5, centerY + 12, 145, 20)
                .build());

        setInitialFocus(promptField);
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        context.fill(0, 0, this.width, this.height, 0xB0000000);
        context.drawCenteredTextWithShadow(textRenderer, Text.literal("SolasAI Objective"), width / 2, height / 2 - 48, 0xFFFFFF);
        context.drawCenteredTextWithShadow(textRenderer, Text.literal("Press Enter to start. Esc+Tab stops AI control."), width / 2, height / 2 - 34, 0xB0B0B0);
        super.render(context, mouseX, mouseY, delta);
    }

    @Override
    public boolean keyPressed(KeyInput input) {
        if (input.key() == GLFW.GLFW_KEY_ENTER || input.key() == GLFW.GLFW_KEY_KP_ENTER) {
            submit();
            return true;
        }
        return super.keyPressed(input);
    }

    private void submit() {
        MinecraftClient client = MinecraftClient.getInstance();
        String objective = promptField != null ? promptField.getText() : "";
        AiController.getInstance().start(client, objective);
        close();
    }

    @Override
    public void close() {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client != null) {
            client.setScreen(null);
        }
    }
}
