#!/usr/bin/env python3
"""Generate SolasAI mod icon – 128×128 cyan-on-dark digital aesthetic."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os

SIZE = 128
cx, cy = SIZE / 2, SIZE / 2

# ─── Base canvas ─────────────────────────────────────────────────────────────
img = Image.new("RGBA", (SIZE, SIZE), (5, 6, 18, 255))
draw = ImageDraw.Draw(img)

# ─── Radial background glow (deep navy → darkness) ───────────────────────────
for r in range(62, 0, -1):
    t = (r / 62) ** 1.4
    c = (int(10 + 20 * (1 - t)), int(14 + 36 * (1 - t)), int(28 + 60 * (1 - t)), 255)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)

# ─── Outer glow ring (multi-pass soft to hard) ───────────────────────────────
ring_radius = 58
for w, col in [
    (20, (0, 140, 255, 15)),
    (14, (0, 170, 255, 25)),
    (8,  (0, 200, 255, 50)),
    (4,  (0, 220, 255, 90)),
    (2,  (100, 235, 255, 160)),
    (1,  (200, 248, 255, 230)),
]:
    draw.ellipse(
        [cx - ring_radius, cy - ring_radius, cx + ring_radius, cy + ring_radius],
        outline=col, width=w,
    )

# ─── Six hex circuit nodes on the ring ──────────────────────────────────────
for deg in range(0, 360, 60):
    rad = math.radians(deg)
    nx, ny = cx + 46 * math.cos(rad), cy + 46 * math.sin(rad)
    # Line spoke to inner hub
    ix, iy = cx + 16 * math.cos(rad), cy + 16 * math.sin(rad)
    draw.line([(nx, ny), (ix, iy)], fill=(0, 180, 220, 80), width=1)
    # Node circle
    for nr, nc in [(5, (0, 150, 220, 60)), (3, (0, 200, 255, 130)), (1, (200, 245, 255, 220))]:
        draw.ellipse([nx - nr, ny - nr, nx + nr, ny + nr], fill=nc)

# ─── Mid-ring tick marks ─────────────────────────────────────────────────────
for deg in range(30, 360, 60):
    rad = math.radians(deg)
    x0, y0 = cx + 51 * math.cos(rad), cy + 51 * math.sin(rad)
    x1, y1 = cx + 57 * math.cos(rad), cy + 57 * math.sin(rad)
    draw.line([(x0, y0), (x1, y1)], fill=(80, 200, 255, 180), width=2)

# ─── Inner hub circle ────────────────────────────────────────────────────────
for r, col in [(14, (0, 120, 200, 40)), (10, (0, 160, 240, 70)), (6, (0, 200, 255, 120))]:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=1)

# ─── Central "S" letter with glow ────────────────────────────────────────────
font = None
for path in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]:
    if os.path.exists(path):
        font = ImageFont.truetype(path, 68)
        break

if font is None:
    font = ImageFont.load_default()

letter = "S"
# Measure using a temp draw
tmp = ImageDraw.Draw(img)
bbox = tmp.textbbox((0, 0), letter, font=font)
lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
lx = int(cx - lw / 2 - bbox[0])
ly = int(cy - lh / 2 - bbox[1]) - 2

# Layered glow passes
for blur, colour in [
    (9,  (0, 160, 255, 20)),
    (6,  (0, 190, 255, 40)),
    (4,  (0, 215, 255, 65)),
    (2,  (60, 230, 255, 100)),
    (1,  (160, 245, 255, 160)),
]:
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.text((lx, ly), letter, font=font, fill=colour)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))
    img = Image.alpha_composite(img, layer)

# Solid bright core of the letter
draw = ImageDraw.Draw(img)
draw.text((lx, ly), letter, font=font, fill=(220, 250, 255, 255))

# ─── Corner pixel accents (Minecraft nod) ────────────────────────────────────
for px, py in [(3, 3), (SIZE - 7, 3), (3, SIZE - 7), (SIZE - 7, SIZE - 7)]:
    for qx, qy in [(0, 0), (4, 0), (0, 4), (4, 4)]:  # 2×2 scattered squares
        draw.rectangle([px + qx, py + qy, px + qx + 1, py + qy + 1],
                        fill=(0, 200, 255, 140))

# ─── Save ─────────────────────────────────────────────────────────────────────
out = "/mnt/data/SolasAI/fabric-mc-ai-agent/src/main/resources/icon.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

# Convert to RGB with black bg for final PNG (Fabric shows on white; keep alpha)
img.save(out, "PNG")
print(f"Saved {out}  ({img.size[0]}x{img.size[1]}px, {os.path.getsize(out)} bytes)")
