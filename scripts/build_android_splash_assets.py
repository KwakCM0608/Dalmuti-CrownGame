"""Build the single branded Android/TWA splash resource.

Android 12 still owns a brief system launch frame, but that frame is configured
as an unbranded solid colour. Bubblewrap then presents this complete recreation
of the original web splash: radial burgundy background, halo, crown, and
wordmark. There is only one visible branded splash.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CROWN = ROOT / "public" / "brand-dalmuti-crown.png"
OUTPUT_ROOT = ROOT / "android-twa" / "custom" / "res"
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/georgiab.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
)


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("A Georgia-compatible serif font is required.")


def draw_spaced_text(
    canvas: Image.Image,
    text: str,
    *,
    center_x: float,
    y: float,
    typeface: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    spacing: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    widths = [draw.textlength(character, font=typeface) for character in text]
    total_width = sum(widths) + spacing * max(0, len(text) - 1)
    cursor_x = center_x - total_width / 2
    for character, width in zip(text, widths, strict=True):
        draw.text((cursor_x, y), character, font=typeface, fill=fill)
        cursor_x += width + spacing


def mix(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int, int]:
    clamped = max(0.0, min(1.0, amount))
    return (
        round(start[0] + (end[0] - start[0]) * clamped),
        round(start[1] + (end[1] - start[1]) * clamped),
        round(start[2] + (end[2] - start[2]) * clamped),
        255,
    )


def build_radial_background(size: int) -> Image.Image:
    # Work at a compact resolution, then resample. This produces a smooth
    # gradient while keeping the asset generator fast and deterministic.
    gradient_size = min(size, 360)
    background = Image.new("RGBA", (gradient_size, gradient_size))
    pixels = background.load()
    center = (gradient_size * 0.5, gradient_size * 0.46)
    radius = gradient_size * 0.68

    for y in range(gradient_size):
        for x in range(gradient_size):
            distance = math.hypot(x - center[0], y - center[1]) / radius
            if distance <= 0.42:
                colour = mix((72, 24, 40), (33, 9, 16), distance / 0.42)
            else:
                colour = mix(
                    (33, 9, 16),
                    (24, 7, 12),
                    (distance - 0.42) / 0.58,
                )

            # Match the restrained purple glow in the former web splash.
            purple_strength = max(0.0, 1.0 - distance / 0.46) * 0.22
            pixels[x, y] = (
                round(colour[0] * (1 - purple_strength) + 126 * purple_strength),
                round(colour[1] * (1 - purple_strength) + 71 * purple_strength),
                round(colour[2] * (1 - purple_strength) + 145 * purple_strength),
                255,
            )

    background = background.resize((size, size), Image.Resampling.LANCZOS)
    texture = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    texture_draw = ImageDraw.Draw(texture)
    stripe_step = max(5, round(size / 170))
    for offset in range(-size, size * 2, stripe_step):
        texture_draw.line(
            ((offset, 0), (offset - size, size)),
            fill=(255, 238, 194, 5),
            width=max(1, round(size / 1200)),
        )
    return Image.alpha_composite(background, texture)


def build_square_splash(size: int) -> Image.Image:
    scale = size / 1200
    canvas = build_radial_background(size)
    draw = ImageDraw.Draw(canvas)

    halo_box = (
        int(size * 0.14),
        int(size * 0.04),
        int(size * 0.86),
        int(size * 0.76),
    )
    halo_glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    halo_glow_draw = ImageDraw.Draw(halo_glow)
    halo_glow_draw.ellipse(
        halo_box,
        outline=(206, 146, 49, 54),
        width=max(3, round(20 * scale)),
    )
    halo_glow = halo_glow.filter(
        ImageFilter.GaussianBlur(max(2, round(32 * scale))),
    )
    canvas = Image.alpha_composite(canvas, halo_glow)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        halo_box,
        outline=(229, 187, 88, 62),
        width=max(1, round(2 * scale)),
    )

    crown = Image.open(SOURCE_CROWN).convert("RGBA")
    crown_size = round(size * 0.37)
    crown = crown.resize((crown_size, crown_size), Image.Resampling.LANCZOS)
    crown_position = (
        (size - crown_size) // 2,
        round(size * 0.205),
    )
    crown_shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    crown_shadow_draw = ImageDraw.Draw(crown_shadow)
    crown_shadow_draw.ellipse(
        (
            crown_position[0] - round(10 * scale),
            crown_position[1] - round(10 * scale),
            crown_position[0] + crown_size + round(10 * scale),
            crown_position[1] + crown_size + round(10 * scale),
        ),
        fill=(230, 189, 92, 38),
    )
    crown_shadow = crown_shadow.filter(
        ImageFilter.GaussianBlur(max(2, round(22 * scale))),
    )
    canvas = Image.alpha_composite(canvas, crown_shadow)
    canvas.alpha_composite(
        crown,
        crown_position,
    )

    draw_spaced_text(
        canvas,
        "DALMUTI",
        center_x=size / 2,
        y=size * 0.615,
        typeface=font(max(12, round(83 * scale))),
        fill=(255, 241, 189, 255),
        spacing=max(2, round(18 * scale)),
    )
    draw_spaced_text(
        canvas,
        "THE GREAT DALMUTI",
        center_x=size / 2,
        y=size * 0.725,
        typeface=font(max(7, round(25 * scale))),
        fill=(226, 196, 126, 150),
        spacing=max(1, round(8 * scale)),
    )
    return canvas

def main() -> None:
    densities = {
        "mdpi": 300,
        "hdpi": 450,
        "xhdpi": 600,
        "xxhdpi": 900,
        "xxxhdpi": 1200,
    }
    master = build_square_splash(max(densities.values()))
    for density, size in densities.items():
        target = OUTPUT_ROOT / f"drawable-{density}" / "splash.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        splash = (
            master
            if size == master.width
            else master.resize((size, size), Image.Resampling.LANCZOS)
        )
        splash.save(target, optimize=True)

    # Remove the old Android 12 branding strip if an earlier generation left
    # it behind. The system frame is intentionally unbranded now.
    (
        OUTPUT_ROOT
        / "drawable-xxxhdpi"
        / "dalmuti_splash_branding.png"
    ).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
