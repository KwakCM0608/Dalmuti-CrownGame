"""Build the branded Android/TWA splash resources.

Bubblewrap normally reuses the launcher icon for its splash image.  These
assets keep the launcher icon unchanged while giving both the Android system
splash and the TWA hand-off the same DALMUTI crown-and-wordmark treatment.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def build_square_splash(size: int) -> Image.Image:
    scale = size / 1200
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    halo_box = (
        int(size * 0.265),
        int(size * 0.155),
        int(size * 0.735),
        int(size * 0.625),
    )
    draw.ellipse(
        halo_box,
        outline=(218, 176, 77, 64),
        width=max(1, round(3 * scale)),
    )

    crown = Image.open(SOURCE_CROWN).convert("RGBA")
    crown_size = round(size * 0.39)
    crown = crown.resize((crown_size, crown_size), Image.Resampling.LANCZOS)
    canvas.alpha_composite(
        crown,
        (
            (size - crown_size) // 2,
            round(size * 0.195),
        ),
    )

    draw_spaced_text(
        canvas,
        "DALMUTI",
        center_x=size / 2,
        y=size * 0.63,
        typeface=font(max(12, round(83 * scale))),
        fill=(255, 241, 189, 255),
        spacing=max(2, round(18 * scale)),
    )
    draw_spaced_text(
        canvas,
        "THE GREAT DALMUTI",
        center_x=size / 2,
        y=size * 0.745,
        typeface=font(max(7, round(25 * scale))),
        fill=(226, 196, 126, 150),
        spacing=max(1, round(8 * scale)),
    )
    return canvas


def build_branding() -> Image.Image:
    width, height = 800, 320
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_spaced_text(
        canvas,
        "DALMUTI",
        center_x=width / 2,
        y=55,
        typeface=font(92),
        fill=(255, 241, 189, 255),
        spacing=20,
    )
    draw_spaced_text(
        canvas,
        "THE GREAT DALMUTI",
        center_x=width / 2,
        y=185,
        typeface=font(30),
        fill=(226, 196, 126, 175),
        spacing=9,
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
    for density, size in densities.items():
        target = OUTPUT_ROOT / f"drawable-{density}" / "splash.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        build_square_splash(size).save(target, optimize=True)

    branding_target = (
        OUTPUT_ROOT / "drawable-xxxhdpi" / "dalmuti_splash_branding.png"
    )
    branding_target.parent.mkdir(parents=True, exist_ok=True)
    build_branding().save(branding_target, optimize=True)


if __name__ == "__main__":
    main()
