"""Resize the approved Android icon and single branded splash resources.

The source artwork is versioned in ``android-twa/assets`` so regenerating the
Bubblewrap project never depends on a temporary Codex image location.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "android-twa" / "assets"
OUTPUT_ROOT = ROOT / "android-twa" / "custom" / "res"
ICON_SOURCE = ASSET_ROOT / "dalmuti-app-icon-v1.png"
SPLASH_SOURCE = ASSET_ROOT / "dalmuti-splash-v2.png"

SPLASH_SIZES = {
    "mdpi": 300,
    "hdpi": 450,
    "xhdpi": 600,
    "xxhdpi": 900,
    "xxxhdpi": 1200,
}
LEGACY_ICON_SIZES = {
    "mdpi": 48,
    "hdpi": 72,
    "xhdpi": 96,
    "xxhdpi": 144,
    "xxxhdpi": 192,
}
ADAPTIVE_ICON_SIZES = {
    "mdpi": 108,
    "hdpi": 162,
    "xhdpi": 216,
    "xxhdpi": 324,
    "xxxhdpi": 432,
}


def load_square(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.width != image.height:
        raise ValueError(f"Android artwork must be square: {path}")
    return image


def save_resized(source: Image.Image, target: Path, size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source.resize((size, size), Image.Resampling.LANCZOS).save(
        target,
        optimize=True,
    )


def save_adaptive_icon(source: Image.Image, target: Path, size: int) -> None:
    # Adaptive launchers apply circles, squircles, and other masks. Keep the
    # full title treatment inside the central safe area while retaining the
    # artwork's own burgundy background around it.
    background_colour = source.getpixel((8, 8))
    canvas = Image.new("RGBA", (size, size), background_colour)
    content_size = round(size * 0.76)
    content = source.resize(
        (content_size, content_size),
        Image.Resampling.LANCZOS,
    )
    inset = (size - content_size) // 2
    canvas.alpha_composite(content, (inset, inset))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)


def main() -> None:
    icon = load_square(ICON_SOURCE)
    splash = load_square(SPLASH_SOURCE)

    for density, size in SPLASH_SIZES.items():
        save_resized(
            splash,
            OUTPUT_ROOT / f"drawable-{density}" / "splash.png",
            size,
        )

    for density, size in LEGACY_ICON_SIZES.items():
        save_resized(
            icon,
            OUTPUT_ROOT / f"mipmap-{density}" / "ic_launcher.png",
            size,
        )

    for density, size in ADAPTIVE_ICON_SIZES.items():
        save_adaptive_icon(
            icon,
            OUTPUT_ROOT / f"mipmap-{density}" / "ic_maskable.png",
            size,
        )

    # This project intentionally presents one complete branded TWA splash.
    # Android 12's mandatory system-owned frame uses a transparent drawable,
    # so no second icon/brand treatment is visible before this image.
    (
        OUTPUT_ROOT
        / "drawable-xxxhdpi"
        / "dalmuti_splash_branding.png"
    ).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
