"""Resize the approved Android launcher icon resources.

The source artwork is versioned in ``android-twa/assets`` so regenerating the
Bubblewrap project never depends on a temporary Codex image location.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "android-twa" / "assets"
OUTPUT_ROOT = ROOT / "android-twa" / "custom" / "res"
ICON_SOURCE = ASSET_ROOT / "dalmuti-app-icon-v3.png"
EXPECTED_SOURCE_HASHES = {
    ICON_SOURCE: "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946",
}
ICON_RESOURCE = "dalmuti_app_icon_v3.png"
MASKABLE_ICON_RESOURCE = "dalmuti_app_icon_maskable_v3.png"

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


def verify_approved_sources() -> None:
    for path, expected_hash in EXPECTED_SOURCE_HASHES.items():
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Android artwork does not match the approved source: {path}",
            )


def remove_obsolete_generated_resources() -> None:
    obsolete_names = {
        "ic_launcher.png",
        "ic_maskable.png",
        "splash.png",
        "splash_glow.png",
        "dalmuti_splash_v4.png",
        "dalmuti_splash_glow_v4.png",
        "dalmuti_splash_branding.png",
    }
    for density in LEGACY_ICON_SIZES:
        for folder in (
            OUTPUT_ROOT / f"drawable-{density}",
            OUTPUT_ROOT / f"mipmap-{density}",
        ):
            for obsolete_name in obsolete_names:
                (folder / obsolete_name).unlink(missing_ok=True)


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
    verify_approved_sources()
    remove_obsolete_generated_resources()
    icon = load_square(ICON_SOURCE)

    for density, size in LEGACY_ICON_SIZES.items():
        save_resized(
            icon,
            OUTPUT_ROOT / f"mipmap-{density}" / ICON_RESOURCE,
            size,
        )

    for density, size in ADAPTIVE_ICON_SIZES.items():
        save_adaptive_icon(
            icon,
            OUTPUT_ROOT / f"mipmap-{density}" / MASKABLE_ICON_RESOURCE,
            size,
        )

if __name__ == "__main__":
    main()
