"""Resize the approved Android icon and single branded splash resources.

The source artwork is versioned in ``android-twa/assets`` so regenerating the
Bubblewrap project never depends on a temporary Codex image location.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "android-twa" / "assets"
OUTPUT_ROOT = ROOT / "android-twa" / "custom" / "res"
ICON_SOURCE = ASSET_ROOT / "dalmuti-app-icon-v3.png"
SPLASH_SOURCE = ASSET_ROOT / "dalmuti-splash-v4.png"
EXPECTED_SOURCE_HASHES = {
    ICON_SOURCE: "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946",
    SPLASH_SOURCE: "13fadbea989e85980994d185b44f4a4215f3df59e075d1bdf6056a820756631f",
}
ICON_RESOURCE = "dalmuti_app_icon_v3.png"
MASKABLE_ICON_RESOURCE = "dalmuti_app_icon_maskable_v3.png"
SPLASH_RESOURCE = "dalmuti_splash_v4.png"
SPLASH_GLOW_RESOURCE = "dalmuti_splash_glow_v4.png"

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
    }
    for density in SPLASH_SIZES:
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


def save_splash_glow(source: Image.Image, target: Path, size: int) -> None:
    resized = source.resize((size, size), Image.Resampling.LANCZOS)
    luminance = resized.convert("L")
    # Only the bright crown, lettering, and existing sparks contribute to the
    # native halo; the dark red frame remains transparent.
    mask = luminance.point(
        lambda value: max(0, min(190, round((value - 46) * 1.85))),
    ).filter(ImageFilter.GaussianBlur(max(4, round(size * 0.028))))
    glow = Image.new("RGBA", (size, size), (255, 157, 38, 0))
    glow.putalpha(mask)
    target.parent.mkdir(parents=True, exist_ok=True)
    glow.save(target, optimize=True)


def main() -> None:
    verify_approved_sources()
    remove_obsolete_generated_resources()
    icon = load_square(ICON_SOURCE)
    splash = load_square(SPLASH_SOURCE)

    for density, size in SPLASH_SIZES.items():
        save_resized(
            splash,
            OUTPUT_ROOT / f"drawable-{density}" / SPLASH_RESOURCE,
            size,
        )
        save_splash_glow(
            splash,
            OUTPUT_ROOT / f"drawable-{density}" / SPLASH_GLOW_RESOURCE,
            size,
        )

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
