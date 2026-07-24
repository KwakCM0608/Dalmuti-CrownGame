from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / ".codex_deps"))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


OUTPUT_SIZE = (1040, 1600)
OUTPUT_SCALE = OUTPUT_SIZE[0] / 520
REFERENCE_SOURCE_SIZE = (2973, 1479)
ROTATED_CARD_NAMES = {"06", "07", "08", "11", "joker"}

# Source-space corners ordered clockwise: top-left, top-right, bottom-right,
# bottom-left. The coordinates target the outer physical edge of each card in
# the user's full-set photograph.
CARD_QUADS: dict[str, tuple[tuple[float, float], ...]] = {
    "01": ((568, 160), (956, 152), (966, 756), (582, 762)),
    "02": ((960, 155), (1345, 145), (1361, 751), (970, 759)),
    "03": ((1348, 151), (1738, 134), (1763, 742), (1372, 756)),
    "04": ((1745, 127), (2139, 124), (2135, 740), (1748, 741)),
    "05": ((2142, 126), (2538, 126), (2530, 744), (2140, 741)),
    "06": ((2542, 122), (2943, 122), (2939, 730), (2536, 727)),
    "07": ((196, 776), (587, 772), (586, 1365), (195, 1371)),
    "08": ((587, 770), (976, 765), (979, 1365), (587, 1368)),
    "09": ((979, 770), (1368, 760), (1379, 1361), (989, 1368)),
    "10": ((1368, 767), (1757, 754), (1778, 1354), (1386, 1365)),
    "11": ((1770, 768), (2142, 766), (2142, 1339), (1772, 1340)),
    "12": ((2153, 765), (2550, 763), (2542, 1348), (2146, 1342)),
    "joker": ((2551, 764), (2951, 764), (2944, 1349), (2545, 1344)),
}


def perspective_coefficients(
    source_quad: tuple[tuple[float, float], ...],
    size: tuple[int, int],
) -> tuple[float, ...]:
    width, height = size
    destination = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    equations: list[list[float]] = []
    values: list[float] = []

    for (x, y), (source_x, source_y) in zip(destination, source_quad):
        equations.append(
            [x, y, 1.0, 0.0, 0.0, 0.0, -source_x * x, -source_x * y],
        )
        values.append(source_x)
        equations.append(
            [0.0, 0.0, 0.0, x, y, 1.0, -source_y * x, -source_y * y],
        )
        values.append(source_y)

    return tuple(np.linalg.solve(np.asarray(equations), np.asarray(values)))


def rectify(
    source: Image.Image,
    source_quad: tuple[tuple[float, float], ...],
) -> Image.Image:
    coefficients = perspective_coefficients(source_quad, OUTPUT_SIZE)
    return source.transform(
        OUTPUT_SIZE,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )


def correct_illumination(card: Image.Image) -> Image.Image:
    rgb = np.asarray(card.convert("RGB"), dtype=np.float32)
    luminance = (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    )
    illumination = np.asarray(
        Image.fromarray(np.clip(luminance, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=72 * OUTPUT_SCALE),
        ),
        dtype=np.float32,
    )
    neutral_level = float(np.median(illumination))
    shade_gain = np.power(neutral_level / np.maximum(illumination, 24.0), 0.28)
    shade_gain = np.clip(shade_gain, 0.88, 1.14)
    rgb *= shade_gain[..., None]

    edge_width = round(42 * OUTPUT_SCALE)
    edge_mask = np.zeros(luminance.shape, dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True
    corrected_luminance = (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    )
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    border_pixels = edge_mask & (corrected_luminance > 155) & (chroma < 62)

    if border_pixels.sum() > 500:
        measured_border = np.median(rgb[border_pixels], axis=0)
        target_border = np.asarray((235.0, 218.0, 196.0), dtype=np.float32)
        channel_gain = np.clip(target_border / np.maximum(measured_border, 1.0), 0.88, 1.14)
        rgb *= channel_gain

    corrected = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    corrected = ImageEnhance.Contrast(corrected).enhance(1.035)
    corrected = ImageEnhance.Color(corrected).enhance(0.98)
    return corrected.filter(
        ImageFilter.UnsharpMask(
            radius=1.8,
            percent=65,
            threshold=3,
        ),
    )


def add_black_edge(card: Image.Image) -> Image.Image:
    edged = card.copy()
    mask = Image.new("L", edged.size, 255)
    draw = ImageDraw.Draw(mask)
    edge_width = round(9 * OUTPUT_SCALE)
    draw.rounded_rectangle(
        (
            edge_width,
            edge_width,
            edged.width - edge_width - 1,
            edged.height - edge_width - 1,
        ),
        radius=round(14 * OUTPUT_SCALE),
        fill=0,
    )
    edged.paste((14, 14, 13), (0, 0, edged.width, edged.height), mask)
    return edged


def repair_top_header_from_matching_bottom(card: Image.Image) -> Image.Image:
    repaired = card.copy()
    header_height = round(112 * OUTPUT_SCALE)
    bottom_header = repaired.crop(
        (0, repaired.height - header_height, repaired.width, repaired.height),
    ).rotate(180)
    replacement = repaired.copy()
    replacement.paste(bottom_header, (0, 0))
    mask = Image.new("L", repaired.size, 0)
    mask_pixels = np.asarray(mask, dtype=np.uint8).copy()
    solid_height = header_height - round(16 * OUTPUT_SCALE)
    mask_pixels[:solid_height, :] = 255
    for y in range(solid_height, header_height):
        progress = (y - solid_height) / max(header_height - solid_height - 1, 1)
        mask_pixels[y, :] = round(255 * (1 - progress))
    return Image.composite(
        replacement,
        repaired,
        Image.fromarray(mask_pixels, "L"),
    )


def make_contact_sheet(cards: dict[str, Image.Image], output_path: Path) -> None:
    thumb_size = (182, 280)
    margin = 22
    columns = 7
    rows = 2
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_size[0] + (columns + 1) * margin,
            rows * thumb_size[1] + (rows + 1) * margin,
        ),
        (31, 33, 29),
    )

    for index, (name, card) in enumerate(cards.items()):
        column = index % columns
        row = index // columns
        x = margin + column * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + margin)
        thumb = ImageOps.fit(card, thumb_size, method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("public/cards"))
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path("artifacts/card-contact-sheet.png"),
    )
    args = parser.parse_args()

    source = Image.open(args.source).convert("RGB")
    scale_x = source.width / REFERENCE_SOURCE_SIZE[0]
    scale_y = source.height / REFERENCE_SOURCE_SIZE[1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, Image.Image] = {}

    for name, quad in CARD_QUADS.items():
        scaled_quad = tuple((x * scale_x, y * scale_y) for x, y in quad)
        card = correct_illumination(rectify(source, scaled_quad))
        if name in {"11", "12", "joker"}:
            card = repair_top_header_from_matching_bottom(card)
        if name in ROTATED_CARD_NAMES:
            card = card.transpose(Image.Transpose.ROTATE_180)
        if name in {"01", "joker"}:
            card = add_black_edge(card)
        output_path = args.output_dir / f"{name}.webp"
        card.save(output_path, "WEBP", quality=98, method=6)
        cards[name] = card

    make_contact_sheet(cards, args.contact_sheet)


if __name__ == "__main__":
    main()
