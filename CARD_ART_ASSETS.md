# Card art assets

The game automatically uses full-card artwork from `public/cards` when these
files are present:

- `01.webp` through `12.webp`
- `joker.webp`

The current set is generated from the project owner's full-deck photograph by
`scripts/build_card_assets.py`. It applies per-card perspective correction,
evens out low-frequency lighting, preserves the photographed paper edge, and
exports every card at `1040×1600` without synthesizing or stretching the side
edges. Rank 1 and the joker receive a thin black outer edge; ranks 2–12 retain
their original cream edge.

Ranks 6, 7, 8, 11, and the joker are rotated 180° during generation so their
character artwork and upper title strip both display upright.

Every copy of a rank references the same file, so alternate print colors in the
source photographs are intentionally normalized to one representative design.
Until an asset is present, the built-in typographic fallback remains visible.

Only add artwork that the project owner is permitted to reproduce and
distribute.
