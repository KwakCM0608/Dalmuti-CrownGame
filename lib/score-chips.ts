export function scoreChipCount(
  score: number,
  highestScore: number,
  maximumChips = 5,
): number {
  if (!Number.isFinite(score) || score <= 0 || maximumChips <= 0) return 0;
  const safeMaximum = Math.max(1, Math.floor(maximumChips));
  const safeHighest = Math.max(1, score, highestScore);
  return Math.max(
    1,
    Math.min(safeMaximum, Math.ceil((score / safeHighest) * safeMaximum)),
  );
}
