export function rankedDealCounts(
  totalCards: number,
  playerCount: number,
): number[] {
  if (!Number.isInteger(totalCards) || totalCards < 0) {
    throw new RangeError("totalCards must be a non-negative integer");
  }
  if (!Number.isInteger(playerCount) || playerCount <= 0) {
    throw new RangeError("playerCount must be a positive integer");
  }

  const baseCount = Math.floor(totalCards / playerCount);
  const remainder = totalCards % playerCount;
  const bonusStart = playerCount - remainder;

  return Array.from(
    { length: playerCount },
    (_, index) => baseCount + (index >= bonusStart ? 1 : 0),
  );
}
