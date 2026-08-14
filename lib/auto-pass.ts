export const AUTO_PASS_STORAGE_KEY = "dalmuti.game.auto-pass.enabled";

export type AutoPassCard = {
  rank: number;
};

export type AutoPassTable = {
  rank: number;
  count: number;
};

/**
 * Returns whether the player can legally submit at least one card set.
 * Only the player's own hand and the public table are inspected.
 */
export function hasLegalCardPlay(
  hand: readonly AutoPassCard[],
  table: AutoPassTable | null,
): boolean {
  if (hand.length === 0) return false;
  if (!table) return true;

  const jokerCount = hand.filter((card) => card.rank === 13).length;
  const naturalCounts = new Map<number, number>();
  for (const card of hand) {
    if (card.rank === 13 || card.rank >= table.rank) continue;
    naturalCounts.set(card.rank, (naturalCounts.get(card.rank) ?? 0) + 1);
  }

  return [...naturalCounts.values()].some(
    (naturalCount) => naturalCount + jokerCount >= table.count,
  );
}
