/**
 * Temporary QA switch requested for the great-revolution test.
 *
 * Set this to false and redeploy to restore normal rank draws and dealing.
 * Keeping every forced outcome behind this single switch makes the temporary
 * production setup straightforward to remove without touching game rules.
 */
export const TEMP_GREAT_REVOLUTION_TEST_MODE = true;

type RankChoice = {
  rank: number;
  claimedByPlayerId: string | null;
};

type RankedCard = {
  id: string;
  rank: number;
};

export function forceClaimedPlayerToLastRank<T extends RankChoice>(
  cards: readonly T[],
  playerId: string,
): T[] {
  if (!TEMP_GREAT_REVOLUTION_TEST_MODE || !playerId || cards.length < 2) {
    return cards.map((card) => ({ ...card }));
  }

  const selectedIndex = cards.findIndex(
    (card) => card.claimedByPlayerId === playerId,
  );
  const lastRank = Math.max(...cards.map((card) => card.rank));
  const lastRankIndex = cards.findIndex((card) => card.rank === lastRank);
  if (
    selectedIndex < 0 ||
    lastRankIndex < 0 ||
    selectedIndex === lastRankIndex
  ) {
    return cards.map((card) => ({ ...card }));
  }

  const next = cards.map((card) => ({ ...card }));
  const selectedRank = next[selectedIndex].rank;
  next[selectedIndex].rank = lastRank;
  next[lastRankIndex].rank = selectedRank;
  return next;
}

export function forceTwoJokersIntoHand<T extends RankedCard>(
  hands: Readonly<Record<string, readonly T[]>>,
  orderedPlayerIds: readonly string[],
  playerId: string,
): Record<string, T[]> {
  const next = Object.fromEntries(
    Object.entries(hands).map(([id, hand]) => [
      id,
      hand.map((card) => ({ ...card })) as T[],
    ]),
  );
  if (
    !TEMP_GREAT_REVOLUTION_TEST_MODE ||
    !playerId ||
    !next[playerId]
  ) {
    return next;
  }

  for (const ownerId of orderedPlayerIds) {
    if (ownerId === playerId) continue;
    const ownerHand = next[ownerId];
    if (!ownerHand) continue;

    while (
      ownerHand.some((card) => card.rank === 13) &&
      next[playerId].filter((card) => card.rank === 13).length < 2
    ) {
      const jokerIndex = ownerHand.findIndex((card) => card.rank === 13);
      const replacementIndex = next[playerId].findIndex(
        (card) => card.rank !== 13,
      );
      if (jokerIndex < 0 || replacementIndex < 0) break;

      const [joker] = ownerHand.splice(jokerIndex, 1);
      const [replacement] = next[playerId].splice(replacementIndex, 1);
      ownerHand.push(replacement);
      next[playerId].push(joker);
    }
  }

  for (const hand of Object.values(next)) {
    hand.sort((left, right) =>
      right.rank - left.rank || left.id.localeCompare(right.id),
    );
  }
  return next;
}
