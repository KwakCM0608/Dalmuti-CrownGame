export interface SelectableCard {
  id: string;
  rank: number;
}

export const JOKER_RANK = 13;

/**
 * Toggles one card while keeping a playable hand selection coherent.
 *
 * Jokers may accompany any normal rank. Selecting a different normal rank,
 * however, replaces only the previously selected normal rank. Jokers are
 * independent wild cards, so they stay selected while the normal rank changes.
 */
export function togglePlayableCardSelection(
  currentIds: readonly string[],
  clickedCard: SelectableCard,
  hand: readonly SelectableCard[],
): string[] {
  if (currentIds.includes(clickedCard.id)) {
    return currentIds.filter((id) => id !== clickedCard.id);
  }

  if (clickedCard.rank === JOKER_RANK) {
    return [...currentIds, clickedCard.id];
  }

  const selectedIdSet = new Set(currentIds);
  const isChangingNormalRank = hand.some(
    (card) =>
      selectedIdSet.has(card.id) &&
      card.rank !== JOKER_RANK &&
      card.rank !== clickedCard.rank,
  );

  return isChangingNormalRank
    ? [
        ...currentIds.filter((id) => {
          const selectedCard = hand.find((card) => card.id === id);
          return selectedCard?.rank === JOKER_RANK;
        }),
        clickedCard.id,
      ]
    : [...currentIds, clickedCard.id];
}

export function toggleWholeRankSelection(
  currentIds: string[],
  rankIds: string[],
): string[] {
  const rankIdSet = new Set(rankIds);
  const isWholeRankSelected =
    rankIds.length > 0 && rankIds.every((id) => currentIds.includes(id));

  return isWholeRankSelected
    ? currentIds.filter((id) => !rankIdSet.has(id))
    : [...rankIds];
}

/**
 * Selects or clears a whole rank without ever mixing two normal ranks.
 * Already selected jokers remain attached to the new normal rank.
 */
export function toggleWholePlayableRankSelection(
  currentIds: readonly string[],
  rankIds: readonly string[],
  hand: readonly SelectableCard[],
): string[] {
  const rankIdSet = new Set(rankIds);
  const isWholeRankSelected =
    rankIds.length > 0 && rankIds.every((id) => currentIds.includes(id));

  if (isWholeRankSelected) {
    return currentIds.filter((id) => !rankIdSet.has(id));
  }

  const clickedRank = hand.find((card) => rankIdSet.has(card.id))?.rank;
  if (clickedRank === JOKER_RANK) {
    return [...new Set([...currentIds, ...rankIds])];
  }

  const selectedJokers = currentIds.filter((id) => {
    const selectedCard = hand.find((card) => card.id === id);
    return selectedCard?.rank === JOKER_RANK;
  });
  return [...new Set([...selectedJokers, ...rankIds])];
}
