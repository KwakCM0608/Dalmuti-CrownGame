export type TaxableCard = {
  rank: number;
};

export function taxationPriority(card: TaxableCard): number {
  // Keep Jesters out of automatic noble returns unless no weaker option exists.
  return card.rank === 13 ? 0 : card.rank;
}

export function selectPeonTaxCards<T extends TaxableCard>(
  cards: readonly T[],
  count: number,
): T[] {
  return cards
    .filter((card) => card.rank !== 13)
    .sort((a, b) => a.rank - b.rank)
    .slice(0, count);
}

export function selectDalmutiReturnCards<T extends TaxableCard>(
  cards: readonly T[],
  count: number,
): T[] {
  return [...cards]
    .sort((a, b) => taxationPriority(b) - taxationPriority(a))
    .slice(0, count);
}
