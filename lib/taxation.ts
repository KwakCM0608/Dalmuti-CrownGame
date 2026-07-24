export type TaxableCard = {
  rank: number;
};

export function taxationPriority(card: TaxableCard): number {
  // House rule: a Jester is the strongest card only when selecting tax cards.
  return card.rank === 13 ? 0 : card.rank;
}

export function selectPeonTaxCards<T extends TaxableCard>(
  cards: readonly T[],
  count: number,
): T[] {
  return [...cards]
    .sort((a, b) => taxationPriority(a) - taxationPriority(b))
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
