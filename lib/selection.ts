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
