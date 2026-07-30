export const MIN_SCORING_PLAYER_COUNT = 4;
export const MAX_SCORING_PLAYER_COUNT = 10;

/**
 * Returns the chips earned for one completed act.
 *
 * The fixed curve keeps the reward comparable across 4–10 player games:
 * first/second earn 4/3, the middle earns 2, and the bottom two earn 1/0.
 */
export function roundChipAward(place: number, playerCount: number): number {
  if (
    !Number.isInteger(playerCount) ||
    playerCount < MIN_SCORING_PLAYER_COUNT ||
    playerCount > MAX_SCORING_PLAYER_COUNT
  ) {
    throw new RangeError(
      `playerCount must be an integer from ${MIN_SCORING_PLAYER_COUNT} to ${MAX_SCORING_PLAYER_COUNT}.`,
    );
  }
  if (!Number.isInteger(place) || place < 1 || place > playerCount) {
    throw new RangeError(
      `place must be an integer from 1 to ${playerCount}.`,
    );
  }

  if (place === 1) return 4;
  if (place === 2) return 3;
  if (place === playerCount - 1) return 1;
  if (place === playerCount) return 0;
  return 2;
}
