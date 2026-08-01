export const V3_ACTION_CATALOGUE_VERSION = 1;
export const V3_PASS_ACTION_INDEX = 0;
export const V3_SOLO_JOKER_ACTION_INDEX = 1;
export const V3_NORMAL_RANK_COUNT = 12;
export const V3_JOKER_COUNT_OPTIONS = 3;

export type V3SemanticAction =
  | { readonly type: "pass" }
  | { readonly type: "solo-joker" }
  | {
      readonly type: "play";
      readonly rank: number;
      readonly count: number;
      readonly jokerCount: number;
    };

function assertIntegerInRange(
  value: number,
  minimum: number,
  maximum: number,
  label: string,
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `${label} must be an integer from ${minimum} to ${maximum}`,
    );
  }
}

function firstPlayIndexForRank(rank: number): number {
  return 2 + (V3_JOKER_COUNT_OPTIONS * (rank - 1) * rank) / 2;
}

function createActionCatalogue(): readonly V3SemanticAction[] {
  const actions: V3SemanticAction[] = [
    Object.freeze({ type: "pass" as const }),
    Object.freeze({ type: "solo-joker" as const }),
  ];
  for (let rank = 1; rank <= V3_NORMAL_RANK_COUNT; rank += 1) {
    for (let naturalCount = 1; naturalCount <= rank; naturalCount += 1) {
      for (
        let jokerCount = 0;
        jokerCount < V3_JOKER_COUNT_OPTIONS;
        jokerCount += 1
      ) {
        actions.push(
          Object.freeze({
            type: "play" as const,
            rank,
            count: naturalCount + jokerCount,
            jokerCount,
          }),
        );
      }
    }
  }
  return Object.freeze(actions);
}

/**
 * Stable V3 action layout:
 *
 * 0 = PASS
 * 1 = one joker played by itself
 * 2..235 = rank 1..12, then natural-card count 1..rank, then
 * joker count 0..2. Every catalogue entry is structurally possible in the
 * 80-card deck; state-specific legality is still supplied by a legal mask.
 */
export const V3_ACTION_CATALOGUE = createActionCatalogue();
export const V3_ACTION_COUNT = V3_ACTION_CATALOGUE.length;

if (V3_ACTION_COUNT !== 236) {
  throw new Error(`V3 action catalogue has ${V3_ACTION_COUNT} entries`);
}

export function encodeV3SemanticAction(action: V3SemanticAction): number {
  if (action.type === "pass") return V3_PASS_ACTION_INDEX;
  if (action.type === "solo-joker") return V3_SOLO_JOKER_ACTION_INDEX;

  assertIntegerInRange(action.rank, 1, V3_NORMAL_RANK_COUNT, "rank");
  assertIntegerInRange(action.jokerCount, 0, 2, "jokerCount");
  assertIntegerInRange(action.count, 1, V3_NORMAL_RANK_COUNT + 2, "count");
  const naturalCount = action.count - action.jokerCount;
  if (naturalCount < 1 || naturalCount > action.rank) {
    throw new RangeError(
      `rank ${action.rank} requires a natural-card count from 1 to ${action.rank}`,
    );
  }
  return (
    firstPlayIndexForRank(action.rank) +
    (naturalCount - 1) * V3_JOKER_COUNT_OPTIONS +
    action.jokerCount
  );
}

export function decodeV3SemanticAction(
  actionIndex: number,
): V3SemanticAction {
  assertIntegerInRange(actionIndex, 0, V3_ACTION_COUNT - 1, "actionIndex");
  return V3_ACTION_CATALOGUE[actionIndex];
}

export const V3_ACTION_FEATURE_LAYOUT = Object.freeze([
  "type.pass",
  "type.solo-joker",
  "type.play",
  ...Array.from(
    { length: V3_NORMAL_RANK_COUNT },
    (_, index) => `rank.${index + 1}`,
  ),
  "joker-count.0",
  "joker-count.1",
  "joker-count.2",
  "rank-strength",
  "natural-count",
  "total-count",
  "joker-fraction",
] as const);

export const V3_ACTION_FEATURE_COUNT = V3_ACTION_FEATURE_LAYOUT.length;

export function encodeV3ActionFeatures(
  actionOrIndex: V3SemanticAction | number,
): readonly number[] {
  const action =
    typeof actionOrIndex === "number"
      ? decodeV3SemanticAction(actionOrIndex)
      : decodeV3SemanticAction(encodeV3SemanticAction(actionOrIndex));
  const features = Array.from(
    { length: V3_ACTION_FEATURE_COUNT },
    () => 0,
  );

  if (action.type === "pass") {
    features[0] = 1;
  } else if (action.type === "solo-joker") {
    features[1] = 1;
    features[16] = 1;
    features[20] = 1 / 14;
    features[21] = 1;
  } else {
    const naturalCount = action.count - action.jokerCount;
    features[2] = 1;
    features[3 + action.rank - 1] = 1;
    features[15 + action.jokerCount] = 1;
    features[18] = (13 - action.rank) / 12;
    features[19] = naturalCount / 12;
    features[20] = action.count / 14;
    features[21] = action.jokerCount / action.count;
  }
  return Object.freeze(features);
}

export const V3_ACTION_FEATURES: readonly (readonly number[])[] =
  Object.freeze(V3_ACTION_CATALOGUE.map(encodeV3ActionFeatures));

