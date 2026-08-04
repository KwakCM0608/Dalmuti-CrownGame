import {
  DOUBLE_JOKER_ACTION_INDEX,
  decodeSemanticAction,
  encodeSemanticAction,
} from "./action-space.ts";
import {
  V3_ACTION_COUNT,
  decodeV3SemanticAction,
  encodeV3SemanticAction,
} from "./v3-action-catalogue.ts";

export const V3_LEGAL_MASK_HEX_LENGTH = V3_ACTION_COUNT / 4;

if (!Number.isInteger(V3_LEGAL_MASK_HEX_LENGTH)) {
  throw new Error("V3 action count must be divisible by four");
}

/** Map a legal legacy 506-space semantic action into the compact V3 space. */
export function legacyActionIndexToV3(actionIndex: number): number {
  const action = decodeSemanticAction(actionIndex);
  if (action.type === "double-joker") {
    throw new RangeError(
      "the frozen V3 catalogue does not contain the later double-joker action",
    );
  }
  return encodeV3SemanticAction(action);
}

/** Map a compact V3 semantic action back to the simulator's legacy space. */
export function v3ActionIndexToLegacy(actionIndex: number): number {
  return encodeSemanticAction(decodeV3SemanticAction(actionIndex));
}

export function legacyLegalActionIndicesToV3(
  actionIndices: readonly number[],
): number[] {
  if (actionIndices.length < 1 || new Set(actionIndices).size !== actionIndices.length) {
    throw new RangeError(
      "legacy legal actions must be a non-empty set of unique indices",
    );
  }
  // The two catalogues use different internal orders. Legacy groups plays by
  // total card count, while V3 groups them by natural-card count. A hand that
  // can mix a joker can therefore map an ascending legacy list to a
  // non-ascending V3 list (for example 44,47,48 -> 5,8,6). Canonicalize only
  // after the semantic one-to-one mapping.
  // V3 stays frozen at 236 actions for artifact compatibility. The production
  // 506-space policy can use the later double-joker action; V3 policies retain
  // their previous legal subset until a separately versioned catalogue exists.
  const representableActionIndices = actionIndices.filter(
    (actionIndex) => actionIndex !== DOUBLE_JOKER_ACTION_INDEX,
  );
  if (representableActionIndices.length < 1) {
    throw new RangeError("no V3-representable legal action remains");
  }
  const result = representableActionIndices
    .map(legacyActionIndexToV3)
    .sort((left, right) => left - right);
  if (
    new Set(result).size !== result.length ||
    result.some((value, index) => index > 0 && value <= result[index - 1])
  ) {
    throw new RangeError(
      "legacy legal actions must map to a unique V3 action set",
    );
  }
  return result;
}

export function encodeV3LegalMaskHex(
  legalActionIndices: readonly number[],
): string {
  if (legalActionIndices.length < 1) {
    throw new RangeError("V3 legal mask requires at least one action");
  }
  const nibbles = new Uint8Array(V3_LEGAL_MASK_HEX_LENGTH);
  let previous = -1;
  for (const actionIndex of legalActionIndices) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= V3_ACTION_COUNT ||
      actionIndex <= previous
    ) {
      throw new RangeError(
        "V3 legal actions must be unique ascending catalogue indices",
      );
    }
    nibbles[Math.floor(actionIndex / 4)] |= 1 << (actionIndex % 4);
    previous = actionIndex;
  }
  return [...nibbles].map((value) => value.toString(16)).join("");
}

export function decodeV3LegalMaskHex(mask: string): number[] {
  if (
    typeof mask !== "string" ||
    mask.length !== V3_LEGAL_MASK_HEX_LENGTH ||
    !/^[0-9a-f]+$/.test(mask)
  ) {
    throw new TypeError(
      `V3 legal mask must be ${V3_LEGAL_MASK_HEX_LENGTH} lowercase hex digits`,
    );
  }
  const result: number[] = [];
  for (let nibbleIndex = 0; nibbleIndex < mask.length; nibbleIndex += 1) {
    const nibble = Number.parseInt(mask[nibbleIndex], 16);
    for (let bit = 0; bit < 4; bit += 1) {
      if ((nibble & (1 << bit)) !== 0) {
        result.push(nibbleIndex * 4 + bit);
      }
    }
  }
  if (result.length < 1) {
    throw new RangeError("V3 legal mask must contain at least one action");
  }
  return result;
}
