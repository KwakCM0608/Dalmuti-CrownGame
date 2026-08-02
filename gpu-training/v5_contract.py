from __future__ import annotations

import hashlib
import json
from typing import Mapping


V5_PUBLIC_CONTRACT_ID = "dalmuti-v5-public-belief-v2"
V5_PUBLIC_CONTRACT_VERSION = 2
V5_PUBLIC_SCHEMA_VERSION = 5

V5_MIN_PLAYERS = 4
V5_MAX_PLAYERS = 10
V5_MAX_OPPONENTS = V5_MAX_PLAYERS - 1
V5_RANK_COUNT = 13
V5_ACTION_COUNT = 236
V5_MAX_HISTORY = 192

# Natural ranks have rank-many copies.  The final entry is the two jokers.
V5_DECK_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2)
V5_DECK_SIZE = sum(V5_DECK_COUNTS)

V5_GLOBAL_FIELDS = (
    "schema-version",
    "player-count",
    "act",
    "actor-role",
    "revolution",
    "truncated-history-count",
)
V5_PLAYER_FIELDS = (
    "relative-offset",
    "remaining-card-count",
    "role",
    "finished",
    "passed",
    "table-leader",
)
V5_TABLE_FIELDS = (
    "present",
    "rank",
    "required-count",
    "natural-count",
    "joker-count",
    "actor-offset",
)
V5_HISTORY_FIELDS = (
    "event-type",
    "actor-offset",
    "hand-count-before",
    "hand-count-after",
    "rank",
    "natural-count",
    "joker-count",
    "total-count",
    "pass-reason",
    "clear-reason",
    "next-leader-offset-plus-one",
    "finish-place",
)

V5_EVENT_TYPES = {
    "padding": 0,
    "play": 1,
    "pass": 2,
    "clear": 3,
    "finish": 4,
}
V5_PASS_REASONS = {
    "not-pass": 0,
    "manual": 1,
    "timeout": 2,
    "insufficient-cards": 3,
    "dalmuti": 4,
}
V5_CLEAR_REASONS = {
    "not-clear": 0,
    "all-passed": 1,
    "dalmuti": 2,
    "act-ended": 3,
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_v5_public_contract() -> dict[str, object]:
    """Return the immutable, privacy-safe V5 actor input contract.

    The digest intentionally covers the input allow-list, compact categorical
    layouts, belief equations, and padding rules.  A data/model bundle cannot
    silently reinterpret an older tensor layout while retaining this digest.
    """

    contract: dict[str, object] = {
        "contract": V5_PUBLIC_CONTRACT_ID,
        "version": V5_PUBLIC_CONTRACT_VERSION,
        "schemaVersion": V5_PUBLIC_SCHEMA_VERSION,
        "playerCounts": [V5_MIN_PLAYERS, V5_MAX_PLAYERS],
        "deckCounts": list(V5_DECK_COUNTS),
        "privacy": {
            "actorPrivateFields": ["ownRankCounts"],
            "publicFields": [
                "globalCodes",
                "publicPlayedCounts",
                "playerCodes",
                "playerMask",
                "tableCodes",
                "historyCodes",
                "historyMask",
                "legalMask",
            ],
            "forbiddenFields": [
                "opponentHands",
                "opponentRankCounts",
                "privateTaxCards",
                "privilegedState",
            ],
            "unknownFields": "reject",
        },
        "arrays": {
            "globalCodes": {
                "dtype": "int32",
                "shape": [len(V5_GLOBAL_FIELDS)],
                "fields": list(V5_GLOBAL_FIELDS),
            },
            "ownRankCounts": {
                "dtype": "uint8",
                "shape": [V5_RANK_COUNT],
            },
            "publicPlayedCounts": {
                "dtype": "uint8",
                "shape": [V5_RANK_COUNT],
            },
            "playerCodes": {
                "dtype": "uint8",
                "shape": [V5_MAX_PLAYERS, len(V5_PLAYER_FIELDS)],
                "fields": list(V5_PLAYER_FIELDS),
                "order": "actor-relative-clockwise",
            },
            "playerMask": {
                "dtype": "bool",
                "shape": [V5_MAX_PLAYERS],
                "padding": "right-contiguous-false",
            },
            "tableCodes": {
                "dtype": "uint8",
                "shape": [len(V5_TABLE_FIELDS)],
                "fields": list(V5_TABLE_FIELDS),
            },
            "historyCodes": {
                "dtype": "uint8",
                "shape": [V5_MAX_HISTORY, len(V5_HISTORY_FIELDS)],
                "fields": list(V5_HISTORY_FIELDS),
                "eventTypes": V5_EVENT_TYPES,
                "passReasons": V5_PASS_REASONS,
                "clearReasons": V5_CLEAR_REASONS,
                "order": "oldest-to-newest",
                "padding": "right-contiguous-zero",
            },
            "historyMask": {
                "dtype": "bool",
                "shape": [V5_MAX_HISTORY],
                "padding": "right-contiguous-false",
            },
            "legalMask": {
                "dtype": "bool",
                "shape": [V5_ACTION_COUNT],
                "catalogue": "dalmuti-fixed-236-v1",
            },
        },
        "belief": {
            "contract": "exchangeable-public-hypergeometric-v1",
            "unknownRankCount": (
                "deckCount-ownRankCount-publicPlayedCount"
            ),
            "populationSize": "sum(unknownRankCount)",
            "opponentDrawSize": "public remaining-card-count",
            "expectedCount": "drawSize*rankCount/populationSize",
            "probabilityAtLeastOne": "hypergeom-tail(threshold=1)",
            "probabilityAtLeastRequired": (
                "hypergeom-tail(threshold=table-required-count-or-1)"
            ),
            "responseFeasibility": (
                "exact multivariate-hypergeometric probability of at least "
                "one legal stronger-rank response, with public unseen jokers"
            ),
            "arithmetic": "exact-integer-combinations-before-float32-output",
            "paddedOpponentRows": V5_MAX_OPPONENTS,
        },
        "packedStorage": {
            "contract": "public-categorical-derived-beliefs-v1",
            "storedBeliefArrays": ["beliefResponseFeasibility"],
            "reconstructedBeliefArrays": {
                "unknownRankCounts": (
                    "deckCount-ownRankCount-publicPlayedCount"
                ),
                "expectedCounts": (
                    "float32(opponentDrawSize*unknownRankCount/populationSize)"
                ),
                "probabilityAtLeastOne": (
                    "float32(hypergeom-tail(threshold=1))"
                ),
                "probabilityAtLeastRequired": (
                    "float32(hypergeom-tail(threshold=table-required-count-or-1))"
                ),
                "opponentMask": "relativeOpponentIndex<playerCount-1",
            },
            "lookupBounds": {
                "population": [0, V5_DECK_SIZE],
                "rankSuccesses": [0, max(V5_DECK_COUNTS)],
                "opponentDrawSize": [0, V5_DECK_SIZE // V5_MIN_PLAYERS],
                "threshold": [0, 14],
            },
            "rounding": "exact-integer-combinations-then-float32",
        },
    }
    digest = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return {**contract, "contractSha256": digest}


V5_PUBLIC_CONTRACT_SHA256 = str(
    canonical_v5_public_contract()["contractSha256"]
)


def validate_v5_public_contract(value: object) -> dict[str, object]:
    expected = canonical_v5_public_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("V5 public-belief contract is missing or non-canonical")
    return expected


def v5_public_contract_fingerprint() -> str:
    return V5_PUBLIC_CONTRACT_SHA256
