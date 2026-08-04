from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_v6_match_splits.py"
SPEC = importlib.util.spec_from_file_location("build_v6_match_splits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def records(matches_per_player: int = 20) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seed = 1000
    for player_count in range(4, 11):
        for match_index in range(matches_per_player):
            result.append({
                "decisionCount": 10 + match_index,
                "decisionEnd": 10 + match_index,
                "decisionStart": 0,
                "localMatchIndex": match_index,
                "matchIndex": match_index,
                "matchSeed": seed,
                "nonforcedDecisionCount": 5,
                "playerCount": player_count,
                "shardManifestSha256": "1" * 64,
                "shardOrdinal": player_count - 4,
                "shardRelativePath": f"shard-p{player_count}",
            })
            seed += 1
    return result


class MatchSplitTests(unittest.TestCase):
    def test_assigns_exact_match_disjoint_80_10_10_per_player(self) -> None:
        source = records()
        split = MODULE.assign_match_splits(source, "a" * 64)
        self.assertEqual([len(split[name]) for name in MODULE.SPLIT_NAMES], [112, 14, 14])
        for player_count in range(4, 11):
            counts = [
                sum(item["playerCount"] == player_count for item in split[name])
                for name in MODULE.SPLIT_NAMES
            ]
            self.assertEqual(counts, [16, 2, 2])
        coordinates = [
            (item["playerCount"], item["matchIndex"])
            for name in MODULE.SPLIT_NAMES
            for item in split[name]
        ]
        self.assertEqual(len(coordinates), len(set(coordinates)))
        self.assertEqual(len(coordinates), len(source))

    def test_assignment_is_deterministic_and_bound_to_corpus_identity(self) -> None:
        source = records(30)
        first = MODULE.assign_match_splits(source, "b" * 64)
        second = MODULE.assign_match_splits(source, "b" * 64)
        self.assertEqual(first, second)
        changed = MODULE.assign_match_splits(source, "c" * 64)
        first_validation = {
            (item["playerCount"], item["matchIndex"])
            for item in first["validation"]
        }
        changed_validation = {
            (item["playerCount"], item["matchIndex"])
            for item in changed["validation"]
        }
        self.assertNotEqual(first_validation, changed_validation)

    def test_rejects_duplicate_coordinates_or_seeds(self) -> None:
        source = records()
        source[1]["matchSeed"] = source[0]["matchSeed"]
        with self.assertRaisesRegex(ValueError, "coordinates or seeds"):
            MODULE.assign_match_splits(source, "d" * 64)


if __name__ == "__main__":
    unittest.main()
