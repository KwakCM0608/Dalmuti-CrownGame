from __future__ import annotations

"""Build immutable, match-disjoint V6 train/validation/test views of V5 data."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
GPU_TRAINING = SOURCE_ROOT / "gpu-training"
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

from v5_dataset import (  # noqa: E402
    V5_INDEX_FORMAT,
    V5_INDEX_VERSION,
)


FORMAT = "dalmuti-v6-match-disjoint-split"
VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")
PLAYER_COUNTS = tuple(range(4, 11))
DOMAIN = b"DALMUTI-V6-MATCH-SPLIT\0"


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ASCII JSON: {path}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError(f"JSON is not canonical: {path}")
    return value


def _verified_mmap_array(
    shard_path: Path,
    shard_manifest: Mapping[str, object],
    partition: str,
    name: str,
) -> np.ndarray:
    partitions = shard_manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("shard partitions are malformed")
    records = partitions.get(partition)
    if not isinstance(records, dict):
        raise ValueError(f"shard {partition} partition is malformed")
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"shard omitted {partition}.{name}")
    expected_relative = f"{partition}/{name}.npy"
    if record.get("path") != expected_relative:
        raise ValueError(f"shard path drifted for {partition}.{name}")
    path = (shard_path / expected_relative).resolve(strict=True)
    if path.parent != (shard_path / partition).resolve(strict=True):
        raise ValueError("shard array path escaped its partition")
    if (
        path.stat().st_size != record.get("byteLength")
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"shard array checksum drifted for {partition}.{name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(array, np.memmap)
        or array.dtype.str != record.get("dtype")
        or list(array.shape) != record.get("shape")
    ):
        raise ValueError(f"shard array header drifted for {partition}.{name}")
    return array


def _split_hash(
    corpus_sha256: str,
    player_count: int,
    match_index: int,
    match_seed: int,
) -> str:
    if len(corpus_sha256) != 64:
        raise ValueError("corpus SHA-256 is malformed")
    return hashlib.sha256(
        DOMAIN
        + bytes.fromhex(corpus_sha256)
        + player_count.to_bytes(1, "little")
        + match_index.to_bytes(4, "little")
        + match_seed.to_bytes(4, "little")
    ).hexdigest()


def assign_match_splits(
    records: Sequence[Mapping[str, object]],
    corpus_sha256: str,
) -> dict[str, list[dict[str, object]]]:
    """SHA-sort whole matches within each p and assign exact 80/10/10 counts."""

    by_player: dict[int, list[dict[str, object]]] = defaultdict(list)
    coordinates: set[tuple[int, int]] = set()
    seeds: set[int] = set()
    for raw in records:
        record = dict(raw)
        player_count = int(record["playerCount"])
        match_index = int(record["matchIndex"])
        match_seed = int(record["matchSeed"])
        if player_count not in PLAYER_COUNTS:
            raise ValueError("match record player count escaped p4..p10")
        coordinate = (player_count, match_index)
        if coordinate in coordinates or match_seed in seeds:
            raise ValueError("match coordinates or seeds are not globally unique")
        coordinates.add(coordinate)
        seeds.add(match_seed)
        record["splitHash"] = _split_hash(
            corpus_sha256, player_count, match_index, match_seed
        )
        by_player[player_count].append(record)
    if set(by_player) != set(PLAYER_COUNTS):
        raise ValueError("corpus must contain every p4..p10 stratum")

    output = {name: [] for name in SPLIT_NAMES}
    for player_count in PLAYER_COUNTS:
        ordered = sorted(
            by_player[player_count],
            key=lambda item: (
                str(item["splitHash"]),
                int(item["matchIndex"]),
                int(item["matchSeed"]),
            ),
        )
        holdout = len(ordered) // 10
        train = len(ordered) - 2 * holdout
        ranges = {
            "train": ordered[:train],
            "validation": ordered[train : train + holdout],
            "test": ordered[train + holdout :],
        }
        if any(not values for values in ranges.values()):
            raise ValueError("each player count needs at least one match per split")
        for name, values in ranges.items():
            for value in values:
                value["split"] = name
            output[name].extend(values)

    assigned = [
        (int(item["playerCount"]), int(item["matchIndex"]))
        for name in SPLIT_NAMES
        for item in output[name]
    ]
    if len(assigned) != len(records) or len(set(assigned)) != len(records):
        raise RuntimeError("split assignment lost or duplicated a complete match")
    for name in SPLIT_NAMES:
        output[name].sort(
            key=lambda item: (
                int(item["playerCount"]),
                str(item["splitHash"]),
                int(item["matchIndex"]),
            )
        )
    return output


def _load_match_records(index_root: Path) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    manifest_path = index_root / "manifest.json"
    manifest = _strict_canonical_json(manifest_path)
    corpus_sha = sha256_file(manifest_path)
    sidecar = manifest_path.with_name("manifest.json.sha256")
    if sidecar.read_bytes() != f"{corpus_sha}  manifest.json\n".encode("ascii"):
        raise ValueError("dataset index checksum sidecar does not match")
    if (
        manifest.get("format") != V5_INDEX_FORMAT
        or manifest.get("version") != V5_INDEX_VERSION
    ):
        raise ValueError("input is not a supported V5 zero-copy index")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("dataset index has no shard records")

    records: list[dict[str, object]] = []
    observed_decisions = 0
    observed_matches = 0
    for shard_ordinal, raw in enumerate(raw_shards):
        if not isinstance(raw, dict):
            raise ValueError("dataset index shard record is malformed")
        relative = raw.get("relativePath")
        expected_manifest_sha = raw.get("manifestSha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("dataset index shard path is unsafe")
        shard_path = (index_root.parent / relative).resolve(strict=True)
        shard_manifest_path = shard_path / "manifest.json"
        if sha256_file(shard_manifest_path) != expected_manifest_sha:
            raise ValueError("dataset shard manifest SHA disagrees with its index")
        shard_manifest = _strict_canonical_json(shard_manifest_path)
        shard_sidecar = shard_manifest_path.with_name("manifest.json.sha256")
        if shard_sidecar.read_bytes() != (
            f"{expected_manifest_sha}  manifest.json\n".encode("ascii")
        ):
            raise ValueError("dataset shard manifest sidecar disagrees with its index")
        arrays = {
            name: _verified_mmap_array(shard_path, shard_manifest, "actor", name)
            for name in ("forced", "match_offsets", "player_counts")
        }
        private = {
            name: _verified_mmap_array(shard_path, shard_manifest, "privileged", name)
            for name in ("match_indices", "match_seeds")
        }
        try:
            offsets = np.asarray(arrays["match_offsets"], dtype=np.int64)
            player_counts = np.asarray(arrays["player_counts"], dtype=np.int64)
            match_indices = np.asarray(private["match_indices"], dtype=np.int64)
            match_seeds = np.asarray(private["match_seeds"], dtype=np.int64)
            forced = np.asarray(arrays["forced"], dtype=np.bool_)
            counts = shard_manifest.get("counts")
            if not isinstance(counts, dict):
                raise ValueError("shard counts are malformed")
            match_count = int(counts["matches"])
            decision_count = int(counts["decisions"])
            if not (
                offsets.shape == (match_count + 1,)
                and player_counts.shape == (match_count,)
                and match_indices.shape == (match_count,)
                and match_seeds.shape == (match_count,)
            ):
                raise ValueError("shard match-level arrays have inconsistent shapes")
            for local_index in range(match_count):
                start = int(offsets[local_index])
                stop = int(offsets[local_index + 1])
                records.append({
                    "decisionEnd": stop,
                    "decisionStart": start,
                    "decisionCount": stop - start,
                    "localMatchIndex": local_index,
                    "matchIndex": int(match_indices[local_index]),
                    "matchSeed": int(match_seeds[local_index]),
                    "nonforcedDecisionCount": int((~forced[start:stop]).sum()),
                    "playerCount": int(player_counts[local_index]),
                    "shardManifestSha256": expected_manifest_sha,
                    "shardOrdinal": shard_ordinal,
                    "shardRelativePath": relative,
                })
            if int(offsets[-1]) != decision_count or forced.shape != (decision_count,):
                raise ValueError("shard decision-level arrays have inconsistent shapes")
            observed_decisions += decision_count
            observed_matches += match_count
        finally:
            for array in (*arrays.values(), *private.values()):
                mapping = getattr(array, "_mmap", None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
    counts = manifest.get("counts")
    if counts != {
        "decisions": observed_decisions,
        "matches": observed_matches,
        "shards": len(raw_shards),
    }:
        raise ValueError("dataset aggregate counts drifted while building splits")
    return corpus_sha, records, manifest


def _summary(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for name in SPLIT_NAMES:
        values = splits[name]
        per_player: dict[str, object] = {}
        for player_count in PLAYER_COUNTS:
            selected = [item for item in values if int(item["playerCount"]) == player_count]
            per_player[str(player_count)] = {
                "decisions": sum(int(item["decisionCount"]) for item in selected),
                "matches": len(selected),
                "nonforcedDecisions": sum(
                    int(item["nonforcedDecisionCount"]) for item in selected
                ),
            }
        output[name] = {
            "decisions": sum(int(item["decisionCount"]) for item in values),
            "matches": len(values),
            "nonforcedDecisions": sum(
                int(item["nonforcedDecisionCount"]) for item in values
            ),
            "perPlayerCount": per_player,
        }
    return output


def build_split_manifest(index_root: Path) -> dict[str, object]:
    corpus_sha, records, source_manifest = _load_match_records(index_root)
    splits = assign_match_splits(records, corpus_sha)
    return {
        "assignment": {
            "domain": "DALMUTI-V6-MATCH-SPLIT",
            "method": "sha256-sort-within-player-count-exact-80-10-10",
            "unit": "complete-five-act-match",
        },
        "corpusIdentitySha256": corpus_sha,
        "format": FORMAT,
        "privacy": {
            "actorInputMayReadPrivilegedPartition": False,
            "privilegedTargetsTrainingOnly": True,
            "splitBeforeOverlappingHistoryPrefixes": True,
        },
        "sourceCounts": source_manifest["counts"],
        "sourceIndex": str(index_root.resolve()),
        "splits": splits,
        "summary": _summary(splits),
        "version": VERSION,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    target = arguments.output.resolve()
    if target.exists():
        raise FileExistsError("V6 split manifest output already exists")
    result = build_split_manifest(arguments.dataset_index.resolve(strict=True))
    raw = canonical_json_bytes(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with target.with_name(target.name + ".sha256").open("xb") as output:
        output.write(f"{digest}  {target.name}\n".encode("ascii"))
    print(json.dumps({
        "corpusIdentitySha256": result["corpusIdentitySha256"],
        "output": str(target),
        "outputSha256": digest,
        "summary": result["summary"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
