from __future__ import annotations

"""Verify TypeScript V4 trace fixtures against the Python reference env."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Mapping, Sequence

from v4_env import (
    ACTION_CATALOGUE,
    ACTION_COUNT,
    DalmutiScalarEnv,
    MAX_HISTORY,
    MEMORY_TRACE_DECAYS,
    ROLES,
    role_for_index,
)


FORMAT = "dalmuti-v4-env-parity-ndjson"
VERSION = 1
PASS_REASONS = ("manual", "timeout", "insufficient-cards", "dalmuti")
CLEAR_REASONS = ("all-passed", "dalmuti", "act-ended")
EVENT_TYPES = ("play", "pass", "clear", "finish")


class ParityError(AssertionError):
    pass


def _fail(location: str, message: str) -> None:
    raise ParityError(f"{location}: {message}")


def _assert_equal(expected: object, actual: object, location: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            _fail(location, f"expected object, got {type(actual).__name__}")
        expected_keys = list(expected)
        actual_keys = list(actual)
        if set(expected_keys) != set(actual_keys):
            _fail(
                location,
                f"object keys differ; expected {expected_keys}, got {actual_keys}",
            )
        for key in expected_keys:
            _assert_equal(expected[key], actual[key], f"{location}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            _fail(location, f"expected list, got {type(actual).__name__}")
        if len(expected) != len(actual):
            _fail(location, f"length differs; expected {len(expected)}, got {len(actual)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_equal(expected_item, actual_item, f"{location}[{index}]")
        return
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        _fail(
            location,
            f"type differs; expected {type(expected).__name__}, got {type(actual).__name__}",
        )
    if expected != actual:
        _fail(location, f"expected {expected!r}, got {actual!r}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalogue_sha256() -> str:
    records: list[dict[str, object]] = []
    for action in ACTION_CATALOGUE:
        if action.type == "pass":
            records.append({"type": "pass"})
        elif action.type == "solo-joker":
            records.append({"type": "solo-joker"})
        else:
            records.append(
                {
                    "type": "play",
                    "rank": action.rank,
                    "count": action.count,
                    "jokerCount": action.joker_count,
                }
            )
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(payload).hexdigest()


def _legal_mask_hex(env: DalmutiScalarEnv) -> str:
    mask = env.legal_mask().detach().cpu().tolist()
    nibbles = [0] * (ACTION_COUNT // 4)
    for action_index, legal in enumerate(mask):
        if legal:
            nibbles[action_index // 4] |= 1 << (action_index % 4)
    return "".join(format(value, "x") for value in nibbles)


def _relative_order(env: DalmutiScalarEnv, actor_id: int) -> list[int]:
    actor_position = env._order.index(actor_id)
    return [
        env._order[(actor_position + offset) % env.player_count]
        for offset in range(env.player_count)
    ]


def _history_token(
    env: DalmutiScalarEnv,
    event: Mapping[str, object],
    actor_id: int,
) -> dict[str, int]:
    relative = _relative_order(env, actor_id)
    event_type = str(event["type"])
    result = {
        "sequence": int(event["sequence"]),
        "type": EVENT_TYPES.index(event_type),
        "actorOffset": relative.index(int(event["actor_id"])),
        "handCountBefore": int(event["hand_before"]),
        "handCountAfter": int(event["hand_after"]),
        "rank": 0,
        "naturalCount": 0,
        "jokerCount": 0,
        "totalCount": 0,
        "passReason": 0,
        "clearReason": 0,
        "nextLeaderOffset": -1,
        "finishPlace": 0,
    }
    if event_type in ("play", "clear"):
        result["rank"] = int(event["rank"])
        result["naturalCount"] = int(event["natural_count"])
        result["jokerCount"] = int(event["joker_count"])
        result["totalCount"] = int(event["total_count"])
    if event_type == "pass":
        result["passReason"] = PASS_REASONS.index(str(event["pass_reason"])) + 1
    elif event_type == "clear":
        result["clearReason"] = CLEAR_REASONS.index(str(event["clear_reason"])) + 1
        next_id = event.get("next_leader_id")
        result["nextLeaderOffset"] = (
            -1 if next_id is None else relative.index(int(next_id))
        )
    elif event_type == "finish":
        result["finishPlace"] = int(event["finish_place"])
    return result


def _memory_trace_feature(player_count: int, token: Mapping[str, int]) -> list[float]:
    features = [0.0] * 20
    features[token["type"]] = 1.0
    features[4] = token["actorOffset"] / max(1, player_count - 1)
    features[5] = token["handCountBefore"] / 20.0
    features[6] = token["handCountAfter"] / 20.0
    features[7] = token["rank"] / 13.0
    features[8] = token["naturalCount"] / 12.0
    features[9] = token["jokerCount"] / 2.0
    features[10] = token["totalCount"] / 14.0
    if token["passReason"] > 0:
        features[10 + token["passReason"]] = 1.0
    if token["clearReason"] > 0:
        features[14 + token["clearReason"]] = 1.0
    features[18] = (
        0.0
        if token["nextLeaderOffset"] < 0
        else (token["nextLeaderOffset"] + 1) / player_count
    )
    features[19] = token["finishPlace"] / player_count
    return features


def _observation_core(env: DalmutiScalarEnv) -> dict[str, object]:
    actor_id = env.current_player_id
    if actor_id < 0:
        raise RuntimeError("terminal state has no actor observation")
    relative = _relative_order(env, actor_id)
    actor_position = env._order.index(actor_id)
    own_counts = [0] * 13
    for card in env._hands[actor_id]:
        own_counts[card.rank - 1] += 1
    table = env._table
    table_value = None
    if table is not None:
        table_value = {
            "actorOffset": relative.index(table.player_id),
            "rank": table.rank,
            "naturalCount": table.natural_count,
            "jokerCount": table.joker_count,
            "totalCount": table.count,
        }
    player_tokens = []
    for offset, player_id in enumerate(relative):
        absolute_position = env._order.index(player_id)
        player_tokens.append(
            {
                "relativeOffset": offset,
                "handCount": len(env._hands[player_id]),
                "finished": int(len(env._hands[player_id]) == 0),
                "passed": int(player_id in env._passed),
                "self": int(offset == 0),
                "tableLeader": int(table is not None and table.player_id == player_id),
                "role": ROLES.index(role_for_index(absolute_position, env.player_count)),
                "score": env._scores[player_id],
            }
        )
    history_tokens = [
        _history_token(env, event, actor_id) for event in env._history
    ]
    truncated = max(0, len(history_tokens) - MAX_HISTORY)
    old_tokens = history_tokens[:truncated]
    recent_tokens = history_tokens[truncated:]
    memory_vectors = []
    for decay in MEMORY_TRACE_DECAYS:
        trace = [0.0] * 20
        for token in old_tokens:
            values = _memory_trace_feature(env.player_count, token)
            for index in range(20):
                trace[index] = decay * trace[index] + (1.0 - decay) * values[index]
        memory_vectors.append(trace)
    history_bytes = json.dumps(
        recent_tokens,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    memory_bytes = b"".join(
        struct.pack("<d", value)
        for trace in memory_vectors
        for value in trace
    )
    return {
        "schemaVersion": 4,
        "playerCount": env.player_count,
        "act": env._act,
        "actorRole": ROLES.index(role_for_index(actor_position, env.player_count)),
        "revolution": env._revolution,
        "ownHandCounts": own_counts,
        "publicPlayedCounts": list(env._public_played),
        "table": table_value,
        "playerTokens": player_tokens,
        "truncatedHistoryCount": truncated,
        "historyTokenCount": len(recent_tokens),
        "historyFirstSequence": -1 if not recent_tokens else recent_tokens[0]["sequence"],
        "historyLastSequence": -1 if not recent_tokens else recent_tokens[-1]["sequence"],
        "historyTokenBytesLength": len(history_bytes),
        "historyTokenBytesSha256": sha256(history_bytes).hexdigest(),
        "memoryTraceFloat64BytesLength": len(memory_bytes),
        "memoryTraceFloat64Sha256": sha256(memory_bytes).hexdigest(),
    }


def _normalize_python_event(event: Mapping[str, object]) -> dict[str, object]:
    event_type = str(event["type"])
    base: dict[str, object] = {
        "type": event_type,
        "sequence": int(event["sequence"]),
        "actorIndex": int(event["actor_id"]),
        "handCountBefore": int(event["hand_before"]),
        "handCountAfter": int(event["hand_after"]),
    }
    if event_type == "play":
        return {
            **base,
            "rank": int(event["rank"]),
            "naturalCount": int(event["natural_count"]),
            "jokerCount": int(event["joker_count"]),
            "totalCount": int(event["total_count"]),
        }
    if event_type == "pass":
        return {**base, "reason": str(event["pass_reason"])}
    if event_type == "clear":
        next_id = event.get("next_leader_id")
        return {
            **base,
            "rank": int(event["rank"]),
            "naturalCount": int(event["natural_count"]),
            "jokerCount": int(event["joker_count"]),
            "totalCount": int(event["total_count"]),
            "reason": str(event["clear_reason"]),
            "nextLeaderIndex": None if next_id is None else int(next_id),
        }
    return {**base, "place": int(event["finish_place"])}


def _taxation_summary(act_result: Mapping[str, object]) -> dict[str, object]:
    taxation = list(act_result["taxation"])
    counts = [len(exchange["tribute_card_ids"]) for exchange in taxation]
    return {
        "applied": bool(taxation),
        "exchangeCounts": counts,
        "transferredEachDirection": sum(counts),
    }


def _act_summary(
    act_result: Mapping[str, object], player_count: int
) -> dict[str, object]:
    chips = act_result["chip_awards"]
    return {
        "act": int(act_result["act"]),
        "playerOrder": list(act_result["player_order"]),
        "finishOrder": list(act_result["finish_order"]),
        "revolution": int(act_result["revolution"]),
        "taxation": _taxation_summary(act_result),
        "chipAwardsByPlayer": [chips[index] for index in range(player_count)],
        "transitions": int(act_result["transitions"]),
    }


def _verify_manifest(
    manifest: Mapping[str, object], repository_root: Path
) -> tuple[dict[str, object], set[tuple[int, int]]]:
    _assert_equal("manifest", manifest.get("type"), "manifest.type")
    _assert_equal(FORMAT, manifest.get("format"), "manifest.format")
    _assert_equal(VERSION, manifest.get("version"), "manifest.version")
    action = manifest.get("actionSpace")
    if not isinstance(action, dict):
        _fail("manifest.actionSpace", "must be an object")
    _assert_equal(ACTION_COUNT, action.get("size"), "manifest.actionSpace.size")
    _assert_equal(
        ACTION_COUNT // 4,
        action.get("legalMaskHexLength"),
        "manifest.actionSpace.legalMaskHexLength",
    )
    _assert_equal(
        _catalogue_sha256(),
        action.get("catalogueSha256"),
        "manifest.actionSpace.catalogueSha256",
    )
    source_hashes = manifest.get("sourceHashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        _fail("manifest.sourceHashes", "must be a non-empty object")
    for relative_path, expected_hash in source_hashes.items():
        path = repository_root / relative_path
        if not path.is_file():
            _fail(f"manifest.sourceHashes.{relative_path}", "bound source is missing")
        _assert_equal(
            expected_hash,
            _sha256_file(path),
            f"manifest.sourceHashes.{relative_path}",
        )
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        _fail("manifest.environment", "must be an object")
    _assert_equal("normal", environment.get("behaviorPolicy"), "manifest.environment.behaviorPolicy")
    players = environment.get("players")
    schedule = environment.get("seedSchedule")
    acts = environment.get("acts")
    if not isinstance(players, list) or not isinstance(schedule, dict):
        _fail("manifest.environment", "players and seedSchedule are invalid")
    if not isinstance(acts, int) or acts < 1:
        _fail("manifest.environment.acts", "must be positive")
    expected_matches: set[tuple[int, int]] = set()
    for player_count in players:
        seeds = schedule.get(str(player_count))
        if not isinstance(player_count, int) or not 4 <= player_count <= 10:
            _fail("manifest.environment.players", "contains an unsupported count")
        if not isinstance(seeds, list) or not seeds:
            _fail(f"manifest.environment.seedSchedule.{player_count}", "has no seeds")
        for seed in seeds:
            if not isinstance(seed, int) or seed < 1:
                _fail("manifest.environment.seedSchedule", "contains an invalid seed")
            pair = (player_count, seed)
            if pair in expected_matches:
                _fail("manifest.environment.seedSchedule", "contains a duplicate match")
            expected_matches.add(pair)
    if manifest.get("testOnly") is not True:
        _assert_equal(list(range(4, 11)), players, "manifest.environment.players")
        if acts != 5 or any(len(schedule[str(player)]) < 5 for player in players):
            _fail("manifest.environment", "production fixture coverage is below p4-p10 x5 seeds x5 acts")
    return environment, expected_matches


def verify_fixture(
    fixture_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, int]:
    fixture = Path(fixture_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent.parent
    )
    checksum_path = Path(f"{fixture}.sha256")
    if not checksum_path.is_file():
        _fail("fixture.sha256", f"missing sidecar {checksum_path}")
    sidecar = checksum_path.read_text(encoding="ascii").strip()
    _assert_equal(sidecar, _sha256_file(fixture), "fixture.sha256")

    record_digest = sha256()
    environment: dict[str, object] | None = None
    expected_matches: set[tuple[int, int]] = set()
    seen_matches: set[tuple[int, int]] = set()
    env: DalmutiScalarEnv | None = None
    match_id: str | None = None
    current_pair: tuple[int, int] | None = None
    expected_decision = 0
    last_step = None
    match_count = 0
    decision_count = 0
    records_before_summary = 0
    saw_summary = False

    with fixture.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.endswith(b"\n"):
                _fail(f"line {line_number}", "record is not newline terminated")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                _fail(f"line {line_number}", f"invalid UTF-8 JSON: {error}")
            if not isinstance(record, dict):
                _fail(f"line {line_number}", "record must be an object")
            record_type = record.get("type")
            if saw_summary:
                _fail(f"line {line_number}", "summary must be the final record")
            if record_type != "summary":
                record_digest.update(raw_line)
                records_before_summary += 1

            if record_type == "manifest":
                if line_number != 1 or environment is not None:
                    _fail("manifest", "must be the first and only manifest")
                environment, expected_matches = _verify_manifest(record, root)
                continue
            if environment is None:
                _fail(f"line {line_number}", "manifest was not read first")

            if record_type == "match-start":
                if env is not None:
                    _fail(f"line {line_number}", "previous match is incomplete")
                player_count = record.get("playerCount")
                seed = record.get("seed")
                acts = record.get("acts")
                if not all(isinstance(value, int) for value in (player_count, seed, acts)):
                    _fail(f"line {line_number}", "match dimensions must be integers")
                current_pair = (player_count, seed)
                if current_pair not in expected_matches or current_pair in seen_matches:
                    _fail(f"line {line_number}", f"unexpected or duplicate match {current_pair}")
                match_id = str(record.get("matchId"))
                _assert_equal(
                    f"p{player_count}-seed-{seed}",
                    match_id,
                    f"{match_id}.matchId",
                )
                _assert_equal(environment["acts"], acts, f"{match_id}.acts")
                env = DalmutiScalarEnv(player_count, acts=acts, seed=seed)
                expected_decision = 0
                last_step = None
                continue

            if record_type == "summary":
                if env is not None:
                    _fail("summary", "last match is incomplete")
                _assert_equal(expected_matches, seen_matches, "summary.matchSchedule")
                _assert_equal(match_count, record.get("matches"), "summary.matches")
                _assert_equal(decision_count, record.get("decisions"), "summary.decisions")
                _assert_equal(records_before_summary, record.get("recordsBeforeSummary"), "summary.recordsBeforeSummary")
                _assert_equal(record_digest.hexdigest(), record.get("recordsBeforeSummarySha256"), "summary.recordsBeforeSummarySha256")
                saw_summary = True
                continue

            if env is None or match_id is None or current_pair is None:
                _fail(f"line {line_number}", "record is outside a match")
            _assert_equal(match_id, record.get("matchId"), f"line {line_number}.matchId")
            location = f"{match_id}.act-{record.get('act', env._act)}"

            if record_type == "act-start":
                _assert_equal(env._act, record.get("act"), f"{location}.act")
                _assert_equal(list(env._order), record.get("playerOrder"), f"{location}.playerOrder")
                _assert_equal(env._revolution, record.get("revolution"), f"{location}.revolution")
                expected_tax = {
                    "applied": bool(env._tax_audit),
                    "exchangeCounts": [
                        len(exchange["tribute_card_ids"]) for exchange in env._tax_audit
                    ],
                    "transferredEachDirection": sum(
                        len(exchange["tribute_card_ids"]) for exchange in env._tax_audit
                    ),
                }
                _assert_equal(expected_tax, record.get("taxation"), f"{location}.taxation")
                _assert_equal(env.current_player_id, record.get("firstActorIndex"), f"{location}.firstActorIndex")
                _assert_equal(
                    _observation_core(env),
                    record.get("initialObservationCore"),
                    f"{location}.initialObservationCore",
                )
                expected_decision = 0
                last_step = None
                continue

            if record_type == "decision":
                decision = record.get("decision")
                decision_location = f"{location}.decision-{decision}"
                _assert_equal(expected_decision, decision, f"{decision_location}.decision")
                _assert_equal(env._act, record.get("act"), f"{decision_location}.act")
                _assert_equal(env.current_player_id, record.get("actorIndex"), f"{decision_location}.actorIndex")
                _assert_equal(env._current_index, record.get("actorSeat"), f"{decision_location}.actorSeat")
                role = role_for_index(env._current_index, env.player_count)
                _assert_equal(role, record.get("actorRole"), f"{decision_location}.actorRole")
                _assert_equal(
                    _observation_core(env),
                    record.get("observationCore"),
                    f"{decision_location}.observationCore",
                )
                legal_hex = _legal_mask_hex(env)
                _assert_equal(legal_hex, record.get("legalMaskHex"), f"{decision_location}.legalMaskHex")
                normal_action = env.normal_action()
                _assert_equal(normal_action, record.get("normalActionIndex"), f"{decision_location}.normalActionIndex")
                history = env._history
                history_length = len(history)
                last_step = env.step(normal_action)
                events = [
                    _normalize_python_event(event)
                    for event in history[history_length:]
                ]
                _assert_equal(events, record.get("eventsAfterAction"), f"{decision_location}.eventsAfterAction")
                expected_decision += 1
                decision_count += 1
                continue

            if record_type == "act-summary":
                if last_step is None or not last_step.act_ended:
                    _fail(location, "act summary arrived before an act-ending step")
                actual = {"type": "act-summary", "matchId": match_id, **_act_summary(last_step.info["act_result"], env.player_count)}
                _assert_equal(actual, record, f"{location}.summary")
                continue

            if record_type == "match-summary":
                if not env.terminated:
                    _fail(match_id, "match summary arrived before termination")
                actual_scores = [env._scores[index] for index in range(env.player_count)]
                _assert_equal(actual_scores, record.get("finalScoresByPlayer"), f"{match_id}.finalScoresByPlayer")
                seen_matches.add(current_pair)
                match_count += 1
                env = None
                match_id = None
                current_pair = None
                last_step = None
                continue

            _fail(f"line {line_number}", f"unknown record type {record_type!r}")

    if not saw_summary:
        _fail("fixture", "summary record is missing")
    return {"matches": match_count, "decisions": decision_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    result = verify_fixture(args.fixture, repository_root=args.repository_root)
    print(
        f"Verified {result['matches']} matches and {result['decisions']} decisions"
    )


if __name__ == "__main__":
    main()


__all__ = ["ParityError", "verify_fixture"]
