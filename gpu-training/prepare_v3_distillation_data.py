from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from actor_critic import ACTION_COUNT as LEGACY_ACTION_COUNT
from actor_critic import OBSERVATION_FEATURES, load_behavior_model
from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURE_LAYOUT,
    V3_ACTION_FEATURES,
)
from v3_distillation_dataset import (
    DISTILLATION_FORMAT,
    DISTILLATION_FORMAT_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    expand_paths,
    file_sha256,
    legacy_action_index_to_v3,
    validate_legacy_sample,
    validate_legacy_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill legacy 506-action PPO observations into a strict "
            "236-action V3 teacher dataset."
        )
    )
    parser.add_argument("--rollout", nargs="+", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--max-samples-per-source", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--include-forced",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _source_metadata(path: Path) -> dict[str, object]:
    digest = file_sha256(path)
    with path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline()
    if not first_line:
        raise ValueError(f"empty legacy rollout: {path}")
    try:
        manifest = json.loads(first_line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid legacy rollout manifest: {path}") from error
    contract = validate_legacy_manifest(manifest, path)
    return {
        "filename": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "manifestSha256": hashlib.sha256(first_line.encode("utf-8")).hexdigest(),
        **contract,
    }


def _write_record(stream, value: dict[str, object]) -> None:
    stream.write(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


def main() -> None:
    args = parse_args()
    if (
        not math.isfinite(args.temperature)
        or args.temperature <= 0
        or args.max_samples_per_source < 1
        or args.batch_size < 1
    ):
        raise ValueError(
            "temperature, max-samples-per-source, and batch-size must be positive"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"distillation dataset directory must be fresh: {output_dir}"
        )
    source_paths = expand_paths(args.rollout)
    teacher_path = Path(args.teacher_model).resolve()
    teacher_sha256 = file_sha256(teacher_path)
    teacher_model, teacher_payload = load_behavior_model(teacher_path)
    if (
        teacher_payload.get("format") != "dalmuti-actor-critic"
        or teacher_payload.get("version") != 1
        or teacher_payload.get("observationFeatures") != OBSERVATION_FEATURES
        or teacher_payload.get("actionCount") != LEGACY_ACTION_COUNT
    ):
        raise ValueError("teacher must be a legacy 506-action actor-critic JSON")
    source_metadata = [
        _source_metadata(path) for path in source_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "v3-distillation.ndjson"
    teacher_model = teacher_model.to(torch.device(args.device)).eval()
    manifest = {
        "type": "manifest",
        "format": DISTILLATION_FORMAT,
        "formatVersion": DISTILLATION_FORMAT_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "filename": teacher_path.name,
            "sha256": teacher_sha256,
            "format": "dalmuti-actor-critic",
            "version": 1,
            "observationFeatures": OBSERVATION_FEATURES,
            "actionCount": LEGACY_ACTION_COUNT,
            "temperature": args.temperature,
            "targets": "legal-action logits, temperature log probabilities, argmax, and value",
        },
        "observation": {
            "schemaVersion": OBSERVATION_SCHEMA_VERSION,
            "featureCount": OBSERVATION_FEATURES,
            "privacy": (
                "own private hand plus public state only; opponent hands excluded"
            ),
        },
        "actionSpace": {
            "legacySize": LEGACY_ACTION_COUNT,
            "catalogueVersion": V3_ACTION_CATALOGUE_VERSION,
            "size": V3_ACTION_COUNT,
            "catalogue": [dict(action) for action in V3_ACTION_CATALOGUE],
            "actionFeatures": V3_ACTION_FEATURE_COUNT,
            "actionFeatureLayout": list(V3_ACTION_FEATURE_LAYOUT),
            "encodedActionFeatures": [
                list(features) for features in V3_ACTION_FEATURES
            ],
            "mapping": (
                "semantic 506-to-236 bijection; legal set sorted in V3 catalogue order"
            ),
        },
        "grouping": {
            "splitUnit": "sourceSha256:episodeId",
            "rule": "all decisions from one source episode stay in one partition",
        },
        "selection": {
            "order": "first eligible samples in each independently shuffled rollout",
            "maxSamplesPerSource": args.max_samples_per_source,
            "includeForced": args.include_forced,
        },
        "sources": source_metadata,
    }
    sample_count = 0
    unique_groups: set[str] = set()
    source_counts: dict[str, int] = {}
    with dataset_path.open("x", encoding="utf-8", newline="\n") as output:
        _write_record(output, manifest)
        for source_index, (source_path, source) in enumerate(
            zip(source_paths, source_metadata)
        ):
            selected = 0
            batch: list[
                tuple[
                    int,
                    list[float],
                    list[int],
                    list[int],
                    str,
                    str,
                ]
            ] = []

            def flush() -> None:
                nonlocal sample_count, selected
                if not batch:
                    return
                observations = torch.tensor(
                    [entry[1] for entry in batch],
                    dtype=torch.float32,
                    device=args.device,
                )
                masks = torch.zeros(
                    (len(batch), LEGACY_ACTION_COUNT),
                    dtype=torch.bool,
                    device=args.device,
                )
                for offset, entry in enumerate(batch):
                    masks[offset, entry[2]] = True
                with torch.no_grad():
                    logits, values = teacher_model(observations, masks)
                logits = logits.detach().cpu()
                values = values.detach().cpu()
                for offset, entry in enumerate(batch):
                    (
                        line_number,
                        observation,
                        legacy_legal,
                        v3_legal,
                        episode_id,
                        trajectory_id,
                    ) = entry
                    mapped_pairs = sorted(
                        (legacy_action_index_to_v3(index), index)
                        for index in legacy_legal
                    )
                    if [pair[0] for pair in mapped_pairs] != v3_legal:
                        raise RuntimeError("legacy/V3 legal mapping changed")
                    legal_logits = torch.tensor(
                        [
                            float(logits[offset, legacy_index])
                            for _, legacy_index in mapped_pairs
                        ],
                        dtype=torch.float64,
                    )
                    log_probabilities = torch.log_softmax(
                        legal_logits / args.temperature, dim=0
                    )
                    maximum = float(legal_logits.max())
                    teacher_argmax = min(
                        v3_index
                        for v3_index, logit in zip(
                            v3_legal, legal_logits.tolist()
                        )
                        if logit == maximum
                    )
                    source_sha = str(source["sha256"])
                    group_key = f"{source_sha}:{episode_id}"
                    sample_id = f"source-{source_index + 1}:line-{line_number}"
                    _write_record(
                        output,
                        {
                            "type": "sample",
                            "sampleId": sample_id,
                            "sourceSha256": source_sha,
                            "episodeId": episode_id,
                            "trajectoryId": trajectory_id,
                            "groupKey": group_key,
                            "observation": observation,
                            "legacyLegalActionIndices": legacy_legal,
                            "legalActionIndices": v3_legal,
                            "teacherLogits": legal_logits.tolist(),
                            "teacherLogProbabilities": (
                                log_probabilities.tolist()
                            ),
                            "teacherArgmaxActionIndex": teacher_argmax,
                            "teacherValue": float(values[offset]),
                        },
                    )
                    sample_count += 1
                    selected += 1
                    unique_groups.add(group_key)
                batch.clear()

            with source_path.open("r", encoding="utf-8") as stream:
                first_line = stream.readline()
                if not first_line:
                    raise ValueError(f"empty legacy rollout: {source_path}")
                for line_number, line in enumerate(stream, start=2):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{source_path}:{line_number}: invalid JSON"
                        ) from error
                    if not isinstance(record, dict) or record.get("type") != "sample":
                        continue
                    (
                        observation,
                        legacy_legal,
                        v3_legal,
                        episode_id,
                        trajectory_id,
                        forced,
                    ) = validate_legacy_sample(
                        record,
                        source_path,
                        line_number,
                        expected_policy_version=(
                            f"sha256:{source['behaviorModelSha256']}"
                        ),
                    )
                    if forced and not args.include_forced:
                        continue
                    batch.append(
                        (
                            line_number,
                            observation,
                            legacy_legal,
                            v3_legal,
                            episode_id,
                            trajectory_id,
                        )
                    )
                    if len(batch) >= args.batch_size:
                        flush()
                    if selected + len(batch) >= args.max_samples_per_source:
                        flush()
                        break
            flush()
            if selected < 1:
                raise ValueError(f"no eligible samples selected from {source_path}")
            source_counts[str(source["sha256"])] = selected
            print(f"selected {selected} samples from {source_path.name}")
        summary = {
            "type": "summary",
            "samples": sample_count,
            "uniqueGroups": len(unique_groups),
            "teacherSha256": teacher_sha256,
            "temperature": args.temperature,
            "sourceSampleCounts": source_counts,
        }
        _write_record(output, summary)
    dataset_sha256 = file_sha256(dataset_path)
    checksum_path = dataset_path.with_suffix(f"{dataset_path.suffix}.sha256")
    with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{dataset_sha256}  {dataset_path.name}\n")
    report = {
        "format": DISTILLATION_FORMAT,
        "version": DISTILLATION_FORMAT_VERSION,
        "dataset": dataset_path.name,
        "datasetSha256": dataset_sha256,
        "datasetBytes": dataset_path.stat().st_size,
        "teacherSha256": teacher_sha256,
        "temperature": args.temperature,
        "samples": sample_count,
        "uniqueGroups": len(unique_groups),
        "sourceSampleCounts": source_counts,
    }
    (output_dir / "dataset-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
