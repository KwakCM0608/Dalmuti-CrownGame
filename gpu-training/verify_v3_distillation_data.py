from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v3_distillation_dataset import (
    file_sha256,
    group_split_mask,
    load_v3_distillation_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly verify V3 distillation data, teacher bindings, and "
            "episode-group split prerequisites."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--binding-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--output")
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    teacher_path = Path(args.teacher_model).resolve()
    loaded = load_v3_distillation_data(
        data_path,
        teacher_model_path=teacher_path,
        binding_tolerance=args.binding_tolerance,
        verify_teacher_bindings=True,
    )
    validation_mask = group_split_mask(
        loaded.group_keys,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    train_mask = ~validation_mask
    train_groups = set(loaded.group_keys[train_mask].tolist())
    validation_groups = set(loaded.group_keys[validation_mask].tolist())
    overlap = train_groups & validation_groups
    legal_counts = loaded.legal_masks.sum(axis=1)
    teacher_probabilities = loaded.teacher_probabilities
    safe_log_probabilities = np.where(
        teacher_probabilities > 0,
        np.log(np.clip(teacher_probabilities, 1.0e-30, None)),
        0.0,
    )
    teacher_entropy = -(
        teacher_probabilities * safe_log_probabilities
    ).sum(axis=1)
    report = {
        "format": "dalmuti-v3-distillation-data-verification",
        "version": 1,
        "data": {
            "filename": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": file_sha256(data_path),
        },
        "teacher": {
            "filename": teacher_path.name,
            "bytes": teacher_path.stat().st_size,
            "sha256": loaded.teacher_sha256,
            "temperature": loaded.temperature,
        },
        "samples": len(loaded),
        "uniqueEpisodeGroups": len(set(loaded.group_keys.tolist())),
        "observationShape": list(loaded.observations.shape),
        "legalMaskShape": list(loaded.legal_masks.shape),
        "legalActions": {
            "minimum": int(legal_counts.min()),
            "maximum": int(legal_counts.max()),
            "mean": float(legal_counts.mean()),
        },
        "teacherDistribution": {
            "meanEntropy": float(teacher_entropy.mean()),
            "minimumEntropy": float(teacher_entropy.min()),
            "maximumEntropy": float(teacher_entropy.max()),
            "probabilityRowSumMaximumAbsoluteError": float(
                np.abs(teacher_probabilities.sum(axis=1) - 1.0).max()
            ),
        },
        "split": {
            "groupSplitKey": "sourceSha256:episodeId",
            "validationFraction": args.validation_fraction,
            "splitSeed": args.split_seed,
            "train": {
                "samples": int(train_mask.sum()),
                "uniqueGroups": len(train_groups),
            },
            "validation": {
                "samples": int(validation_mask.sum()),
                "uniqueGroups": len(validation_groups),
            },
            "overlappingGroups": len(overlap),
        },
        "bindings": {
            "datasetChecksum": "verified",
            "teacherSha256": "verified",
            "teacherLogits": "recomputed-and-verified",
            "teacherTemperatureLogProbabilities": "recomputed-and-verified",
            "teacherArgmax": "recomputed-and-verified",
            "teacherValue": "recomputed-and-verified",
            "legacyToV3LegalSet": "semantic-round-trip-and-observation-verified",
            "absoluteTolerance": args.binding_tolerance,
        },
        "finite": bool(
            np.isfinite(loaded.observations).all()
            and np.isfinite(loaded.teacher_probabilities).all()
            and np.isfinite(loaded.teacher_values).all()
            and np.isfinite(teacher_entropy).all()
        ),
    }
    if overlap or not report["finite"]:
        raise ValueError("distillation verification failed")
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
