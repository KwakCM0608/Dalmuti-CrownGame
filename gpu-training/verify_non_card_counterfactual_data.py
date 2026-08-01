"""CLI preflight for non-card counterfactual NDJSON files."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from non_card_counterfactual_dataset import (
    load_non_card_counterfactuals,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify DALMUTI non-card counterfactual data."
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        help="Allow mixed source temperatures because training will recompute targets.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.policy_temperature is not None and (
        not math.isfinite(args.policy_temperature)
        or args.policy_temperature <= 0
    ):
        parser.error("--policy-temperature must be finite and greater than zero")
    datasets = load_non_card_counterfactuals(
        args.data,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        allow_mixed_policy_temperatures=args.policy_temperature is not None,
    )
    decisions = {}
    for name, split in (
        ("tax-return", datasets.tax_return),
        ("revolution", datasets.revolution),
    ):
        if split is None:
            continue
        decisions[name] = {
            "trainSamples": len(split.train),
            "validationSamples": len(split.validation),
            "trainUniqueEpisodes": len(set(split.train.episode_ids)),
            "validationUniqueEpisodes": len(set(split.validation.episode_ids)),
            "trainUniqueWorlds": len(set(split.train.world_keys)),
            "validationUniqueWorlds": len(set(split.validation.world_keys)),
            "trainTargetBestEqualsBaselineRate": (
                float(
                    np.mean(
                        split.train.best_actions
                        == split.train.baseline_actions
                    )
                )
                if len(split.train)
                else None
            ),
            "validationTargetBestEqualsBaselineRate": (
                float(
                    np.mean(
                        split.validation.best_actions
                        == split.validation.baseline_actions
                    )
                )
                if len(split.validation)
                else None
            ),
        }
    report = {
        "format": "dalmuti-non-card-counterfactual-verification",
        "version": 1,
        "groupSplitKey": datasets.group_split_key,
        "splitSeed": datasets.split_seed,
        "validationFraction": datasets.validation_fraction,
        "policyTemperatureOverride": args.policy_temperature,
        "sourceActs": sorted(
            {
                report.manifest["collection"]["acts"]
                for report in datasets.files
            }
        ),
        "sourcePolicyTemperatures": sorted(
            {
                report.manifest["collection"]["policyTemperature"]
                for report in datasets.files
            }
        ),
        "files": [
            {
                "path": item.path,
                "bytes": item.bytes,
                "sha256": item.sha256,
                "decisions": item.decisions,
                "actionEvaluations": item.action_evaluations,
            }
            for item in datasets.files
        ],
        "decisions": decisions,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Verified {len(datasets.files)} counterfactual file(s) with "
            f"canonical-world-grouped split seed {datasets.split_seed}."
        )
        for decision, counts in decisions.items():
            print(
                f"{decision}: {counts['trainSamples']} train / "
                f"{counts['validationSamples']} validation samples"
            )


if __name__ == "__main__":
    main()
