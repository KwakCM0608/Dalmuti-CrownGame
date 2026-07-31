from __future__ import annotations

import argparse
import json

from dataset import iter_action_histogram, load_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rollout files before GPU training.",
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--supervised-weight", type=float, default=5.0)
    args = parser.parse_args()
    loaded = load_rollouts(
        args.data,
        validation_fraction=args.validation_fraction,
        max_samples=args.max_samples,
        supervised_weight=args.supervised_weight,
    )
    histogram = sorted(
        iter_action_histogram(loaded.train),
        key=lambda entry: entry[1],
        reverse=True,
    )
    result = {
        "files": list(loaded.files),
        "totalSamplesSeen": loaded.total_samples,
        "forcedSamplesSkipped": loaded.forced_samples_skipped,
        "trainSamples": len(loaded.train),
        "validationSamples": len(loaded.validation),
        "observationShape": list(loaded.train.observations.shape),
        "legalMaskShape": list(loaded.train.legal_masks.shape),
        "weightedTrainSamples": float(loaded.train.weights.sum()),
        "uniqueTrainActions": len(histogram),
        "mostCommonTrainActions": [
            {"actionIndex": action, "count": count}
            for action, count in histogram[:10]
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
