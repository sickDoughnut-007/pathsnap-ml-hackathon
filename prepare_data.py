"""Export the evaluated PathSnap features and estimated labels, one file per trip."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from map_matching import MAX_LABEL_DISTANCE_M, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to Student folder.zip")
    parser.add_argument("--output", type=Path, default=Path("processed"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new --output directory to regenerate the data.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(__file__).with_name("splits.json").read_text(encoding="utf-8"))
    groups = {"train": splits["fitting"] + splits["validation"], "student_test": splits["student_test"]}
    with ZipFile(args.archive) as archive, TemporaryDirectory(dir=args.output.parent) as temp:
        prepared = Path(temp) / "processed"
        for split, trips in groups.items():
            folder = prepared / split
            folder.mkdir(parents=True)
            for trip in trips:
                data = prepare(archive, split, trip)
                obs, reference = data["obs"], data["reference"]
                positive = data["arc"] == reference[obs]
                covered = np.bincount(obs[positive], minlength=len(reference)) > 0
                fit_mask = (covered & (data["error"] <= MAX_LABEL_DISTANCE_M))[obs]
                np.savez_compressed(folder / f"{trip}.npz", **data, y=positive, fit_mask=fit_mask)
            print(f"Prepared {len(trips)} {split} trips", flush=True)
        prepared.rename(args.output)


if __name__ == "__main__":
    main()
