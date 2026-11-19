# Distance-spaced guidance for the KNN route decoder

Giving the original decoder a route estimated from more widely spaced GPS readings improved student-test observation top-1 from **96.26% to 97.22%**. The new decoder made **184 additional correct road assignments**. It uses the existing fitted KNN, so the branch’s candidate classification scores apply to both decoders. [Team scorecard](../../../../README.md#team-comparison-scorecard)

## How the route is selected

1. Split each trip at the same time gaps and missing-candidate readings as the first decoder. Within each segment, keep the endpoints and select a new intermediate reading when its displacement from the last selected reading reaches the spacing threshold.
2. Run Viterbi on those selected readings using their GPS displacements. For each pair of successive selected roads, identify their directed connector arcs where the graph contains a path within the existing 500 m search limit.
3. Return to the full GPS sequence. At each reading, add a fixed bonus to candidates on the two surrounding selected roads or their connectors. Decode again and refine the timing of road changes. A candidate outside the coarse route can still win. The inputs at prediction time are the GPS track, the road graph and KNN scores. [Guidance implementation](../../../../pathsnap/knn_anchors.py) · [Full-rate decoder](../../../../pathsnap/knn_sequence.py)

## Selection on training trips

We compared the original decoder with 18 combinations of anchor spacing (15, 35 or 70 m), bonus (1, 4 or 12) and reversal penalty (35 or 90). Each training trip received KNN scores from a fit that excluded its entire fold. We selected by the arithmetic mean of five fold accuracies, counting each eligible GPS reading once within a fold. Trials 3 and 4 tied for first. The first maximum selected **15 m, bonus 4, reversal penalty 35**. These fold scores include the effect of selecting settings on the same folds. [All trials](trials.csv) · [Fold counts](folds.csv) · [Selected settings](settings.json)

| Training fold | First decoder | Second decoder |
| --- | ---: | ---: |
| 0 | 97.82% | 98.01% |
| 1 | 91.17% | 93.32% |
| 2 | 95.44% | 96.23% |
| 3 | 96.68% | 98.07% |
| 4 | 95.50% | 96.56% |
| **Mean** | **95.32%** | **96.44%** |

## Student-test result

With the configuration fixed, **18,601 of 19,133 eligible readings** matched the estimated reference arc. One-hop continuity reached **90.13%** and two-hop continuity reached **98.37%**. Six trips gained correct matches and four tied the first decoder. Trip `00000077` contributed the largest gain (+48) and still had the lowest top-1 (3,644 of 3,911, or 93.17%). The 98% top-1 target requires 18,751 correct matches on this split. [Metrics](metrics.json) · [Counts and timing by trip](student_trips.csv) · [First decoder on the same trips](../student_trips.csv)

## Reproduce locally

As a separate check, prediction from the raw GPS tracks and road graphs reproduced the recorded eligible and correct counts for each of the ten student-test trips.

Use the environment and fitted model from the [KNN training guide](../../README.md) and the [local preprocessing instructions](../../../../SHARED_PIPELINE.md). From the repository root, run the following commands. The first command can be skipped if the out-of-fold scores for the same preprocessing and KNN fit are already in `outputs/knn_sequence/oof.parquet`.

```powershell
python models/distance_weighted_knn/sequence/oof.py
python models/distance_weighted_knn/sequence/anchors.py select --data-root "data/raw/Student folder"
python models/distance_weighted_knn/sequence/anchors.py test --data-root "data/raw/Student folder"
```

To predict a raw trip, pass `--iteration 2`. Omitting the flag runs the first decoder. The CSV contains one predicted arc ID per reading, with −1 for a reading without candidates. The predictor does not load the reference route.

```powershell
python models/distance_weighted_knn/sequence/predict.py "data/raw/Student folder/student_test/00000052" "route_52.csv" --iteration 2
```
