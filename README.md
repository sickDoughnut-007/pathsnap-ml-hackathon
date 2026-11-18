# PathSnap map matching with distance-weighted KNN

This branch contains the **distance-weighted KNN** experiment for the PES University machine learning hackathon. Given a noisy GPS track and a road map, KNN scores nearby directed road segments at each reading. A Viterbi decoder uses those scores and road connectivity to reconstruct the trip. **Observation top-1** measures how often the selected road and direction match the estimated reference arc.

**Best result on this branch:** KNN with the second route decoder selected the correct directed road at **18,601 of 19,133 eligible student-test readings (97.22%)**. It reached **96.44%** mean top-1 across the five training folds used for selection. The reserved evaluator trips have yet to be scored.

## Team comparison scorecard

### Candidate classifier

Both branches use the same 37 features, five whole-trip training folds and ten student-test trips. CV is the arithmetic mean of the five fold scores. Student-test results pool predictions across trips. Both models scored 169,408 eligible candidate rows. A positive candidate matches the estimated reference arc. Hard classification uses a probability threshold of 0.5.

| Metric | KNN CV | LightGBM CV | KNN student test | LightGBM student test |
| --- | ---: | ---: | ---: | ---: |
| Candidate accuracy | 96.67% | 96.98% | 97.05% | 97.68% |
| Macro precision | 0.9306 | 0.9383 | 0.9305 | 0.9479 |
| Macro recall | 0.9164 | 0.9238 | 0.9204 | 0.9347 |
| **Macro F1** | 0.9231 | 0.9307 | 0.9254 | 0.9412 |
| Match-class precision | 0.8826 | 0.8963 | 0.8793 | 0.9110 |
| Match-class recall | 0.8495 | 0.8626 | 0.8558 | 0.8804 |
| Match-class F1 | 0.8653 | 0.8787 | 0.8674 | 0.8954 |
| Log loss ↓ | 0.0987 | 0.0729 | 0.0870 | 0.0572 |
| **Pointwise observation top-1** | 87.49% | 88.29% | 87.57% | 90.04% |
| Candidate miss rate ↓ | 0.483% | 0.483% | 0.031% | 0.031% |

Candidate accuracy counts correct binary decisions. Precision measures how many predicted matches are correct, recall measures how many actual matches are found, and F1 balances the two. Macro scores average the two classes equally. Log loss evaluates probabilities and penalizes confident errors. Observation top-1 compares the single highest-scoring road with the reference at each eligible GPS reading.

LightGBM leads on pointwise student-test top-1 by 2.47 percentage points. Its published one-hop and two-hop continuity scores are 79.91% and 89.75% before route decoding. The team would need a decoded LightGBM run to compare full routes. The earlier logistic regression experiment used seven features and a single held-out set of 16 trips. It would need a run on the shared benchmark to enter this table. [Published LightGBM run](https://github.com/sickDoughnut-007/pathsnap-ml-hackathon/blob/3c212155e5bb32d5e8cc998c73ff1b2943d3e414/models/lightgbm/metrics.json) · [Earlier logistic regression experiment](https://github.com/sickDoughnut-007/pathsnap-ml-hackathon/blob/b58e69cefaf2a88fad358b8667a7e29144159221/README.md)

Six of the 19,133 eligible student-test readings lacked the reference candidate and count as top-1 misses. Another 27 readings had unreliable references and are excluded by the shared evaluator. Most candidate rows describe other roads, so accuracy alone can hide mistakes on the matching-road class. The KNN student-test confusion matrix contains **16,369 true positives, 2,246 false positives, 2,758 false negatives and 148,035 true negatives**. Its fold-to-fold sample standard deviations were 0.0181 for macro F1 and 3.39 percentage points for pointwise top-1. [Machine-readable metrics](models/distance_weighted_knn/metrics.json) · [Fold results](models/distance_weighted_knn/cv_folds.csv)

### Route quality

| Metric | Pointwise KNN | First decoder | Second decoder |
| --- | ---: | ---: | ---: |
| Mean training-fold observation top-1 | 87.49% | 95.32% | 96.44% |
| **Student-test observation top-1** | **87.57%** | **96.26%** | **97.22%** |
| Student-test correct / eligible readings | 16,755 / 19,133 | 18,417 / 19,133 | 18,601 / 19,133 |
| One-hop route continuity | 76.78% | 89.12% | 90.13% |
| Two-hop route continuity | 87.43% | 97.68% | 98.37% |

Continuity measures the fraction of **changes of predicted arc** within a GPS segment that the road graph connects in one or two hops. The decoder outputs a selected arc per reading. Candidate F1 and log loss come from the underlying KNN scores. Compare classifiers using pointwise top-1, candidate macro F1 and log loss. Compare complete pipelines using decoded top-1 and continuity on the same eligible readings. The folds used to select decoder settings can give optimistic results.

### Compute cost

| Measured stage | Distance-weighted KNN | LightGBM |
| --- | ---: | ---: |
| Final classifier fit on 80 trips | 3.20 s | 46.08 s |
| Scoring per 1,000 student-test candidate rows | 101.81 ms | 8.28 ms |

The KNN laptop has an Intel Core i5-1135G7 CPU. Its saved classifier occupies 42.30 MiB and used 1,479,905 rows for its final fit. The first decoder added 4.30 seconds for ten trips, excluding preprocessing, file loading and KNN scoring. The timings come from separate runs and hardware conditions may differ. [KNN training environment](models/distance_weighted_knn/README.md) · [First decoder timing](models/distance_weighted_knn/sequence/student_metrics.json) · [LightGBM timings](https://github.com/sickDoughnut-007/pathsnap-ml-hackathon/blob/3c212155e5bb32d5e8cc998c73ff1b2943d3e414/models/lightgbm/metrics.json)

The second decoder's recorded wall time was 56.04 seconds across the ten trips, including 45.21 seconds on trip `00000007`. That single trip dominates the measurement, so the ratio between the two recorded times is a poor estimate of typical slowdown. [Per-trip timings](models/distance_weighted_knn/sequence/iteration2/student_trips.csv)

## Why road-selection accuracy rose from 87.57% to 96.26%

On the same ten student-test trips, pointwise KNN selected the estimated reference road at **16,755 of 19,133 eligible GPS readings**. KNN with route decoding matched **18,417 of 19,133**. That is **1,662 additional matches**, a gain of **8.69 percentage points**. Both runs used the same fitted KNN, the same 37-feature candidate tables, and the same reference labels. The improvement came from how the system chose a sequence of roads from the KNN scores. [Original model and results](legacy/distance_weighted_knn/) · [Per-trip comparison](models/distance_weighted_knn/sequence/student_trips.csv)

The original model picked the highest-scoring road at each reading. On a two-way street, opposite directions can have exactly the same distance to the GPS point. When the vehicle slows down, GPS jitter can also make its estimated heading swing around. A pointwise choice can therefore reverse direction for one reading and switch back for the next. The decoder considers whether the vehicle could plausibly travel that sequence on the directed road network.

| Change | Why it helps | Evidence from the five training folds |
| --- | --- | --- |
| Choose a sequence with Viterbi | Combine KNN scores and GPS distance with the road distance between successive choices. Penalize implausible travel and immediate reversals. | Mean top-1 rose from **87.49%** for pointwise KNN to **92.92%** in trial 10. |
| Refine the timing of road changes | Once a route is predicted, use GPS distance to determine which reading crossed onto the next arc, while preserving route order. | Trials 11 and 15 differed only in this refinement. Accuracy rose from **93.19% to 94.01%**. |
| Include short connecting roads | A vehicle can cross a short arc between readings. The alignment step can consider that connector when it belongs to a reading’s candidate set. | Enabling connectors in trial 21 raised trial 15’s **94.01% to 94.25%**. |
| Reduce the influence of GPS jitter | At low movement speeds, give individual readings less influence over the route choice. | Trials 26 and 30 differed only in this setting. Accuracy rose from **94.78% to 95.27%**. |

We selected the settings with the highest mean top-1 across the five training folds. Each trip’s KNN scores came from a fit that excluded that entire trip. The best of 38 configurations reached **95.32%** on those folds. The matched trial pairs above isolate individual changes at specific settings. Their gains depend on those settings and should not be added together. [All 38 trials](models/distance_weighted_knn/sequence/trials.csv) · [Selected settings](models/distance_weighted_knn/sequence/settings.json)

With settings fixed, accuracy improved on **all ten student-test trips**. Trip `00000077` went from **2,827 to 3,596** correct out of **3,911**, or **72.28% to 91.95%**. Across the ten trips, the share of predicted road changes connected within one network hop rose from **76.78% to 89.12%**. The **96.26%** result is pooled across eligible student-test readings. The reserved evaluator trips have yet to be scored. [Decoder, results and reproduction commands](models/distance_weighted_knn/sequence/README.md)

## How the second decoder reached 97.22%

The first decoder can follow a plausible but incorrect road through a cluster of noisy readings. The second decoder first traces a route using readings at least 15 metres apart, plus each segment’s endpoints. It then scores every reading with the original decoder, giving extra weight to candidates on that coarse route and its short connecting roads. Every original candidate can still win. The fitted KNN and its candidate macro F1 stay the same.

We tested the first decoder and 18 variants on out-of-fold training predictions. The selected settings improved all five folds. On the student test, six trips improved and four tied. Trip `00000077` gained the most matches, from **3,596 to 3,644 of 3,911**. At **97.22%**, this iteration is **150 correct matches short** of the 18,751 needed for 98%. The same ten student-test trips have been reported across iterations, so the team should use the reserved evaluator for its final performance claim. [Second decoder, fold results and prediction command](models/distance_weighted_knn/sequence/iteration2/README.md)

## Reproduce a team comparison

1. Build the local Parquet files from the student archive and use the supplied feature list, eligibility flags, weights and trip folds. [Shared pipeline and dataset guide](SHARED_PIPELINE.md)
2. Fit preprocessing inside each training fold, then refit the selected classifier on all 80 training trips. Keep predictions, fit time and inference time with the measured scores. [Training template](models/_template_train.py) · [KNN source and exact environment](models/distance_weighted_knn/README.md)
3. Score all ten student-test trips with the shared evaluator. Record the classifier and sequence-decoder results separately, including trip-level counts so difficult trips remain visible. [Shared evaluator](pathsnap/evaluate.py) · [Original pointwise experiment](legacy/distance_weighted_knn/) · [First sequence decoder](models/distance_weighted_knn/sequence/)
