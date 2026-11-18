# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University. The project matches noisy GPS observations to road segments. A candidate classifier scores the nearby segments, and HMM/Viterbi decoding then picks a connected route. The trained model is served through a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

This README currently covers the **shared data pipeline**: EDA, preprocessing, candidate generation, labels, features and evaluation. Every model in the team is trained on its output.

## Repository layout

```
config.yaml                 shared settings (radius, K, gap threshold, CV seed) - do not change alone
pathsnap/                   shared Python package
  io.py                     read .track/.nodes/.arcs/.route, config
  geo.py                    local projection (metres), bearings, point-to-segment distance
  network.py                road graph: duplicate-arc canonicalisation, connectivity
  preprocess.py             time ordering, gap segmentation, speed, heading, jitter indicators
  candidates.py             KD-tree candidate generation (radius + top-K)
  labels.py                 per-observation reference labels by monotone route alignment (DP)
  features.py               37 location-independent features per (observation, candidate)
  dataset.py                load the processed tables  <- use this in model code
  evaluate.py               shared metrics             <- use this in model code
scripts/
  01_eda.py                 dataset exploration -> results/eda/
  02_build_features.py      builds data/processed/ (the shared dataset)
  03_validate_dataset.py    sanity checks + non-learned baselines -> results/preprocessing/
models/_template_train.py   copy this for each model
data/processed/             shared dataset (parquet) + folds.csv + feature_list.txt
results/                    EDA tables, plots, Folium maps, quality reports
```

## Setup

```
pip install -r requirements.txt
```

The raw data is expected at `../Student folder/Student folder/{train,student_test}`, next to the repo, which is how the provided zip extracts. If yours is elsewhere, pass `--data-root PATH` or set `PATHSNAP_DATA_ROOT`. The raw data is not committed.

## Rebuilding everything (about 1 minute)

```
python scripts/01_eda.py
python scripts/02_build_features.py
python scripts/03_validate_dataset.py
```

Generate `data/processed/` locally with the feature-building script above. The [distance-weighted KNN experiment](models/distance_weighted_knn/README.md) uses these tables and includes its measured results and saved model.

## Training a model (for every team member)

```
mkdir models/<model_name>
cp models/_template_train.py models/<model_name>/train.py
# edit make_model() only
python models/<model_name>/train.py
```

The template loads the shared tables, runs the fixed trip-level 5-fold CV, refits on all 80 training trips, scores the 10 student-testing trips, and writes `metrics.json`, `cv_folds.csv` and `model.joblib`.

Rules:
- Use `FEATURES` from `pathsnap.dataset`, the precomputed `fold` column, and the functions in `pathsnap/evaluate.py`. Never write your own split or metric.
- Train on `model_rows(...)`, i.e. labelled observations only (`label_ok == 1`).
- Pass `trip_weight` as the sample weight, so that the three 15k–19k-point trips don't dominate.
- Do not use `class_weight="balanced"` on the model that feeds the HMM. It distorts the probabilities, and the HMM uses them as emission probabilities.
- If a feature change is needed, change `pathsnap/` and rebuild. Then everyone retrains.

## The shared dataset

One row per (GPS observation, candidate road segment), labelled 1 if that segment is the observation's reference segment.

| file | rows | content |
|---|---|---|
| `train_candidates_fold{0..4}.parquet` | 1.51 M | 80 training trips, split by CV fold (about 25 MB each, under the GitHub limit) |
| `student_test_candidates.parquet` | 0.17 M | 10 student-testing trips |
| `*_points.parquet` | 208 k / 19 k | one row per observation: reference arc, alignment distance, label_ok, candidate count |
| `folds.csv` | 80 | trip to fold (seed 42, balanced by labelled observations) |

Columns: `trip_id, point_idx, arc_id, <37 features>, label, label_ok, segment, t, fold, trip_weight`. About 12% of rows are positives.

Feature groups (all in metres, seconds, degrees or counts, so they are independent of location):
- **geometry:** perpendicular distance, distance relative to the nearest candidate, rank, number of candidates, projection position, overshoot past the segment end, segment length
- **direction:** GPS heading compared with segment bearing (directed and undirected)
- **kinematics:** speed, turn, time gaps, path straightness, window speed
- **topology:** node degrees, two-way flag
- **temporal context:** distance of neighbouring observations to the same segment, whether the segment is also a candidate for neighbouring observations, link to the neighbour's nearest segment, signed progress along the segment

## Key preprocessing decisions (from EDA)

The trips come from different places around the world (Europe, Russia, the Americas, Asia, Africa), and each has its own road network. Arc ids are local to a trip. So the task is framed as binary classification of candidate segments using location-independent features, not as prediction of arc ids.

- **Distances:** a local equirectangular projection centred on each trip. The largest trip extent is about 75 km, giving a scale error of at most 0.8%, which is negligible for the GPS-to-road distances used here (under 50 m).
- **Gaps:** a time gap over 30 s splits a trajectory into segments. Context features never cross a segment boundary, and the HMM should restart there. Gaps reach 42 h (trip 90), 14.7 h (trip 78) and 36 min (trip 21).
- **Duplicate arcs:** 60 trips list some node pairs more than once. These are collapsed to the lowest arc id.
- **Candidates:** every segment within 50 m, keeping the nearest 16 (both travel directions count separately; about 95% of streets are two-way). Candidate miss rate: 0.48% on train, 0.03% on student_test.
- **Labels:** `.route` gives the ordered arcs but not which observation is on which arc. Each observation is assigned a route position by a monotone dynamic-programming alignment (positions never move backwards) that minimises total distance. Two refinements:
  - each point's cost is capped at 100 m, so points far from the route can't pull the alignment;
  - ties go to the earliest route position.

  This settles two-way streets, U-turns and repeated passes, which plain "nearest arc" labelling gets wrong.
- **Unreliable references:** observations more than 50 m from their aligned route arc get `label_ok = 0`. Some GPS tracks continue well past the end of their reference route: trip 54 (route 31 km, track 169 km, 11% labelled), trip 33 (41%), and trips 2, 43 and 66 (88–93%). This is a limitation of the reference data. These points are excluded from training and metrics but still get predictions.
- **GPS jitter:** when the vehicle is stopped, GPS noise fakes speeds of 2–8 m/s and random headings. About 15% of correct labels at 2–8 m/s point against the computed heading for this reason, not because the labels are wrong. The features `straightness` and `speed_window` let models recognise it.

## Reference numbers (`results/preprocessing/validation.json`)

| | train | student_test |
|---|---|---|
| Top-1, nearest-segment rule | 55.7% | 54.1% |
| Top-1, distance + heading rule | 83.7% | 83.9% |
| Top-1, logistic regression (template default) | 88.3% (CV) | 88.6% |
| Route continuity of the reference labels, 1 hop / 2 hops | 91.9% / 98.5% | 90.8% / 98.5% |

The nearest-segment rule is near 50% because a two-way street has two arcs at exactly the same distance. The reference labels themselves are only 91–92% continuous at 1 hop, because 1 Hz sampling skips short segments. Report continuity at both 1 and 2 hops.

## Data-leakage rules

The pipeline uses only the 80 training trips for fitting and the 10 student-testing trips for comparison. Features never read `.route`; it is used only to make labels. At prediction time, load trips with `load_trip(..., with_route=False)` or `load_trip_dir()`. Evaluator trips must never be used for anything except the final demo.
