# PathSnap local baseline

This experiment ranks directed road segments for each GPS observation. It compares a nearest-road rule with a CPU-trained logistic regression model. The saved model performs pointwise scoring. Sequence decoding is a later experiment.

## Reproduce the run

From this branch, use Python 3.12 and PowerShell. Set `$dataset` to your copy of the student archive. The code reads trips directly from the ZIP.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$dataset = "$env:USERPROFILE\OneDrive\Desktop\ml\Student folder.zip"
.\.venv\Scripts\python.exe baseline.py $dataset
```

The run produces `outputs/metrics.json`, per-trip counts in `outputs/trip_metrics.csv`, a scaler and model in `outputs/logistic.joblib`, and validation predictions for the example map. The model bundle includes the ordered feature names. The committed files in `results/` record the measured baseline.

To regenerate the offline map, fetch the Leaflet display library once and run the renderer. Open `outputs/trip_00000001.html` in a browser. Its layers show road geometry, GPS points, the supplied route, and the segments each method selected. Both directions of a road share geometry, so visually overlapping lines can have different directed arc IDs.

```powershell
Invoke-WebRequest https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz -OutFile outputs/leaflet-1.9.4.tgz
.\.venv\Scripts\python.exe trip_map.py $dataset
```

## Data and evaluation

1. The validation run fits 64 complete training trips and holds out 16. Trip 00000001 is reserved for the map, and a seeded draw selects the other 15. A second fit uses all 80 training trips for the 10-trip student-test comparison. Exact partitions, the archive digest, configuration, and timings are in `results/metrics.json`.
2. GPS observations and nodes are projected to a local plane in metres. An STRtree finds directed segments within 60 m. Each observation retains up to the eight closest segments. Distance ties use arc ID. Candidate generation uses only the track and network.
3. Each supplied route is an ordered list of traversed arcs. A separate dynamic program estimates a reference arc per GPS observation by minimizing the total point-to-segment distance with nondecreasing positions along that list. It permits skipped and repeated positions. Parallel roads, overlapping geometry, gaps and disconnected route entries can produce uncertain labels. Scores measure agreement with this alignment.
4. Training uses observations whose estimated reference arc is in the candidate set and within 30 m. One candidate is positive and the rest are negative. The scaler is fitted only with the training rows. Features include distance, heading agreement, recent speed, segment length, and proximity to an endpoint. Heading uses only the current and preceding GPS points.
5. Top-1 accuracy compares the selected directed arc with the estimated reference arc across **all** observations. A missing candidate counts as incorrect. Candidate macro F1 uses a 0.5 probability threshold over generated candidate rows. Candidate log loss measures binary probability quality on those same rows.

| Split | GPS observations | Nearest top-1 | Logistic top-1 | Candidate miss rate | Candidate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 held-out training trips | 50,190 | 38.05% | 48.84% | 35.30% | 0.793 |
| 10 student-test trips | 19,160 | 54.41% | 75.83% | 0.41% | 0.806 |

Trip 00000054 accounts for 16,173 of the 16,215 validation observations with no candidate within 60 m. It remains in the benchmark. Weighting every validation trip equally gives logistic regression 70.81% mean top-1. The pooled score above weights every GPS point equally.

## Comparing team branches

Use the recorded trip lists, reference alignment, and scoring denominators to compare candidate scorers. Record changes to candidate generation separately because they change the fraction of observations a model can match. This first run uses a single grouped holdout. The five-model comparison, grouped cross-validation, Bayesian tuning, and HMM/Viterbi comparison are subsequent stages of the hackathon.
