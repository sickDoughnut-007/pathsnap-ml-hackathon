# PathSnap local baseline

This experiment uses **scikit-learn logistic regression**, one of the examples in the hackathon guidelines' **linear and probabilistic** family. The classifier learns whether a candidate directed road segment matches a GPS observation. At prediction time, it scores each nearby segment and selects the highest-scoring one. A nearest-road rule provides the comparison. Both methods select a segment independently for each observation. Sequence decoding is a later experiment.

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

## Preprocessing before training

| Step | What we did |
| --- | --- |
| 1. Validate each trip | Read the GPS track, road nodes, directed segments and reference route. Checked column counts, finite coordinates and timestamps, increasing timestamps, and valid node and segment IDs. |
| 2. Convert coordinates to metres | Projected GPS points and road nodes onto a local plane centred on the trip so distances to segments could be measured in metres. |
| 3. Find candidate roads | Used a spatial index to find segments within 60 m of each GPS point. Kept up to the eight closest, breaking distance ties by arc ID. Opposite directions on the same road can be separate candidates. |
| 4. Estimate a training label | Trip 00000001 has 1,315 GPS points and 525 reference-route entries. To associate them, we minimized total point-to-segment distance while preserving route order and allowing skipped or repeated positions. These per-point labels are estimates and can be ambiguous where roads overlap. |
| 5. Build and scale features | Computed seven values per candidate: distance, log-transformed distance, heading agreement, a heading-available indicator, recent speed, log-transformed segment length, and distance to the nearer endpoint. Motion uses the current and preceding GPS points. The model pipeline fits its scaler only on eligible training rows. |
| 6. Filter training rows and split by trip | Trained on observations whose estimated reference arc appeared among the candidates and lay within 30 m. Held out 16 whole trips and fitted on 64. Then refitted on all 80 training trips for the 10-trip student-test comparison. Every evaluation observation counts, including candidate misses. |

The shared [processed trip files and loading example](https://github.com/sickDoughnut-007/pathsnap-ml-hackathon/tree/main#load-a-trip) contain the features before scaling so each model can fit its own preprocessing on its training fold. The exact partitions, archive digest, configuration, and timings for this experiment are in [the run record](results/metrics.json).

## Evaluation

Top-1 accuracy compares the selected directed arc with the estimated reference arc across **all** observations. A missing candidate counts as incorrect. Candidate macro F1 uses a 0.5 probability threshold over generated candidate rows. Candidate log loss measures binary probability quality on those same rows.

| Split | GPS observations | Nearest top-1 | Logistic top-1 | Candidate miss rate | Candidate macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 held-out training trips | 50,190 | 38.05% | 48.84% | 35.30% | 0.793 |
| 10 student-test trips | 19,160 | 54.41% | 75.83% | 0.41% | 0.806 |

Trip 00000054 accounts for 16,173 of the 16,215 validation observations with no candidate within 60 m. It remains in the benchmark. Weighting every validation trip equally gives logistic regression 70.81% mean top-1. The pooled score above weights every GPS point equally.

## Comparing team branches

Use the recorded trip lists, reference alignment, and scoring denominators to compare candidate scorers. Record changes to candidate generation separately because they change the fraction of observations a model can match. This first run uses a single grouped holdout. The five-model comparison, grouped cross-validation, Bayesian tuning, and HMM/Viterbi comparison are subsequent stages of the hackathon.
