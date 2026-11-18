# PathSnap shared preprocessing

Start a model branch from `main` to use the same candidate roads, features and estimated labels as the logistic-regression experiment. The ready-to-load NumPy files in [processed/](processed/) cover 80 training trips and 10 student-test trips. You can train directly from these files.

## Load a trip

Install the dependencies with `python -m pip install -r requirements.txt`. Run this example from the repository root.

```python
import json
from pathlib import Path

import numpy as np
from map_matching import FEATURES

splits = json.loads(Path("splits.json").read_text())
trip = splits["fitting"][0]
with np.load(Path("processed/train") / f"{trip}.npz", allow_pickle=False) as data:
    keep = data["fit_mask"]
    X_train_trip = data["x"][keep]
    y_train_trip = data["y"][keep]
    print(trip, X_train_trip.shape, FEATURES)
```

For a comparable first run, combine the 64 trips in `fitting` and fit your scaler and model on their eligible rows. Validate on the 16 whole trips in `validation`. The 10 `student_test` trips are for development-stage comparison after refitting on all 80 training trips. [Exact trip IDs](splits.json)

## File contents

| Array | Meaning |
| --- | --- |
| `x` | Seven unscaled features per candidate, in column order: distance in metres, log(1 + distance), direction cosine, heading-available indicator, speed in m/s, log(1 + segment length in metres), and distance from the projected point to the nearer segment endpoint in metres. |
| `obs`, `arc` | GPS observation index and directed road-segment ID for each candidate row. Both are local to the trip. |
| `reference`, `error` | One estimated reference arc and its GPS-to-segment distance in metres per observation, including observations with no candidates. |
| `y` | Binary candidate target, true when `arc == reference[obs]`. |
| `fit_mask` | Candidate rows eligible for fitting. The estimated reference must be among the candidates and within 30 m of the GPS observation. |

Use only `x` as the candidate model input. The reference, alignment error, target and fitting mask depend on the supplied route. For top-1 evaluation, score **all** candidate rows, select the highest-scoring arc for each observation and divide correct matches by `len(reference)`. An observation with no candidates counts as a miss.

## Regenerate from the student archive

The preprocessing projects coordinates into local metres, searches up to 60 m for the eight nearest directed road segments, and computes the seven features. Separately, it aligns GPS points with the ordered reference route to estimate one label per observation. These labels can be ambiguous where roads overlap. Feature scaling happens later within each training fold.

```powershell
python prepare_data.py "C:\path\to\Student folder.zip" --output rebuilt
```

The export uses the same [geometry and alignment code](map_matching.py) as the baseline. [Logistic-regression experiment](https://github.com/sickDoughnut-007/pathsnap-ml-hackathon/tree/codex/local-map-matching-baseline)
