# Data

Raw PathSnap datasets are maintained outside this repository and must not be committed.

Expected dataset partitions:
- train: 80 trips
- student_test: 10 trips
- evaluator_test: 10 trips

Configure dataset locations using:
- PATHSNAP_DATA_ROOT
- PATHSNAP_EVALUATOR_ROOT

The processed training and student-testing tables are generated using scripts/02_build_features.py. Local processed artifacts are excluded from Git.
