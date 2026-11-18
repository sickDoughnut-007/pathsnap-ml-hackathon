# Distance-weighted KNN visualizations

This folder contains charts generated only from recorded experimental results. It excludes artificial epoch curves because this branch does not contain KNN training-history epochs.

## Outputs

- `model_comparison_cv_student.png` - cross-validation and student-test model comparisons
- `macro_f1_and_log_loss.png` - macro F1 and log-loss comparisons
- `knn_confusion_matrix.png` - candidate-class confusion matrix for the student-test split
- `decoder_accuracy_comparison.png` - pointwise KNN and both decoder iterations across folds and trips
- `decoder_trial_search.png` - measured configuration-search performance
- `trip_00000052.html` - interactive candidate, reference, and KNN route map
- `model_comparison.csv` - cross-model metrics used by the summary charts

Regenerate the static plots and refresh the map copy:

```powershell
python models/distance_weighted_knn/visualizations/generate_visualizations.py
```
