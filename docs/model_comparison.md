# Model Comparison

## 1. Purpose

This document records the development-stage comparison of the supervised map-matching models.

The comparison uses the student-test metrics stored in results/model_comparison/student_test_metrics.csv.

## 2. Results

| Model | Top-1 score | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.883761 | 0.926637 |
| Random Forest | 0.888465 | 0.935139 |
| MLP | 0.884075 | 0.930020 |
| Distance-Weighted KNN | 0.875712 | 0.925398 |
| LightGBM | 0.900434 | 0.941172 |
| LightGBM (Optuna Tuned) | 0.903727 | 0.942521 |

## 3. Observations

- Tuned LightGBM achieved the highest Top-1 score in this comparison.
- Tuned LightGBM achieved the highest Macro F1 in this comparison.
- Random Forest was the next strongest listed model by both reported measures.
- Distance-Weighted KNN had the lowest Top-1 score among the listed models.
- All five classifier families are retained in the repository for comparison and reproducibility.

## 4. Selected Model

The tuned LightGBM model is the selected classifier for the current API implementation.

Its recorded student-test metrics are:

| Measure | Value |
|---|---:|
| Top-1 | 0.903727 |
| Accuracy | 0.977132 |
| Macro F1 | 0.942521 |
| Log loss | 0.056768 |
| Candidate miss rate | 0.000314 |

The API loads the tuned model from models/lightgbm/tuning/model.joblib.

## 5. Evaluation Boundary

These results are development-stage student-test results. The separate evaluator partition is reserved for evaluator testing and must not be used to select models or tune hyperparameters.

The evaluator API results are documented separately in experiments.md.