# Experiments and Results

## 1. Experimental Setup

The project uses 80 training trips for model development, five-fold group-aware cross-validation, and 10 student-test trips for development-stage comparison.

A separate set of 10 evaluator trips is retained for evaluator testing. It is not used for model selection or hyperparameter tuning.

The trained models and evaluation outputs are stored under models/ and results/.

## 2. Supervised Model Comparison

The following values are taken from results/model_comparison/student_test_metrics.csv.

| Model | Top-1 score | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.883761 | 0.926637 |
| Random Forest | 0.888465 | 0.935139 |
| MLP | 0.884075 | 0.930020 |
| Distance-Weighted KNN | 0.875712 | 0.925398 |
| LightGBM | 0.900434 | 0.941172 |
| LightGBM (Optuna Tuned) | 0.903727 | 0.942521 |

The tuned LightGBM configuration achieved the highest Top-1 score and Macro F1 among the listed models on the student-test comparison.

## 3. LightGBM Hyperparameter Optimization

Optuna tuning used the TPE sampler and completed eight trials.

Recorded best cross-validation results:

| Measure | Value |
|---|---:|
| Mean CV Top-1 | 0.883255 |
| CV Top-1 standard deviation | 0.030883 |
| CV Macro F1 | 0.931166 |
| CV log loss | 0.073399 |

The selected tuned model's student-test results were:

| Measure | Value |
|---|---:|
| Top-1 | 0.903727 |
| Accuracy | 0.977132 |
| Macro F1 | 0.942521 |
| Log loss | 0.056768 |
| Candidate miss rate | 0.000314 |
| Route continuity, 1-hop | 0.800546 |
| Route continuity, 2-hop | 0.899818 |

The student-test evaluation contained 19,133 observations and 169,408 candidate rows.

## 4. HMM/Viterbi Experiment

The HMM/Viterbi experiment compared classifier-only predictions with predictions after sequence decoding.

| Measure | Classifier only | Classifier + HMM |
|---|---:|---:|
| Mean sequence point accuracy | 0.905948 | 0.914556 |
| Exact sequence matches | 0/13 | 0/13 |

The recorded results show an improvement in mean sequence point accuracy after HMM decoding. However, neither approach achieved an exact match for any of the 13 evaluated sequences.

This distinction is important: the experiment supports an improvement in point-level sequence accuracy, but does not establish perfect route reconstruction.

## 5. API Evaluator Test

The FastAPI service was tested against all 10 evaluator trips. All 10 prediction requests returned HTTP 200.

| Trip ID | Observations | Candidates | Inference time (s) | Top-1 |
|---|---:|---:|---:|---:|
| 00000015 | 6854 | 51382 | 34.246 | 0.794380 |
| 00000017 | 1346 | 17118 | 11.673 | 0.924963 |
| 00000023 | 458 | 2842 | 3.336 | 0.954149 |
| 00000030 | 654 | 8831 | 4.907 | 0.830275 |
| 00000049 | 2642 | 13494 | 18.276 | 0.992430 |
| 00000059 | 1031 | 12218 | 8.154 | 0.951503 |
| 00000067 | 916 | 6216 | 6.528 | 0.959607 |
| 00000073 | 1144 | 10382 | 7.451 | 0.980769 |
| 00000087 | 1399 | 11750 | 7.567 | 0.987849 |
| 00000094 | 3230 | 31549 | 11.547 | 0.934056 |

The weakest Top-1 result in this evaluator run was trip 00000015, at approximately 0.7944. This should remain visible in error analysis rather than being omitted.

The API health check also confirmed that the service was healthy and that the model, HMM configuration, and evaluator data were available.

## 6. Folium Visualization

The repository includes Folium map-generation code and generated HTML maps.

The visualizations support inspection of GPS observations, road-network candidates, reference routes, and reconstructed routes.

Relevant implementation and outputs are located under:

- models/hmm_viterbi/visualize_folium.py
- models/hmm_viterbi/results/
- results/eda/maps/
- models/distance_weighted_knn/

## 7. Interpretation

The student-test comparison identifies tuned LightGBM as the strongest listed classifier according to Top-1 and Macro F1.

The HMM experiment shows an improvement in mean point-level sequence accuracy, but exact sequence reconstruction remains challenging.

The evaluator API test demonstrates that all 10 evaluator trips were processed successfully by the backend. Prediction quality varies by trip, which motivates trip-level error analysis and continued route-quality evaluation.