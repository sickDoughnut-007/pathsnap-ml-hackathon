# Methodology

## 1. Dataset Organization

The project separates the available trajectories into three groups:

| Dataset partition | Number of trips | Purpose |
|---|---:|---|
| Training data | 80 | Model development and cross-validation |
| Student test data | 10 | Development-stage model comparison |
| Evaluator test data | 10 | Separate evaluator testing |

The evaluator test partition is kept separate from model selection and hyperparameter tuning.

Dataset locations are configurable. The repository does not require the raw datasets to be committed to Git.

The data root can be supplied through the PATHSNAP_DATA_ROOT environment variable. The evaluator dataset root can be supplied through PATHSNAP_EVALUATOR_ROOT.

## 2. Data Processing

The processing pipeline prepares GPS observations and road-network information for candidate-level map matching.

The main processing stages are:

1. Load trajectory observations.
2. Organize observations into trips.
3. Apply trajectory preprocessing and segmentation.
4. Generate candidate road segments for each observation.
5. Calculate features for each observation-candidate pair.
6. Prepare training and evaluation inputs.

The shared configuration is maintained in config.yaml.

## 3. Feature Engineering

The project uses 37 features to describe the relationship between GPS observations and candidate road segments.

The configured processing parameters include:

| Parameter | Configured value |
|---|---:|
| Trajectory gap split threshold | 30 seconds |
| Heading calculation window | 2 |
| Minimum heading displacement | 3 metres |
| Jitter window | 3 |
| Candidate search radius | 50 metres |
| Maximum candidates per observation | 16 |
| Road-network index step | 10 metres |

These values are configuration settings used by the project pipeline.

## 4. Model Families

Five supervised classifier families are implemented:

1. Logistic Regression
2. Random Forest
3. Multi-Layer Perceptron
4. Distance-Weighted K-Nearest Neighbors
5. LightGBM

The models estimate the likelihood that a candidate road segment corresponds to a GPS observation.

## 5. Cross-Validation and Model Selection

Model development uses the 80 training trips.

The evaluation workflow uses five-fold group-aware cross-validation. Grouping by trip helps prevent observations from the same trip from being distributed across training and validation folds.

The 10 student-test trips are used for development-stage comparison.

The separate 10-trip evaluator partition is not used for model selection or hyperparameter tuning.

## 6. Hyperparameter Optimization

Optuna is used to tune the LightGBM model.

The recorded optimization used the Tree-structured Parzen Estimator (TPE) sampler and completed eight trials.

The optimization objective was based on five-fold trip-grouped observation-level Top-1 performance.

The selected configuration is stored with the tuning outputs under models/lightgbm/tuning/.

## 7. Sequence Decoding

Candidate-level classifier scores do not independently guarantee a coherent route.

The HMM/Viterbi stage incorporates observation likelihoods and transition constraints to identify a likely sequence of road segments.

The project evaluates classifier-only predictions against predictions that include HMM/Viterbi decoding.

## 8. Evaluation Measures

The project records classification and route-related measures, including:

- Top-1 candidate accuracy
- Accuracy
- Macro F1
- Log loss
- Candidate miss rate
- Route continuity
- Sequence-level point accuracy
- Exact sequence matches

The reported measures should be interpreted according to their evaluation partition and aggregation level.