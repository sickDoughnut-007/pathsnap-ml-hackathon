# PathSnap — ML-Based Map Matching and Route Reconstruction

**Machine Learning Hackathon | PES University**

PathSnap reconstructs realistic vehicle trajectories by matching noisy GPS observations to road-network segments.

The system combines GPS preprocessing, candidate generation, feature engineering, machine-learning-based candidate classification, and Hidden Markov Model (HMM) decoding using the Viterbi algorithm.

A FastAPI backend exposes route predictions and evaluation results to an evaluator frontend.

---

## 1. Project Objectives

- Reconstruct vehicle routes from noisy GPS observations.
- Identify the most likely road-network segment for each GPS observation.
- Compare multiple machine-learning approaches for candidate selection.
- Improve route consistency using HMM and Viterbi sequence decoding.
- Evaluate model performance using observation-level and route-level metrics.
- Provide an API and geographical interface for route visualisation.

## 2. System Architecture

GPS Dataset → Data Preprocessing → Candidate Generation → Feature Engineering → Candidate Classification → HMM/Viterbi Decoding → Reconstructed Route → Evaluation and Visualisation → FastAPI Backend → Frontend

## 3. Repository Structure

- `api/` — FastAPI inference backend.
- `frontend/` — Evaluator frontend integration.
- `config.yaml` — Shared pipeline configuration.
- `data/` — Dataset documentation and local data organisation.
- `docs/` — Project architecture, methodology and experiment documentation.
- `models/` — Model implementations, saved models, configurations and experiment outputs.
- `pathsnap/` — Shared preprocessing, candidate generation, feature engineering and evaluation modules.
- `results/` — EDA, preprocessing, model comparison and baseline results.
- `scripts/` — EDA, feature generation and dataset validation scripts.
- `requirements.txt` — Python dependencies.

## 4. Dataset Organisation

The project uses three dataset partitions:

| Partition | Purpose | Number of trips |
|---|---|---:|
| Training | Model training and cross-validation | 80 |
| Student testing | Model evaluation and comparison | 10 |
| Evaluator testing | Final evaluation and demonstration | 10 |

Raw datasets and locally generated processed tables are maintained outside Git where appropriate.

Configure local dataset locations using:

- `PATHSNAP_DATA_ROOT`
- `PATHSNAP_EVALUATOR_ROOT`

Refer to `data/README.md` for the expected local dataset arrangement.

Evaluator-testing data must not be used for model training or hyperparameter optimisation.

## 5. Shared Processing Pipeline

The `pathsnap/` package contains reusable components used across the experiments.

| Module | Responsibility |
|---|---|
| `candidates.py` | Candidate road-segment generation |
| `dataset.py` | Dataset loading and preparation |
| `evaluate.py` | Model and route evaluation |
| `features.py` | Candidate feature extraction |
| `geo.py` | Geographical calculations |
| `io.py` | Input/output utilities |
| `labels.py` | Candidate labelling |
| `network.py` | Road-network operations |
| `preprocess.py` | Data preprocessing |
| `modeling.py` | Shared model-related utilities |
| `knn.py` | KNN prediction functionality |
| `knn_anchors.py` | KNN anchor utilities |
| `knn_sequence.py` | Sequence-level KNN utilities |

The shared pipeline promotes consistent data preparation and feature representation across model experiments.

## 6. Machine Learning Models

The project evaluates five classifier families:

| Model | Implementation directory |
|---|---|
| Logistic Regression | `models/logistic_regression/` |
| Random Forest | `models/random_forest/` |
| Multi-Layer Perceptron | `models/mlp/` |
| Distance-Weighted KNN | `models/distance_weighted_knn/` |
| LightGBM | `models/lightgbm/` |

The respective model directories maintain their training implementations, notebooks where applicable, saved model artifacts, configurations and evaluation outputs.

### 6.1 Hyperparameter Optimisation

Bayesian hyperparameter optimisation is performed for the selected LightGBM model using Optuna.

Search configurations, trial results and the tuned model are maintained under:

`models/lightgbm/tuning/`

### 6.2 HMM and Viterbi Decoding

The HMM/Viterbi implementation uses classifier probabilities to reconstruct a geographically and topologically consistent sequence of road segments.

The implementation and experiment outputs are maintained under:

`models/hmm_viterbi/`

The sequence-decoding stage is evaluated against classifier-only predictions to assess its contribution to route reconstruction.

HMM/Viterbi is the sequence-decoding component, not an additional independent classifier.

## 7. Exploratory Data Analysis and Preprocessing

The main data exploration and preparation scripts are:

- `scripts/01_eda.py`
- `scripts/02_build_features.py`
- `scripts/03_validate_dataset.py`

Run them from the repository root:

    python scripts/01_eda.py
    python scripts/02_build_features.py
    python scripts/03_validate_dataset.py

These scripts support dataset inspection, feature construction and validation.

Generated reports and visualisations are maintained under `results/`.

## 8. Model Evaluation and Comparison

Model evaluation includes observation-level and route-level measurements where applicable.

The evaluation outputs include:

- Cross-validation performance.
- Top-1 candidate accuracy.
- Macro F1-score.
- Log Loss.
- Route continuity.
- Inference time.
- Confusion matrices and diagnostic plots.

Model comparison outputs are maintained under:

`results/model_comparison/`

Non-learned baseline evaluation outputs are maintained under:

`results/baselines/`

The comparison should distinguish classifier-only predictions from classifier predictions followed by HMM/Viterbi decoding.

Reported experimental values must be traceable to actual experiment outputs.

## 9. Geographical Visualisation

Folium is used to generate geographical visualisations of GPS trajectories and reconstructed routes.

The visualisation implementation is maintained alongside the relevant model or experiment code.

Maps are intended to support comparison of:

- Original GPS observations.
- Relevant road-network geometry.
- Reference routes, where available.
- Predicted routes.

Generated maps and their underlying generation code must correspond to actual project predictions and data.

## 10. API Backend

The FastAPI backend is implemented in:

`api/main.py`

Install dependencies:

    python -m pip install -r requirements.txt

Configure the dataset locations in PowerShell:

    $env:PATHSNAP_DATA_ROOT = "D:\Path\To\Student folder\Student folder"
    $env:PATHSNAP_EVALUATOR_ROOT = "D:\Path\To\evaluator_test"

Start the API:

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

API documentation:

`http://localhost:8000/docs`

The backend provides health checking, trip listing, prediction and evaluation functionality.

Evaluator ground truth must remain on the evaluation side and must not be used for model training or tuning.

## 11. Frontend

The `frontend/` directory is reserved for the evaluator interface.

The frontend implementation is maintained separately and will be integrated with the FastAPI backend.

The intended interface supports:

- Evaluator trip selection.
- Prediction requests.
- GPS trajectory and reconstructed-route visualisation.
- Display of evaluation metrics.
- Communication with the backend API.

## 12. Installation

Use Python 3.10 or later in a virtual environment.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Additional model-specific dependencies, if required, should be installed according to the relevant model directory instructions.

## 13. Reproducibility and Data Integrity

The following practices are followed:

- Training and evaluation use the designated dataset partitions.
- Evaluator-testing data is excluded from model training and hyperparameter optimisation.
- Model comparisons use consistent evaluation partitions and metrics wherever applicable.
- Saved models are accompanied by their training implementations and relevant configurations.
- Reported results must be reproducible from the submitted source code and experiment outputs.
- Secrets, credentials and private dataset files must not be committed.

## 14. Project Documentation

Additional project documentation, architecture descriptions, methodology and experiment notes are maintained in `docs/`.

Each model directory contains model-specific implementation and experiment materials.

---

**PathSnap — Machine Learning-Based Map Matching and Route Reconstruction**
