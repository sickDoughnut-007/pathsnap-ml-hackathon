# System Architecture

## 1. Overview

PathSnap is an ML-based map-matching and route-reconstruction system. It processes noisy GPS observations, generates plausible road-network candidates, extracts features describing the relationship between observations and candidate road segments, and predicts the most likely road segment for each observation.

The system combines supervised machine-learning classifiers with sequence-level decoding using a Hidden Markov Model (HMM) and the Viterbi algorithm.

## 2. Main Components

### 2.1 Data and preprocessing

The data pipeline reads GPS trajectory data and road-network information. It prepares observations, handles trajectory segmentation, generates candidate road segments, and constructs model-ready features.

### 2.2 Feature engineering

The shared feature pipeline produces features describing the relationship between GPS observations and candidate road segments. The project uses 37 features for candidate-level prediction.

### 2.3 Candidate generation

For each GPS observation, the system retrieves nearby road-network candidates. Candidate generation uses a configured search radius and maximum candidate count.

### 2.4 Machine-learning models

The repository contains five supervised classifier families:

- Logistic Regression
- Random Forest
- Multi-Layer Perceptron
- Distance-Weighted K-Nearest Neighbors
- LightGBM

LightGBM is also evaluated after hyperparameter optimization using Optuna.

### 2.5 Sequence decoding

The classifier produces candidate-level scores. The HMM/Viterbi component uses observation likelihoods and transition constraints to decode a coherent sequence of road segments.

### 2.6 API

The FastAPI service loads the selected LightGBM model and HMM decoder configuration. It exposes endpoints for health checks, evaluator-trip discovery, and trip prediction.

### 2.7 Evaluator interface

The evaluator interface is intended to run separately from the model/API service. The interface sends a trip identifier to the backend and displays the returned route reconstruction.

The frontend directory is reserved for the separately maintained evaluator interface.

## 3. High-Level Processing Flow

1. Load GPS trajectory observations and road-network data.
2. Preprocess and segment the trajectories.
3. Generate nearby road-segment candidates.
4. Extract candidate-level features.
5. Apply a trained classifier to estimate candidate scores.
6. Decode the candidate sequence using HMM/Viterbi.
7. Return predictions and route information through the API.
8. Display the reconstructed route in the evaluator interface.

## 4. Deployment Arrangement

The intended demonstration uses two machines.

- Machine 1 hosts the trained model, road-network resources, and FastAPI backend.
- Machine 2 hosts the evaluator interface and sends evaluator-trip requests to Machine 1.

The backend must be reachable over the local network. The API should bind to 0.0.0.0 rather than only localhost when requests originate from another machine.

## 5. Repository Components

| Component | Location |
|---|---|
| API service | api/ |
| Shared pipeline modules | pathsnap/ |
| Model implementations | models/ |
| Training and processing scripts | scripts/ |
| Evaluation outputs | results/ |
| Dataset instructions | data/README.md |
| Project documentation | docs/ |
| Evaluator interface placeholder | frontend/ |