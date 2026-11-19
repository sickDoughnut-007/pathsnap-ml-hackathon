# Reproducibility and Execution

## 1. Requirements

The project uses Python and the dependencies listed in requirements.txt.

Use a virtual environment where possible.

From the repository root, install the dependencies:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

The commands above are for Windows PowerShell. On Linux or WSL, activate the environment using the appropriate shell activation script.

## 2. Dataset Configuration

Raw datasets are not committed to the repository.

Set the environment variable pointing to the local training and student-test dataset root:

    $env:PATHSNAP_DATA_ROOT = "D:\Path\To\Student folder\Student folder"

Set the evaluator dataset root separately:

    $env:PATHSNAP_EVALUATOR_ROOT = "D:\Path\To\evaluator_test"

Use the actual dataset locations on the machine running the project.

The shared configuration is stored in config.yaml. The repository uses data/README.md for dataset placement and setup guidance.

## 3. Processing Workflow

The main feature-building script is:

    python scripts/02_build_features.py

Run the processing and training scripts from the repository root so that project-relative paths resolve correctly.

The model-specific training and tuning implementations are stored under models/.

## 4. API Startup

Start the backend from the repository root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Binding to 0.0.0.0 allows another device on the same network to reach the backend, subject to firewall and network configuration.

For a local health check, open:

    http://127.0.0.1:8000/health

The API documentation is available at:

    http://127.0.0.1:8000/docs

## 5. Two-Machine Demonstration

The intended demonstration separates the backend and evaluator interface.

Machine 1:
- Holds the repository, model artifacts, road-network resources, and evaluator data.
- Starts the FastAPI backend.
- Allows inbound access to the selected API port on the local network.

Machine 2:
- Runs the evaluator interface.
- Sends evaluator-trip requests to the backend machine's LAN IP address.
- Displays the route information returned by the API.

The interface must use the backend machine's reachable LAN address rather than 127.0.0.1 when the machines are separate.

## 6. Model and Decoder Artifacts

The API uses the tuned LightGBM model:

    models/lightgbm/tuning/model.joblib

The HMM decoder configuration is stored at:

    models/hmm_viterbi/results/best_decoder_config.json

Keep the model artifact and configuration available at their expected repository paths when starting the API.

## 7. Generated Data and Results

Processed datasets and local raw data are not intended to be committed to Git.

Training outputs, evaluation metrics, plots, and visualizations are stored in the relevant model and results directories.

Use the existing result files for documented metrics rather than assuming that a new training run will reproduce identical values across different environments.