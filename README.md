# Mini AutoML Tool

<p align="center">
  <img alt="Mini AutoML Tool" src="https://img.shields.io/badge/Python-3.11%2B-blue" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115.5-009688" />
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.5.2-F7931E" />
</p>

A lightweight AutoML application for uploading tabular datasets, exploring their structure, selecting a target column, and running a focused model search for either classification or regression tasks.

This project combines a FastAPI backend with a React + Vite frontend to provide a simple workflow for:

- dataset upload and preview
- basic EDA and correlation checks
- task detection
- model selection
- random or grid hyperparameter search
- metric comparison
- best-model selection
- model artifact download and reuse

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Main Capabilities](#main-capabilities)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Frontend and Backend Structure](#frontend-and-backend-structure)
- [Installation and Setup](#installation-and-setup)
- [Environment Variables](#environment-variables)
- [Running the App](#running-the-app)
- [Complete User Workflow](#complete-user-workflow)
- [Dataset Upload and EDA](#dataset-upload-and-eda)
- [Classification and Regression Support](#classification-and-regression-support)
- [Model Selection and Hyperparameter Search](#model-selection-and-hyperparameter-search)
- [Multiple Evaluation Metrics](#multiple-evaluation-metrics)
- [Model Comparison and Best Model Selection](#model-comparison-and-best-model-selection)
- [Model Download and Reuse](#model-download-and-reuse)
- [Input Data Template](#input-data-template)
- [Prediction Workflow](#prediction-workflow)
- [API Overview](#api-overview)
- [Example Usage](#example-usage)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Future Improvements](#future-improvements)

---

## Project Overview

Mini AutoML Tool is a compact end-to-end machine learning interface designed for quick experimentation with tabular datasets. It is intentionally simple and transparent, with the following flow:

1. Upload a CSV or Excel file.
2. Inspect the first rows and column metadata.
3. Choose a target column.
4. Let the app suggest whether the task is classification or regression.
5. Select one or more models and metrics.
6. Run a random or grid search with cross-validation.
7. Compare results by primary metric.
8. Download the trained artifact and an input template.

The project is meant for learning, prototyping, and lightweight model exploration rather than full production AutoML.

---

## Key Features

- Upload CSV, XLSX, and XLS files
- Detect target column and suggest task type
- Support classification and regression workflows
- Run model search with random or grid strategies
- Evaluate multiple metrics per task
- View progress while jobs are running in the background
- Compare candidate models by score and training time
- Download the best model as a `.joblib` artifact
- Download a sample CSV template for new inputs
- Display dataset preview and column information
- Show correlation matrix for numeric columns

---

## Main Capabilities

### Dataset handling
- Read CSV and Excel files with `pandas`
- Allow files up to 10 MB
- Validate empty or invalid uploads
- Generate dataset preview and basic column metadata

### Task detection
- If the target appears categorical or low-cardinality, the app suggests `classification`
- Otherwise, it suggests `regression`

### AutoML search
- Model search runs asynchronously in the backend
- Uses cross-validation with sklearn
- Supports random and grid search
- Tracks evaluation progress and job state

### Model artifacts
- Saves the best trained pipeline to `backend/artifacts`
- Saves an input template CSV for reuse
- Exposes download endpoints for both artifacts

---

## Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Backend | Python, FastAPI | API server and training orchestration |
| ML | scikit-learn, pandas, NumPy | preprocessing, validation, model search |
| Model persistence | joblib | Save/load trained pipelines |
| Frontend | React, Vite | User interface and interaction flow |
| Visualization | Chart.js, react-chartjs-2 | Correlation display |
| HTTP client | Axios | API calls from frontend |
| Data parsing | pandas | CSV/Excel ingestion |
| Deployment helper | Docker Compose | Container orchestration support |

---

## Project Architecture

```text
autoML-tool/
├── backend/
│   ├── artifacts/                  # trained models and input templates
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── registry.py             # supported model registry and hyperparameter spaces
│   │   └── search.py               # scoring, CV, random/grid search logic
│   ├── tests/
│   │   └── test_regression_pipeline.py
│   ├── main.py                    # FastAPI app and orchestration logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # main UI logic and dataset workflow
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── nginx.conf
│   └── Dockerfile
├── data/                          # project data folder
├── docker-compose.yml
├── .venv/                         # local Python virtual environment
├── .conda/                       # local conda metadata if present
├── tmp_upload.csv                # sample file used during debugging
└── README.md
```

---

## Frontend and Backend Structure

### Backend responsibilities
The backend in [backend/main.py](backend/main.py) is responsible for:

- upload validation and parsing
- dataset storage in memory
- preview generation
- target detection
- task validation
- random and grid model search
- background job tracking
- saving trained artifacts
- serving download endpoints

### Frontend responsibilities
The frontend in [frontend/src/App.jsx](frontend/src/App.jsx) is responsible for:

- file selection and upload
- dataset preview render
- column display and target selection
- task selection
- model and metric configuration
- triggering AutoML jobs
- monitoring progress
- displaying model comparison results
- downloading the best model and template

---

## Installation and Setup

### Prerequisites

- Python 3.11 or newer
- Node.js 18+ and npm
- Optional: Docker + Docker Compose

### 1) Clone the repository

```bash
git clone <repository-url>
cd autoML-tool
```

### 2) Create and activate a Python environment

Using venv:

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate  # Windows PowerShell
```

Using conda:

```bash
conda create -n autuml-env python=3.11 -y
conda activate autuml-env
```

### 3) Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4) Install frontend dependencies

```bash
cd ../frontend
npm install
```

---

## Environment Variables

The current implementation does not require a custom `.env` file for the normal app flow.

The project uses these defaults:

| Variable | Default | Notes |
| --- | --- | --- |
| `API_BASE_URL` | `http://localhost:8000` | Set in the frontend app |
| `BACKEND_PORT` | `8000` | Used by the FastAPI app |
| `FRONTEND_PORT` | `5173` | Vite dev server default |
| `DOCKER_FRONTEND_PORT` | `80` | Used in the provided Docker Compose config |

There are no required environment variables in the current codebase beyond the local app ports.

---

## Running the App

### Run the backend

From the repository root:

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is then available at:

- http://localhost:8000
- root endpoint: http://localhost:8000/

### Run the frontend

From the repository root:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

- http://localhost:5173

### Run with Docker Compose

```bash
docker compose up --build
```

This starts:

- backend on port `8000`
- frontend on port `80`

---

## Complete User Workflow

### 1) Upload a dataset
The user selects a CSV, XLSX, or XLS file from the UI.

Supported file types:
- `.csv`
- `.xlsx`
- `.xls`

The backend validates the extension and file size before storing the dataset.

### 2) Review the dataset
After upload, the app displays:
- preview of the first rows
- column metadata
- missing counts
- unique counts
- dtype information
- numeric correlation matrix when possible

### 3) Select a target column
The user chooses the column to predict.

The backend returns a suggested task:
- `classification` for categorical or low-cardinality targets
- `regression` for numeric targets

### 4) Configure AutoML
The user selects:
- task type
- one or more models
- search strategy (`random` or `grid`)
- evaluation metrics
- primary metric

### 5) Run the search
The backend creates a background job and periodically updates progress. The frontend polls the job status endpoint.

### 6) Compare and rank results
Once complete, the app shows a comparison table with:
- model name
- primary score
- selected metric values
- training time

The top result is highlighted as the recommended model.

### 7) Download the model
The user can download:
- the trained `.joblib` pipeline artifact
- a CSV input template for feeding new data into the saved pipeline

---

## Dataset Upload and EDA

### Upload rules
The backend currently enforces:
- only CSV/Excel files
- maximum size of 10 MB
- non-empty dataset validation

### Preview generation
The upload route returns:

```json
{
  "dataset_id": "uuid",
  "preview": [
    {"col1": 1, "col2": "a"},
    {"col1": 2, "col2": "b"}
  ],
  "columns": [
    {
      "column": "col1",
      "dtype": "int64",
      "missing": 0,
      "unique": 10
    }
  ],
  "shape": [100, 5]
}
```

### EDA included
The current UI provides:
- first five rows preview
- column inspection table
- missing-value overview
- numeric correlation matrix

This is a lightweight exploratory step, not a full statistical notebook.

---

## Classification and Regression Support

The project supports both task types with separate model registries and metric definitions.

### Classification models
The registry currently includes:

- `logistic_regression`
- `random_forest_classifier`
- `svc`
- `decision_tree_classifier`
- `knn_classifier`
- `gradient_boosting_classifier`

### Regression models
The registry currently includes:

- `linear_regression`
- `random_forest_regressor`
- `svr`
- `decision_tree_regressor`
- `knn_regressor`
- `gradient_boosting_regressor`

The task is validated server-side and model-task mismatches are rejected.

---

## Model Selection and Hyperparameter Search

### Search strategies
The backend supports:

- `random` search
- `grid` search

### Search behavior
- Cross-validation is performed using scikit-learn
- For classification, `StratifiedKFold` is used
- For regression, `KFold` is used
- A progress callback updates the running job status

### Hyperparameter spaces
Each model has a parameter space defined in [backend/ml/registry.py](backend/ml/registry.py). Parameters are configured as:

- continuous numeric values
- integer ranges
- categorical values

The grid generator and random sampler create parameter combinations from those definitions.

---

## Multiple Evaluation Metrics

### Classification metrics
The current implementation supports:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `log_loss`
- `balanced_accuracy`
- `specificity`
- `mcc`

### Regression metrics
The current implementation supports:

- `mae`
- `mse`
- `rmse`
- `r2`
- `mape`
- `median_absolute_error`

Metrics are validated server-side before starting a run, and the primary metric is used to rank candidates.

---

## Model Comparison and Best Model Selection

After the search completes, the API returns a results payload similar to:

```json
{
  "results": [
    {
      "model_name": "random_forest_regressor",
      "params": {"n_estimators": 200, "max_depth": 12},
      "mean_score": 0.91,
      "std_score": 0.04,
      "primary_metric": "r2",
      "metric_scores": {
        "r2": {"mean": 0.91, "std": 0.04},
        "mae": {"mean": 1.23, "std": 0.10}
      },
      "training_time": 1.13
    }
  ]
}
```

The UI then:
- sorts results using the selected primary metric
- highlights the best model
- displays a side-by-side comparison table
- surfaces the selected metrics and score values

---

## Model Download and Reuse

The backend saves the fitted pipeline and metadata under `backend/artifacts`.

Saved outputs include:
- `best_model_<job_id>.joblib`
- `best_model_<job_id>_input_template.csv`

The frontend exposes download buttons for:
- the model artifact
- the input template CSV

These are served from:

- `/api/jobs/{job_id}/model/download`
- `/api/jobs/{job_id}/input-template`

The saved model includes the preprocessing pipeline and fitted estimator; it is intended for reuse in follow-up Python workflows.

---

## Input Data Template

The input template is generated from the dataset used during model training and saved as a CSV file.

This template is useful for:
- preparing a new inference batch
- verifying the expected feature columns
- reusing the model with the same schema

The template includes the feature columns only, without the target column.

---

## Prediction Workflow

The current project does not expose a dedicated live prediction API endpoint. The implemented flow is instead:

1. Train a model using the AutoML workflow.
2. Download the trained `.joblib` artifact.
3. Download the corresponding CSV input template.
4. Prepare a new dataset in the same schema.
5. Load the saved pipeline in Python and call `predict(...)` or `predict_proba(...)` as appropriate.

Example:

```python
import joblib

model = joblib.load("best_model_<job_id>.joblib")
new_data = ...
predictions = model.predict(new_data)
print(predictions)
```

If the model supports probability output, the pipeline may also expose `predict_proba` for classification tasks.

---

## API Overview

The current FastAPI contract is:

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Health check and app status |
| POST | `/api/upload` | Upload a CSV/XLS/XLSX dataset |
| GET | `/api/datasets/{dataset_id}/correlation` | Return numeric feature correlation matrix |
| POST | `/api/datasets/{dataset_id}/target` | Select a target column and infer task |
| POST | `/api/search` | Start an AutoML job |
| GET | `/api/jobs/{job_id}` | Get current job status and progress |
| GET | `/api/jobs/{job_id}/results` | Fetch completed model search results |
| GET | `/api/jobs/{job_id}/model/download` | Download the trained model artifact |
| GET | `/api/jobs/{job_id}/input-template` | Download the input template CSV |

---

## Example Usage

### Upload a dataset with curl

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@example.csv"
```

### Select a target column

```bash
curl -X POST "http://localhost:8000/api/datasets/<dataset_id>/target" \
  -H "Content-Type: application/json" \
  -d '{"target_column":"target"}'
```

### Start an AutoML search

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_id>",
    "target_column": "target",
    "task": "classification",
    "models": ["logistic_regression", "random_forest_classifier"],
    "search_strategy": "random",
    "evaluation_metrics": ["accuracy", "f1"],
    "primary_metric": "accuracy",
    "n_iter": 10,
    "cv": 5
  }'
```

### Check job status

```bash
curl "http://localhost:8000/api/jobs/<job_id>"
```

### Fetch results

```bash
curl "http://localhost:8000/api/jobs/<job_id>/results"
```

---

## Error Handling

The backend raises HTTP errors for invalid or unsupported input, including:

- unsupported file extensions
- file size too large
- empty datasets
- missing target columns
- incorrect model-task combinations
- invalid metric definitions

The frontend catches API errors and surfaces the server-provided message when available, instead of only showing a generic network error. For example, the app displays the detail string from the backend when available.

Examples of current error conditions:

- `413` — file too large
- `400` — invalid or malformed dataset
- `404` — dataset or job not found
- `422` — invalid metric or missing required metric input

---

## Testing

The project includes a focused regression validation test suite in [backend/tests/test_regression_pipeline.py](backend/tests/test_regression_pipeline.py).

Run the tests with:

```bash
cd backend
python -m unittest tests.test_regression_pipeline -v
```

This checks that:
- regression loss metrics are ranked correctly
- scaling is present in the regression pipeline
- data preview sanitization handles `NaN` and infinity values safely

---

## Future Improvements

The current app is intentionally lightweight, but these are good next steps for broader use:

- dedicated prediction API endpoint
- persisted datasets and job history
- richer preprocessing and feature engineering
- more models and ensembles
- model explainability views
- user authentication and saved projects
- better dashboarding and downloadable reports
- larger-scale dataset handling and async queueing

---

## Summary

Mini AutoML Tool is a practical, easy-to-understand AutoML starter that helps a user upload tabular data, understand it quickly, choose a target, run a model search, compare candidates, and download the best trained pipeline for reuse.

For a small project with a simple architecture, it provides a full and usable experimentation cycle without requiring a complex production stack.

---

## Contributors and Notes

This project was built as a compact local AutoML tool using Python and React. It is best suited for:

- learning AutoML concepts
- small-medium tabular experiments
- rapid model comparison and prototype workflows

If you are running this locally, the typical flow is:

```bash
# terminal 1
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# terminal 2
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open http://localhost:5173 to use the app.
