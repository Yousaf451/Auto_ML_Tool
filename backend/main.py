import io
import uuid
import threading
import traceback
import itertools
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.search import run_search, generate_param_grid, validate_metrics, normalize_metric, metric_definitions
from ml.registry import MODELS

app = FastAPI(
    title="Mini AutoML API",
    description="Backend API for the mini AutoML tool",
    version="0.6.0"
)

# CORS
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DATASETS: Dict[str, pd.DataFrame] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Pydantic schemas
class TargetSelectionRequest(BaseModel):
    target_column: str = Field(..., description="Name of the target column")

class SearchRequest(BaseModel):
    dataset_id: str
    target_column: str
    task: str
    models: List[str]
    search_strategy: str
    evaluation_metrics: Optional[List[str]] = None
    evaluation_metric: Optional[str] = None
    metric: Optional[str] = None
    primary_metric: Optional[str] = None
    n_iter: Optional[int] = Field(default=10, gt=0, le=500)
    cv: Optional[int] = Field(default=5, ge=2, le=10)

# Helper functions
def read_dataset(file: UploadFile) -> pd.DataFrame:
    filename = file.filename
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file extension '{ext}'. Only CSV/Excel allowed.")
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File size exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit.")
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    if df.empty:
        raise HTTPException(400, "Uploaded file is empty or has no columns.")
    return df

def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json_value(v) for v in value.tolist()]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (float, int, np.floating, np.integer)) and not np.isfinite(float(value)):
        return None
    return value


def dataframe_preview(df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
    preview = df.head(rows).to_dict(orient="records")
    preview = _sanitize_json_value(preview)
    columns_info = []
    for col in df.columns:
        columns_info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique())
        })
    return {"preview": preview, "columns": columns_info}

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = df.copy()
    df = df.dropna(subset=[target_column])
    y = df[target_column]
    X = df.drop(columns=[target_column])

    warnings = []
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    cols_to_drop = []
    for col in categorical_cols:
        unique_count = X[col].nunique()
        if unique_count <= 100:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)
        else:
            cols_to_drop.append(col)
            warnings.append(f"Column '{col}' had {unique_count} unique values (>100) and was dropped.")
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    remaining_cat_cols = [c for c in categorical_cols if c not in cols_to_drop]
    if remaining_cat_cols:
        X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

    info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_categorical_cols": cols_to_drop,
        "encoded_features": X.columns.tolist(),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "warnings": warnings
    }
    return X, y, info

def build_model_pipeline(df: pd.DataFrame, target_column: str, best_result: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    training_df = df.dropna(subset=[target_column]).copy()
    X_raw = training_df.drop(columns=[target_column])
    numeric_features = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    dropped_features = [column for column in categorical_features if X_raw[column].nunique() > 100]
    categorical_features = [column for column in categorical_features if column not in dropped_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ], remainder="drop")
    model_info = MODELS[best_result["model_name"]]
    model = model_info["class"](**best_result["params"])
    fitted_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    fitted_pipeline.fit(X_raw, training_df[target_column])

    feature_dtypes = {column: str(X_raw[column].dtype) for column in X_raw.columns}
    metadata = {
        "model_name": best_result["model_name"],
        "task": model_info["task"],
        "target_column": target_column,
        "target_type": str(training_df[target_column].dtype),
        "target_values": [str(value) for value in training_df[target_column].dropna().unique()[:100]],
        "feature_columns": X_raw.columns.tolist(),
        "feature_dtypes": feature_dtypes,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "dropped_features": dropped_features,
        "training_rows": len(X_raw),
        "training_columns": len(X_raw.columns),
        "preprocessing": "Mean imputation for numeric columns; most-frequent imputation and one-hot encoding for categorical columns; unknown categories ignored.",
        "hyperparameters": best_result["params"],
        "supports_predict_proba": hasattr(model, "predict_proba"),
    }
    return fitted_pipeline, metadata, X_raw

def safe_artifact_path(job_id: str, suffix: str) -> Path:
    if not re.fullmatch(r"[0-9a-f-]{36}", job_id):
        raise HTTPException(404, "Job not found")
    return ARTIFACT_DIR / f"best_model_{job_id}{suffix}"

def update_job(job_id: str, **kwargs):
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)

def run_search_job(job_id: str, config: dict):
    try:
        update_job(job_id, status="running", progress=0.0)
        df = DATASETS.get(config["dataset_id"])
        if df is None:
            raise ValueError("Dataset not found")

        X, y, info = preprocess_data(df, config["target_column"])
        update_job(job_id, info=info)

        total_evals = 0
        model_list = config["models"]
        if config["search_strategy"] == "grid":
            for model_name in model_list:
                if model_name in MODELS and MODELS[model_name]["task"] == config["task"]:
                    param_grid = generate_param_grid(MODELS[model_name]["params"])
                    total_evals += len(list(itertools.product(*param_grid.values())))
        else:
            total_evals = len(model_list) * config["n_iter"]
        total_evals = max(total_evals, 1)
        completed_evals = 0

        def progress_callback(completed, total):
            nonlocal completed_evals
            completed_evals = completed
            progress = min(completed_evals / total_evals, 1.0)
            update_job(job_id, progress=progress, completed_evaluations=completed_evals,
                       total_evaluations=total_evals)

        results = run_search(
            task=config["task"],
            model_names=config["models"],
            X_train=X,
            y_train=y,
            strategy=config["search_strategy"],
            metrics=config["evaluation_metrics"],
            primary_metric=config["primary_metric"],
            n_iter=config["n_iter"],
            cv=config["cv"],
            progress_callback=progress_callback
        )

        best_result = results[0]
        fitted_pipeline, model_metadata, input_df = build_model_pipeline(df, config["target_column"], best_result)
        model_path = safe_artifact_path(job_id, ".joblib")
        template_path = safe_artifact_path(job_id, "_input_template.csv")
        joblib.dump(fitted_pipeline, model_path)
        input_df.head(1).to_csv(template_path, index=False)
        model_metadata.update({
            "job_id": job_id,
            "primary_metric": config["primary_metric"],
            "primary_score": best_result["mean_score"],
            "selected_metrics": config["evaluation_metrics"],
            "model_file": model_path.name,
            "input_template_file": template_path.name,
        })
        update_job(job_id, status="completed", progress=1.0, results=results,
                   completed_evaluations=completed_evals, total_evaluations=total_evals,
                   info=info, model_path=str(model_path), template_path=str(template_path),
                   model_metadata=model_metadata)
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        update_job(job_id, status="failed", error=error_msg)

# API endpoints
@app.get("/")
async def root():
    return {"message": "AutoML API is running"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    df = read_dataset(file)
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    info = dataframe_preview(df, rows=5)
    return {
        "dataset_id": dataset_id,
        "preview": info["preview"],
        "columns": info["columns"],
        "shape": df.shape
    }

@app.get("/api/datasets/{dataset_id}/correlation")
async def get_correlation(dataset_id: str):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {"columns": [], "matrix": []}
    corr = numeric_df.corr().round(4)
    corr_values = np.nan_to_num(corr.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return {"columns": list(corr.columns), "matrix": corr_values.tolist()}

@app.post("/api/datasets/{dataset_id}/target")
async def select_target(dataset_id: str, request: TargetSelectionRequest):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    target = request.target_column
    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' does not exist")

    missing_percent = (df[target].isna().sum() / len(df)) * 100
    warning = None
    if missing_percent > 50:
        warning = f"Target column has {missing_percent:.1f}% missing values."

    unique_count = df[target].nunique()
    dtype = df[target].dtype
    if dtype == "object" or dtype.name == "category" or unique_count <= 10:
        suggested_task = "classification"
    else:
        suggested_task = "regression"

    return {
        "target_column": target,
        "suggested_task": suggested_task,
        "unique_values": int(unique_count),
        "dtype": str(dtype),
        "warning": warning
    }

@app.post("/api/search")
async def create_search_job(request: SearchRequest, background_tasks: BackgroundTasks):
    if request.dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")
    df = DATASETS[request.dataset_id]
    if request.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{request.target_column}' not found")

    if request.task not in ["classification", "regression"]:
        raise HTTPException(400, "Invalid task")

    requested_metrics = request.evaluation_metrics or []
    requested_metrics.extend(
        metric for metric in [request.evaluation_metric, request.metric]
        if metric
    )
    if not requested_metrics:
        raise HTTPException(422, "At least one evaluation metric is required")
    try:
        evaluation_metrics = validate_metrics(request.task, requested_metrics)
        primary_metric = normalize_metric(request.primary_metric or evaluation_metrics[0])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if primary_metric not in evaluation_metrics:
        raise HTTPException(422, "Primary metric must be one of the selected evaluation metrics")

    target_dtype = df[request.target_column].dtype
    unique_count = df[request.target_column].nunique()
    if request.task == "classification":
        if unique_count < 2:
            raise HTTPException(400, "Classification requires at least 2 classes.")
    elif request.task == "regression":
        if target_dtype == "object" or target_dtype.name == "category":
            raise HTTPException(400, "Regression target must be numeric.")

    valid_models = []
    for model in request.models:
        if model not in MODELS:
            raise HTTPException(400, f"Unknown model: {model}")
        if MODELS[model]["task"] != request.task:
            raise HTTPException(400, f"Model '{model}' is for {MODELS[model]['task']} but task is {request.task}")
        valid_models.append(model)
    if not valid_models:
        raise HTTPException(400, "No valid models selected")

    if request.search_strategy == "grid":
        total_combinations = 0
        for model_name in valid_models:
            param_grid = generate_param_grid(MODELS[model_name]["params"])
            total_combinations += len(list(itertools.product(*param_grid.values())))
        if total_combinations > 1000:
            raise HTTPException(400, f"Grid search would evaluate {total_combinations} combinations (>1000). Use random search.")

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "completed_evaluations": 0,
        "total_evaluations": 0,
        "results": None,
        "error": None,
        "config": {**request.dict(), "evaluation_metrics": evaluation_metrics, "primary_metric": primary_metric}
    }
    with JOB_LOCK:
        JOBS[job_id] = job_data

    config = {**request.dict(), "evaluation_metrics": evaluation_metrics, "primary_metric": primary_metric}
    background_tasks.add_task(run_search_job, job_id, config)
    return {"job_id": job_id, "status": "pending", "evaluation_metrics": evaluation_metrics, "primary_metric": primary_metric}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "completed_evaluations": job["completed_evaluations"],
        "total_evaluations": job["total_evaluations"],
        "error": job["error"] if job["status"] == "failed" else None,
        "results_ready": job["status"] == "completed"
    }

@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job is not completed. Status: {job['status']}")
    return {
        "job_id": job_id,
        "results": job["results"],
        "info": job.get("info"),
        "evaluation_metrics": job["config"].get("evaluation_metrics", []),
        "primary_metric": job["config"].get("primary_metric"),
        "model_metadata": job.get("model_metadata")
    }

@app.get("/api/jobs/{job_id}/model/download")
async def download_best_model(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.get("status") != "completed":
        raise HTTPException(404, "Completed job not found")
    model_path = Path(job.get("model_path", ""))
    if not model_path.is_file() or model_path.parent != ARTIFACT_DIR:
        raise HTTPException(404, "Model artifact not found")
    return FileResponse(model_path, media_type="application/octet-stream", filename=model_path.name)

@app.get("/api/jobs/{job_id}/input-template")
async def download_input_template(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.get("status") != "completed":
        raise HTTPException(404, "Completed job not found")
    template_path = Path(job.get("template_path", ""))
    if not template_path.is_file() or template_path.parent != ARTIFACT_DIR:
        raise HTTPException(404, "Input template not found")
    return FileResponse(template_path, media_type="text/csv", filename=template_path.name)