"""
Hyperparameter Search Functions with progress callback support.
"""

import itertools
import random
import time
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    make_scorer, accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, matthews_corrcoef, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, precision_score, r2_score,
    recall_score, confusion_matrix, roc_auc_score
)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true)
    safe_denominator = np.where(denominator < 1e-8, 1.0, denominator)
    errors = np.abs((y_true - y_pred) / safe_denominator)
    errors = np.where(np.isfinite(errors), errors, 0.0)
    return float(np.mean(errors))

from .registry import MODELS

CLASSIFICATION_METRICS = {
    "accuracy": {"label": "Accuracy", "higher_is_better": True, "scorer": "accuracy"},
    "precision": {"label": "Precision", "higher_is_better": True, "scorer": "precision_macro"},
    "recall": {"label": "Recall", "higher_is_better": True, "scorer": "recall_macro"},
    "f1": {"label": "F1 Score", "higher_is_better": True, "scorer": "f1_macro"},
    "roc_auc": {"label": "ROC-AUC", "higher_is_better": True, "scorer": "roc_auc_ovr"},
    "log_loss": {"label": "Log Loss", "higher_is_better": False, "scorer": "neg_log_loss"},
    "balanced_accuracy": {"label": "Balanced Accuracy", "higher_is_better": True, "scorer": "balanced_accuracy"},
    "specificity": {"label": "Specificity", "higher_is_better": True, "scorer": make_scorer(
        lambda y_true, y_pred: float(np.mean([
            (cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) /
            max(cm.sum() - cm[:, i].sum(), 1) for i in range(len(cm))
        ])) if (cm := confusion_matrix(y_true, y_pred)).size else 0.0)},
    "mcc": {"label": "Matthews Correlation Coefficient", "higher_is_better": True, "scorer": make_scorer(matthews_corrcoef)},
}

REGRESSION_METRICS = {
    "mae": {"label": "MAE", "higher_is_better": False, "scorer": "neg_mean_absolute_error"},
    "mse": {"label": "MSE", "higher_is_better": False, "scorer": "neg_mean_squared_error"},
    "rmse": {"label": "RMSE", "higher_is_better": False, "scorer": "neg_root_mean_squared_error"},
    "r2": {"label": "R²", "higher_is_better": True, "scorer": "r2"},
    "mape": {"label": "MAPE", "higher_is_better": False, "scorer": make_scorer(_safe_mape, greater_is_better=False)},
    "median_absolute_error": {"label": "Median Absolute Error", "higher_is_better": False, "scorer": "neg_median_absolute_error"},
}

METRIC_ALIASES = {
    "neg_mean_absolute_error": "mae",
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_mean_absolute_percentage_error": "mape",
    "neg_median_absolute_error": "median_absolute_error",
}

def metric_definitions(task: str) -> Dict[str, Dict[str, Any]]:
    if task == "classification":
        return CLASSIFICATION_METRICS
    if task == "regression":
        return REGRESSION_METRICS
    raise ValueError(f"Unknown task: {task}")

def normalize_metric(metric: str) -> str:
    return METRIC_ALIASES.get(metric, metric)

def validate_metrics(task: str, metrics: List[str]) -> List[str]:
    definitions = metric_definitions(task)
    normalized = list(dict.fromkeys(normalize_metric(metric) for metric in metrics))
    invalid = [metric for metric in normalized if metric not in definitions]
    if invalid:
        raise ValueError(f"Unsupported {task} metric(s): {', '.join(invalid)}")
    return normalized

def get_scorer(task: str, metric: str):
    canonical = normalize_metric(metric)
    definitions = metric_definitions(task)
    if canonical not in definitions:
        raise ValueError(f"Unsupported {task} metric: {metric}")
    return definitions[canonical]["scorer"]

def _metric_scores(task: str, metrics: List[str], cv_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    scores = {}
    definitions = metric_definitions(task)
    for metric in metrics:
        canonical = normalize_metric(metric)
        if canonical not in definitions:
            raise ValueError(f"Unsupported {task} metric: {metric}")

        scorer_name = definitions[canonical]["scorer"]
        if callable(scorer_name):
            test_key = f"test_{canonical}"
        else:
            test_key = f"test_{scorer_name}"
            if test_key not in cv_result:
                test_key = f"test_{canonical}"

        values = np.asarray(cv_result[test_key], dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        raw_mean = float(np.nanmean(values))
        raw_std = float(np.nanstd(values))

        if not definitions[canonical]["higher_is_better"]:
            raw_mean = -raw_mean
        scores[canonical] = {"mean": raw_mean, "std": raw_std}
    return scores

def generate_param_grid(param_space: Dict[str, Any], max_points_per_param: int = 4) -> Dict[str, List]:
    # (same as before)
    grid = {}
    for param, spec in param_space.items():
        if spec["type"] == "float":
            low, high = spec["low"], spec["high"]
            if spec.get("log_scale", False):
                values = np.logspace(np.log10(low), np.log10(high), max_points_per_param)
            else:
                values = np.linspace(low, high, max_points_per_param)
            values = [round(float(v), 4) for v in values]
            grid[param] = values
        elif spec["type"] == "int":
            low, high = spec["low"], spec["high"]
            if high - low + 1 <= max_points_per_param:
                values = list(range(low, high + 1))
            else:
                values = list(np.linspace(low, high, max_points_per_param, dtype=int))
            grid[param] = values
        elif spec["type"] == "categorical":
            grid[param] = spec["values"]
        else:
            raise ValueError(f"Unknown param type: {spec['type']}")
    return grid

def sample_random_params(param_space: Dict[str, Any]) -> Dict[str, Any]:
    # (same as before)
    params = {}
    for param, spec in param_space.items():
        if spec["type"] == "float":
            low, high = spec["low"], spec["high"]
            if spec.get("log_scale", False):
                value = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            else:
                value = np.random.uniform(low, high)
            params[param] = float(value)
        elif spec["type"] == "int":
            low, high = spec["low"], spec["high"]
            params[param] = int(np.random.randint(low, high + 1))
        elif spec["type"] == "categorical":
            params[param] = random.choice(spec["values"])
        else:
            raise ValueError(f"Unknown param type: {spec['type']}")
    return params

def get_cv(task: str, n_splits: int = 5):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

def grid_search(
    model_name: str,
    param_grid: Dict[str, List],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: Optional[List[str]] = None,
    primary_metric: str = "accuracy",
    cv: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    model_info = MODELS[model_name]
    model_class = model_info["class"]
    task = model_info["task"]
    metrics = validate_metrics(task, metrics or [primary_metric])
    primary_metric = normalize_metric(primary_metric)
    scoring = {metric: get_scorer(task, metric) for metric in metrics}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    results = []
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        start_time = time.time()
        model = model_class(**params)
        cv_obj = get_cv(task, cv)
        cv_result = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=-1, error_score=np.nan)
        metric_scores = _metric_scores(task, metrics, cv_result)
        mean_score = metric_scores[primary_metric]["mean"]
        std_score = metric_scores[primary_metric]["std"]
        training_time = time.time() - start_time

        results.append({
            "model_name": model_name,
            "params": params,
            "mean_score": mean_score,
            "std_score": std_score,
            "primary_metric": primary_metric,
            "metric_scores": metric_scores,
            "training_time": training_time
        })

        if progress_callback:
            progress_callback(idx + 1, total)

    return results

def random_search(
    model_name: str,
    param_space: Dict[str, Any],
    n_iter: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: Optional[List[str]] = None,
    primary_metric: str = "accuracy",
    cv: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    model_info = MODELS[model_name]
    model_class = model_info["class"]
    task = model_info["task"]
    metrics = validate_metrics(task, metrics or [primary_metric])
    primary_metric = normalize_metric(primary_metric)
    scoring = {metric: get_scorer(task, metric) for metric in metrics}

    results = []
    for i in range(n_iter):
        params = sample_random_params(param_space)
        start_time = time.time()
        model = model_class(**params)
        cv_obj = get_cv(task, cv)
        cv_result = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=-1, error_score=np.nan)
        metric_scores = _metric_scores(task, metrics, cv_result)
        mean_score = metric_scores[primary_metric]["mean"]
        std_score = metric_scores[primary_metric]["std"]
        training_time = time.time() - start_time

        results.append({
            "model_name": model_name,
            "params": params,
            "mean_score": mean_score,
            "std_score": std_score,
            "primary_metric": primary_metric,
            "metric_scores": metric_scores,
            "training_time": training_time
        })

        if progress_callback:
            progress_callback(i + 1, n_iter)

    return results

def run_search(
    task: str,
    model_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "random",
    metrics: Optional[List[str]] = None,
    primary_metric: str = "accuracy",
    n_iter: int = 10,
    cv: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    metrics = validate_metrics(task, metrics or [primary_metric])
    primary_metric = normalize_metric(primary_metric)
    all_results = []
    total_evals = 0
    # First pass to calculate total evaluations for progress
    if strategy == "grid":
        for model_name in model_names:
            if model_name in MODELS and MODELS[model_name]["task"] == task:
                grid = generate_param_grid(MODELS[model_name]["params"])
                total_evals += len(list(itertools.product(*grid.values())))
    else:
        total_evals = len(model_names) * n_iter

    completed = 0
    for model_name in model_names:
        if model_name not in MODELS or MODELS[model_name]["task"] != task:
            continue
        param_space = MODELS[model_name]["params"]
        if strategy == "grid":
            param_grid = generate_param_grid(param_space)
            results = grid_search(model_name, param_grid, X_train, y_train, metrics, primary_metric, cv,
                                  lambda c, t: progress_callback(completed + c, total_evals) if progress_callback else None)
            completed += len(results)
        else:
            results = random_search(model_name, param_space, n_iter, X_train, y_train, metrics, primary_metric, cv,
                                    lambda c, t: progress_callback(completed + c, total_evals) if progress_callback else None)
            completed += len(results)
        all_results.extend(results)

    sort_reverse = metric_definitions(task)[primary_metric]["higher_is_better"]
    all_results.sort(key=lambda x: x["mean_score"], reverse=sort_reverse)
    return all_results