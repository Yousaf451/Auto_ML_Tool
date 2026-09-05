"""
Model Registry for the Mini AutoML Tool
Defines supported models and their hyperparameter spaces.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Hyperparameter space format:
# Each parameter is a dict with:
#   "type": "float", "int", or "categorical"
#   For float/int: "low", "high", "log_scale" (optional)
#   For categorical: "values" (list)

MODELS = {
    # ------------------- Classification Models -------------------
    "logistic_regression": {
        "class": LogisticRegression,
        "task": "classification",
        "params": {
            "C": {"type": "float", "low": 0.01, "high": 10, "log_scale": True},
            "penalty": {"type": "categorical", "values": ["l1", "l2"]},
            "solver": {"type": "categorical", "values": ["liblinear", "saga"]}
        }
    },
    "random_forest_classifier": {
        "class": RandomForestClassifier,
        "task": "classification",
        "params": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
        }
    },
    "svc": {
        "class": SVC,
        "task": "classification",
        "params": {
            "probability": {"type": "categorical", "values": [True]},
            "C": {"type": "float", "low": 0.1, "high": 100, "log_scale": True},
            "kernel": {"type": "categorical", "values": ["linear", "rbf", "poly"]},
            "gamma": {"type": "float", "low": 1e-4, "high": 1, "log_scale": True}
        }
    },
    "decision_tree_classifier": {
        "class": DecisionTreeClassifier,
        "task": "classification",
        "params": {
            "max_depth": {"type": "int", "low": 2, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "criterion": {"type": "categorical", "values": ["gini", "entropy"]}
        }
    },
    "knn_classifier": {
        "class": KNeighborsClassifier,
        "task": "classification",
        "params": {
            "n_neighbors": {"type": "int", "low": 1, "high": 30},
            "weights": {"type": "categorical", "values": ["uniform", "distance"]},
            "metric": {"type": "categorical", "values": ["euclidean", "manhattan", "minkowski"]}
        }
    },
    "gradient_boosting_classifier": {
        "class": GradientBoostingClassifier,
        "task": "classification",
        "params": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0}
        }
    },

    # ------------------- Regression Models -------------------
    "linear_regression": {
        "class": LinearRegression,
        "task": "regression",
        "params": {
            "fit_intercept": {"type": "categorical", "values": [True, False]}
        }
    },
    "random_forest_regressor": {
        "class": RandomForestRegressor,
        "task": "regression",
        "params": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
        }
    },
    "svr": {
        "class": SVR,
        "task": "regression",
        "params": {
            "C": {"type": "float", "low": 0.1, "high": 100, "log_scale": True},
            "kernel": {"type": "categorical", "values": ["linear", "rbf", "poly"]},
            "gamma": {"type": "float", "low": 1e-4, "high": 1, "log_scale": True},
            "epsilon": {"type": "float", "low": 0.01, "high": 1, "log_scale": True}
        }
    },
    "decision_tree_regressor": {
        "class": DecisionTreeRegressor,
        "task": "regression",
        "params": {
            "max_depth": {"type": "int", "low": 2, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20}
        }
    },
    "knn_regressor": {
        "class": KNeighborsRegressor,
        "task": "regression",
        "params": {
            "n_neighbors": {"type": "int", "low": 1, "high": 30},
            "weights": {"type": "categorical", "values": ["uniform", "distance"]},
            "metric": {"type": "categorical", "values": ["euclidean", "manhattan", "minkowski"]}
        }
    },
    "gradient_boosting_regressor": {
        "class": GradientBoostingRegressor,
        "task": "regression",
        "params": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0}
        }
    }
}