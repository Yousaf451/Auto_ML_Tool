import unittest

import numpy as np
import pandas as pd

from main import build_model_pipeline, dataframe_preview, preprocess_data
from ml.search import run_search


class RegressionPipelineTests(unittest.TestCase):
    def test_regression_loss_metrics_are_ranked_by_minimization(self):
        df = pd.DataFrame({
            'feature_a': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'feature_b': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        results = run_search(
            task='regression',
            model_names=['linear_regression'],
            X_train=df[['feature_a', 'feature_b']],
            y_train=df['target'],
            strategy='grid',
            metrics=['mae'],
            primary_metric='mae',
            cv=3,
        )

        self.assertGreater(len(results), 0)
        self.assertLess(results[0]['mean_score'], results[-1]['mean_score'])

    def test_regression_pipeline_applies_scaling_to_numeric_features(self):
        df = pd.DataFrame({
            'location': ['A', 'B', 'A', 'B', 'C', 'C'],
            'size': [10, 20, 30, 40, 50, 60],
            'price': [100, 200, 300, 400, 500, 600],
        })

        pipeline, _, _ = build_model_pipeline(df, 'price', {'model_name': 'linear_regression', 'params': {'fit_intercept': True}})
        numeric_transformer = pipeline.named_steps['preprocessor'].transformers_[0][1]
        self.assertTrue(any(step_name == 'scaler' for step_name, _ in numeric_transformer.steps))

    def test_dataframe_preview_sanitizes_nan_and_inf_values(self):
        df = pd.DataFrame({
            'x': [1.0, np.nan, np.inf, -np.inf],
            'label': ['a', 'b', 'c', 'd'],
        })

        preview = dataframe_preview(df, rows=4)
        values = [row['x'] for row in preview['preview']]

        self.assertEqual(values[0], 1.0)
        self.assertIsNone(values[1])
        self.assertIsNone(values[2])
        self.assertIsNone(values[3])

    def test_preprocessing_converts_infinite_numeric_values_to_missing(self):
        df = pd.DataFrame({
            'feature': [1.0, np.inf, -np.inf, 4.0],
            'target': [2.0, 4.0, 6.0, 8.0],
        })

        X, _, _ = preprocess_data(df, 'target')

        self.assertTrue(np.isfinite(X.to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
