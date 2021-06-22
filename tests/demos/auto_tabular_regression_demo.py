from mlsolver.ml_solver import MLSolver

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from mlsolver.metrics.regression_metrics import RegressionMetrics

import pandas as pd

if __name__ == '__main__':
    boston = load_boston()

    y = boston['target']
    X = pd.DataFrame(boston['data'], columns=boston.feature_names)
    data = pd.concat([X, y], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLSolver(problem_category='tabular', problem_type='regression', metric_to_optimize='mae')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    mae = RegressionMetrics('mae', y_test, y_pred)

    print(f'Best Model: {model.best_estimator}')
    print()
    print(f'Best Pipeline: {model.best_pipeline}')
    print()
    print(f'Mean Absolute Error (MAE) ', mae)