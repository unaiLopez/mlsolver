from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from src import MLSolver

import pandas as pd

if __name__ == '__main__':
    boston = load_boston()

    y = pd.Series(boston['target'])

    X = pd.DataFrame(boston['data'], columns=boston.feature_names)
    data = pd.concat([X, y], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLSolver(problem_category='tabular', problem_type='regression', scoring='mae', n_trials=10, feature_engineering=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    print(f'Best Model: {model.best_estimator}')
    print()
    print(f'Mean Absolute Error (MAE) ', mae)