from tabular import auto_tabular, auto_time_series
from nlp import auto_nlp
from vision import auto_vision

class MLSolver:
    def __init__(self, problem_category, problem_type, metric_to_optimize, algorithms=['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetworks'], data_cleaning=True, feature_engineering=True, hyperparameter_tuning=True, ensembles=True):
        self.problem_category = problem_category
        self.problem_type = problem_type
    
    def fit(self, X, y):
        if self.problem_category == 'tabular':
            if self.problem_type == 'regression':
                tabular_regressor(X, y, self.problem_type, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles)
            elif self.problem_type == 'classification':
                tabular_classifier(X, y, self.problem_type, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles)
            else:
                raise Exception(f'{self.problem_type} does not exist or is not supported.')

        elif problem_category == 'time series':
            self.auto_time_series(X, y, self.problem_type, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles)
        elif problem_category == 'nlp':
            self.auto_nlp(X, y, self.problem_type, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles)
        elif problem_category == 'vision':
            self.auto_vision(X, y, self.problem_type, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles)
        else:
            raise Exception(f'{self.problem_category} does not exist or is not supported.')
    
    def get_best_pipeline():
        pass

    def get_best_model():
        pass