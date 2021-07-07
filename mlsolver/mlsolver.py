from .tabular import TabularRegressor, TabularClassifier
#from tabular.auto_time_series
#from nlp.auto_nlp
#from vision.auto_vision

#'XGBoost', 'LightGBM', 'CatBoost'

class MLSolver:
    def __init__(self, problem_category, problem_type, metric_to_optimize,
                 models=['SGD', 'SVM', 'AdaBoost', 'LinearRegression', 'KNN', 'RandomForest', 'LightGBM'],
                 feature_engineering=True, hyperparameter_tuning=True, ensembles=True, n_jobs=-1):
                 
        self.problem_category = problem_category
        self.problem_type = problem_type
        self.metric_to_optimize = metric_to_optimize
        self.models = models
        self.feature_engineering = feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensembles = ensembles
        self.n_jobs = n_jobs
        self.best_estimator = None
    
    def fit(self, X, y):
        if self.problem_category == 'tabular':
            if self.problem_type == 'regression':
                model = TabularRegressor(self.metric_to_optimize, self.models, self.feature_engineering,
                                         self.hyperparameter_tuning, self.ensembles, self.n_jobs)
            elif self.problem_type == 'classification':
                model = TabularClassifier(self.metric_to_optimize, self.models, self.feature_engineering,
                                          self.hyperparameter_tuning, self.ensembles, self.n_jobs)
            else:
                raise Exception(f'{self.problem_type} does not exist or is not supported.')
        
        model.fit(X, y)

        self.best_estimator = model.best_estimator


        #elif problem_category == 'time series':
        #    self.auto_time_series(self.metric_to_optimize, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles, self.n_jobs)
        #elif problem_category == 'nlp':
        #    self.auto_nlp(self.metric_to_optimize, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles, self.n_jobs)
        #elif problem_category == 'vision':
        #    self.auto_vision(self.metric_to_optimize, self.algorithms, self.data_cleaning, self.feature_engineering, self.hyperparameter_tuning, self.ensembles, self.n_jobs)
        #else:
        #    raise Exception(f'{self.problem_category} does not exist or is not supported.')

    def predict(self, X):

        return self.best_estimator.predict(X)
    
    def get_best_pipeline():
        pass

    def get_best_model():
        pass