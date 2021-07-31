from .auto_tabular import AutoTabular

class TabularClassifier(AutoTabular):
    def __init__(self, scoring, n_splits, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        super().__init__(scoring, n_splits, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs)
        self.best_estimator = None
        
    def _create_feature_engineering_pipeline(self):
        pass

    def _create_model_pipeline(self):
        pass
    
    def fit(self, X, y):
        pass