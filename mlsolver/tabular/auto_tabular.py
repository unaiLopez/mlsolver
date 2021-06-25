class AutoTabular:
    def __init__(self, metric_to_optimize, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        self.metric_to_optimize = metric_to_optimize
        self.models = models
        self.feature_engineering = feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensembles = ensembles
        self.n_jobs = n_jobs
        self.pipeline = None
        self.best_estimator = None

    def _create_feature_engineering_pipeline(self):
        pass

    def _create_model_pipeline(self):
        pass

    def _create_ensembles(self):
        pass
    
    def fit(self, X, y):
        pass