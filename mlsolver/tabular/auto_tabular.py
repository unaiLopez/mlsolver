from abc import ABC, abstractmethod

class AutoTabular(ABC):
    @abstractmethod
    def __init__(self, scoring, n_splits, n_trials, timeout, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.timeout = timeout
        self.models = models
        self.feature_engineering = feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensembles = ensembles
        self.n_jobs = n_jobs
        self.best_estimator = None

    @abstractmethod
    def _create_feature_engineering_pipeline(self):
        pass

    @abstractmethod
    def _create_model_pipeline(self):
        pass
    
    @abstractmethod
    def _create_ensembles(self):
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass