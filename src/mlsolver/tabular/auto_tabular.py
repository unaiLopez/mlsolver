class AutoTabular:
    def __init__(self, X, y, problem_type, algorithms, data_cleaning, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.algorithms = algorithms
        self.data_cleaning = data_cleaning
        self.feature_engineering = feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensembles = ensembles
        self.n_jobs = n_jobs

    def _create_data_cleaning_pipeline():
        #it returns a pipeline with all the data cleaning preprocessing steps
        pass

    def _create_feature_engineering_pipeline():
        pass

    def _create_hyperparameter_tuning_pipeline():
        pass

    def _create_ensembles():
        pass
    
    def fit(self, X, y):
        pass
