class tabular_classifier:
    def __init__(self, X, y, problem_type, algorithms, data_cleaning, feature_engineering, hyperparameter_tuning, ensembles):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.algorithms = algorithms
        self.data_cleaning = data_cleaning
        self.feature_engineering = feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensembles = ensembles
    
    def fit(self, X, y):
        pass