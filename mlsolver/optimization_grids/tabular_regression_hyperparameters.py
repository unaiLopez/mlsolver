from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder

import numpy as np

class TabularRegressionHyperparameters:
    def __init__(self, models, feature_engineering):
        self.models = models
        self.feature_engineering = feature_engineering
        self.numerical_clean_strategies = ['mean', 'median']
        self.categorical_clean_strategies = ['most_frequent']
        self.scalers = [None, StandardScaler(), MinMaxScaler(), RobustScaler()]
        self.model_hyperparameters = {
            'RandomForest': self._load_random_forest_hyperparameters(),
            'KNN': self._load_knn_hyperparameters(),
            'LinearRegression': self._load_linear_regression_hyperparameters(),
            'AdaBoost': self._load_adaboost_hyperparameters(),
            'SGD': self._load_sgd_hyperparameters(),
            'SVM': self._load_svm_hyperparameters()
            #'XGBoost': self._load_xgboost_hyperparameters,
            #'LightGBM': self._load_lightgbm_hyperparameters
        }
    
    def __call__(self):
        optimization_grid = list()

        for model in self.models:
            model_hyperparameters = self.model_hyperparameters[model]
            optimization_grid.append(model_hyperparameters)
        
        return optimization_grid
    
    def _load_random_forest_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [RandomForestRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(100, 800, 10)
        hyperparameters_dict['estimator__criterion'] = ['mse','mae']
        hyperparameters_dict['estimator__min_samples_split'] = np.arange(2, 10, 4)
        hyperparameters_dict['estimator__min_samples_leaf'] = np.arange(1, 5, 4)

        return hyperparameters_dict

    def _load_knn_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [KNeighborsRegressor()]
        hyperparameters_dict['estimator__weights'] = ['uniform','distance']
        hyperparameters_dict['estimator__algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
        hyperparameters_dict['estimator__leaf_size'] = np.arange(10, 50, 10)
        hyperparameters_dict['estimator__metric'] = ['euclidean', 'manhattan', 'minkowski']
        hyperparameters_dict['estimator__n_neighbors'] = np.arange(1, 100, 8)

        return hyperparameters_dict

    def _load_svm_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [SVR()]
        hyperparameters_dict['estimator__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        hyperparameters_dict['estimator__gamma'] = ['scale', 'auto']
        hyperparameters_dict['estimator__C'] = np.arange(0.1, 1, 5)

        return hyperparameters_dict

    def _load_linear_regression_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [LinearRegression()]
        
        return hyperparameters_dict

    def _load_sgd_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [SGDRegressor()]
        hyperparameters_dict['estimator__loss'] = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        hyperparameters_dict['estimator__penalty'] = ['l2', 'l1', 'elasticnet']
        hyperparameters_dict['estimator__alpha'] = np.linspace(1e-5, 1e-2, 6)

        return hyperparameters_dict

    def _load_adaboost_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]

        hyperparameters_dict['estimator'] = [AdaBoostRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(50, 1000, 10)
        hyperparameters_dict['estimator__learning_rate'] = np.linspace(0.01, 0.9, 20)
        hyperparameters_dict['estimator__loss'] = ['linear', 'square']
        
        return hyperparameters_dict

    def _load_xgboost_hyperparameters(self):
        pass

    def _load_lightgbm_hyperparameters(self):
        pass
        