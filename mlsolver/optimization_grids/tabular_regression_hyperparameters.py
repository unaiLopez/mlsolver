from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from optuna import trial

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
            'SVM': self._load_svm_hyperparameters(),
            #'XGBoost': self._load_xgboost_hyperparameters()
            'LightGBM': self._load_lightgbm_hyperparameters()
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
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [RandomForestRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(150, 1200, 10)
        hyperparameters_dict['estimator__criterion'] = ['mse','mae']
        hyperparameters_dict['estimator__min_samples_split'] = np.arange(2, 100, 10)
        hyperparameters_dict['estimator__min_samples_leaf'] = np.arange(1, 10, 6)
        hyperparameters_dict['estimator__max_depth'] = np.arange(1, 60, 10)
        hyperparameters_dict['estimator__max_features'] = [None, 'log2', 'sqrt']

        return hyperparameters_dict

    def _load_knn_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
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
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
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
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [LinearRegression()]
        hyperparameters_dict['estimator__fit_intercept'] = [True, False]
        hyperparameters_dict['estimator__normalize'] = [True, False]
        
        return hyperparameters_dict

    def _load_sgd_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = self.scalers
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
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]

        hyperparameters_dict['estimator'] = [AdaBoostRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(50, 1000, 10)
        hyperparameters_dict['estimator__learning_rate'] = np.linspace(0.01, 0.9, 20)
        hyperparameters_dict['estimator__loss'] = ['linear', 'square']
        
        return hyperparameters_dict

    def _load_xgboost_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [XGBRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(150, 1200, 10)
        hyperparameters_dict['estimator__eta'] = np.linspace(0.01, 1.0, 8)
        hyperparameters_dict['estimator__gamma'] = np.linspace(0.05, 1.0, 8)
        hyperparameters_dict['estimator__max_depth'] = np.arange(3, 30, 10)
        hyperparameters_dict['estimator__min_child_weight'] = np.arange(1, 10, 5)
        hyperparameters_dict['estimator__subsample'] = np.linspace(0.1, 1.0, 10)
        hyperparameters_dict['estimator__colsample_bytree'] = np.linspace(0.1, 1.0, 10)
        hyperparameters_dict['estimator__reg_lambda'] = np.linspace(0.01, 1.0, 10)
        hyperparameters_dict['estimator__reg_alpha'] = np.linspace(0, 1.0, 6)

        return hyperparameters_dict


    def _load_lightgbm_hyperparameters(self):
        hyperparameters_dict = {}
        if self.feature_engineering:
            hyperparameters_dict['preprocessor__numerical__outlier_capper__quantile_range'] = [(0, 100), (1, 99), (5, 95)]
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__interpolation'] = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
            #hyperparameters_dict['preprocessor__numerical__outlier_capper__discard_infs'] = [True, False]
            hyperparameters_dict['preprocessor__numerical__cleaner__strategy'] = self.numerical_clean_strategies
            hyperparameters_dict['preprocessor__numerical__scaler'] = [None]
            hyperparameters_dict['preprocessor__categorical__cleaner__strategy'] = self.categorical_clean_strategies
            hyperparameters_dict['preprocessor__categorical__encoder'] = [OneHotEncoder(sparse=False)]
            hyperparameters_dict['feature_selector__k'] = ['all', 3, 6, 12]
        
        hyperparameters_dict['estimator'] = [LGBMRegressor()]
        hyperparameters_dict['estimator__n_estimators'] = np.arange(150, 1200, 10)
        hyperparameters_dict['estimator__eta'] = np.linspace(0.01, 1.0, 8)
        hyperparameters_dict['estimator__learning_rate'] = np.linspace(0.0005, 0.5, 10)
        hyperparameters_dict['estimator__feature_fraction'] = np.linspace(0.0005, 0.5, 10)
        hyperparameters_dict['estimator__bagging_fraction'] = np.linspace(0.05, 0.9, 10)
        hyperparameters_dict['estimator__bagging_freq'] = np.arange(1, 20, 10)
        hyperparameters_dict['estimator__max_bin'] = np.arange(128, 1024, 10)
        hyperparameters_dict['estimator__num_leaves'] = np.arange(15, 200, 10)
        hyperparameters_dict['estimator__gamma'] = np.linspace(0.05, 1.0, 8)
        hyperparameters_dict['estimator__max_depth'] = np.arange(3, 30, 10)
        hyperparameters_dict['estimator__min_child_weight'] = np.linspace(1e-4, 10, 10)
        hyperparameters_dict['estimator__subsample'] = np.linspace(0.1, 1.0, 10)
        hyperparameters_dict['estimator__colsample_bytree'] = np.linspace(0.1, 1.0, 10)
        hyperparameters_dict['estimator__reg_lambda'] = np.linspace(0.01, 1.0, 10)
        hyperparameters_dict['estimator__reg_alpha'] = np.linspace(0, 1.0, 6)
        hyperparameters_dict['estimator__importance_type'] = ['split', 'gain']
        hyperparameters_dict['estimator__boosting_type'] = ['gbdt', 'dart', 'goss', 'rf']
        hyperparameters_dict['estimator__min_split_gain'] = np.linspace(0, 3.5, 10)

        return hyperparameters_dict
        