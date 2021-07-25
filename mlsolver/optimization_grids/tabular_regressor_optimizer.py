from optuna import create_study

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class TabularRegressorOptimizer:
    def __init__(self, pipeline, X, y, scorer, cv, n_jobs, models, feature_engineering):
        self.pipeline = pipeline
        self.X = X
        self.y = y
        self.scorer = scorer
        self.cv = cv
        self.n_jobs = n_jobs
        self.models = models
        self.feature_engineering = feature_engineering
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        self.encoders = {
            'OneHotEncoding': OneHotEncoder(sparse=False)
        }
        self.estimators = {
            'RandomForest': RandomForestRegressor(),
            'KNN': KNeighborsRegressor(),
            'LinearRegression': LinearRegression(),
            'AdaBoost': AdaBoostRegressor(),
            'SGD': SGDRegressor(),
            'SVM': SVR(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }

        #self.model_hyperparameters = {
            #'RandomForest': self._load_random_forest_hyperparameters(),
            #'KNN': self._load_knn_hyperparameters(),
            #'LinearRegression': self._load_linear_regression_study(trial)
            #'AdaBoost': self._load_adaboost_hyperparameters(),
            #'SGD': self._load_sgd_hyperparameters(),
            #'SVM': self._load_svm_hyperparameters(),
            #'XGBoost': self._load_xgboost_hyperparameters()
            #'LightGBM': self._load_lightgbm_hyperparameters()
        #}
    
    def optimize(self, direction, n_trials, timeout, n_jobs):
        study = create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        if 'preprocessor__numerical__scaler' in study.best_params:
            scaler = self.estimators[study.best_params['preprocessor__numerical__scaler']]
        else:
            scaler = None

        best_estimator_params = study.best_params

        estimator = self.estimators[study.best_params['estimator']]
        encoder = self.encoders[study.best_params['preprocessor__categorical__encoder']]

        best_estimator_params['estimator'] = estimator
        best_estimator_params['preprocessor__numerical__scaler'] = scaler
        best_estimator_params['preprocessor__categorical__encoder'] = encoder

        return study, best_estimator_params

    def _suggest_linear_regression_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:

            numerical_cleaner_strategy = trial.suggest_categorical('preprocessor__numerical__cleaner__strategy', ['mean', 'median'])
            numerical_scaler_name = trial.suggest_categorical('preprocessor__numerical__scaler', list(self.scalers.keys()))
            categorical_cleaner_strategy = trial.suggest_categorical('preprocessor__categorical__cleaner__strategy', ['most_frequent'])
            categorical_encoder_name = trial.suggest_categorical('preprocessor__categorical__encoder', list(self.encoders.keys()))
            feature_selector_k = trial.suggest_int('feature_selector__k', 1, self.X.shape[1], 3)

            numerical_scaler_object = self.scalers[numerical_scaler_name]
            categorical_encoder_object = self.encoders[categorical_encoder_name]

            feature_engineering_params = {
                'preprocessor__numerical__cleaner__strategy': numerical_cleaner_strategy,
                'preprocessor__numerical__scaler': numerical_scaler_object,
                'preprocessor__categorical__cleaner__strategy': categorical_cleaner_strategy,
                'preprocessor__categorical__encoder': categorical_encoder_object,
                'feature_selector__k': feature_selector_k
            }

        fit_intercept = trial.suggest_categorical('estimator__fit_intercept', [True, False])
        normalize = trial.suggest_categorical('estimator__normalize', [True, False])

        model_params = {
            'estimator': LinearRegression(),
            'estimator__fit_intercept': fit_intercept,
            'estimator__normalize': normalize
        }
        
        return feature_engineering_params, model_params

    def _suggest_random_forest_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:

            numerical_cleaner_strategy = trial.suggest_categorical('preprocessor__numerical__cleaner__strategy', ['mean', 'median'])
            categorical_cleaner_strategy = trial.suggest_categorical('preprocessor__categorical__cleaner__strategy', ['most_frequent'])
            categorical_encoder_name = trial.suggest_categorical('preprocessor__categorical__encoder', list(self.encoders.keys()))
            feature_selector_k = trial.suggest_int('feature_selector__k', 1, self.X.shape[1], 3)

            categorical_encoder_object = self.encoders[categorical_encoder_name]

            feature_engineering_params = {
                'preprocessor__numerical__cleaner__strategy': numerical_cleaner_strategy,
                'preprocessor__categorical__cleaner__strategy': categorical_cleaner_strategy,
                'preprocessor__categorical__encoder': categorical_encoder_object,
                'feature_selector__k': feature_selector_k
            }

        n_estimators = trial.suggest_int("estimator__n_estimators", 50, 250)
        max_depth = trial.suggest_int("estimator__max_depth", 5, 20)
        min_samples_split = trial.suggest_int("estimator__min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("estimator__min_samples_leaf", 2, 10)
        max_features = trial.suggest_categorical("estimator__max_features", ['auto', 'sqrt', 'log2'])

        model_params = {
            'estimator': RandomForestRegressor(),
            'estimator__n_estimators': n_estimators,
            'estimator__max_depth': max_depth,
            'estimator__min_samples_split': min_samples_split,
            'estimator__min_samples_leaf': min_samples_leaf,
            'estimator__max_features': max_features
        }
        
        return feature_engineering_params, model_params
        
    def _objective(self, trial):
        feature_engineering_params = {}
        model_params = {}

        model = trial.suggest_categorical('estimator', self.models)

        if model == 'LinearRegression':
            feature_engineering_params, model_params = self._suggest_linear_regression_hyperparameters(trial)
        
        elif model == 'RandomForest':
            feature_engineering_params, model_params = self._suggest_random_forest_hyperparameters(trial)

        pipeline_params = {**feature_engineering_params, **model_params}
        pipeline = self.pipeline.set_params(**pipeline_params)

        scores = -cross_val_score(pipeline, self.X, self.y, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv)
        score = scores.mean()

        return score