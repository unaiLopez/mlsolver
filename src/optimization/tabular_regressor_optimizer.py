import optuna

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder

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
            'RobustScaler': RobustScaler(),
            'None': None
        }
        self.encoders = {
            'OneHotEncoding': OneHotEncoder(sparse=False),
            'LabelEncoding': LabelEncoder()
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

    
    def optimize(self, direction, n_trials, timeout, n_jobs):
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        best_params = self._parse_string_hyperparameters_to_objects(study)

        return study, best_params

    def _objective(self, trial):
        feature_engineering_params = dict()
        model_params = dict()

        model = trial.suggest_categorical('estimator', self.models)

        if model == 'LinearRegression':
            feature_engineering_params, model_params = self._suggest_linear_regression_hyperparameters(trial)
        
        elif model == 'RandomForest':
            feature_engineering_params, model_params = self._suggest_random_forest_hyperparameters(trial)
        
        elif model == 'AdaBoost':
            feature_engineering_params, model_params = self._suggest_adaboost_hyperparameters(trial)
        
        elif model == 'XGBoost':
            feature_engineering_params, model_params = self._suggest_xgboost_hyperparameters(trial)

        pipeline_params = {**feature_engineering_params, **model_params}
        pipeline = self.pipeline.set_params(**pipeline_params)

        scores = -cross_val_score(pipeline, self.X, self.y, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv)
        score = scores.mean()

        return score

    def _parse_string_hyperparameters_to_objects(self, study):
        best_params = study.best_params

        estimator = self.estimators[best_params['estimator']]
        scaler = self.scalers[best_params['preprocessor__numerical__scaler']]
        encoder = self.encoders[best_params['preprocessor__categorical__encoder']]

        best_params['estimator'] = estimator
        best_params['preprocessor__numerical__scaler'] = scaler
        best_params['preprocessor__categorical__encoder'] = encoder

        return best_params

    #######################################################################################
    #                  FEATURE ENGINEERING HYPERPARAMETER SUGGESTION
    #######################################################################################

    def _suggest_feature_engineering_hyperparameters(self, trial):
        numerical_cleaner_strategy = trial.suggest_categorical('preprocessor__numerical__cleaner__strategy', ['mean', 'median'])
        numerical_scaler_name = trial.suggest_categorical('preprocessor__numerical__scaler', list(self.scalers.keys()))
        categorical_cleaner_strategy = trial.suggest_categorical('preprocessor__categorical__cleaner__strategy', ['most_frequent'])
        categorical_encoder_name = trial.suggest_categorical('preprocessor__categorical__encoder', list(self.encoders.keys()))
        feature_selector_k = trial.suggest_int('feature_selector__k', 1, self.X.shape[1])

        numerical_scaler_object = self.scalers[numerical_scaler_name]
        categorical_encoder_object = self.encoders[categorical_encoder_name]

        feature_engineering_params = {
            'preprocessor__numerical__cleaner__strategy': numerical_cleaner_strategy,
            'preprocessor__numerical__scaler': numerical_scaler_object,
            'preprocessor__categorical__cleaner__strategy': categorical_cleaner_strategy,
            'preprocessor__categorical__encoder': categorical_encoder_object,
            'feature_selector__k': feature_selector_k
        }

        return feature_engineering_params

    #######################################################################################
    #                   LINEAR REGRESSION HYPERPARAMETER SUGGESTION
    #######################################################################################

    def _suggest_linear_regression_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:
            feature_engineering_params = self._suggest_feature_engineering_hyperparameters(trial)

        fit_intercept = trial.suggest_categorical('estimator__fit_intercept', [True, False])
        normalize = trial.suggest_categorical('estimator__normalize', [True, False])

        model_params = {
            'estimator': LinearRegression(),
            'estimator__fit_intercept': fit_intercept,
            'estimator__normalize': normalize
        }
        
        return feature_engineering_params, model_params

    #######################################################################################
    #                   RANDOM FOREST HYPERPARAMETER SUGGESTION
    #######################################################################################

    def _suggest_random_forest_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:
            feature_engineering_params = self._suggest_feature_engineering_hyperparameters(trial)

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
    
    #######################################################################################
    #                   ADABOOST HYPERPARAMETER SUGGESTION
    #######################################################################################

    def _suggest_adaboost_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:
            feature_engineering_params = self._suggest_feature_engineering_hyperparameters(trial)

        n_estimators = trial.suggest_int("estimator__n_estimators", 50, 250)
        learning_rate = trial.suggest_float("estimator__learning_rate", 0.001, 0.5)
        loss = trial.suggest_categorical("estimator__loss", ['linear', 'square'])

        model_params = {
            'estimator': AdaBoostRegressor(),
            'estimator__n_estimators': n_estimators,
            'estimator__learning_rate': learning_rate,
            'estimator__loss': loss
        }
        
        return feature_engineering_params, model_params

    #######################################################################################
    #                            XGBOOST HYPERPARAMETER SUGGESTION
    #######################################################################################

    def _suggest_xgboost_hyperparameters(self, trial):
        feature_engineering_params = {}

        if self.feature_engineering:
            feature_engineering_params = self._suggest_feature_engineering_hyperparameters(trial)

        n_estimators = trial.suggest_int("estimator__n_estimators", 50, 250)
        eta = trial.suggest_float("estimator__eta", 0.01, 1.0)
        gamma = trial.suggest_float("estimator__gamma", 0.05, 1.0)
        max_depth = trial.suggest_int("estimator__max_depth", 3, 30)
        min_child_weight = trial.suggest_int("estimator__min_child_weight", 1, 10)
        subsample = trial.suggest_float("estimator__subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("estimator__colsample_bytree", 0.1, 1.0)
        reg_lambda = trial.suggest_float("estimator__reg_lambda", 0, 1.0)
        reg_alpha = trial.suggest_float("estimator__reg_alpha", 0, 1.0)

        model_params = {
            'estimator': XGBRegressor(),
            'estimator__n_estimators': n_estimators,
            'estimator__eta': eta,
            'estimator__gamma': gamma,
            'estimator__max_depth': max_depth,
            'estimator__min_child_weight': min_child_weight,
            'estimator__subsample': subsample,
            'estimator__colsample_bytree': colsample_bytree,
            'estimator__reg_lambda': reg_lambda,
            'estimator__reg_alpha': reg_alpha
        }
        
        return feature_engineering_params, model_params