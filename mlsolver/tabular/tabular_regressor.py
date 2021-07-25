from .auto_tabular import AutoTabular
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklego.preprocessing import ColumnCapper

import numpy as np
import pandas as pd
from ..folds.create_folds import create_stratified_kfolds_for_regression
from ..scorers import RegressionScorers
from ..optimization_grids import TabularRegressorOptimizer

class TabularRegressor(AutoTabular):
    def __init__(self, scoring, n_splits, n_trials, timeout, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        super().__init__(scoring, n_splits, n_trials, timeout, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs)
        self.best_estimator = None

    def _create_feature_engineering_pipeline(self):
        numerical_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('outlier_capper', ColumnCapper()),
            ('scaler', StandardScaler())
        ])
        
        #Create a pipeline to substitute NaN categorical values by the most frequent categorical value and encoding categorical values with OneHotEncoder()
        categorical_pipeline = Pipeline([
            ('cleaner',SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False))
        ])

        #Apply the previous pipelines: Apply the first to numerical values and the second to categorical values
        feature_engineering_preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
            ('categorical', categorical_pipeline, make_column_selector(dtype_include=['object','category'])),
        ])

        return feature_engineering_preprocessor

    def _create_model_pipeline(self):
        pass

    def _create_ensembles(self):
        pass
    
    def fit(self, X, y):
        pipeline = Pipeline(steps=[
            ('preprocessor', self._create_feature_engineering_pipeline()),
            ('feature_selector', SelectKBest(f_regression, k=10)),
            ('estimator', KNeighborsRegressor())
        ])

        data = pd.concat([X, y], axis=1)
        folds = create_stratified_kfolds_for_regression(data=data, target_column=data.columns[-1], n_splits=self.n_splits) #THIS WILL CRASH IF THE PROBLEM IS A MULTIPLE COLUMN REGRESSION
        
        #Retrieve scorer and optimization direction
        regression_scorers = RegressionScorers()
        scorer, direction = regression_scorers.get_scorer(self.scoring), regression_scorers.get_direction(self.scoring)
        print(scorer)
        print(direction)

        optimizer = TabularRegressorOptimizer(pipeline, X, y, scorer, folds, self.n_jobs, self.models, self.feature_engineering)
        study, best_params = optimizer.optimize(direction=direction, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)

        self.best_estimator = pipeline.set_params(**best_params)
        self.best_estimator.fit(X, y)

        print(study.trials_dataframe().head(10))

        return self