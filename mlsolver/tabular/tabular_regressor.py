from .auto_tabular import AutoTabular
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

import numpy as np
import pandas as pd
from ..folds.create_folds import create_stratified_kfolds_for_regression
from ..metrics import RegressionMetrics
from ..optimization_grids import TabularRegressionHyperparameters
from mlsolver.tabular import auto_tabular

class TabularRegressor(AutoTabular):
    def __init__(self, metric_to_optimize, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs):
        super().__init__(metric_to_optimize, models, feature_engineering, hyperparameter_tuning, ensembles, n_jobs)
        self.best_estimator = None

    def _create_feature_engineering_pipeline(self, X):
        numerical_pipeline = Pipeline([
            ('cleaner',SimpleImputer()),
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

        auto_tabular_pipeline_steps = list()

        if self.feature_engineering:
            auto_tabular_pipeline_steps.append(('preprocessor', self._create_feature_engineering_pipeline(X)))

        auto_tabular_pipeline_steps.append(('feature_selector', SelectKBest(f_regression, k=10)))
        auto_tabular_pipeline_steps.append(('estimator', KNeighborsRegressor()))
        self.pipeline = Pipeline(auto_tabular_pipeline_steps)
        optimization_grid = TabularRegressionHyperparameters(self.models, self.feature_engineering)()

        data = pd.concat([X, y], axis=1)

        cv = create_stratified_kfolds_for_regression(data=data, target_column=data.columns[-1], n_splits=5) #THIS WILL CRASH IF THE PROBLEM IS A MULTIPLE COLUMN REGRESSION
        
        my_metric = RegressionMetrics().get_metric_function(self.metric_to_optimize)
        my_scorer = make_scorer(my_metric)

        search = RandomizedSearchCV(self.pipeline ,
                                    optimization_grid,
                                    n_iter=100,
                                    scoring=my_scorer,
                                    n_jobs=self.n_jobs,
                                    refit=True,
                                    verbose=3,
                                    cv=cv)
        
        search.fit(X, y)

        print(search.cv_results_)

        self.best_estimator = search.best_estimator_

        return self