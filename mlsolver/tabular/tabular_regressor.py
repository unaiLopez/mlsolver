
from sklearn.model_selection import RandomizedSearchCV
from folds.create_folds import create_stratified_kfolds_for_regression

class TabularRegressor(AutoTabular):
    def __init__(self):
        super().__init__()

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
        auto_tabular_pipeline = list()
        auto_tabular_pipeline.append(_create_data_cleaning_pipeline())
        auto_tabular_pipeline.append(_create_feature_engineering_pipeline())
        auto_tabular_pipeline.append(_create_hyperparameter_tuning_pipeline())

        data = pd.concat([X, y], axis=1)

        cv = create_stratified_kfolds_for_regression(data=data, target_column=data.columns[X.shape[0]:]) #THIS CAN CRASH IF THE PROBLEM IS A MULTIPLE COLUMN REGRESSION

        search = RandomizedSearchCV(auto_tabular_pipeline,
                                    n_iter=20,
                                    scoring=self.metric_to_optimize,
                                    n_jobs=self.n_jobs,
                                    cv=cv)
        
        search.fit(X, y)

        self.best_estimator = search.best_estimator_
        self.best_pipeline = search.best_params_