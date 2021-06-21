import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

def create_kfolds(data, n_splits=5, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    folds = list()
    for train, test in enumerate(kf.split(data)):
        folds.append((train, test))
    
    return folds

def create_stratified_kfolds_for_classification(data, target_column, n_splits=5, shuffle=True):
    kf = StratifiedKFold(n_splits, shuffle=shuffle)

    folds = list()
    for train, test in kf.split(X, data[target_column]):
        folds.append((train, test))
    
    return folds


def create_stratified_kfolds_for_regression(data, target_column, n_splits=5):
    data = data.sample(frac=1).reset_index(drop=False)

    #Apply Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data['bins'] = pd.cut(data[target_column], bins=num_bins, labels=False)

    kf = StratifiedKFold(n_splits=n_splits)

    folds = list()
    for fold, (train, test) in enumerate(kf.split(X=data, y=data.bins.values)):        
        folds.append((train, test))

    return folds