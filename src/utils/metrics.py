from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, accuracy_score, recall_score, precision_score, auc, f1_score

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(f)(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def auc(y_true, y_pred):
    return auc(y_true, y_pred)