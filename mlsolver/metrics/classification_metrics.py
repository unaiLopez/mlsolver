from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, f1_score

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            'accuracy': self._accuracy,
            'recall': self._recall,
            'precision': self._precision,
            'f1': self._f1,
            'auc': self._auc
        }
    
    def __call__(self, metric, y_true, y_pred):
        if metric in self.metrics:
            return self.metrics[metric](y_true, y_pred)
        else:
            raise Exception(f'{metric} is not a binary classification metric or is not supported.')

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred)(y_true, y_pred)

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred)

    @staticmethod
    def auc(y_true, y_pred):
        return auc(y_true, y_pred)