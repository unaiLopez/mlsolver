from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            'mae': self._mae,
            'mse': self._mse,
            'mape': self._mape
        }
    
    def __call__(self, metric, y_true, y_pred):
        if metric in self.metrics:
            return self.metrics[metric](y_true, y_pred)
        else:
            raise Exception(f'{metric} is not a binary classification metric or is not supported.')

    def get_metric_function(self, metric):
        return self.metrics[metric]

    @staticmethod
    def _mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def _mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)