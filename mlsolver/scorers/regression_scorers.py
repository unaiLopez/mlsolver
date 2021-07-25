from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer

class RegressionScorers:
    def __init__(self):
        self.scorers = {
            'mae': {'scorer': self._mae_scorer(), 'direction': 'minimize'},
            'mse': {'scorer': self._mse_scorer(), 'direction': 'minimize'},
            'mape': {'scorer': self._mape_scorer(), 'direction': 'minimize'},
            'r2': {'scorer': self._r2_scorer(), 'direction': 'maximize'}
        }

    def get_scorer(self, metric):
        return self.scorers[metric]['scorer']

    def get_direction(self, metric):
        return self.scorers[metric]['direction']

    def _mae_scorer(self):
        return make_scorer(mean_absolute_error, greater_is_better=False)

    def _mse_scorer(self):
        return make_scorer(mean_squared_error, greater_is_better=False)

    def _mape_scorer(self):
        return make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    
    def _r2_scorer(self):
        return make_scorer(r2_score, greater_is_better=True)