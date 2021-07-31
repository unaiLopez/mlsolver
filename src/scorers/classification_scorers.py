from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, f1_score, make_scorer

class ClassificationScorers:
    def __init__(self):
        self.scorers = {
            'accuracy': {'scorer': self._accuracy_scorer(), 'direction': 'maximize'},
            'recall': {'scorer': self._recall_scorer(), 'direction': 'maximize'},
            'precision': {'scorer': self._precision_scorer(), 'direction': 'maximize'},
            'f1': {'scorer': self._f1_scorer(), 'direction': 'maximize'},
            'auc': {'scorer': self._auc_scorer(), 'direction': 'maximize'}
        }
    
    def get_scorer(self, metric):
        return self.scorers[metric]['scorer']

    def get_direction(self, metric):
        return self.scorers[metric]['direction']

    def _accuracy_scorer(self):
        return make_scorer(accuracy_score, greater_is_better=True)

    def _recall_scorer(self):
        return make_scorer(recall_score, greater_is_better=True)

    def _precision_scorer(self):      
        return make_scorer(precision_score, greater_is_better=True)

    def _f1_scorer(self):
        return make_scorer(f1_score, greater_is_better=True)

    @staticmethod
    def _auc_scorer(self):
        return make_scorer(auc, greater_is_better=True)