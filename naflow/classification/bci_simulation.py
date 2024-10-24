import copy
import numpy as np
import sklearn

import tag_mne as tm

class TrialClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self,
                 classifier,
                 vectorizer,
                 dynamic_stopping = False,
                 p_th = 0.05,
                 alternative = 'greater',
                 min_nstims = 5,
                 groups = 'best-second',
                 transient_state = False):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.dynamic_stopping = dynamic_stopping
        self.p_th = p_th
        self.alternative = alternative
        self.min_nstims = min_nstims
        self.groups = groups
        self.transient_state = transient_state

    def check_nstims(self, distances, events):
        val = list()
        for event in events:
            val.append(len(distances[event]))
        return min(val)
    
    def test_distances(self, distances, events, method = 'mean'):
        from scipy import stats
        distance_class = list()
        for event in events:
            if method == 'mean':
                distance_class.append(np.mean(np.array(distances[event])))
        I = np.argsort(distance_class)
        best = events[I[-1]]
        second_best = events[I[-2]]
        rest = events.copy()
        rest.remove(best) # modified in place

        if self.groups == 'best-rest':
            best_group = np.array(distances[best])
            another_group = list()
            for event in rest:
                another_group += distances[event]
            another_group = np.array(another_group)
        elif self.groups == 'best-second':
            best_group = np.array(distances[best])
            another_group = np.array(distances[second_best])
        else:
            raise ValueError("groups '%s' is not yet implemented."%self.groups)

        _, p = stats.ttest_ind(best_group, another_group, equal_var = False, alternative = self.alternative)
        
        pred = best
        
        return pred, p


    def fit(self, X, y = None):
        return self
    
    def _predict_static(self, X):
        events = tm.get_values_list(X, "event")
        
        if self.transient_state:

            transient = list()
            distances = dict()
            for event in events:
                distances[event] = list()

            for m in range(X.__len__()):
                epoch = X[m]
                event = tm.get_values_list(epoch, "event")[0]
                
                vec = self.vectorizer.transform(epoch)
                distances[event].append(self.classifier.decision_function(vec))

                n_epochs = m+1
                distances_save = copy.copy(distances)
                for event in events:
                    distances_save[event] = np.array(distances_save[event]).tolist()

                transient.append([n_epochs, distances_save])

            for event in events:
                distances[event] = np.mean(distances[event])
            y = max(distances, key = distances.get)
            
            return y, None, transient
            
        else:
            distances = dict()
            for event in events:
                data = X['event:%s'%event]               
                data = self.vectorizer.transform(data)
                distances[event] = np.mean(self.classifier.decision_function(data))
            y = max(distances, key = distances.get)
            return y, None, None
    
    def _predict_dynamic_stopping(self, X):
        print(X['target'])

        events = tm.get_values_list(X, "event")

        distances = dict()
        for event in events:
            distances[event] = list()
            
        for m in range(X.__len__()):
            epoch = X[m]
            event = tm.get_values_list(epoch, "event")[0]
            
            vec = self.vectorizer.transform(epoch)
            distances[event].append(self.classifier.decision_function(vec))
            
            if self.check_nstims(distances, events) >= self.min_nstims:
                pred, p = self.test_distances(distances, events, 'mean')
                if p < self.p_th:
                    break

        n_epochs = m+1
        
        return pred, {'p':p, 'n_epochs': n_epochs}, None
        

    
    def predict(self, X):
        trials = tm.get_values_list(X, "trial")
        if len(trials) != 1:
            raise RuntimeError("Number of trials contained in X should be one.")

        if self.dynamic_stopping:
            return self._predict_dynamic_stopping(X)
        else:
            return self._predict_static(X)

class BCISimulation(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self,
                 vectorizer,
                 classifier,
                 picks = 'eeg',
                 dynamic_stopping = False,
                 dynamic_stopping_params = {},
                 transient_state = False):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.picks = picks
        self.dynamic_stopping = dynamic_stopping
        self.dynamic_stopping_params = dynamic_stopping_params
        self.transient_state = transient_state
        self.trial_classifier = TrialClassifier(classifier = self.classifier,
                                                vectorizer = self.vectorizer,
                                                dynamic_stopping = self.dynamic_stopping,
                                                transient_state = self.transient_state,
                                                **self.dynamic_stopping_params)
    
    def fit(self, X, y = None):
        X = X.copy().pick(picks = self.picks)
        
        X, Y = tm.get_binary_epochs(X)

        X = self.vectorizer.transform(X)
        self.classifier.fit(X, Y)

        return self
    
    def predict(self, X):
        X = X.copy().pick(picks = self.picks)
        runs = tm.get_values_list(X, "run")
        
        results = dict()
        results['labels'] = list()
        results['preds'] = list()
        results['dynamic_stopping'] = list()
        results['transient'] = list()
        for run in runs:
            trials = tm.get_values_list(X['run:%s'%run], "trial")
            for trial in trials:

                epochs_trial = X['run:%s/trial:%s'%(run, trial)]

                results['labels'].append(tm.get_values_list(epochs_trial['target'], "event")[0])

                y, ds, transient = self.trial_classifier.predict(epochs_trial)
                results['preds'].append(y)
                results['dynamic_stopping'].append(ds)
                results['transient'].append(transient)

        return results
