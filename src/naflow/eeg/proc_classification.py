import copy
import numpy as np
import sklearn

import tag_mne as tm

class EpochsVectorizer():
    def __init__(self,
                 ivals,
                 channel_prime = True,
                 type = 'mne',
                 tmin = None,
                 tmax = None,
                 include_tmax = True,
                 fs = None):
        self.ivals = ivals
        self.channel_prime = channel_prime
        self.type = type
        self.tmin = tmin
        self.tmax = tmax
        self.include_tmax = include_tmax
        self.fs = fs
        
        """
        
        type: 'mne', 'ndarray'
        
        channel_prime:
        if True,  vec = [ch1_t1, ch2_t1, ch3_t1, ch1_t2, ch2_t2, ch3_t2, ...]
        if False, vec = [ch1_t1, ch1_t2, ch1_t3, ch2_t1, ch2_t2, ch2_t3, ...]


        """

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X):
        
        if self.type == 'ndarray':
            tmin = self.tmin
            tmax = self.tmax
            fs = self.fs
            data = X
            if self.include_tmax:
                times = np.linspace(start=tmin, stop=tmax, num=int((tmax-tmin)*fs)+1)
            else:
                times = np.linspace(start=tmin, stop=tmax, num=int((tmax-tmin)*fs))
            if data.shape[2] != times.size:
                raise ValueError("number of time samples is invalid.")
        elif self.type == 'mne':
            tmin = X.times[0]
            tmax = X.times[-1]
            data = X.get_data(copy = True)
        else:
            raise ValueError("Unknown type: %s"%str(self.type))
        
        
        if np.min(np.array(self.ivals).flatten()) < tmin:
            raise ValueError("minimum time value of ivals belows tmin of epochs.")
        
        if np.max(np.array(self.ivals).flatten()) > tmax:
            raise ValueError("maximum time value of ivals exceeds tmax of epochs.")
        
        vec = np.zeros((data.shape[0], data.shape[1], len(self.ivals)))

        for m, ival in enumerate(self.ivals):
            if self.type == 'ndarray':
                idx = list()
                for t in ival:
                    I = np.argmin(np.absolute(times - t))
                    idx.append(I)
            elif self.type == 'mne':
                idx = X.time_as_index(ival)
            idx = list(range(idx[0], idx[1]+1))
            vec[:, :, m] = np.mean(data[:, :, idx], axis=2)

        if self.channel_prime:
            vec = np.reshape(vec, (vec.shape[0], -1), order='F')
        else:
            vec = np.reshape(vec, (vec.shape[0], -1), order='C')
            
        return vec 

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
                 dynamic_stopping_params = None,
                 transient_state = False):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.picks = picks
        self.dynamic_stopping = dynamic_stopping
        self.dynamic_stopping_params = dynamic_stopping_params
        self.trial_classifier = TrialClassifier(classifier = self.classifier,
                                                vectorizer = self.vectorizer,
                                                dynamic_stopping = self.dynamic_stopping,
                                                transient_state = transient_state,
                                                **dynamic_stopping_params)
    
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

def calc_itr(n_classes, accuracy, trial_duration=None):
    """
    compute ITR (Information Transfer Rate).

    Parameters
    ==========
    n_classes : int
        Number of classes of bci system
    accuracy : float
        Classification accuracy
    trial_duration : float, default=None
        Time duration of single trial in minutes. If it's NOT None, ITR will be returned in unit of (bits/min).
        Otherwise, it will be (bit/trial). Default value is None.    

    References
    ==========
    [1] Jonathan R. Wolpaw, Niels Birbaumer, et al., 
        BrainЃomputer Interface Technology: A Review of the First International Meeting, 
        IEEE TRANSACTIONS ON REHABILITATION ENGINEERING, VOL. 8, NO. 2, JUNE 2000
    """
    N = n_classes
    P = accuracy

    log2 = np.log2

    if P == 1:
        itr = log2(N)
    elif P == 0:
        itr = log2(N) + (1-P)*log2((1-P)/(N-1))
    else:
        itr = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))

    if trial_duration is not None:
        return itr / trial_duration
    else:
        return itr