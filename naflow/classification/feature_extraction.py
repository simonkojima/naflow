import numpy as np

class EpochsVectorizer():
    """
    
    type: 'mne', 'ndarray'
    
    channel_prime:
    if True,  vec = [ch1_t1, ch2_t1, ch3_t1, ch1_t2, ch2_t2, ch3_t2, ...]
    if False, vec = [ch1_t1, ch1_t2, ch1_t3, ch2_t1, ch2_t2, ch2_t3, ...]


    """
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