import copy
import numpy as np
import sklearn

def subtract_classwise_mean(X, y):
    xTr = copy.copy(X).T
    X = np.zeros((xTr.shape[0], 0))
    for cl_num in np.unique(y):
        I = np.where(y == cl_num)[0]
        _mean = np.mean(xTr[:, I], axis = 1, keepdims = True)
        X = np.concatenate([X, xTr[:, I] - np.tile(_mean, (1, I.size))], axis = 1)
    return X

class ShrinkageLDA(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, gamma = 'shrinkage', scaling = 2):
        """
        LDA (Linear Discriminant Analysis) implementation with scikit-learn interface.

        Parameters
        ----------
        gamma : float or str, default = 'shrinkage'
        scaling : float, default = 2
                  the distance between the projeted means become specified number.
        
        """
        self.w = None
        self.b = None
        self.gamma = gamma
        self.scaling = scaling
        self._Cw = None
        
    def gamma_shrinkage(self, X, T = None):
        X = copy.copy(X)

        p, n = X.shape

        Xn = X - np.tile(np.mean(X, axis = 1, keepdims = True), (1, n))
        
        S = np.dot(Xn, Xn.T)
        Xn2 = np.square(Xn)
        
        if T is None:
            nu = np.mean(S[np.diag_indices(p)])
            T = nu * np.eye(p, p)

        V = 1/(n-1) * (np.dot(Xn2, Xn2.T) - np.square(S)/n)
        gamma = n * np.sum(np.sum(V)) / np.sum(np.sum(np.square(S - T)))
        
        return gamma, T, S, n
    
    def fit(self, X, y = None):

        classes = np.unique(y)
        if len(classes) != 2:
            raise RuntimeError("Number of classes should be 2. Seems y contains more than 3 labels.")

        cl = list()
        for cl_num in classes:
            I = np.where(y == cl_num)[0]
            cl.append(X[I, :])
        
        mean = list()
        for cl_data in cl:
            mean.append(np.mean(cl_data, axis = 0))

        C = list()
        for cl_data in cl:
            C.append(np.cov(cl_data.T))

        gamma, T, S, n = self.gamma_shrinkage(X = subtract_classwise_mean(X = X, y = y))

        if self.gamma == 'shrinkage':
            self.gamma = gamma

        Cw = (np.dot(self.gamma, T) + np.dot((1-self.gamma), S))/(n-1)
            
        self._Cw = Cw
        Cw_inv = np.linalg.inv(Cw)
        self._Cw_inv = Cw_inv

        w = np.dot(Cw_inv,(mean[1]-mean[0]))
        #print(mean)
        #w = np.dot(Cw_inv, mean[0]) - np.dot(Cw_inv, mean[1])
        if self.scaling is not None:
            scaling_factor = self.scaling / (np.dot(w.T, mean[0]) - np.dot(w.T, mean[1]))
            w = np.squeeze(w*np.absolute(scaling_factor))
            #w = np.squeeze(w / np.linalg.norm(w))
            #w = w/(np.dot(w.T, mean[0]-mean[1]))*self.scaling
            #print(mean[0]-mean[1])

        b = -0.5 * (np.dot(w.T, mean[0])+np.dot(w.T, mean[1]))
            
        self.w = w
        self.b = b
        
        return self
    
    def decision_function(self, X):
        return np.dot(self.w, X.T)
    
    def predict(self, X):
        pass
    
