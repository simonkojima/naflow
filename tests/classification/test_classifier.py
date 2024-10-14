import unittest
import os

import sklearn.discriminant_analysis
import naflow

import numpy as np
import scipy

class Test_ShrinkageLDA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_ShrinkageLDA, self).__init__(*args, **kwargs)
        base_dir = os.path.dirname(__file__)
        self.bbci_plain_lda = scipy.io.loadmat(os.path.join(base_dir, "plain_lda.mat"))
        self.bbci_shrinkage_lda = scipy.io.loadmat(os.path.join(base_dir, "shrinkage_lda.mat"))
        
        self.X = self.bbci_plain_lda['xTr']
        self.Y = self.bbci_plain_lda['yTr'][:, 1]
        
    def compare_bbci(self, clf, bbci):
        self.assertAlmostEqual(clf.gamma, np.squeeze(bbci['gamma']))
        
        V1 = clf._Cw.flatten()
        V2 = np.squeeze(bbci['cov']).flatten()
        for v1, v2 in zip(V1, V2):
            self.assertAlmostEqual(v1, v2)

        V1 = clf._Cw_inv.flatten()
        V2 = np.squeeze(bbci['invcov']).flatten()
        for v1, v2 in zip(V1, V2):
            self.assertAlmostEqual(v1, v2)
        
        w = np.squeeze(bbci['w'])
        for v1, v2 in zip(w, clf.w):
            self.assertAlmostEqual(v1, v2)

        self.assertAlmostEqual(np.squeeze(bbci['b']), clf.b)

    def test_fit(self):
        y_neg = np.unique(self.Y)[0]
        y_pos = np.unique(self.Y)[1]
        
        X_pos = self.X[np.where(self.Y == y_pos)[0], :]
        X_neg = self.X[np.where(self.Y == y_neg)[0], :]

        # plain lda
        clf = naflow.classification.ShrinkageLDA(gamma = 0, scaling = 2)
        clf.fit(self.X, self.Y)
        self.compare_bbci(clf, self.bbci_plain_lda)

        scores = clf.decision_function(X_pos)
        for score in scores:
            self.assertGreater(score, 0)
        
        scores = clf.decision_function(X_neg)
        for score in scores:
            self.assertLess(score, 0)
        
        # shrinkage lda
        clf = naflow.classification.ShrinkageLDA(gamma = 'shrinkage', scaling = 2)
        clf.fit(self.X, self.Y)
        self.compare_bbci(clf, self.bbci_shrinkage_lda)