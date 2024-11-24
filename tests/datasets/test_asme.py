import os
import unittest

import naflow.datasets

class Test_asme(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_asme, self).__init__(*args, **kwargs)
        
    def test_asme_speller_30chars_copychar(self):
        origin_base = os.path.join(os.path.expanduser('~'),
                                   "Documents",
                                   "eeg",
                                   "copychar")
        offline, online = naflow.datasets.asme_speller_30chars_copychar(subject = 'A',
                                                      origin_base=origin_base,
                                                      tmin = -0.1,
                                                      tmax = 1.0,
                                                      baseline = None,
                                                      resample = 250)
        
        print(offline['target'])