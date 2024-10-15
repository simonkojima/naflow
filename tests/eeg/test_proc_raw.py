import unittest
import os

import naflow

import numpy as np
import scipy

class Test_RemoveEOG(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_RemoveEOG, self).__init__(*args, **kwargs)
        base_dir = os.path.dirname(__file__)