import unittest
import os
import requests
import tqdm

import naflow

import numpy as np
import scipy

class Test_get_raw_from_streams(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_get_raw_from_streams, self).__init__(*args, **kwargs)
        base_dir = os.path.dirname(__file__)
        
        self.url = "https://dataverse.harvard.edu/api/access/datafile/10614085"
        self.save_dir = naflow.utils.mkdir(os.path.join(os.path.expanduser('~'), "naflow_data"))
        self.file = "oddball.xdf"
        self.name_eeg_stream = 'BrainAmpSeries'
        self.name_marker_stream = 'scab-c'
        self.picks = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'vEOG', 'hEOG']

        self.fname = os.path.join(self.save_dir, self.file)
        
    def get_database(self):
        isExist = os.path.exists(self.fname)
        if isExist is False:
            data = requests.get(self.url, stream = True)
            with open(os.path.join(self.save_dir, self.file), 'wb') as f:
                for chunk in tqdm.tqdm(data.iter_content(chunk_size = 1024)):
                    if chunk:
                        f.write(chunk)
                        f.flush()

    def read_raws(self):
        raw, events, event_id = naflow.read.get_raw_from_streams(self.fname,
                                                               name_eeg_stream = self.name_eeg_stream,
                                                               name_marker_stream = self.name_marker_stream,
                                                               channel_type={'eog':['vEOG', 'hEOG']})
        raw = raw.pick(picks = self.picks)
        raw.set_montage('standard_1020')

        raws = list()
        raws.append(raw)
        raws.append(raw.copy())
        
        return raws
    
    def test_raw(self):
        raws = self.read_raws()
        raw = raws[0]
        print(raw.get_data(picks = 'eeg', units = 'uV'))
        raw.filter(l_freq = 1, h_freq = 40)
        print(raw)
        raw.plot(block = True)
