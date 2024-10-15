import unittest
import os
import requests
import tqdm
import mne

import naflow

import numpy as np
import scipy

class Test_RemoveEOG(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_RemoveEOG, self).__init__(*args, **kwargs)
        base_dir = os.path.dirname(__file__)
        
        self.url = "https://dataverse.harvard.edu/api/access/datafile/10614085"
        self.save_dir = naflow.utils.mkdir(os.path.join(os.path.expanduser('~'), "naflow_data"))
        self.file = "oddball_2.vhdr"
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

    def read_raws_vhdr(self):
        raw = mne.io.read_raw_brainvision(self.fname,
                                          eog = ['vEOG', 'hEOG'],
                                          preload = True)
        raw = raw.pick(picks = self.picks)
        
        raw.set_montage('standard_1020')

        raws = list()
        raws.append(raw)
        raws.append(raw.copy())
        
        return raws 

    def read_raws_xdf(self):
        raw, events, event_id = naflow.read.get_raw_from_streams(self.fname,
                                                               name_eeg_stream = self.name_eeg_stream,
                                                               name_marker_stream = self.name_marker_stream)
        raw = raw.pick(picks = self.picks)

        mapping = dict()
        for ch in raw.ch_names:
            if ch in ['vEOG', 'hEOG']:
                mapping[ch] = 'eog'
            else:
                mapping[ch] = 'eeg'

        raw = raw.set_channel_types(mapping = mapping)
        raw.set_montage('standard_1020')

        raws = list()
        raws.append(raw)
        raws.append(raw.copy())
        
        return raws
    
    def read_raws(self):
        if self.fname.split(".")[-1] == 'vhdr':
            return self.read_raws_vhdr()
        elif self.fname.split(".")[-1] == 'xdf':
            return self.read_raws_xdf()
        else:
            raise RuntimeError("file type error.")
    
    def test_eog_removal(self):
        raws = self.read_raws()
        id_list = [id(raw) for raw in raws]
        ch_names_list = [raw.info['ch_names'] for raw in raws]
        highpass_list = [raw.info['highpass'] for raw in raws]
        lowpass_list = [raw.info['lowpass'] for raw in raws]

        remove_eog = naflow.RemoveEOG(l_freq = 1.0, len_transition = 0.5)
        remove_eog.fit(raws)
        
        for idx, raw in enumerate(raws):
            self.assertEqual(id_list[idx], id(raw))
            self.assertEqual(ch_names_list[idx], raw.info['ch_names'])
            self.assertEqual(highpass_list[idx], raw.info['highpass'])
            self.assertEqual(lowpass_list[idx], raw.info['lowpass'])

        print(remove_eog.scores)
        print(remove_eog.exclude)

        remove_eog.ica.plot_scores(remove_eog.scores)
        remove_eog.ica.plot_properties(raws[0].copy().filter(l_freq = 0.1, h_freq = 40), picks = remove_eog.exclude)
        remove_eog.ica.plot_sources(raws[0].copy().filter(l_freq = 0.1, h_freq = 40))
        remove_eog.ica.plot_components()
            
        

        