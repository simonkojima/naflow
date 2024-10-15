import numpy as np
import scipy
import sklearn

import mne

import tag_mne as tm

def reconstruct_raw(raw):
    raw = mne.io.RawArray(raw.get_data(), mne.create_info(raw.ch_names, raw.info['sfreq']))
    return raw

def concatenate_raws(raws, l_freq, order = 2, len_transition = 0.5):
    from ..utils.proc_temporal import round_edge

    cat_raws = list()
    for idx, raw in enumerate(raws):
        Fs = raw.info['sfreq']
        #raw.apply_function(apply_sosfilter, picks = 'all', n_jobs = -1, channel_wise = True, sos=sos, zero_phase = True)
        raw.filter(l_freq = l_freq, h_freq = None, picks = 'all', method = 'iir', iir_params = {'order': order, 'ftype':'butter'}, phase = 'zero', n_jobs = -1)
        raw.apply_function(round_edge, picks = 'all', n_jobs = -1, channel_wise = True, Fs = Fs, len_transition = len_transition)
        
        cat_raws.append(raw)

    cat_raw = mne.concatenate_raws(cat_raws)
    cat_raw.info.highpass = l_freq

    return cat_raw

class RemoveEOG(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Remove EOG artifacts from raw signals.

    Parameters
    ----------
    l_freq : float, default = 1.0
        cutoff frequency for the highpass filter applied to raw signal before applying ICA.
    len_transition : float, default = 0.5
        duration of the taper in second for applying window function before concatenating multiple raws.
    """
    def __init__(self, l_freq = 1.0, len_transition = 0.5):
        self.l_freq = l_freq
        self.len_transition = len_transition
        self.ica = None
        self.scores = None
        self.exclude = None

    def find_bad_eog(self, raw, ica, h_freq = 10, threshold = 0.9):
        """
        Parameters
        ----------

        raw : raw instance contains eog channels.
        ica : ica instance
        filter : filter range will be used for eog channels
        threshold : numerical or 'max': 
        
        Returns
        -------
        scores : pearson correlation bet. each IC and EOG channel
        indices : indices of ICs which should be removed
        
        """

        raw_eog = raw.copy().pick(picks = ['eog'])
        raw_eeg = raw.copy().pick(picks = ['eeg'])

        raw_eog = reconstruct_raw(raw_eog) 
        raw_eeg = reconstruct_raw(raw_eeg)

        IC = ica.get_sources(raw_eeg)
        

        if h_freq is not None:
            raw_eog.filter(picks = 'all', l_freq = None, h_freq = h_freq, method = 'iir', iir_params = {'order': 2, 'ftype':'butter'}, phase = 'zero', n_jobs = -1)
            raw_eeg.filter(picks = 'all', l_freq = None, h_freq = h_freq, method = 'iir', iir_params = {'order': 2, 'ftype':'butter'}, phase = 'zero', n_jobs = -1)

        scores = list()
        indices = list()
        for ch in raw_eog.ch_names:
            data_eog = raw_eog.get_data(picks = ch)

            score = list() 
            for idx, ic in enumerate(IC.ch_names):
                a = scipy.stats.pearsonr(x = np.squeeze(data_eog), y = np.squeeze(IC.get_data(picks = ic)))

                score.append(a[0])
                
            if threshold == 'max':
                I = np.argmax(np.absolute(np.array(score)))
                indices.append(I)
            else:
                I = np.where(np.absolute(np.array(score)) >= threshold)
                indices += I[0].tolist()
                    
            scores.append(score)

        scores = np.array(scores)
        
        return scores, indices

    def fit(self, X, y = None):
        X = [x.copy() for x in X]
            
        raw = concatenate_raws(X, l_freq = self.l_freq, order = 2, len_transition = 0.5)

        ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=42)
        ica.fit(raw.copy().pick(picks = 'eeg'))
        
        scores, indices = self.find_bad_eog(raw, ica, h_freq = 10, threshold = 'max')
        ica.exclude = indices

        self.exclude = indices
        self.scores = scores
        self.ica = ica
        
        return self

    def transform(self, X):
        if type(X) is list:
            X = [self.ica.apply(x.copy(), exclude = self.ica.exclude) for x in X]
        else:
            X = self.ica.apply(X.copy(), exclude = self.ica.exclude)
        return X

class ExtractEpochs(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Extract Epochs from raws
    
    Parameters
    ----------
    l_freq : float, default = 0.1 
        cutoff frequency for high-pass filter
    h_freq : float, default = 8
        cutoff frequency for low-pass filter
    filter_params : dict, default = {'method': 'iir', 'phase': 'zero', 'iir_params':{'order':2, 'ftype':'butter'}, 'n_jobs': -1}
        pass to filter() method of mne.raw object
    tmin : float, default = -0.1
        start time of the epochs in seconds relative to the event onset.
    tmax : float, default = 1.2
        end time of the epochs in seconds relative to the event onset.
    baseline : None or tuple of length 2, default = None
        time duration for baseline correction. No baseline correction will be applied if None. see details for baseline arguments for mne.Epochs()
    resample : None or float, default = None
        convert sampling frequency. No resampling will be applied if None.
    event_names : None or dict, default = None
        name of each event.
        e.g., if {'A':['1', '101'], 'B': ['2', '102']}, event '1' and '101' will be tagged as 'A', and event '2' and '102' will be tagged as 'B'. 'misc' will be tagged if there's no matched element and if not None.
    marker_trial : None or list
        event id which indecates start of the trial.
        e.g., if [str(val) for val in range(201, 300)], epochs between event '201' to '299' will be tagged as different trial.
    marker_tnt : None or dict
        event id for target and nontarget event.
        dict should have the following structure. {'target': list(), 'nontarget':list()}
        e.g., if {'target': [str(val) for val in range(101, 200)], 'nontarget': [str(val) for val in range(1, 100)]}. events '101' to '199' will be tagged as target and events '1' to '99' will be tagged as nontarget
    add_run : Bool, default = False
        run number will be tagged if True.
        if True, raw.info['description'] should include run tag, e.g., raw.info['description'] = 'run:1'
    remove_misc : Bool, default = True
        remove misc event which was not specified with event_names parameter.
    """

    def __init__(self,
                 l_freq = 0.1,
                 h_freq = 8,
                 filter_params = {'method': 'iir', 'phase': 'zero', 'iir_params':{'order':2, 'ftype':'butter'}, 'n_jobs': -1},
                 tmin = -0.1,
                 tmax = 1.2,
                 baseline = None,
                 resample = None,
                 event_names = None,
                 marker_trial = None,
                 marker_tnt = None,
                 add_run = False,
                 remove_misc = True):
        self.l_freq = l_freq 
        self.h_freq = h_freq
        self.filter_params = filter_params
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.resample = resample
        
        self.event_names = event_names
        self.marker_trial = marker_trial
        self.marker_tnt = marker_tnt
        self.add_run = add_run
        self.remove_misc = remove_misc
        
    def fit(self, X, y = None):
        """fit"""
        return self
    
    def transform(self, X):
        """transform"""
        if type(X) != list:
            raise RuntimeError("type(X) should be list.")

        epochs_list = list()
        for x in X:
            
            if self.filter_params is None:
                x.filter(l_freq = self.l_freq, h_freq = self.h_freq) 
            else:
                x.filter(l_freq = self.l_freq, h_freq = self.h_freq, **self.filter_params)

            events, event_id = mne.events_from_annotations(x) 
            samples, markers = tm.markers_from_events(events, event_id)
            
            if self.event_names is not None:
                markers = tm.add_event_names(markers, self.event_names)
            if self.add_run:
                descs = x.info['description']
                for desc in descs.split("/"):
                    if 'run:' in desc:
                        run = desc.split(":")[1]
                markers = tm.add_tag(markers, "run:%d"%(int(run)))
            if self.marker_trial is not None:
                markers = tm.split_trials(markers, trial = self.marker_trial)
            if self.marker_tnt is not None:
                markers = tm.add_tnt(markers, target = self.marker_tnt['target'], nontarget = self.marker_tnt['nontarget'])
            if self.remove_misc:
                samples, markers = tm.remove(samples, markers, "misc")
            
            events, event_id = tm.events_from_markers(samples, markers)

            epochs = mne.Epochs(raw = x,
                                events = events,
                                tmin = self.tmin,
                                tmax = self.tmax,
                                baseline = self.baseline,
                                event_id = event_id)
            
            epochs_list.append(epochs)
        epochs = tm.concatenate_epochs(epochs_list)
        
        if self.resample is not None:
            epochs.resample(self.resample, n_jobs = -1)
        
        return epochs