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
    def __init__(self, l_freq = 1.0, len_transition = 0.5):
        self.l_freq = l_freq
        self.len_transition = len_transition
        self.ica = None
        self.scores = None
        self.exclude = None

    def find_bad_eog(self, raw, ica, h_freq = 10, threshold = 0.9):
        """
        Parameters
        ==========

        raw : raw instance contains eog channels.
        ica : ica instance
        filter : filter range will be used for eog channels
        threshold, numerical or 'max': 
        
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
                #data_ic = ica.get_data(picks = ic)
            
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

class ExtractERP(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
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
                 add_run = True,
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
        return self
    
    def transform(self, X):
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