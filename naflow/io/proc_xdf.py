import numpy as np
import pyxdf
import mne

class numpy_data():
    def __init__(self, data, ch_names, times, fs):
        self.data = data
        self.ch_names = ch_names
        self.times = times
        self.fs = fs


def get_stream_names_xdf(fname, print_name = True):
    streams, header = pyxdf.load_xdf(fname)
    
    names = list()
    for idx, stream in enumerate(streams):
        name = stream['info']['name'][0]
        names.append(name)
        if print_name:
            print("Name of Stream %d: %s"%(idx, name))
            
    return names

def read_numpy_xdf(fname):
    streams, header = pyxdf.load_xdf(fname)

    data = dict()
    for idx, stream in enumerate(streams):
        name = stream['info']['name'][0]
        
        try:
            ch_names = stream['info']['desc'][0]['channels'][0]['channel']
        except:
            ch_names = None

        fs = float(stream['info']['nominal_srate'][0])

        data[name] = numpy_data(data = stream['time_series'],
                                times = stream['time_stamps'],
                                ch_names = ch_names,
                                fs = fs)

        
    return data
    

def read_raw_xdf(fname, name_eeg_stream, name_marker_stream, channel_type = None):
    """
    Parameters
    ----------
    
    channel_type: None or dict, default None
        e.g., `{'eog': ['vEOG', 'hEOG'], 'ecg': ['ECG']}`
    """
    
    streams, header = pyxdf.load_xdf(fname)
    
    eeg = None
    marker = None

    for stream in streams:
        name = stream['info']['name'][0]

        if name == name_eeg_stream:
            eeg = stream
        elif name == name_marker_stream:
            marker = stream
    
    if eeg is None:
        raise ValueError("'%s' was not found"%(name_eeg_stream))

    if marker is None:
        raise ValueError("'%s' was not found"%(name_marker_stream))
    
    data = eeg['time_series'].T.astype(np.float64)
    times = eeg['time_stamps']

    events = marker['time_series']
    mrk_times = marker['time_stamps']
    
    events = [int(val[0]) for val in events]
    
    times = np.array(times)
    mrk_times = np.array(mrk_times)
    
    events_mne = list()
    for idx, event in enumerate(events):
        diff = np.abs(times - mrk_times[idx])
        I = np.argmin(diff)
        
        events_mne.append([I, 0, event])
        
    events = np.array(events_mne)
    channels = eeg['info']['desc'][0]['channels'][0]['channel']
    mne_ch_types = mne.io.get_channel_type_constants(True)
    ch_names = [ch['label'][0] for ch in channels]

    ch_types = list()
    for ch in channels:
        if ch['type'] and ch['type'][0].lower() in mne_ch_types:
            ch_types.append(ch['type'][0].lower())
        else:
            ch_types.append('misc')
    
    if channel_type is not None:
        for idx, ch in enumerate(ch_names):
            for key, value in channel_type.items():
                if ch in value:
                    ch_types[idx] = key.lower()

    sfreq = float(eeg['info']['nominal_srate'][0])
    units = [ch['unit'][0] for ch in channels]
    
    for idx, unit in enumerate(units):
        match unit:
            case 'microvolts':
                data[idx, :] *= 1e-6
            case _:
                raise RuntimeError("unit '%s' is unknown"%str(unit))

    info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
    raw = mne.io.BaseRaw(preload = data, info = info, filenames = [fname])
    
    event_id = dict()
    for event in events:
        event = event[2]
        event_id[str(event)] = event
    
    return raw, events, event_id