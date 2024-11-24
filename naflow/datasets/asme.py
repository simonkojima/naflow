import os
import mne
import tag_mne as tm

import naflow.utils
import naflow.io

try:
    import tomllib
except:
    import toml as tomllib

def asme_4class_frontiers():
    #url =
    pass

def extract_epochs_asme_speller_30chars_copychar(files,
                                                 subject,
                                                 l_freq = 1.0,
                                                 h_freq = 40.0,
                                                 tmin = -0.1,
                                                 tmax = 1.0,
                                                 baseline = None,
                                                 resample = 250,
                                                 online = False):
    from . import common

    name_datasets = "asme_speller_30chars_copychar"
    eog_channels = ['vEOG', 'hEOG']

    fname_config = os.path.join(common.naflow_data_base, name_datasets, "config.toml")
    try:
        with open(fname_config, "r") as f:
            config = tomllib.load(f)   
    except:
        with open(fname_config, "rb") as f:
            config = tomllib.load(f)   

    epochs_list = list()
    for file in files:
        raw, events, event_id = naflow.io.read_raw_xdf(os.path.join(common.naflow_data_base,
                                                                    name_datasets,
                                                                    "sub-%s"%subject,
                                                                    "ses-S001",
                                                                    "eeg",
                                                                    file),
                                                       name_eeg_stream = "BrainAmpSeries",
                                                       name_marker_stream = "scab-c",
                                                       channel_type={'eog': eog_channels})
        raw = raw.pick(picks = config['channels']['64_eog'])
        
        raw.filter(l_freq = l_freq,
                   h_freq = h_freq,
                   method = 'iir',
                   iir_params = {"order": 2, "ftype": 'butter', "btype": 'bandpass'},
                   n_jobs = -1)
        
        samples, markers = tm.markers_from_events(events, event_id)

        markers = tm.add_tag(markers, "subject:sub-%s"%subject)

        markers = tm.add_event_names(markers, config['events'])
        
        markers = tm.add_tag(markers, "run:%d"%(int(common.get_run(file))))

        if online is False:
            markers = tm.split_trials(markers, trial = [str(val) for val in range(201, 300)])
            markers = tm.add_tnt(markers, target = [str(val) for val in range(101, 200)], nontarget = [str(val) for val in range(1, 100)] )
        else:
            markers = tm.split_trials(markers, trial = ['200'])
    
        samples, markers = tm.remove(samples, markers, "event:misc")
    
        events, event_id = tm.events_from_markers(samples, markers)

        epochs = mne.Epochs(raw = raw,
                            events = events,
                            event_id = event_id,
                            tmin = tmin,
                            tmax = tmax,
                            baseline = baseline)
    
        epochs_list.append(epochs)
        
    return epochs_list

def asme_speller_30chars_copychar(subject,
                                  origin_base = None,
                                  force_redownload = False,
                                  l_freq = 1.0,
                                  h_freq = 40.0,
                                  tmin = -0.1,
                                  tmax = 1.0,
                                  baseline = None,
                                  resample = None):
    from . import common
    
    name_datasets = "asme_speller_30chars_copychar"

    isExist = os.path.exists(os.path.join(common.naflow_data_base, name_datasets))
    
    if not isExist or force_redownload:
        if origin_base is not None:
            common.cp_local_datasets(origin_base = origin_base,
                                     dir_name = name_datasets)
        else:
            raise ValueError("Please specify origin_base")
        
    
    offline_fname = "sub-%s_offline_lfreq-%s_hfreq-%s_tmin-%s_tmax-%s_baseline-%s_resample-%s-epo.fif"%(subject, str(l_freq), str(h_freq), str(tmin), str(tmax), str(baseline), str(resample))
    online_fname = "sub-%s_online_lfreq-%s_hfreq-%s_tmin-%s_tmax-%s_baseline-%s_resample-%s-epo.fif"%(subject, str(l_freq), str(h_freq), str(tmin), str(tmax), str(baseline), str(resample))
    
    offlineExist = os.path.exists(os.path.join(common.naflow_data_base, name_datasets, "epochs", offline_fname))
    onlineExist = os.path.exists(os.path.join(common.naflow_data_base, name_datasets, "epochs", online_fname))
    
    if offlineExist and onlineExist:
        offline_epochs = mne.read_epochs(os.path.join(common.naflow_data_base, name_datasets, "epochs", offline_fname))
        online_epochs = mne.read_epochs(os.path.join(common.naflow_data_base, name_datasets, "epochs", online_fname))
        
        return offline_epochs, online_epochs

    fname_config = os.path.join(common.naflow_data_base, name_datasets, "config.toml")
    try:
        with open(fname_config, "r") as f:
            config = tomllib.load(f)   
    except:
        with open(fname_config, "rb") as f:
            config = tomllib.load(f)   
    
    #subjects = config['subjects']['list']


    offline_epochs = list()
    files = os.listdir(os.path.join(common.naflow_data_base, name_datasets, "sub-%s"%subject, "ses-S001", "eeg"))
    offline_files = list()
    online_files = list()
    
    for file in files:
        if "asmeoffline" in file:
            offline_files.append(file)
        if "asmeOnlineCopyChar" in file:
            online_files.append(file)
    
    offline_files = naflow.utils.sort_list(offline_files)
    online_files = naflow.utils.sort_list(online_files)
    
    print(offline_files)
    offline_epochs = extract_epochs_asme_speller_30chars_copychar(files = offline_files,
                                                                  subject = subject,
                                                                 online = False,
                                                                 l_freq = l_freq,
                                                                 h_freq = h_freq,
                                                                 tmin = tmin,
                                                                 tmax = tmax,
                                                                 baseline = baseline)

    print("start concatenating...")
    offline_epochs = tm.concatenate_epochs(offline_epochs)
    print(offline_epochs['subject:sub-A/run:2/trial:1/target'])
    
    if resample is not None:
        offline_epochs.resample(resample)

    print(online_files)
    online_epochs = extract_epochs_asme_speller_30chars_copychar(files = online_files,
                                                                 subject = subject,
                                                                 online = True,
                                                                 l_freq = l_freq,
                                                                 h_freq = h_freq,
                                                                 tmin = tmin,
                                                                 tmax = tmax,
                                                                 baseline = baseline)

    print("start concatenating...")
    online_epochs = tm.concatenate_epochs(online_epochs)

    if resample is not None:
        online_epochs.resample(resample)
        
    naflow.utils.mkdir(os.path.join(common.naflow_data_base,
                                    name_datasets,
                                    "epochs"))

    fname = "sub-%s_offline_lfreq-%s_hfreq-%s_tmin-%s_tmax-%s_baseline-%s_resample-%s-epo.fif"%(subject, str(l_freq), str(h_freq), str(tmin), str(tmax), str(baseline), str(resample))
    offline_epochs.save(os.path.join(common.naflow_data_base,
                                     name_datasets,
                                     "epochs",
                                     fname),
                        overwrite = True)

    fname = "sub-%s_online_lfreq-%s_hfreq-%s_tmin-%s_tmax-%s_baseline-%s_resample-%s-epo.fif"%(subject, str(l_freq), str(h_freq), str(tmin), str(tmax), str(baseline), str(resample))
    online_epochs.save(os.path.join(common.naflow_data_base,
                                    name_datasets,
                                    "epochs",
                                    fname),
                       overwrite = True)
    
    return offline_epochs, online_epochs




    
