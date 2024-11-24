import os
import shutil

import naflow.utils

naflow_data_base = os.path.join(os.path.expanduser('~'), ".naflow", "datasets")

class GetERPDatasets():
    def __init__(self,
                 l_freq = 1,
                 h_freq = 40,
                 tmin = -0.1,
                 tmax = 1.0,
                 baseline = None):
        pass
    

def download_datasets(url, save_base):
    naflow.utils.mkdir(naflow_data_base)

def cp_local_datasets(origin_base, dir_name):
    naflow.utils.mkdir(naflow_data_base)
    shutil.copytree(origin_base, os.path.join(naflow_data_base, dir_name))
    
def get_run(file):
    parts = file.split("_")
    for part in parts:
        if "run-" in part:
            break
    run = part.split("-")[1]
    return run
    
    


