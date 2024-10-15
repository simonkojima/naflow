import os
import re

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    return dir

def sort_list(data):
    return sorted(data, key=natural_keys)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def remove_from_list(data, remove):
    return [val for val in data if val not in remove]

def invert_dict(dict):
    return {value: key for key, value in dict.items()}