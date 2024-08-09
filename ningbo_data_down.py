import sys 
sys.path.append("./code")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os
from os.path import isdir
import subprocess
import wget
import shutil
import pdb

data_root=Path("./data/")
data_folder_ptb_xl=Path("./data/ningbo")

def download(data_url, dataset_dir):
    filename = wget.download(data_url, out=str(data_root))
    shutil.unpack_archive(str(filename), dataset_dir)
    # os.remove(filename)

ptb_xl_url = 'https://physionet.org/files/challenge-2021/1.0.3/'

download(ptb_xl_url, data_folder_ptb_xl)