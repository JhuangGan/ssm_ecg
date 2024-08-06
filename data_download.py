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

target_fs=100
data_root=Path("./data/")
target_root=Path("./data")
if not isdir(data_root):
    os.makedirs(data_root)
if not isdir(target_root):
    os.makedirs(target_root)

def download(data_url, dataset_dir, flatten=True):
    filename = wget.download(data_url, out=str(data_root))
    shutil.unpack_archive(str(filename), dataset_dir)
    if flatten:
        source = str(dataset_dir/os.listdir(dataset_dir)[0])
        destination = str(dataset_dir)
        files = os.listdir(source)
        for file in files:
            file_name = os.path.join(source, file)
            shutil.move(file_name, destination)
        os.rmdir(source)
    # os.remove(filename)

if __name__ == '__main__':
    data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
    target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs))  # 
    # ptb_xl_url='https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip'
    # 直接替换链接就可以
    ptb_xl_url='https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip'
    
    download(ptb_xl_url, data_folder_ptb_xl)
    
