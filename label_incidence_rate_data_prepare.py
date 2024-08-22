# 生成发生率大于0.5%的label的数据， min_cnt= 21837*0.5=110

import sys 
sys.path.append("./code")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os
from os.path import isdir

if __name__ == '__main__':
    

    def data_prepare(target_fs):
        data_root=Path("./data/")
        target_root=Path("./data")

        if not isdir(data_root):
            os.makedirs(data_root)
        if not isdir(target_root):
            os.makedirs(target_root)
        
        data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
        target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs)+"0p5")  # 
        
        df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=110, target_fs=target_fs, 
                                                                            channels=12, channel_stoi=channel_stoi_default, 
                                                                            target_folder=target_folder_ptb_xl, recreate_data=True)
        reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=False)

    data_prepare(target_fs=100)
    data_prepare(target_fs=500)