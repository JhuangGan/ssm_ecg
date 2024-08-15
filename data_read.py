import os
import sys
sys.path.append("./code")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from os.path import isdir
from pathlib import Path


if __name__ == '__main__':

    target_root=Path("./data")
    if not isdir(target_root):
        os.makedirs(target_root)

    target_folder_ptb_xl = target_root/("Ningbo_npys_fs500")
    
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)

    print(df_ptb_xl)
    # print(lbl_itos_ptb_xl)
    # print(mean_ptb_xl)
    # print(std_ptb_xl)

    # df_memmap, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=True)
    # print(df_memmap.iloc[1:4, -6:-1])

    # memap_meta.npy文件这个文件怎么来的呢？

    # memap_meta.npz文件读取，看看npz到底是什么样子
    # import numpy as np
    # # import os
    # # current_path = os.getcwd()
    # # print(current_path)
    # memmap_meta_npz = np.load('./data/ptb_xl_fs100/memmap_meta.npz')
    # for i, memmap in enumerate(memmap_meta_npz):
    #     print(f'{i},{memmap}')
    #     if i<4:
    #         print(memmap_meta_npz[str(memmap)])
    # # print(memmap_meta_npz['file_idx'])
    



