# 用于生成互斥的5标签数据，保存为label_diag_superclass_mutual，数据量不一样，保存为新的文件

import sys 
# sys.path.append("./code")
sys.path.append("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/code")
sys.path.append("D:/Works/FW/WorkNoteLog/2407/paper10/")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os
from os.path import isdir


if __name__ == '__main__':

    target_fs=100
    # data_root=Path("./data/")
    # target_root=Path("./data")

    data_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/data/")
    target_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/data")

    if not isdir(data_root):
        os.makedirs(data_root)
    if not isdir(target_root):
        os.makedirs(target_root)
    
    data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
    target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs)+"_mutual")  #还是重新来个新的文件夹，这样不用改里面的东西
    # 得先把df等文件copy到目标文件

    # 已经复制完了
    # # 复制整个文件夹，包括所有子文件夹和文件
    # import shutil
    # import os

    # # 源文件夹路径
    # src_folder = target_root/("ptb_xl_fs"+str(target_fs))
    # # 目标文件夹路径
    # dest_folder = target_folder_ptb_xl


    # try:
    #     # 检查目标文件夹是否存在，如果存在则删除
    #     if os.path.exists(dest_folder):
    #         shutil.rmtree(dest_folder)
    #     shutil.copytree(src_folder, dest_folder)
    #     print(f'Copied {src_folder} to {dest_folder}')
    # except FileExistsError:
    #     print(f'Error: Destination folder {dest_folder} already exists')
    
    df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                           channels=12, channel_stoi=channel_stoi_default, 
                                                                           target_folder=target_folder_ptb_xl, recreate_data=False)
    
    # 500hz的npy保存在fs500里，但
    # 在这里，把多标签的删去，仅仅保留mutual的数据
    # print(df_ptb_xl['label_diag_superclass'].shape)

    
    # 先对一下总的，还未多标签互斥的时候

    # 重新对df_ptb_xl superclass生成五标签的label看看，

    df_ptb_xl['label_diag_superclass'] = df_ptb_xl['label_diag_superclass'].apply(lambda x: list(set(x)))
    df_ptb_xl = df_ptb_xl[df_ptb_xl['label_diag_superclass'].apply(lambda x: len(x)<=1)]
                
    def val_count(df):
        # 计算各分类的个数
        value_counts = df['label_diag_superclass'].value_counts().to_frame()
        print(value_counts)

    df_train = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) <= 8)]
    df_val = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) == 9)]
    df_test = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) == 10)]

    print('train value counts:')
    val_count(df_train)
    print('val value counts:')
    val_count(df_val)
    print('test value counts:')
    val_count(df_test)

    # reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=False)
