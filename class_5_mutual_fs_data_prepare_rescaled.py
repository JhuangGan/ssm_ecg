# 用于生成互斥的5标签数据，保存为label_diag_superclass_mutual，数据量不一样，保存为新的文件

import sys 
sys.path.append("./code")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os
from os.path import isdir
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--fs", type=str, default='')

    args = parser.parse_args()

    target_fs=args.fs
    
    data_root=Path("./data/")
    target_root=Path("./data")

    # data_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/data/")
    # target_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/data")

    if not isdir(data_root):
        os.makedirs(data_root)
    if not isdir(target_root):
        os.makedirs(target_root)
    
    data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
    target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs)+"_mutual_rescaled")  #还是重新来个新的文件夹，这样不用改里面的东西
    # 得先把df等文件copy到目标文件
    # 在linux里操作 mkdir ./data/ptb_xl_fs100_mutual
    # cp -r ./data/ptb_xl_fs100/* ./data/ptb_xl_fs100_mutual/

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
    
    # 不需要重新创造数据，直接读取即可
    df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                           channels=12, channel_stoi=channel_stoi_default, 
                                                                           target_folder=target_folder_ptb_xl, recreate_data=True, rescaled=1000)
    
    
    # 对原五分类的，去重，然后仅保留互斥的数据
    df_ptb_xl['label_diag_superclass'] = df_ptb_xl['label_diag_superclass'].apply(lambda x: list(set(x)))
    df_ptb_xl = df_ptb_xl[df_ptb_xl['label_diag_superclass'].apply(lambda x: len(x)<=1)]  ## 仅保留五标签互斥的数据
                
    # def val_count(df):
    #     # 计算各分类的个数
    #     value_counts = df['label_diag_superclass'].value_counts().to_frame()
    #     print(value_counts)

    # df_train = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) <= 8)]
    # df_val = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) == 9)]
    # df_test = df_ptb_xl[df_ptb_xl['strat_fold'].apply(lambda x: int(x) == 10)]

    # print('train value counts:')
    # val_count(df_train)
    # print('val value counts:')
    # val_count(df_val)
    # print('test value counts:')
    # val_count(df_test)

    # 在mutual文件夹下，创建对应所需df与memmap文件
    reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=False)
