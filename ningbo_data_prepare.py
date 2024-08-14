# 用于处理宁波数据为可用的df.pkl, lbl_itos.pkl, mean.npy, std.npy, df_mapped.pkl, memmap.npy, memmap.npz

# ningbo_data_save_pth运行后，
# 数据存放于./data/Ningbo_pth, ./data/Ningbo_pth_label

import sys
sys.path.append("./code")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os

from os.path import isdir
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch 
import resampy

def prepare_data_ningbo(data_path, min_cnt=10, target_fs=100, channels=12, target_folder=None, recreate_data=True):
    # data_path = Path(data_path)
    target_root_ningbo = Path(".") if target_folder is None else target_folder
    
    target_root_ningbo.mkdir(parents=True, exist_ok=True)
    
    
    filenames = os.listdir(data_path)


    if(recreate_data is True):

        df_ningbo = {'pid':[], 'age':[], 'sex':[], 'label':[], 'data':[], 'dataset':[], 'strat_fold':[]}

        for filename in tqdm(filenames):
            # data_path='D:/Works/FW/WorkNoteLog/2407/paper10/gitclone_repo/ssm_ecg/data/Ningbo_pth'
            
            file = torch.load(data_path/filename, weights_only=False)
            
            
            data = file['image']  ## 仅仅保存ecg的部分

            # 横纵相反
            data = np.array([[row[i] for row in data] for i in range(len(data[0]))])
            df_ningbo['age'].append(file['age'])
            df_ningbo['sex'].append(file['sex'])
            df_ningbo['pid'].append(file['pid'])
            df_ningbo['label'].append(file['label'])

            df_ningbo['dataset'].append('ningbo')
            df_ningbo['data'].append(filename+".npy")
            df_ningbo['strat_fold'].append(-1)  ## 全部设为-1无标签

            # 降采样部分
            resampled_data = resampy.resample(data, 500, target_fs, axis=1)

            # print(filename)
            # 转成npy，仅仅保存ecg的部分
            np.save(target_root_ningbo/(filename+".npy"), resampled_data)
            # os.chdir(target_folder_ningbo)
            # print(os.getcwd())
            # np.save('./'+filename+'.npy', data)

        df_ningbo =pd.DataFrame(df_ningbo)

        df_ningbo, lbl_itos =map_and_filter_labels(df_ningbo,min_cnt=1,lbl_cols=["label"])

        # 要保存的位置
        # target_root_ningbo = 'D:/Works/FW/WorkNoteLog/2407/paper10/gitclone_repo/ssm_ecg/Ningbo_PTBXL_CS_Ga_CPSC_pth/'

            #add means and std
        dataset_add_mean_col(df_ningbo,data_folder=target_root_ningbo)
        dataset_add_std_col(df_ningbo,data_folder=target_root_ningbo)
        dataset_add_length_col(df_ningbo,data_folder=target_root_ningbo)

        #save means and stds
        mean_sph, std_sph = dataset_get_stats(df_ningbo)

        #save 保存为df.pkl, lbl_itos.pkl, mean.npy, std.npy
        save_dataset(df_ningbo,lbl_itos,mean_sph,std_sph,target_root_ningbo)

    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ningbo,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl




if __name__ == '__main__':
    
    def ningbo_data(target_fs=500):
        target_fs=target_fs 

        data_root=Path("./data/")
        target_root=Path("./data")
        # data_root=os.path.join(os.getcwd(), 'data')
        # target_root=os.path.join(os.getcwd(), 'data')

        if not isdir(data_root):
            os.makedirs(data_root)
        if not isdir(target_root):
            os.makedirs(target_root)
        
        data_folder_ningbo = data_root/'Ningbo_pth/' # 数据存放文件夹
        target_folder_ningbo = target_root/('Ningbo_npys_fs'+str(target_fs))  # npys 存放文件夹

        if not isdir(target_folder_ningbo):
            os.makedirs(target_folder_ningbo)


        # 生成所需列，转成降采样，并转成npys
        df_ptb_ningbo, lbl_itos_ningbo,  mean_ptb_ningbo, std_ptb_ningbo = prepare_data_ningbo(data_folder_ningbo, min_cnt=1, target_fs=target_fs,
                                                                            channels=12, 
                                                                            target_folder=target_folder_ningbo, recreate_data=True)
        
        reformat_as_memmap(df_ptb_ningbo, target_folder_ningbo/("memmap.npy"),data_folder=target_folder_ningbo,delete_npys=False)


    ningbo_data(target_fs=500)

