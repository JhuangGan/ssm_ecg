### 用于将内网数据转成npy的同时，不通过memmap，转换为Timeseries类
### 最后返回dataloader

import sys 
sys.path.append("./code")
# from clinical_ts.timeseries_utils import *
# from clinical_ts.ecg_utils import *  ##
from pathlib import Path
import os
from os.path import isdir

#### 新增
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
import resampy

try:
    import pickle5 as pickle
except ImportError as e:
    import pickle


channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}


def map_and_filter_labels(df,min_cnt,lbl_cols):
    #filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in lbl_cols:
        if(min_cnt>0):
            label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
            df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
            lbl_itos_ptb_xl[selection+"_filtered"] = np.array(sorted(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist]))))
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection+"_filtered"])}
            df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
        #also lbl_itos and ..._numeric col for original label column
        lbl_itos_ptb_xl[selection]= np.array(sorted(list(set([x for sublist in df_ptb_xl[selection] for x in sublist]))))
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection+"_numeric"]=df_ptb_xl[selection].apply(lambda x:[lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):#,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs  # fs是500hz
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                #if(skimage_transform):
                #    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                #else:
                #    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
                data[:,channel_stoi[cl]] = resampy.resample(sigbufs[:,i], fs, target_fs).astype(np.float32)
    else:
        #if(skimage_transform):
        #    data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        #else:
        #    data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data



def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))


def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))


def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x, allow_pickle=True)))


def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)


def save_dataset(df,lbl_itos,mean,std,target_root,filename_postfix="",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)

    if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:#array
        np.save(target_root/("lbl_itos"+filename_postfix+".npy"),lbl_itos)

    np.save(target_root/("mean"+filename_postfix+".npy"),mean)
    np.save(target_root/("std"+filename_postfix+".npy"),std)

def load_dataset(target_root,filename_postfix="",df_mapped=True):
    target_root = Path(target_root)

    if(df_mapped):
        df = pickle.load(open(target_root/("df_memmap"+filename_postfix+".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root/("df"+filename_postfix+".pkl"), "rb"))


    if((target_root/("lbl_itos"+filename_postfix+".pkl")).exists()):#dict as pickle
        infile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "rb")
        lbl_itos=pickle.load(infile)
        infile.close()
    else:#array
        lbl_itos = np.load(target_root/("lbl_itos"+filename_postfix+".npy"))


    mean = np.load(target_root/("mean"+filename_postfix+".npy"))
    std = np.load(target_root/("std"+filename_postfix+".npy"))
    return df, lbl_itos, mean, std



def prepare_data_ptb_xl(data_path, min_cnt=10, target_fs=100, channels=12, channel_stoi=channel_stoi_default, target_folder=None, recreate_data=True,rescaled=1):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        ptb_xl_csv = data_path/"ptbxl_database.csv"
        df_ptb_xl=pd.read_csv(ptb_xl_csv,index_col="ecg_id")
        #print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path/"scp_statements.csv")
        ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic >0]
        ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form >0]
        ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm >0]

        diag_class_mapping={}
        diag_subclass_mapping={}
        for id,row in ptb_xl_label_diag.iterrows():
            if(isinstance(row["diagnostic_class"],str)):
                diag_class_mapping[id]=row["diagnostic_class"]
            if(isinstance(row["diagnostic_subclass"],str)):
                diag_subclass_mapping[id]=row["diagnostic_subclass"]

        df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"]="ptb_xl"
        #filter and map (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl =map_and_filter_labels(df_ptb_xl,min_cnt=min_cnt,lbl_cols=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"])

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            # always start from 500Hz and sample down
            filename = data_path/row["filename_hr"] #data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root_ptb_xl/(filename.stem+".npy"),data*rescaled)
            filenames.append(Path(filename.stem+".npy"))
        df_ptb_xl["data"] = filenames

        #add means and std
        dataset_add_mean_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        #save
        save_dataset(df_ptb_xl,lbl_itos_ptb_xl,mean_ptb_xl,std_ptb_xl,target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl







def get_dataset(batch_size, num_workers, target_folder, test=False, label_class="label_all", 
                input_size=250, shuffle_train=False, nomemmap=False, test_folds=[8, 9], 
                combination='both', filter_label=None, val_stride=None, normalize=False, use_meta_information_in_head=False):
    
    dataset = ECGDataSetWrapper(
        batch_size, num_workers, target_folder, test=test, label=label_class, 
        input_size=input_size, shuffle_train=shuffle_train, nomemmap=nomemmap,
        normalize=normalize, test_folds=test_folds, combination=combination,
        filter_label=filter_label, val_stride=val_stride, use_meta_information_in_head=use_meta_information_in_head)

    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader



def train_dataloader(self):
    _, train_loader, _ = get_dataset(self.batch_size, self.num_workers, self.target_folder, label_class=self.label_class,
                                        input_size=self.data_input_size, shuffle_train=self.shuffle_train, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head)
    return train_loader

def val_dataloader(self):
    _, _, valid_loader = get_dataset(self.batch_size, self.num_workers, self.target_folder, label_class=self.label_class,
                                        input_size=self.data_input_size, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, val_stride=self.val_stride, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head)
    return valid_loader

def test_dataloader(self):
    dataset, _, valid_loader = get_dataset(self.batch_size, self.num_workers, self.target_folder, test=True,
                                        label_class=self.label_class, input_size=self.data_input_size, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head)
    self.test_idmap = dataset.val_ds_idmap
    return valid_loader






if __name__ == '__main__':
    target_fs=500
    data_root=Path("./data/")
    target_root=Path("./data")

    if not isdir(data_root):
        os.makedirs(data_root)
    if not isdir(target_root):
        os.makedirs(target_root)
    
    data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
    target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs)+"_npydata")  # 
    
    df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                           channels=12, channel_stoi=channel_stoi_default, 
                                                                           target_folder=target_folder_ptb_xl, recreate_data=True, rescaled=1)

    # 首先，将dataset.pkl 生成对应的df，与datafold
    # 貌似不需要，操作，可以直接npy文件输入，那就变成，处理上一步的结果，为npy就可以了。





