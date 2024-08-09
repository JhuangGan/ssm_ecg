import sys 
sys.path.append("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main/code")
sys.path.append("D:/Works/FW/WorkNoteLog/2407/paper10/")
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import os
from os.path import isdir
import subprocess
import wget
import shutil
import pdb
from tqdm import tqdm
import pandas as pd
import numpy as np
import wfdb


target_fs=100
data_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main//data/")
target_root=Path("D:/Works/FW/WorkNoteLog/2407/paper10/ssm_ecg-main/ssm_ecg-main//data")
if not isdir(data_root):
    os.makedirs(data_root)
if not isdir(target_root):
    os.makedirs(target_root)

def prepare_data_ptb_xl(data_path, min_cnt=10, target_fs=100, channels=12, channel_stoi=channel_stoi_default, target_folder=None, recreate_data=True):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        ptb_xl_csv = data_path/"ptbxl_database.csv"
        df_ptb_xl=pd.read_csv(ptb_xl_csv,index_col="ecg_id")
        #print(df_ptb_xl.columns)
        # 将scp_codes列中所有的字符串 "nan" 替换为NumPy的np.nan
        df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))  

        # preparing labels 
        ptb_xl_label_df = pd.read_csv(data_path/"scp_statements.csv")
        ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])  # 将各种描述的缩写定为index
        # 提取有值的地方，分别提取诊断，波，节律的df
        ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic >0]
        ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form >0]
        ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm >0]

        # 将每个ecg的index为key，diagnostic_class为value保存为字典diag_class_mapping
        # 将每个ecg的index为key，diagnostic_subclass为value保存为字典diag_subclass_mapping
        diag_class_mapping={}
        diag_subclass_mapping={}
        for id,row in ptb_xl_label_diag.iterrows():
            # 分别将对应的str保存起来，有些是空的
            if(isinstance(row["diagnostic_class"],str)):
                diag_class_mapping[id]=row["diagnostic_class"]
            if(isinstance(row["diagnostic_subclass"],str)):
                diag_subclass_mapping[id]=row["diagnostic_subclass"]

        # scp_codes是一个字典，key是scp-ecg语句，value是可能性
        df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()]) ## 保存所有的scp的key也就是所有的判断，并作为list保存
        df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])   ## 保存出现的诊断，
        df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])  ## 保存出现的form
        df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])  ## 保存出现的rhythm节律

        # 将刚生成的label_diag的列，通过将label_diag与前面刚生成的diag_subclass_mapping和diag_class_mapping进行对照，然后取出所有的diag_sub_class和diag_class作为sub_class和super_class列
        df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])
        
        # 增加一个数据标识的列
        df_ptb_xl["dataset"]="ptb_xl"
        
        # 
        #filter and map (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl =map_and_filter_labels(df_ptb_xl,min_cnt=min_cnt,lbl_cols=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"])
        # 获取增加筛选（超过min_cnt）和未筛选后的label的进一步的元素拆分，每个为一个list，与key为label，value为所有去重元素的np.array的字典

        filenames = []  # 用以保存所有的filename的list
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            # 对所有ecg文件降赫兹
            # always start from 500Hz and sample down
            # filename_hr 是records500/00000/00001_hr 类似这种的，records500/00000/00999_hr， 然后每一1000个为一个文件夹，
            # 总共21w个数据，22个文件夹，然后这个就是其路径及文件名
            filename = data_path/row["filename_hr"] #data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))  ## 读取数据 hea和dat一起读，500hz的话是5000*12导联的数据，
            # 重采样，将500hz变100hz，即将5000个数据，重采样为1000个数据
            # 'sig_name':导联名，fs赫兹数，
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])

            # 保存为对应的npy文件
            np.save(target_root_ptb_xl/(filename.stem+".npy"),data)  # stem获取文件名的茎（不包含后缀）
            filenames.append(Path(filename.stem+".npy"))
        df_ptb_xl["data"] = filenames  # 把对应的文件名保存为df的列

        #add means and std
        # 得到每个文件的mean和std和length，共12个导联，所以n*12
        dataset_add_mean_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        # 得到所有文件的mean和std
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        #save
        # 把df_ptb_xl保存为df.pkl,lbl_itos_ptb_xl(key为label，包含未筛选min_cnt和筛选的，value为对应去重的值的list)保存为lbl_itos.pkl
        # 将mean_ptb_xl,std_ptb_xl保存为mean.npy,std.npy
        save_dataset(df_ptb_xl,lbl_itos_ptb_xl,mean_ptb_xl,std_ptb_xl,target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl

def map_and_filter_labels(df,min_cnt,lbl_cols):
    # lbl_cols=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"]
    # 对于df里的这些列都做操作，先得到出现个数多于mincnt的label，然后赋予label_filtered的列，
    #filter labels
    def select_labels(labels, min_cnt=10):
        # 将labels两层展开，并取unique，并得到各个value对应的count，然后返回出现个数多于mincnt的value
        # 这个两层展开，对于一个df的lie来说，就是该列的所有单元，并且该单元里的元素再展开，即该列所有单元的各种value值出现的次数的统计
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    df_ptb_xl = df.copy()
    ## 用于保存，未经过筛选和经过筛选的label的所有元素的list为value，key分别为label和label_filtered.
    ## 对于label_all, label_diag,label_form, label_rhyth,label_diag_subclass, label_diag_superclass都做这个操作，保存上述字典里
    lbl_itos_ptb_xl = {}  # itos究竟是什么的缩写呢？还有stoi（一个数字编码字典）
    for selection in lbl_cols:
        if(min_cnt>0):
            # 得到该列的所有单元格的单个元素的出现个数超过min_cnt的元素label
            label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
            # 生成超过min_cnt个数的元素label的列，为selection_filtered
            df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
            # 将新生成的selection_filtered的列里的所有元素展开，并去重排序，作为value，key为selection_filtered的str，保存在字典lbl_itos_ptb_xl里
            lbl_itos_ptb_xl[selection+"_filtered"] = np.array(sorted(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist]))))
            # 将上面的np.array生成一个元素和index对应的dic，lbl_stoi保存起来，方便之后对应
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection+"_filtered"])}
            # 然后，生成筛选后的元素对应的数字对应的列，即将其编码，并保存在新的列selection_filtered_numeric
            df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
        #also lbl_itos and ..._numeric col for original label column
        # 同时也将没有经过最小min_cnt的筛选的label的进行展开拆分，去重，该列所有单元的元素展开，去重，保存为np.array
        lbl_itos_ptb_xl[selection]= np.array(sorted(list(set([x for sublist in df_ptb_xl[selection] for x in sublist]))))
        # 同样的，生成对应的数字编码
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        # 然后生成数字编码对应的列
        df_ptb_xl[selection+"_numeric"]=df_ptb_xl[selection].apply(lambda x:[lbl_stoi[y] for y in x])

    ## 输出的df_ptb_xl比输入的时候，多了经过筛选和未筛选最小出现个数的label，及其数字对应编码，
    # 然后所有label为key，value为对应label的所有去重元素的list的字典
    return df_ptb_xl, lbl_itos_ptb_xl



if __name__ == '__main__':
    data_folder_ptb_xl = data_root/"ptb_xl/"
    target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs))
    channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}

    # 为了避免之后的label_all_filtered_numeric等numeric有值，min_cnt=1，原先是0
    df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                           channels=12, channel_stoi=channel_stoi_default, 
                                                                           target_folder=target_folder_ptb_xl, recreate_data=True)
    
    # 把df_ptb_xl保存为df.pkl,lbl_itos_ptb_xl(key为label，包含未筛选min_cnt和筛选的，value为对应去重的值的list)保存为lbl_itos.pkl
    # 将mean_ptb_xl,std_ptb_xl保存为mean.npy, std.npy
    # 且已经将对应的文件保存为npy文件了

    # 将npy文件，读取并通过memmap，然后生成memmap.npy文件以及对应的memmap_meta.npz
    #reformat everything as memmap for efficiency

    reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=False)



