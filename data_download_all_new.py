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


if __name__ == '__main__':

    def download_all_data_fs(target_fs=100):
        target_fs=target_fs
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


        data_folder_ptb_xl = data_root/"ptb_xl/" # 数据存放文件夹
        target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs))  # 
        # ptb_xl_url='https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip'
        # 直接替换链接就可以
        ptb_xl_url='https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip'
        
        # download(ptb_xl_url, data_folder_ptb_xl)
        
        df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                            channels=12, channel_stoi=channel_stoi_default, 
                                                                            target_folder=target_folder_ptb_xl, recreate_data=True)
        reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=True)

        # ribeiro 2020
        data_folder_ribeiro_test = data_root/"ribeiro2020_test"
        target_folder_ribeiro_test = target_root/("ribeiro_fs"+str(target_fs))
        ribeiro_test_url='https://zenodo.org/record/3765780/files/data.zip?download=1'

        # download and unzip dataset 
        if target_fs == 100:
            download(ribeiro_test_url, data_folder_ribeiro_test)
        # 这一步做降采样，所以这里需要都进行一次，上面的数据不能删除，这一步产生的npy可以删除
        df_ribeiro_test, lbl_itos_ribeiro_test,  mean_ribeiro_test, std_ribeiro_test = prepare_data_ribeiro_test(data_folder_ribeiro_test, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_ribeiro_test)
        #reformat everything as memmap for efficiency
        reformat_as_memmap(df_ribeiro_test, target_folder_ribeiro_test/("memmap.npy"),data_folder=target_folder_ribeiro_test,delete_npys=True)

        # zheng 2020
        data_folder_chapman = data_root/"chapman/"
        target_folder_chapman = target_root/("chapman_fs"+str(target_fs))
        chapman_url = 'https://figshare.com/ndownloader/files/15651326'
        # download and unzip dataset 
        if target_fs == 100:
            download(chapman_url, data_folder_chapman, flatten=False)
        condition = 'https://figshare.com/ndownloader/files/15651293'
        rhythm = 'https://figshare.com/ndownloader/files/15651296'
        diagnostic = 'https://figshare.com/ndownloader/files/15653771'
        attributes = 'https://figshare.com/ndownloader/files/15653762'
        for url in [condition, rhythm, diagnostic, attributes]:
            wget.download(url, out=str(data_folder_chapman))

        df_chapman, lbl_itos_chapman,  mean_chapman, std_chapman = prepare_data_chapman(data_folder_chapman, denoised=False, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_chapman)
        reformat_as_memmap(df_chapman, target_folder_chapman/("memmap.npy"),data_folder=target_folder_chapman,delete_npys=True)

        # # cinc
        # data_folder_cinc = data_root/"cinc2020/"
        # if not isdir(data_folder_cinc):
        #     os.makedirs(data_folder_cinc)
        # target_folder_cinc = target_root/("cinc_fs"+str(target_fs))

        # filenames = ['PhysioNetChallenge2020_Training_CPSC.tar.gz','PhysioNetChallenge2020_Training_2.tar.gz',
        #         'PhysioNetChallenge2020_Training_StPetersburg.tar.gz', 'PhysioNetChallenge2020_Training_PTB.tar.gz',
        #         'PhysioNetChallenge2020_Training_PTB-XL.tar.gz', 'PhysioNetChallenge2020_Training_E.tar.gz']
        
        # for fname in filenames:
        #     shutil.unpack_archive(fname, data_folder_cinc)

        # for fname in filenames:
        #     os.remove(fname)

        # df_cinc, lbl_itos_cinc,  mean_cinc, std_cinc = prepare_data_cinc(data_folder_cinc, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_cinc)
        # #reformat everything as memmap for efficiency
        # reformat_as_memmap(df_cinc, target_folder_cinc/("memmap.npy"),data_folder=target_folder_cinc,delete_npys=True)

        # sph
        data_folder_sph = data_root/"sph/"
        target_folder_sph = target_root/("sph_fs"+str(target_fs))
        sph_url='https://springernature.figshare.com/ndownloader/files/32630684'

        # download and unzip dataset 
        if target_fs == 100:
            download(sph_url, data_folder_sph)

        attributes = 'https://springernature.figshare.com/ndownloader/files/34793152'
        diagnostic = 'https://springernature.figshare.com/ndownloader/files/32630954'

        for url in [diagnostic, attributes]:
            wget.download(url, out=str(data_folder_sph))

        df_sph, lbl_itos_sph,  mean_sph, std_sph = prepare_data_sph(data_folder_sph, min_cnt=0, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_sph)

        #reformat everything as memmap for efficiency
        reformat_as_memmap(df_sph, target_folder_sph/("memmap.npy"),data_folder=target_folder_sph,delete_npys=True)
    
    download_all_data_fs(target_fs=100)
    # download_all_data_fs(target_fs=500)
