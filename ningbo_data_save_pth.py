# -*- coding: utf-8 -*-

# load the data

import os

global CUR_DIR
global PTH_FOLDERNAMES
global PTH_LABEL_FOLDERNAMES
global SUFFIX
global FILE_dx_all


CUR_DIR = os.path.join(os.getcwd(), 'data')
PTH_FOLDERNAMES= 'Ningbo_pth'
PTH_LABEL_FOLDERNAMES = "Ningbo_pth_labels"

SUFFIX = '.pth'

FILE_dx_all = "dx_all"+SUFFIX

if not os.path.exists(os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES)):
    os.mkdir(os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES))
if not os.path.exists(os.path.join(CUR_DIR, PTH_FOLDERNAMES)):
    os.mkdir(os.path.join(CUR_DIR, PTH_FOLDERNAMES))
    


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id
# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age

# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

def get_dx(data):
    dx = None
    for l in data.split('\n'):
        if l.startswith('#Dx:'):
            try:
                dx = l.split(': ')[1].strip()
            except:
                pass
    return dx


# merge the original .mat and .hea data to pth
def mat_and_hea2pth_new():
    import os, scipy.io, scipy.io.wavfile
    from tqdm import tqdm
    import torch
    import numpy as np
    
    foldernames = ["WFDB_Ningbo"]


    # PTH_FOLDERNAMES= 'Ningbo_PTBXL_CS_Ga_pth'
    # PTH_LABEL_FOLDERNAMES = "Ningbo_PTBXL_CS_Ga_pth_labels"
    # CUR_DIR = '/root/yzhou/data_paper'

    if not os.path.exists(os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES)):
        os.mkdir(os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES))
    if not os.path.exists(os.path.join(CUR_DIR, PTH_FOLDERNAMES)):
        os.mkdir(os.path.join(CUR_DIR, PTH_FOLDERNAMES))

    save_dir = os.path.join(CUR_DIR, PTH_FOLDERNAMES)
    print('merge the original .mat and .hea data to pth')
    print('source {} and target {}'.format("_and_".join(str(i) for i in foldernames), save_dir))
    dx_all = []
    freq_all = []
    total_stat = {}
    total_ecg = {}
    total_lead_zero = {}
    for folder in foldernames:
        folder_tem = []
        zero_pid = {}
        
        os.chdir(os.path.join(CUR_DIR, folder))
        filenames = os.listdir()
        for filename in tqdm(filenames):
            if filename.endswith("hea"):
                try:
                    file = load_patient_data(filename)
                    pid = get_patient_id(file)
                    # debug their records
                    if (pid == 'S23074'):
                        pid = 'JS23074'
                    freq = get_frequency(file)
                    age = get_age(file)
                    sex = get_sex(file)
                    ecg = scipy.io.loadmat(pid+".mat")['val']
                    dx = get_dx(file).split(',')
                    
                    # v20230111
                    if ecg.shape == (12, 5000):
                        # some leads may be all zeros
                        valid = 1
                        for i in range(ecg.shape[0]):
                            if np.all(np.array(ecg[i]) == 0):
                                valid = 0
                                if zero_pid.get(pid, None) is None:
                                    zero_pid[pid] = [i]
                                else:
                                    zero_pid[pid].append(i)    
                                
                        if valid == 1:
                            sample = {'image': ecg,
                                      'dx': dx,
                                      'pid': pid,
                                      'age': age,
                                      'sex': sex,
                                      'freq': freq,
                                      'label': torch.tensor([0.0])
                                      }
                            dx_all.extend(dx)
                            freq_all.append(freq)
                            torch.save(sample, os.path.join(save_dir,pid))
                            folder_tem.append(pid)      
                        
                    '''
                    else:
                        print(ecg.shape)
                    if freq!=500:
                        print(freq)
                    '''
                except:
                    # print("read problem file:", filename)
                    pass

        print("the total ecgs for {} is {}".format(folder, len(folder_tem)))
        total_stat[folder] = len(folder_tem)
        total_ecg[folder] = folder_tem
        total_lead_zero[folder] = zero_pid
        print(folder, 'has ', len(zero_pid), 'ecgs including leads all zeros')

    dx_all =  sorted(list(set(dx_all)))
    dx_all = dict(zip(dx_all, range(len(dx_all))))
    freq_all = set(freq_all)
    dx_all['total_stat'] = total_stat
    dx_all['freq_all'] = freq_all
    dx_all['total_ecg'] = total_ecg
    dx_all['total_lead_zero'] = total_lead_zero
    
    
    torch.save(dx_all, os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES, 
                                     FILE_dx_all))
    print("all the diagnosises are saved in:", 
          os.path.join(CUR_DIR, PTH_LABEL_FOLDERNAMES, 
                        FILE_dx_all))
    print("the frequency of the data is:", freq_all)
    return dx_all


if __name__ == '__main__':

    # v20240814------
    '''
    # # transform the mat and hea to pth
    _ = mat_and_hea2pth_new()
    '''
    # transform the mat and hea to pth
    _ = mat_and_hea2pth_new()




    
    
