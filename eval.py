import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
def print_info(info=''):
  dic = torch.load('./'+str(info)+'_preds_targs_agg'+'.pth')
  
  macro_auc = np.round(roc_auc_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
  macro_aupr = np.round(average_precision_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
  # macro_f1 = np.round(f1_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
  
  print(f'{valortest} {sig},auc:{macro_auc}, aupr:{macro_aupr}')

print_info(info='val_')
print_info(info='val_sig_')
print_info(info='testinfo_')
print_info(info='test')
