import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
def print_info(info=''):
  dic = torch.load('./'+str(info)+'preds_targs'+'.pth')
  
  macro_auc = np.round(roc_auc_score(dic['targs_agg'], dic['preds_agg'], average='macro'), 3)
  macro_aupr = np.round(average_precision_score(dic['targs_agg'], dic['preds_agg'], average='macro'), 3)
  # macro_f1 = np.round(f1_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
  
  print(f'{info},auc:{macro_auc}, aupr:{macro_aupr}')

print_info(info='val_')
print_info(info='val_agg_')
print_info(info='val_sig_')

print_info(info='test_')
print_info(info='test_agg_')
print_info(info='test_sig_agg_')