import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

dic = torch.load('./preds_targs_agg.pth')

macro_auc = np.round(roc_auc_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
macro_aupr = np.round(average_precision_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)
macro_f1 = np.round(f1_score(dic['targs-agg'], dic['preds_agg'], average='macro'), 3)

print(f'auc:{macro_auc}, aupr:{macro_aupr}, f1:{macro_f1}')