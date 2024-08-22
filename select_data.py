import torch

filename = "ptb_xl_fs100_250_label_allpreds_targs"

test_dic = torch.load('./test_'+str(filename)+'.pth')
test_preds = test_dic['preds']
test_targs = test_dic['targs']

n_select =9
test_preds = test_dic['preds'][:, test_dic['targs'].sum(axis=0)>=n_select]
test_targs = test_dic['targs'][:, test_dic['targs'].sum(axis=0)>=n_select]


test_dic['preds'] = test_preds
test_dic['targs'] = test_targs

torch.save(test_dic, './test_ptb_xl_fs100_250_label_allpreds_targs.pth')

