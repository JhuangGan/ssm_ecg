import torch

filename = "ptb_xl_fs100_250_label_allpreds_targs"

test_dic = torch.load('./test_'+str(filename)+'.pth')
test_preds = test_dic['preds']
test_targs = test_dic['targs']




