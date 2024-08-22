import torch

def select(filename, n_select):
    test_dic = torch.load('./'+str(filename)+'.pth')
    test_preds = test_dic['preds']
    test_targs = test_dic['targs']


    test_preds = test_dic['preds'][:, test_dic['targs'].sum(axis=0)>=n_select]
    test_targs = test_dic['targs'][:, test_dic['targs'].sum(axis=0)>=n_select]


    test_dic['preds'] = test_preds
    test_dic['targs'] = test_targs

    torch.save(test_dic, './'+filename+'_select_'+str(n_select)+'_.pth')

filename1 = "ptb_xl_fs100_250_label_allpreds_targs"
select(filename1, n_select=9)

