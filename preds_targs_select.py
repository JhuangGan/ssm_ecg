# 挑出71分类里的测试集中样本量大于等于9的诊断的指标
import torch

filename = "ptb_xl_fs100_250_label_allpreds_targs"

test_dic = torch.load('./test_'+str(filename)+'.pth')
test_preds = test_dic['preds']
test_targs = test_dic['targs']

# 转为列为unit
test_preds = [[test_preds[i][j] for i in range(len(test_preds))] for j in range(len(test_preds[0]))]
test_targs = [[test_targs[i][j] for i in range(len(test_targs))] for j in range(len(test_targs[0]))]

n_select = 9

label_list = []

for i in range(len(test_targs)):
    if sum(test_targs[i]) >= n_select:
        label_list.append(i)

select_preds = []
select_targs = []

for i in label_list:
    select_preds.append(test_preds)
    select_targs.append(test_targs)

select_preds = [[select_preds[i][j] for i in range(len(select_preds))] for j in range(len(select_preds[0]))]
select_targs = [[select_targs[i][j] for i in range(len(select_targs))] for j in range(len(select_targs[0]))]

test_dic['preds'] = select_preds
test_dic['targs'] = select_targs

test_dic.save('./test_ptb_xl_fs100_250_label_allpreds_targs_select_'+str(n_select))




