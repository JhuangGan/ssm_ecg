import torch
import numpy as np
from argparse import ArgumentParser

def select(filename, n_select):
    test_dic = torch.load('./'+str(filename)+'.pth')
    test_preds = np.array(test_dic['preds'])
    test_targs = np.array(test_dic['targs'])


    test_preds = test_preds[:, test_dic['targs'].sum(axis=0)>=n_select]
    test_targs = test_targs[:, test_dic['targs'].sum(axis=0)>=n_select]


    test_dic['preds'] = test_preds
    test_dic['targs'] = test_targs

    torch.save(test_dic, './'+filename+'_select_'+str(n_select)+'_.pth')

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--n_select", type=int, default=9)
    parser.add_argument("--test_file", type=str, default=9)

    args = parser.parse_args()
    filename1 = args.test_file
    select(filename1, n_select=args.n_select)

