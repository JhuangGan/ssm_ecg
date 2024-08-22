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

    torch.save(test_dic, './'+filename+'_select_'+str(n_select)+'.pth')

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--n_select", type=int, default=9)
    parser.add_argument("--test_file", type=str, default='')
    parser.add_argument("--val_file", type=str, default='')

    args = parser.parse_args()
    select(args.test_file, n_select=args.n_select)

    select(args.val_filename1, n_select=args.n_select)
