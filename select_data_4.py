import torch
import numpy as np
from argparse import ArgumentParser

def select(filename, n_select):
    dic = torch.load('./'+str(filename)+'.pth')
    preds = np.array(dic['preds'])
    targs = np.array(dic['targs'])


    preds = preds[:, dic['targs'].sum(axis=0)>=n_select]
    targs = targs[:, dic['targs'].sum(axis=0)>=n_select]


    dic['preds'] = preds
    dic['targs'] = targs

    torch.save(dic, './'+filename+'_select_'+str(n_select)+'.pth')

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--n_select", type=int, default=9)
    parser.add_argument("--test_file", type=str, default='')
    parser.add_argument("--val_file", type=str, default='')

    args = parser.parse_args()
    select(args.test_file, n_select=args.n_select)

    select(args.val_file, n_select=args.n_select)
