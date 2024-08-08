# 读取preds和targets，并得到对应的metric
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, average_precision_score


def bestf1score(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict,)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score, thresholds[best_f1_score_index]


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--preds_targs_path", type=str, default='')
    parser.add_argument("--info", type=str, default='')

    args = parser.parse_args()

    dic = torch.load('./'+str(args.preds_targs_path)+'.pth')
    preds = dic['preds']
    targs = dic['targs']

    best_f1_score, thresholds = bestf1score(targs, preds)

    macro_auc = np.round(roc_auc_score(targs, preds, average='macro'), 3)
    macro_aupr = np.round(average_precision_score(targs, preds, average='macro'), 3)
    print(f'info:{args.info},f1:{best_f1_score},auc:{macro_auc}, aupr:{macro_aupr}')

