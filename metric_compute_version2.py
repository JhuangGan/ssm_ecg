from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def bestf1score(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score, thresholds[best_f1_score_index]

def multilabel_f1score(label, predict):
    # label = [[label[i][j] for i in range(len(label))] for j in range(len(label[0]))]
    # predict = [[predict[i][j] for i in range(len(predict))] for j in range(len(predict[0]))]

    

    best_f1_score_list = []
    threshold_list = []
    for i in range(len(label)):
        best_f1_score, threshold = bestf1score(label[i], predict[i])
        best_f1_score_list.append(best_f1_score)
        threshold_list.append(threshold)
    
    macro_best_f1_score = best_f1_score.mean()

    return macro_best_f1_score, threshold_list


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--val_preds_targs_path", type=str, default='')
    parser.add_argument("--test_preds_targs_path", type=str, default='')

    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--macro_acc", action="store_true", default=False)
    parser.add_argument("--n_bootstraps", type=int, default=1)


    args = parser.parse_args()

    val_dic = torch.load('./'+str(args.val_preds_targs_path)+'.pth')
    val_preds = np.array(val_dic['preds'])
    val_targs = np.array(val_dic['targs'])

    test_dic = torch.load('./'+str(args.test_preds_targs_path)+'.pth')
    test_preds = np.array(test_dic['preds'])
    test_targs = np.array(test_dic['targs'])

    val_macro_auc = np.round(roc_auc_score(val_targs, val_preds, average='macro'), 5)
    val_macro_aupr = np.round(average_precision_score(val_targs, val_preds, average='macro'), 5)

    