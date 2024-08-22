# 读取preds和targets，并得到对应的metric
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, average_precision_score


def bestf1score(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score, f1_scores, best_f1_score_index

def multilabel_f1score(label, predict, label_test, predict_test):
    label = [[label[i][j] for i in range(len(label))] for j in range(len(label[0]))]
    predict = [[predict[i][j] for i in range(len(predict))] for j in range(len(predict[0]))]

    label_test = [[label_test[i][j] for i in range(len(label_test))] for j in range(len(label_test[0]))]
    predict_test = [[predict_test[i][j] for i in range(len(predict_test))] for j in range(len(predict_test[0]))]
    
    best_f1_score_list = []
    threshold_list = []

    best_f1_score_list_test = []
    threshold_list_test = []

    for i in range(len(label)):
        best_f1_score, f1_score_list, threshold_index = bestf1score(label[i], predict[i])
        
        best_f1_score_test, f1_score_list_test, threshold_index_test = bestf1score(label_test[i], predict_test[i])
        
        test_f1 = f1_score_list_test[threshold_index]

        best_f1_score_list_test.append(test_f1)

        best_f1_score_list.append(best_f1_score)
        # threshold_list.append(threshold)
    
    macro_best_f1_score = best_f1_score_list.mean()
    macro_best_f1_score_test = best_f1_score_list_test.mean()

    return macro_best_f1_score, macro_best_f1_score_test


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--val_preds_targs_path", type=str, default='')
    # parser.add_argument("--info", type=str, default='')
    parser.add_argument("--test_preds_targs_path", type=str, default='')

    args = parser.parse_args()

    val_dic = torch.load('./'+str(args.val_preds_targs_path)+'.pth')
    val_preds = val_dic['preds']
    val_targs = val_dic['targs']
    test_dic = torch.load('./'+str(args.test_preds_targs_path)+'.pth')
    test_preds = test_dic['preds']
    test_targs = test_dic['targs']


    val_macro_auc = np.round(roc_auc_score(val_targs, val_preds, average='macro'), 5)
    val_macro_aupr = np.round(average_precision_score(val_targs, val_preds, average='macro'), 5)

    test_macro_auc = np.round(roc_auc_score(test_targs, test_preds, average='macro'), 5)
    test_macro_aupr = np.round(average_precision_score(test_targs, test_preds, average='macro'), 5)

    macro_best_f1_score, macro_best_f1_score_test = multilabel_f1score(val_targs, val_preds, test_preds, test_targs)


    print(f'val_auc:{val_macro_auc}, val_f1:{macro_best_f1_score}, val_aupr:{val_macro_aupr}')
    print(f'test_auc:{test_macro_auc}, test_f1:{macro_best_f1_score_test}, test_aupr:{test_macro_aupr}')


