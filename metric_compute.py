# 读取preds和targets，并得到对应的metric
from sklearn.metrics import precision_recall_curve, f1_score
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

def multilabel_f1score(label, predict):
    label = [[label[i][j] for i in range(len(label))] for j in range(len(label[0]))]
    predict = [[predict[i][j] for i in range(len(predict))] for j in range(len(predict[0]))]
    
    best_f1_score_list = []
    threshold_list = []
    for i in range(len(label)):
        best_f1_score, threshold = bestf1score(label[i], predict[i])
        best_f1_score_list.append(best_f1_score)
        threshold_list.append(threshold)
    
    macro_best_f1_score = best_f1_score.mean()

    return macro_best_f1_score, threshold_list

def test_f1_score(label, predict, threshold):
    print(threshold)
    result = np.greater(predict, threshold)

    predict = result.astype(int)
    print(predict)
    label = [[label[i][j] for i in range(len(label))] for j in range(len(label[0]))]
    predict = [[predict[i][j] for i in range(len(predict))] for j in range(len(predict[0]))]

    # best_f1_score_list = []
    # for i in range(len(label)):
    #     f1 = f1_score(label[:,i], predict[:,i])
    #     best_f1_score_list.append(f1)
    
    # return best_f1_score_list.mean()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--preds_targs_path", type=str, default='')
    parser.add_argument("--info", type=str, default='')

    args = parser.parse_args()

    dic = torch.load('./val_'+str(args.preds_targs_path)+'.pth')
    preds = dic['preds']
    targs = dic['targs']

    macro_best_f1_score, thresholds_list = multilabel_f1score(targs, preds)

    macro_auc = np.round(roc_auc_score(targs, preds, average='macro'), 5)
    macro_aupr = np.round(average_precision_score(targs, preds, average='macro'), 5)

    test_dic = torch.load('./test_'+str(args.preds_targs_path)+'.pth')
    test_preds = test_dic['preds']
    test_targs = test_dic['targs']

    test_macro_auc = np.round(roc_auc_score(test_targs, test_preds, average='macro'), 5)
    test_macro_aupr = np.round(average_precision_score(test_targs, test_preds, average='macro'), 5)

    test_preds = [[1 if pred > thresh else 0 for pred, thresh in zip(pred_row, thresholds_list)] for pred_row in test_preds]
    # print(test_preds)

    test_f1_list = []
    for i in range(len(test_targs)):
        f1 = f1_score(test_targs[i], test_preds[i])
        test_f1_list.append(f1)

    

    print(f'val_f1:{macro_best_f1_score},val_auc:{macro_auc}, val_aupr:{macro_aupr}')
    print(f'test_f1:{sum(test_f1_list)/len(test_f1_list)}, test_auc:{test_macro_auc}, test_aupr:{test_macro_aupr}')

