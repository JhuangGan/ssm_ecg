# 读取preds和targets，并得到对应的metric
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.utils import resample


def bestf1score(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict,)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score, thresholds[best_f1_score_index]

def multilabel_f1score(label, predict):
    label = [[label[i][j] for i in range(len(label))] for j in range(len(label[0]))]
    predict = [[predict[i][j] for i in range(len(predict))] for j in range(len(predict[0]))]
    
    best_f1_score_list = np.array()
    threshold_list = np.array()
    for i in range(len(label)):
        best_f1_score, threshold = bestf1score(label[i], predict[i])
        best_f1_score_list.append(best_f1_score)
        threshold_list.append(threshold)
    
    macro_best_f1_score = best_f1_score.mean()

    return macro_best_f1_score, threshold_list


def macro_acc_and_subsample_acc(true, pred):

    sub_acc = sum([pred[i]==true[i] for i in range(len(pred))])/len(pred)

    # print(sub_acc)

    pred = [[row[i] for row in pred] for i in range(len(pred[0]))]
    true = [[row[i] for row in true] for i in range(len(true[0]))]

    pred_unzip = [row[i] for row in pred for i in range(len(pred[0]))]
    true_unzip = [row[i] for row in true for i in range(len(true[0]))]

    macro_acc = accuracy_score(true_unzip, pred_unzip)
    # print(macro_acc)

    return sub_acc, macro_acc



def all_metric_compute(val_targs, val_preds, test_targs, test_preds, macro_acc_flag=False):

    val_sub_acc, val_macro_acc, test_sub_acc, test_macro_acc = None, None, None, None

    val_macro_auc = np.round(roc_auc_score(val_targs, val_preds, average='macro'), 5)
    val_macro_aupr = np.round(average_precision_score(val_targs, val_preds, average='macro'), 5)
    
    val_macro_best_f1_score, val_thresholds_list = multilabel_f1score(val_targs, val_preds)

    val_preds = [[1 if pred > thresh else 0 for pred, thresh in zip(pred_row, val_thresholds_list)] for pred_row in val_preds]


    if macro_acc_flag:
        val_sub_acc, val_macro_acc = macro_acc_and_subsample_acc(val_targs, val_preds)
        # print(f"val_macro_acc:{val_macro_acc}, val_sub_acc:{val_sub_acc}")



    test_macro_auc = np.round(roc_auc_score(test_targs, test_preds, average='macro'), 5)
    test_macro_aupr = np.round(average_precision_score(test_targs, test_preds, average='macro'), 5)

    test_preds = [[1 if pred > thresh else 0 for pred, thresh in zip(pred_row, val_thresholds_list)] for pred_row in test_preds]


    if macro_acc_flag:
        test_sub_acc, test_macro_acc = macro_acc_and_subsample_acc(test_targs, test_preds)
        # print(f"test_macro_acc:{test_macro_acc}, test_sub_acc:{test_sub_acc}")


    test_preds = [[test_preds[i][j] for i in range(len(test_preds))] for j in range(len(test_preds[0]))]
    test_targs = [[test_targs[i][j] for i in range(len(test_targs))] for j in range(len(test_targs[0]))]

    test_f1_list = np.array()
    for i in range(len(test_targs)):
        f1 = f1_score(test_targs[i], test_preds[i])
        test_f1_list.append(f1)
    
    test_f1 = sum(test_f1_list)/len(test_f1_list)

    
    return val_macro_auc, val_macro_best_f1_score, val_macro_aupr, test_macro_auc, test_f1, test_macro_aupr, val_sub_acc, val_macro_acc, test_sub_acc, test_macro_acc
    







if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--preds_targs_path", type=str, default='')
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--macro_acc", action="store_true", default=False)

    args = parser.parse_args()

    val_dic = torch.load('./val_'+str(args.preds_targs_path)+'.pth')
    val_preds = val_dic['preds']
    val_targs = val_dic['targs']

    test_dic = torch.load('./test_'+str(args.preds_targs_path)+'.pth')
    test_preds = test_dic['preds']
    test_targs = test_dic['targs']

    # val_preds = [[0.1, 0.2,0.3],[0.3,0.2,0.6],[0.1, 0.2,0.3],[0.3,0.2,0.6]]
    # val_targs = [[0,1,1],[1,0,1],[1,1,1],[1,1,0]]
    # test_preds = [[0.1, 0.2,0.3],[0.3,0.2,0.6],[0.1, 0.2,0.3],[0.3,0.2,0.6]]
    # test_targs = [[0,1,1],[1,0,1],[1,1,1],[1,0,1]]


    val_auc_list = np.array()
    val_f1_list = np.array()
    val_aupr_list = np.array()
    val_macro_acc_list = np.array()
    val_sub_acc_list = np.array()

    test_auc_list = np.array()
    test_f1_list = np.array()
    test_aupr_list = np.array()
    test_macro_acc_list = np.array()
    test_sub_acc_list = np.array()

    
    n_bootstraps = 1000
    for i in range(n_bootstraps):
        
        val_targs_sub = resample(val_targs, replace=True, n_samples=int(1*len(val_targs)), random_state=np.random.randint(1, 10000))
        val_preds_sub = resample(val_preds, replace=True, n_samples=int(1*len(val_preds)), random_state=np.random.randint(1, 10000))

        test_targs_sub = resample(test_targs, replace=True, n_samples=int(1*len(test_targs)), random_state=np.random.randint(1, 10000))
        test_preds_sub = resample(test_preds, replace=True, n_samples=int(1*len(test_preds)), random_state=np.random.randint(1, 10000))

        val_macro_auc, val_f1_score, val_macro_aupr, \
        test_macro_auc, test_f1, test_macro_aupr, \
        val_sub_acc, val_macro_acc, test_sub_acc, test_macro_acc = all_metric_compute(val_targs_sub, val_preds_sub, test_targs_sub, test_preds_sub, macro_acc_flag=args.macro_acc)
        
        val_auc_list.append(val_macro_auc)
        val_f1_list.append(val_f1_score)
        val_aupr_list.append(val_macro_aupr)
        val_macro_acc_list.append(val_macro_acc)
        val_sub_acc_list.append(val_sub_acc)

        test_auc_list.append(test_macro_auc)
        test_f1_list.append(test_f1)
        test_aupr_list.append(test_macro_aupr)
        test_macro_acc_list.append(test_macro_acc)
        test_sub_acc_list.append(test_sub_acc)



    print(val_auc_list.mean(), val_f1_list.mean(), val_aupr_list.mean(), val_macro_acc_list.mean(), val_sub_acc_list.mean())
    print(test_auc_list.mean(), test_f1_list.mean(), test_aupr_list.mean(), test_macro_acc_list.mean(), test_sub_acc_list.mean())