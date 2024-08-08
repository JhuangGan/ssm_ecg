# 读取preds和targets，并得到对应的metric
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch


def bestf1score(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score, thresholds[best_f1_score_index]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--preds_targs_path", type=int, default=100)

    dic = torch.load('./val_preds_targs.pth')
    preds = dic['preds']
    targs = dic['targs']

    best_f1_score, thresholds = bestf1score(targs, preds)
    print()

