from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from collections import defaultdict
import numpy as np
import math


nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def genome_acc(grps, pred_grps, y_true, n_cluters):
    groups_cluster_lb = assign_cluster_2_reads(grps, pred_grps)
    prec, recall = eval_quality(y_true, groups_cluster_lb, n_clusters=n_cluters)
    f1_score = 2*((prec*recall)/(prec+recall))
    return prec, recall, f1_score

def genome_acc_per_class(grps, pred_grps, y_true, n_cluters):
    groups_cluster_lb = assign_cluster_2_reads(grps, pred_grps)

    prec, recall = eval_quality_per_class(y_true, y_pred, n_clusters=n_clusters)
    f1_score = [2*((prec[i]*recall[i])/(prec[i]+recall[i])) for i in range(len(prec))]
    # f1_score = 2*((prec*recall)/(prec+recall))
    return prec, recall, f1_score

def assign_cluster_2_reads(groups, y_grp_cl):
    label_cl_dict=dict()

    for idx, g in enumerate(groups):
        for r in g:
            label_cl_dict[r]=y_grp_cl[idx]
    
    y_cl=[]
    for i in sorted(label_cl_dict):
        y_cl.append(label_cl_dict[i])
    
    return y_cl

def eval_quality(y_true, y_pred, n_clusters):
    A = confusion_matrix(y_pred, y_true)
    if len(A) == 1:
        return 1, 1
    prec = sum([max(A[:,j]) for j in range(0,n_clusters)])/sum([sum(A[i,:]) for i in range(0,n_clusters)])
    rcal = sum([max(A[i,:]) for i in range(0,n_clusters)])/sum([sum(A[i,:]) for i in range(0,n_clusters)])

    return prec, rcal

def eval_quality_per_class(y_true, y_pred, n_clusters):
    A = confusion_matrix(y_pred, y_true)
    if len(A) == 1:
        return 1, 1

    denom = [sum(A[i,:]) for i in range(0,n_clusters)]
    prec_nom = [max(A[:,i]) for i in range(0,n_clusters)]
    recall_nom = [max(A[i,:]) for i in range(0,n_clusters)]

    print(denom)
    print(prec_nom)
    print(recall_nom)

    prec = [prec_nom[i]/denom[i] for i in range(len(denom))]
    recall = [recall_nom[i]/denom[i] for i in range(len(denom))]

    # prec = np.array([max(A[:,j]) for j in range(0,n_clusters)])/np.array()
    # recall = np.array([max(A[i,:]) for i in range(0,n_clusters)])/np.array([sum(A[i,:]) for i in range(0,n_clusters)])

    return prec, recall
    
def precision_per_class(grps, pred_grps, y_true):
    groups_cluster_lb = assign_cluster_2_reads(grps, pred_grps)
    y_pred = np.array(groups_cluster_lb)
    y_true = np.array(y_true)
    unique_labels = np.unique(y_true)

    result = {}

    for lb in unique_labels:
        # number of preds for label lb
        lb_pred = np.sum(y_pred == lb)

        if lb_pred == 0:
            result[lb] = 0
            continue

        tp = 0
        for i in range(y_true.shape[0]):
            if y_pred[i] == lb:
                if y_pred[i] == y_true[i]:
                    tp += 1

        result[lb] = np.around(tp / lb_pred, 3)

    return result
        
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == "__main__":
    A = np.array([
                    [2, 0, 0],
                    [0, 0, 1],
                    [1, 0, 2]
                ])

    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 2, 2]
    A = confusion_matrix(y_true, y_pred)

    report = defaultdict()
    n_clusters = 3
    # prec = np.array([max(A[:,j]) for j in range(0,n_clusters)])/np.array([sum(A[i,:]) for i in range(0,n_clusters)])
    # recall = np.array([max(A[i,:]) for i in range(0,n_clusters)])/np.array([sum(A[i,:]) for i in range(0,n_clusters)])

    # prec, recall = eval_quality_per_class(y_true, y_pred, 3)
    # f1 = [2*((prec[i]*recall[i])/(prec[i]+recall[i])) for i in range(len(prec))]
    # prec, recall, f1 = genome_acc_per_class(y_pred, y_true, 3)
    prec = precision_per_class(y_pred, y_true, 3)
    # f1 = 2*((prec*recall)/(prec+recall))
    print('prec: ', prec)
    # print('recall: ', recall)
    # print('f1: ', f1)
