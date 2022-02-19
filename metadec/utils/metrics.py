from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def genome_acc(grps, pred_grps, y_true, n_cluters):
    groups_cluster_lb = assign_cluster_2_reads(grps, pred_grps)
    prec, recall = eval_quality(y_true, groups_cluster_lb, n_clusters=n_cluters)
    f1_score = 2*((prec*recall)/(prec+recall))
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