from sklearn.metrics import (
    roc_curve, 
    auc, 
    balanced_accuracy_score
)



def auc_metric(y_true, y_score, pos_label=None):
    """
    This function calculates area under ROC curve

    Parameters
    ----------
    y_true:
        True binary labels. If labels are not either {-1, 1} or {0, 1}, 
        then pos_label should be explicitly given.
    y_score:
        Target scores, probability estimates of the positive class
    pos_label: default None
        The label of the positive class.

    Returns
    -------
    auc:
        area under the roc curve

    """

    # check positive label
    if pos_label is None:
        pos_label = 1
    if pos_label not in y_true:
        raise ValueError("pos_label: {} is not in y_true".format(pos_label))
    

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)

    return auc(fpr, tpr)

def bac_metric(y_true, y_pred):
    """
    This function calculates balanced accuracy score

    Parameters
    ----------
    y_true:
        Ground truth (correct) target values.
    y_pred:
        Estimated targets as returned by a classifier.

    Returns
    -------
    balanced_accuracy:
        Balanced accuracy score.

    """
    return balanced_accuracy_score(y_true, y_pred)



