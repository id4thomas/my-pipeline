import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.metrics import recall_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

def make_roc_plot(fpr, tpr, thresholds, pos_label = 1):
    fig=plt.figure()
    roc=fig.add_subplot(1,1,1)
    lw = 2
    roc.plot(fpr, tpr, color='darkorange', lw=lw, )
    roc.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    roc.set_xlim([0.0, 1.0])
    roc.set_ylim([0.0, 1.05])
    roc.set_xlabel('False Positive Rate')
    roc.set_ylabel('True Positive Rate')
    roc.set_title('ROC-curve')
    roc.legend(loc="lower right")
    return fig

def calc_roc(y_true, y_scores, pos_label = 1, make_plot = False):
    # y_scores are logits, probs
    auc_score = metrics.roc_auc_score(y_true, y_scores)
    if make_plot:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=2)
        fig = make_roc_plot(fpr, tpr, thresholds)
        return auc_score, fig
    else:
        return auc_score

# Accuracy
def calc_acc(y_true, y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    return accuracy

# Precision, Recall, F-score
def calc_prf(y_true, y_pred, pos_label=1, average='binary', beta = 1.0):
    #beta: The strength of recall versus precision in the F-score.
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, beta = beta, pos_label=pos_label, average=average)
    return precision,recall,f_score

#False Positive Rate
def calc_fpr(y_true, y_pred, ans_label=1):
    # 1- TN/TN+FP
    return 1-recall_score(y_true,y_pred,pos_label=0)

#Matthews Correlation Coefficient
def calc_mcc(y_true, y_pred, ans_label=1):
    return matthews_corrcoef(y_true,y_pred)

#Confusion matrix
def calc_conf_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)