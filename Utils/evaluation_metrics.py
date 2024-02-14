from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from statistics import mean

def compute_evaluation_metrics(target_values, predic_values):
    """
    Compute model performance: precision, recall and f1, both
    for the positive and the negative label value.
    :param target_values: list of real values for the predicted label
    :param predic_values: list of predicted values
    :return: a dictionary with the computed performance values
    """
    tn, fp, fn, tp = confusion_matrix(target_values, predic_values).ravel()

    pos_precision = tp / (tp + fp)
    neg_precision = tn / (tn + fn)

    pos_recall = tp / (tp + fn)
    neg_recall = tn / (tn + fp)

    pos_f1 = (2 * (pos_precision * pos_recall)) / (pos_precision + pos_recall)
    neg_f1 = (2 * (neg_precision * neg_recall)) / (neg_precision + neg_recall)

    return {
        'precision': [pos_precision, neg_precision],
        'recall': [pos_recall, neg_recall],
        'f1': [pos_f1, neg_f1],
    }
    
def compute_evaluation_metrics_mean(target_values, predic_values, len_test):
    """
    Compute model performance: precision, recall and f1, both
    for the positive and the negative label value.
    :param target_values: list of real values for the predicted label
    :param predic_values: list of predicted values
    :return: a dictionary with the computed performance values
    """
    pos_prec = []
    neg_prec = []
    pos_rec = []
    neg_rec = []
    posi_f1 = []
    nega_f1 = []
    for j in range(0,10):
        
        tn, fp, fn, tp = confusion_matrix(target_values[j*len_test[j] : j*len_test[j]+len_test[j]], predic_values[j*len_test[j]:j*len_test[j]+len_test[j]]).ravel()

        pos_precision = tp / (tp + fp)
        neg_precision = tn / (tn + fn)

        pos_recall = tp / (tp + fn)
        neg_recall = tn / (tn + fp)

        pos_f1 = (2 * (pos_precision * pos_recall)) / (pos_precision + pos_recall)
        neg_f1 = (2 * (neg_precision * neg_recall)) / (neg_precision + neg_recall)
        pos_prec.append(pos_precision)
        neg_prec.append(neg_precision)
        pos_rec.append(pos_recall)
        neg_rec.append(neg_recall) 
        posi_f1.append(pos_f1) 
        nega_f1.append(neg_f1) 

    return {
        'precision': [mean(pos_prec), mean(neg_prec)],
        'recall': [mean(pos_rec), mean(neg_rec)],
        'f1': [mean(posi_f1), mean(nega_f1)],
    }

def normalize(n1,n2):
    return n1/(n1+n2), n2/(n1+n2)

# _____________________________________ PLOT __________________________________________
def printResult(y_pred, y_prob, y_test):
    """
    A function to plot and print result
    :param y_pred:
    :param y_prob:
    :param y_test:
    :return:
    """
    acc = accuracy_score(y_test, y_pred)
    # Result
    print("Accuracy: {:.2f}".format(acc*100),end='\n\n')
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix:\n', cm)
    print(classification_report(y_test,y_pred))
    # Plot
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc='lower right')

