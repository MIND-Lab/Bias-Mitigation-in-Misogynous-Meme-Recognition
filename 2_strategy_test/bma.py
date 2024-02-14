
from statistics import mean
import pandas as pd

def normalize(n1,n2):
    return n1/(n1+n2), n2/(n1+n2)

def ubma_neu(data_score, probs_path, results_path, dataset):
    
    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')

        y_test = data_probs["GROUND TRUTH"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
            label_norm_0, label_norm_1 = normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            else:
              labels_bma.append(1)