import sys
sys.path.append('/home/jimmy/Documenti/tesi/bma/bias/')
#import load_data
import model_bias_analysis

import numpy as np
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import json
from statistics import mean
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
#import stanza
#import spacy_stanza
import string
####################

import pandas as pd
import string
from collections import Counter
import stanza
#import spacy_stanza
import re
from Utils import load_data, project_paths, evaluation_metrics, preprocessing
import os
from tqdm import tqdm
import ast


###################################
import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description="BIAS",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-s", "--score_path",  default="../data/results2strategy/TAGS/sintest/score_sin_test_TAGS10fold.csv", type=str, help="path of output models")
#parser.add_argument("-p", "--probs_path",  default="../data/results2strategy/TAGS/sintest/probs_sin_test_fold_", type=str, help="path where prob results are located")
#parser.add_argument("-r", "--results_path",  default='results/TAGS_bma_neu/sintest/probs_sintest_', type=str, help="syn results path")

parser.add_argument("-d", "--data_to_eval",  default='syn', type=str, help="dataset test o syntest")
parser.add_argument("-c", "--correction_strategy",  default='neu', type=str, help="test")
parser.add_argument('--mitigation', type=str_to_bool, nargs='?', const=True, default=False)



args = parser.parse_args()
config_arg= vars(args)
print(config_arg)
data_to_eval = args.data_to_eval
correction_strategy = args.correction_strategy
data_path = ""
dataset = pd.DataFrame()
result_path = ""
score_path = ""
out_path = ""


fold_mitigation = args.mitigation
if fold_mitigation:
    key_of_folds = "mitigation"
else:
    key_of_folds = "measure"
    

import pickle
with open('identity_tags_mis', 'rb') as f:
    id_tags_mis = pickle.load(f)    
with open('identity_tags_notmis', 'rb') as f:
    id_tags_notmis = pickle.load(f)

identity_tags_mis = id_tags_mis[:10]
identity_tags_notmis = id_tags_notmis[:10]


##################BMA####################
 
 
def init_dataset(dataset, tag_column):
    #taggy = []
    #for i in range(0, len(dataset)):
    #    #img_tag = []
#
    #    lista_tags_row=ast.literal_eval(dataset["tag list"][i])
    #    #print(lista_tags_row)
    #    #print("lista tags riga ", i, " : ", lista_tags_row
    #    #conf > 0.5 aPPEND(0 1)
    #    stringa = ' '.join(lista_tags_row)
    #    print(stringa)
    #    #img_tag.append(item["class"])
    #    #unique_tags = list(set(img_tag))
    #    taggy.append(stringa)
    #    #max_conf = max(img_tag)
    #    #print("max conf ", max_conf)    
    #dataset["tag_string"] = taggy
    #print(dataset.info()) 
    identity_tags_mis = id_tags_mis[:10]
    identity_tags_notmis = id_tags_notmis[:10]
    identity_tags = identity_tags_mis + identity_tags_notmis
    meme_tags = []
    for index, row in dataset.iterrows():
        tmp = []
        #print("tag unico: ", tag)
        tags = ast.literal_eval(row[tag_column])
        for i in range(0, len(tags)):
            tmp.append(tags[i]["class"])
            #print("max conf ", max_conf)    
        meme_tags.append(list(set(tmp)))
        #result[tag] = tmp
    dataset["tag list"] = meme_tags     
     
    for id_tag in identity_tags:
        temp = []
        for index, row in dataset.iterrows():
            tags = row["tag list"]
            #print(tokens)
            if id_tag in tags:
                temp.append(1)
            else:
                temp.append(0)
        #if not check_integrity(temp):
        #    #print("OOOOOOOOOOOOOOOOOOOO")
        dataset[id_tag] = temp
    return dataset
                                                                             
                                                                                                            
def check_integrity(presence):
    integrity = False
    for item in presence:
        if item == 1:
            integrity= True
            return integrity

    return integrity


def bma_biased(data_score, probs_path, results_path, dataset):
    
    predictions_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []
    verit_assoluta = []

    y_test_complete = []
    labels_bma_complete=[]

    #data_score =  pd.read_csv(project_paths.csv_uni_tags_syn_scores, sep="\t")
    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        #print(data_probs.info())
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)

            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
              labels_bma.append(0)
            else:
              labels_bma.append(1)

        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma
        
        print("result path ", results_path)
        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")

        results = evaluation_metrics.compute_evaluation_metrics(y_test, labels_bma)

        rec_pos.append(results['recall'][0]) 
        rec_neg.append(results['recall'][1]) 
        f1_pos.append(results['f1'][0])   
        f1_neg.append(results['f1'][1])
        prec_pos.append( results['precision'][0])
        prec_neg.append(results['precision'][1])

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)

        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        #print("ACC BMA ", acc_bma)
        #print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        verit_assoluta.append(y_test)

    verit_assoluta = [item for sublist in verit_assoluta for item in sublist]
    predictions_bma = [item for sublist in predictions_bma for item in sublist]

    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", mean(prec_pos))
    print("precision class 0 of kfold BMA ", mean(prec_neg))
    print("prec ", mean([mean(prec_pos), mean(prec_neg)]))

    print("recall class 1 k fold BMA", mean(rec_pos))
    print("recall class 0 k fold BMA ", mean(rec_neg))
    print("rec ", mean([mean(rec_pos),mean(rec_neg)]))

    print("f1 pos BMA ", mean(f1_pos))
    print("f1 neg BMA ", mean(f1_neg))
    print("f1 ", mean([ mean(f1_pos), mean(f1_neg)]))

def bma_biased_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    
    predictions_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []
    verit_assoluta = []

    y_test_complete = []
    labels_bma_complete=[]

    #data_score =  pd.read_csv(project_paths.csv_uni_tags_syn_scores, sep="\t")
    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        #print(data_probs.info())
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)

            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
              labels_bma.append(0)
            else:
              labels_bma.append(1)

        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma
        
        print("result path ", results_path)
        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")

        results = evaluation_metrics.compute_evaluation_metrics(y_test, labels_bma)

        rec_pos.append(results['recall'][0]) 
        rec_neg.append(results['recall'][1]) 
        f1_pos.append(results['f1'][0])   
        f1_neg.append(results['f1'][1])
        prec_pos.append( results['precision'][0])
        prec_neg.append(results['precision'][1])

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)

        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        #print("ACC BMA ", acc_bma)
        #print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        verit_assoluta.append(y_test)
        j+=1

    verit_assoluta = [item for sublist in verit_assoluta for item in sublist]
    predictions_bma = [item for sublist in predictions_bma for item in sublist]

    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", mean(prec_pos))
    print("precision class 0 of kfold BMA ", mean(prec_neg))
    print("prec ", mean([mean(prec_pos), mean(prec_neg)]))

    print("recall class 1 k fold BMA", mean(rec_pos))
    print("recall class 0 k fold BMA ", mean(rec_neg))
    print("rec ", mean([mean(rec_pos),mean(rec_neg)]))

    print("f1 pos BMA ", mean(f1_pos))
    print("f1 neg BMA ", mean(f1_neg))
    print("f1 ", mean([ mean(f1_pos), mean(f1_neg)]))


def ubma_pos_corr(data_score, probs_path, results_path, dataset):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []



    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path +f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))

def ubma_pos_corr_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []

    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        bias_pos_svm_tags = [bias_tags_dict["svm_pos"]] * len(foldsizes)
        bias_neg_svm_tags = [bias_tags_dict["svm_neg"]] * len(foldsizes)
        bias_pos_KNN_tags = [bias_tags_dict["knn_pos"]] * len(foldsizes)
        bias_neg_KNN_tags = [bias_tags_dict["knn_neg"]] * len(foldsizes)
        bias_pos_NBY_tags = [bias_tags_dict["nby_pos"]] * len(foldsizes)
        bias_neg_NBY_tags = [bias_tags_dict["nby_neg"]] * len(foldsizes)
        bias_pos_DTR_tags = [bias_tags_dict["dtr_pos"]] * len(foldsizes)
        bias_neg_DTR_tags = [bias_tags_dict["dtr_neg"]] * len(foldsizes)
        bias_pos_MLP_tags = [bias_tags_dict["mlp_pos"]] * len(foldsizes)
        bias_neg_MLP_tags = [bias_tags_dict["mlp_neg"]] * len(foldsizes)
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma
        

        probs_name = results_path +f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))

def ubma_neg_corr(data_score, probs_path, results_path, dataset):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []



    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')

        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path +f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma])) 

def ubma_neg_corr_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []



    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        bias_pos_svm_tags = [bias_tags_dict["svm_pos"]] * len(foldsizes)
        bias_neg_svm_tags = [bias_tags_dict["svm_neg"]] * len(foldsizes)
        bias_pos_KNN_tags = [bias_tags_dict["knn_pos"]] * len(foldsizes)
        bias_neg_KNN_tags = [bias_tags_dict["knn_neg"]] * len(foldsizes)
        bias_pos_NBY_tags = [bias_tags_dict["nby_pos"]] * len(foldsizes)
        bias_neg_NBY_tags = [bias_tags_dict["nby_neg"]] * len(foldsizes)
        bias_pos_DTR_tags = [bias_tags_dict["dtr_pos"]] * len(foldsizes)
        bias_neg_DTR_tags = [bias_tags_dict["dtr_neg"]] * len(foldsizes)
        bias_pos_MLP_tags = [bias_tags_dict["mlp_pos"]] * len(foldsizes)
        bias_neg_MLP_tags = [bias_tags_dict["mlp_neg"]] * len(foldsizes)
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')

        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path +f'{j+1}.csv'
        print("result ", results_path)
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma])) 

def ubma_dyn_sub_models(data_score, probs_path, results_path, dataset):
    #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis
    print("###############################mis#############################################")
    print(identity_mis)
    print("###############################non mis#############################################")
    print(identity_notmis)
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []
    
    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        #pres_mis = False
        #pres_not_mis = False
        #for identity in identity_tags_mis:
        #    if data
        for i in range(len(dataset)):
            pres_mis = False
            pres_not_mis = False
            for id_tag in identity_mis:
                if data_probs[id_tag][i] > 0:
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] > 0:
                    pres_not_mis = True

            if pres_mis and pres_not_mis:
              print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
              correzione.append("neu")
            elif pres_mis and not(pres_not_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
              correzione.append("pos")
            elif pres_not_mis and not(pres_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
              correzione.append("neg")
            elif not (pres_not_mis) and not (pres_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
              correzione.append("nan")
            #print("MARGINALE 0 ", marginale_0_)
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
        ##########################   
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN

        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["CORREZIONE USATA"] = correzione
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma
        print("result path ", results_path)
        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
  
def ubma_dyn_sub_models_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis
    print("###############################mis#############################################")
    print(identity_mis)
    print("###############################non mis#############################################")
    print(identity_notmis)
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []

    
    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        bias_pos_svm_tags = [bias_tags_dict["svm_pos"]] * len(foldsizes)
        bias_neg_svm_tags = [bias_tags_dict["svm_neg"]] * len(foldsizes)
        bias_pos_KNN_tags = [bias_tags_dict["knn_pos"]] * len(foldsizes)
        bias_neg_KNN_tags = [bias_tags_dict["knn_neg"]] * len(foldsizes)
        bias_pos_NBY_tags = [bias_tags_dict["nby_pos"]] * len(foldsizes)
        bias_neg_NBY_tags = [bias_tags_dict["nby_neg"]] * len(foldsizes)
        bias_pos_DTR_tags = [bias_tags_dict["dtr_pos"]] * len(foldsizes)
        bias_neg_DTR_tags = [bias_tags_dict["dtr_neg"]] * len(foldsizes)
        bias_pos_MLP_tags = [bias_tags_dict["mlp_pos"]] * len(foldsizes)
        bias_neg_MLP_tags = [bias_tags_dict["mlp_neg"]] * len(foldsizes)
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        #pres_mis = False
        #pres_not_mis = False
        #for identity in identity_tags_mis:
        #    if data
        for i in range(len(foldsizes)):
            pres_mis = False
            pres_not_mis = False
            for id_tag in identity_mis:
                if data_probs[id_tag][i] > 0:
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] > 0:
                    pres_not_mis = True

            if pres_mis and pres_not_mis:
              print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
              correzione.append("neu")
            elif pres_mis and not(pres_not_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
              correzione.append("pos")
            elif pres_not_mis and not(pres_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
              correzione.append("neg")
            elif not (pres_not_mis) and not (pres_mis):
              marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
              marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j]) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])
              correzione.append("nan")
            #print("MARGINALE 0 ", marginale_0_)
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
        ##########################   
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)

        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["CORREZIONE USATA"] = correzione
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma
        print("result path ", results_path)
        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))  
    
def ubma_neu(data_score, probs_path, results_path, dataset):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []



    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')

        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) +  (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path +f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
    
def ubma_neu_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    
    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []


    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        bias_pos_svm_tags = [bias_tags_dict["svm_pos"]] * len(foldsizes)
        bias_neg_svm_tags = [bias_tags_dict["svm_neg"]] * len(foldsizes)
        bias_pos_KNN_tags = [bias_tags_dict["knn_pos"]] * len(foldsizes)
        bias_neg_KNN_tags = [bias_tags_dict["knn_neg"]] * len(foldsizes)
        bias_pos_NBY_tags = [bias_tags_dict["nby_pos"]] * len(foldsizes)
        bias_neg_NBY_tags = [bias_tags_dict["nby_neg"]] * len(foldsizes)
        bias_pos_DTR_tags = [bias_tags_dict["dtr_pos"]] * len(foldsizes)
        bias_neg_DTR_tags = [bias_tags_dict["dtr_neg"]] * len(foldsizes)
        bias_pos_MLP_tags = [bias_tags_dict["mlp_pos"]] * len(foldsizes)
        bias_neg_MLP_tags = [bias_tags_dict["mlp_neg"]] * len(foldsizes)
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path + f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')

        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) +  (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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
            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################
        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)
        data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_tags
        data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_tags
        data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_tags
        data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_tags
        data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_tags
        data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_tags
        data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_tags
        data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_tags
        data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_tags
        data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_tags
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path +f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        data_probs.to_csv(probs_name, sep="\t")
        #data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
    
def ubma_dyn_corr_bma(data_score, probs_path, results_path, dataset):
        #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis


    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []

    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path +f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        for i in range(len(dataset)):
            pres_mis = False
            pres_not_mis = False
            for id_tag in identity_mis:
                if data_probs[id_tag][i] >0:
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] >0:
                    pres_not_mis = True

            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] ) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])

            label_0, label_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            if pres_mis and pres_not_mis:
                label_0_corr = label_0 * BMA_BIAS_NEG
                label_1_corr = label_1 * BMA_BIAS_POS
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis and not(pres_not_mis):
                label_1_corr = label_1 * BMA_BIAS_POS
                label_0_corr = label_0
            elif pres_not_mis and not(pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0 * BMA_BIAS_NEG
            elif not (pres_not_mis) and not (pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
            sum_prob0_bma.append(label_norm_0) 
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(label_norm_1)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################


        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN

        #data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_TAGS
        #data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_TAGS
        #data_probs["CORREZIONE USATA"] = correzione
        #data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_TAGS
        #data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_TAGS
        #data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_TAGS
        #data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_TAGS
        #data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_TAGS
        #data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_TAGS
        #data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_TAGS
        #data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_TAGS
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
    
def ubma_dyn_corr_bma_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
        #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis


    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []

    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path +f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        for i in range(len(foldsizes)):
            pres_mis = False
            pres_not_mis = False
            for id_tag in identity_mis:
                if data_probs[id_tag][i] >0:
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] >0:
                    pres_not_mis = True

            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] ) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])

            label_0, label_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            if pres_mis and pres_not_mis:
                label_0_corr = label_0 * BMA_BIAS_NEG
                label_1_corr = label_1 * BMA_BIAS_POS
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis and not(pres_not_mis):
                label_1_corr = label_1 * BMA_BIAS_POS
                label_0_corr = label_0
            elif pres_not_mis and not(pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0 * BMA_BIAS_NEG
            elif not (pres_not_mis) and not (pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
            sum_prob0_bma.append(label_norm_0) 
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(label_norm_1)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################


        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)

        #data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_TAGS
        #data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_TAGS
        #data_probs["CORREZIONE USATA"] = correzione
        #data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_TAGS
        #data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_TAGS
        #data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_TAGS
        #data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_TAGS
        #data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_TAGS
        #data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_TAGS
        #data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_TAGS
        #data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_TAGS
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))


def ubma_term_corr_bma(data_score, probs_path, results_path, dataset):
    
    #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis


    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []
    id_dict = {
        "pos": {
        "Woman": 0.109053,
        "Earring": 0.106626,
        "Lip": 0.1052,
        "Strap": 0.1050,
        "Tire": 0.1029,
        "Eyebrow":0.095350,
        "Girl": 0.085626,
        "Teeth": 0.083316,
        "Short": 0.079309,
        "Dress": 0.075707,
        },
        "neg":{
        "Penguin": 0.2726,
        "Cat": 0.2601,
        "Whisker": 0.2282,
        "Beak": 0.1819,
        "Gun": 0.1661,
        "Dog": 0.155689,
        "Toy": 0.1469,
        "Paw": 0.1452,
        "Animal": 0.1445,
        "Bear": 0.1426
        }
    
    }




    for j in range(0,10):

        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni = []

        for i in range(len(dataset)):
            pres_mis = False
            pres_not_mis = False
            terms_pos_to_correct = []
            terms_neg_to_correct = []

            terms_scores_neg = []
            terms_scores_pos =  []
            for id_tag in identity_mis:
                if data_probs[id_tag][i] >0:
                    terms_pos_to_correct.append(id_tag)
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] >0:
                    terms_neg_to_correct.append(id_tag)
                    pres_not_mis = True

            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] ) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])

            label_0, label_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)

            if pres_mis and pres_not_mis:
                for term in terms_pos_to_correct:
                    score_term = id_dict["pos"][term]
                    terms_scores_pos.append(score_term)
                for term in terms_neg_to_correct:
                    score_term = id_dict["neg"][term]
                    terms_scores_neg.append(score_term)
                penalization_pos = mean(terms_scores_pos)
                penalization_neg = mean(terms_scores_neg)
                penalizzazioni.append((penalization_neg, penalization_pos))
                label_0_corr = label_0 * penalization_neg
                label_1_corr = label_1 * penalization_pos
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis and not(pres_not_mis):
                for term in terms_pos_to_correct:
                    score_term = id_dict["pos"][term]
                    terms_scores_pos.append(score_term)
                penalization = mean(terms_scores_pos)
                penalizzazioni.append((0, penalization))
                label_1_corr = label_1 * penalization
                label_0_corr = label_0
            elif pres_not_mis and not(pres_mis):
                for term in terms_neg_to_correct:
                    score_term = id_dict["neg"][term]
                    terms_scores_neg.append(score_term)
                penalization = mean(terms_scores_neg)
                penalizzazioni.append((penalization, 1))
                label_1_corr = label_1
                label_0_corr = label_0 * penalization
            elif not(pres_not_mis) and not(pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
            sum_prob0_bma.append(label_norm_0) 
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(label_norm_1)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)
            

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################


        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*DATA_LEN
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*DATA_LEN
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*DATA_LEN
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*DATA_LEN
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*DATA_LEN

        #data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_TAGS
        #data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_TAGS
        #data_probs["CORREZIONE USATA"] = correzione
        #data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_TAGS
        #data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_TAGS
        #data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_TAGS
        #data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_TAGS
        #data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_TAGS
        #data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_TAGS
        #data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_TAGS
        #data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_TAGS
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
    
def ubma_term_corr_bma_sintest(data_score, probs_path, results_path, dataset, syn_folds, keyfold):
    
    #y_bma_pred = []
    identity_mis = identity_tags_mis
    identity_notmis = identity_tags_notmis


    predictions_bma = []
    tp_bma = []
    tn_bma = []
    fn_bma = []
    fp_bma = []
    auc_bma_list = []
    acc_bma_list = []
    rec_pos = []
    rec_neg = []
    prec_pos = []
    prec_neg = []
    f1_pos = []
    f1_neg = []
    id_dict = {
        "pos": {
        "Woman": 0.109053,
        "Earring": 0.106626,
        "Lip": 0.1052,
        "Strap": 0.1050,
        "Tire": 0.1029,
        "Eyebrow":0.095350,
        "Girl": 0.085626,
        "Teeth": 0.083316,
        "Short": 0.079309,
        "Dress": 0.075707,
        },
        "neg":{
        "Penguin": 0.2726,
        "Cat": 0.2601,
        "Whisker": 0.2282,
        "Beak": 0.1819,
        "Gun": 0.1661,
        "Dog": 0.155689,
        "Toy": 0.1469,
        "Paw": 0.1452,
        "Animal": 0.1445,
        "Bear": 0.1426
        }
    
    }



    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold = probs_path+f"{j+1}.csv"
        result = pd.read_csv(n_fold, sep="\t")
        data_probs = pd.merge(dataset,result,on='file_name')
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni = []

        for i in range(len(foldsizes)):
            pres_mis = False
            pres_not_mis = False
            terms_pos_to_correct = []
            terms_neg_to_correct = []

            terms_scores_neg = []
            terms_scores_pos =  []
            for id_tag in identity_mis:
                if data_probs[id_tag][i] >0:
                    terms_pos_to_correct.append(id_tag)
                    pres_mis = True
            for id_tag in identity_notmis:
                if data_probs[id_tag][i] >0:
                    terms_neg_to_correct.append(id_tag)
                    pres_not_mis = True

            marginale_0_ = (data_probs["SVM PROB 0"][i]* data_score["SCORE 0 SVM"][j]) + (data_probs["KNN PROB 0"][i]* data_score["SCORE 0 KNN"][j]) + (data_probs["NB PROB 0"][i]* data_score["SCORE 0 NB"][j]) +  (data_probs["DT PROB 0"][i]* data_score["SCORE 0 DT"][j]) +  (data_probs["MLP PROB 0"][i]* data_score["SCORE 0 MLP"][j])
            marginale_1_ = (data_probs["SVM PROB 1"][i]* data_score["SCORE 1 SVM"][j] ) + (data_probs["KNN PROB 1"][i]* data_score["SCORE 1 KNN"][j]) + (data_probs["NB PROB 1"][i]* data_score["SCORE 1 NB"][j]) +  (data_probs["DT PROB 1"][i]* data_score["SCORE 1 DT"][j]) +  (data_probs["MLP PROB 1"][i]* data_score["SCORE 1 MLP"][j])

            label_0, label_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)

            if pres_mis and pres_not_mis:
                for term in terms_pos_to_correct:
                    score_term = id_dict["pos"][term]
                    terms_scores_pos.append(score_term)
                for term in terms_neg_to_correct:
                    score_term = id_dict["neg"][term]
                    terms_scores_neg.append(score_term)
                penalization_pos = mean(terms_scores_pos)
                penalization_neg = mean(terms_scores_neg)
                penalizzazioni.append((penalization_neg, penalization_pos))
                label_0_corr = label_0 * penalization_neg
                label_1_corr = label_1 * penalization_pos
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis and not(pres_not_mis):
                for term in terms_pos_to_correct:
                    score_term = id_dict["pos"][term]
                    terms_scores_pos.append(score_term)
                penalization = mean(terms_scores_pos)
                penalizzazioni.append((0, penalization))
                label_1_corr = label_1 * penalization
                label_0_corr = label_0
            elif pres_not_mis and not(pres_mis):
                for term in terms_neg_to_correct:
                    score_term = id_dict["neg"][term]
                    terms_scores_neg.append(score_term)
                penalization = mean(terms_scores_neg)
                penalizzazioni.append((penalization, 1))
                label_1_corr = label_1
                label_0_corr = label_0 * penalization
            elif not(pres_not_mis) and not(pres_mis):
                label_1_corr = label_1
                label_0_corr = label_0
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
            sum_prob0_bma.append(label_norm_0) 
            sum_prob1_bma.append(label_norm_1)
            #y_neg = nb_probs_neg[i] + svm_probs_neg[i] +rf_probs_neg[i]
            #y_pos = nb_probs_pos[i] +svm_probs_pos[i] +rf_probs_pos[i]
            y_prob_auc.append(label_norm_1)
            if label_norm_0 > label_norm_1:
            #if probs_sum_0[i] > probs_sum_1[i]:
              labels_bma.append(0)
            elif label_norm_1 > label_norm_0:
              labels_bma.append(1)
            

            #if y_neg > y_pos:
            #  y_bma_pred.append(0)
            #else:
          #  y_bma_pred.append(1)
    ##########################


        data_probs["SCORE 1 SVM"] = [data_score["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 SVM"] = [data_score["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs["SCORE 0 KNN"] = [data_score["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 1 KNN"] = [data_score["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs["SCORE 0 NB"] =  [data_score["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs["SCORE 1 NB"] =  [data_score["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs["SCORE 0 DT"] =  [data_score["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs["SCORE 1 DT"] =  [data_score["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs["SCORE 0 MLP"] = [data_score["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs["SCORE 1 MLP"] = [data_score["SCORE 1 MLP"][j]]*len(foldsizes)

        #data_probs["AUC_FINAL POS SVM TAGS"]= bias_pos_svm_TAGS
        #data_probs["AUC_FINAL NEG SVM TAGS"]= bias_neg_svm_TAGS
        #data_probs["CORREZIONE USATA"] = correzione
        #data_probs["AUC_FINAL POS KNN TAGS"]= bias_pos_KNN_TAGS
        #data_probs["AUC_FINAL NEG KNN TAGS"]= bias_neg_KNN_TAGS
        #data_probs["AUC_FINAL POS NBY TAGS"]= bias_pos_NBY_TAGS
        #data_probs["AUC_FINAL NEG NBY TAGS"]= bias_neg_NBY_TAGS
        #data_probs["AUC_FINAL POS DTR TAGS"]= bias_pos_DTR_TAGS
        #data_probs["AUC_FINAL NEG DTR TAGS"]= bias_neg_DTR_TAGS
        #data_probs["AUC_FINAL POS MLP TAGS"]= bias_pos_MLP_TAGS
        #data_probs["AUC_FINAL NEG MLP TAGS"]= bias_neg_MLP_TAGS
        data_probs["BMA PROB 0"] = sum_prob0_bma
        data_probs["BMA PROB 1"] = sum_prob1_bma
        data_probs["BMA LABELS"] = labels_bma

        probs_name = results_path+f'{j+1}.csv'
        #result = pd.merge(dataset,data_probs,on='file_name')
        #result.to_csv(probs_name, sep="\t")
        data_probs.to_csv(probs_name, sep="\t")
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, labels_bma).ravel()
        tp_bma.append(tp_b)
        tn_bma.append(tn_b)
        fn_bma.append(fn_b)
        fp_bma.append(fp_b)
        rec_pos_bm = tp_b/ (tp_b + fn_b) ###True postive rate recall classe 1
        fn_rate_bm = fn_b/ (tp_b+ fn_b)
        prec_mis_bm =  tp_b/ (tp_b + fp_b)
        prec_notmis_bm = tn_b/ (tn_b + fn_b)
        rec_neg_bm = tn_b / (tn_b+ fp_b)
        f1_1_bm= (2* (prec_mis_bm * rec_pos_bm)) / (prec_mis_bm+ rec_pos_bm) 
        f1_0_bm = (2* (prec_notmis_bm * rec_neg_bm)) / (prec_notmis_bm + rec_neg_bm)
        false_positive_rate_bma = fp_b / (fp_b+ tn_b)
        rec_pos.append(rec_pos_bm) 
        rec_neg.append(rec_neg_bm) 
        f1_pos.append(f1_1_bm)   
        f1_neg.append(f1_0_bm)
        prec_pos.append(prec_mis_bm)
        prec_neg.append(prec_notmis_bm)

        fpr_bma, tpr_bma, thresholds_bma = roc_curve(y_test, y_prob_auc)
        roc_auc_bma = auc(fpr_bma, tpr_bma)
        #printResult(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        j+=1
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)

    #auc_score_bma = roc_auc_score(y_test, y_prob_auc)

    #print("################  BMA calcolo alternativo #############################")
    rec_pos_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fn_bma)) ###True postive rate recall classe 1
    fn_rate_bma = sum(fn_bma)/ (sum(tp_bma)+ sum(fn_bma))
    prec_mis_bma = sum(tp_bma)/ (sum(tp_bma) + sum(fp_bma))
    prec_notmis_bma = sum(tn_bma)/ (sum(tn_bma) + sum(fn_bma))
    rec_neg_bma = sum(tn_bma) / (sum(tn_bma)+ sum(fp_bma))
    false_positive_rate_bma = sum(fp_bma) / (sum(fp_bma)+ sum(tn_bma))
    f1_1_bma= (2* (prec_mis_bma * rec_pos_bma)) / (prec_mis_bma+ rec_pos_bma) 
    f1_0_bma = (2* (prec_notmis_bma * rec_neg_bma)) / (prec_notmis_bma + rec_neg_bma)


    print("################  BMA #############################")
    print("ACC BMA ", acc_bma_list)
    print("ACC BMA ", sum(acc_bma_list)/10)
    print("AUC BMA ", auc_bma_list)
    print("AUC BMA ", sum(auc_bma_list)/10)

    print("precision class 1 of k fold BMA ", prec_mis_bma)
    print("precision class 0 of kfold BMA ", prec_notmis_bma)
    print("prec ", mean([prec_mis_bma, prec_notmis_bma]))

    print("recall class 1 k fold BMA", rec_pos_bma)
    print("recall class 0 k fold BMA ", rec_neg_bma)
    print("rec ", mean([rec_pos_bma, rec_neg_bma]))

    print("f1 pos BMA ", f1_1_bma)
    print("f1 neg BMA ", f1_0_bma)
    print("f1 ", mean([f1_0_bma, f1_1_bma]))
############END BMA######################


if (correction_strategy == "neu"):
    result_path = project_paths.out_corr_bma_neu_tags
elif (correction_strategy == "pos"):
    result_path = project_paths.out_corr_bma_pos_tags
if (correction_strategy == "neg"):
    result_path = project_paths.out_corr_bma_neg_tags
elif (correction_strategy == "dyn_base"):
    result_path = project_paths.out_corr_dyn_base_tags
elif (correction_strategy == "dyn_bma"):
    result_path = project_paths.out_corr_dyn_bma_tags
elif (correction_strategy == "terms_base"):
    result_path = project_paths.out_corr_terms_base_tags
elif (correction_strategy == "terms_bma"):
    result_path = project_paths.out_corr_terms_bma_tags
elif(correction_strategy == "none"):
    result_path = project_paths.out_bma_biased_tags
                
if (data_to_eval =="syn"): 
    dataset = load_data.load_syn_data_tag()
    dataset.rename(columns={'0': 'file_name'}, inplace= True)
    dataset.drop(columns=dataset.columns[8:], axis=1,inplace=True)
    dataset.drop(columns = ["Unnamed: 0.1", "cleaned", "lemmas", "tag list"], axis=1, inplace=True)  
    tag_column = "1"
    print(dataset.info())
    print(dataset["file_name"][0])
    print(dataset.info())
    if key_of_folds == "mitigation":
        probs_path = project_paths.csv_uni_tags_syn_probs
        score_path = project_paths.csv_uni_tags_syn_scores
        out_path = result_path + "sintest/probs_sin_test_fold_"
    if key_of_folds == "measure":
        probs_path = "../data/results2strategy/tags/new_sintest/measure/probs_sin_test_fold_"
        score_path =  project_paths.csv_uni_tags_syn_scores
        out_path = result_path + "new_sintest/measure/probs_sin_test_fold_"
elif(data_to_eval =="test"):
    dataset = load_data.load_test_data_tag()
    dataset.drop(columns=dataset.columns[3:], axis=1,inplace=True)
    print(dataset.info())
    tag_column = "1_y"
    probs_path = project_paths.csv_uni_tags_test_probs
    score_path = project_paths.csv_uni_tags_test_scores
    out_path = result_path + "test/probs_test_fold_"
    
if (correction_strategy != "neu" and correction_strategy != "none" and correction_strategy != "pos" and correction_strategy != "neg" ):
    dataset = init_dataset(dataset, tag_column)
    print(dataset.info())
    #model_bias_analysis.add_subgroup_columns_from_text(dataset, 'tag_string', model_bias_analysis.identity_tags)

DATA_LEN = len(dataset)
print(DATA_LEN)


with open('../data/datasets/synthetic_folds.pkl', 'rb') as f:
    syn_folds = pickle.load(f)
    

    
with open('../data/datasets/bias_mitigation_tags.pkl', 'rb') as f:
    bias_tags_dict = pickle.load(f)

#bias_tags_dict["svm_neg"] = 0.5725
#bias_tags_dict["knn_neg"] = 0.5575
#bias_tags_dict["nby_neg"] = 0.5592
#bias_tags_dict["dtr_neg"] = 0.5474
#bias_tags_dict["mlp_neg"] = 0.5884
#
#bias_tags_dict["svm_pos"] = 0.6303
#bias_tags_dict["knn_pos"] = 0.6256
#bias_tags_dict["nby_pos"] = 0.5983
#bias_tags_dict["dtr_pos"] = 0.6168
#bias_tags_dict["mlp_pos"] = 0.6439
#
bias_pos_svm_tags = [bias_tags_dict["svm_pos"]] * DATA_LEN
bias_neg_svm_tags = [bias_tags_dict["svm_neg"]] * DATA_LEN
bias_pos_KNN_tags = [bias_tags_dict["knn_pos"]] * DATA_LEN
bias_neg_KNN_tags = [bias_tags_dict["knn_neg"]] * DATA_LEN
bias_pos_NBY_tags = [bias_tags_dict["nby_pos"]] * DATA_LEN
bias_neg_NBY_tags = [bias_tags_dict["nby_neg"]] * DATA_LEN
bias_pos_DTR_tags = [bias_tags_dict["dtr_pos"]] * DATA_LEN
bias_neg_DTR_tags = [bias_tags_dict["dtr_neg"]] * DATA_LEN
bias_pos_MLP_tags = [bias_tags_dict["mlp_pos"]] * DATA_LEN
bias_neg_MLP_tags = [bias_tags_dict["mlp_neg"]] * DATA_LEN

BMA_BIAS_POS = 0.6171
BMA_BIAS_NEG = 0.5868




print(score_path)

data_score =  load_data.load_data_score(score_path)

#if (correction_strategy == "neu"):
#    ubma_neu(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "pos"):
#    ubma_pos_corr(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "neg"):
#    ubma_neg_corr(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "dyn_base"):
#    ubma_dyn_sub_models(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "dyn_bma"):
#    ubma_dyn_corr_bma(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "terms_base"):
#    ubma_term_corr_bma(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "terms_bma"):
#    ubma_term_corr_bma(data_score, probs_path, out_path, dataset)
#elif (correction_strategy == "none"):
#    bma_biased(data_score, probs_path, out_path, dataset)
    


if (correction_strategy == "neu" and data_to_eval == "test"):
    ubma_neu(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "neu" and data_to_eval == "syn"):
    ubma_neu_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif(correction_strategy == "neg"and data_to_eval == "test" ):
    ubma_neg_corr(data_score, probs_path, out_path, dataset)
elif(correction_strategy == "neg"and data_to_eval == "syn" ):
    ubma_neg_corr_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif(correction_strategy == "pos" and data_to_eval == "test"):
    ubma_pos_corr(data_score, probs_path, out_path, dataset)
elif(correction_strategy == "pos" and data_to_eval == "syn"):
    ubma_pos_corr_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif (correction_strategy == "dyn_base" and data_to_eval == "test"):
    ubma_dyn_sub_models(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "dyn_base" and data_to_eval == "syn"):
    ubma_dyn_sub_models_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif (correction_strategy == "dyn_bma" and data_to_eval == "test"):
    ubma_dyn_corr_bma(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "dyn_bma" and data_to_eval == "syn"):
    ubma_dyn_corr_bma_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif (correction_strategy == "terms_base" and data_to_eval == "test"):
    ubma_term_corr_bma(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "terms_base" and data_to_eval == "syn"):
    ubma_term_corr_bma_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds) 
elif (correction_strategy == "terms_bma" and data_to_eval == "test"):
    ubma_term_corr_bma(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "terms_bma" and data_to_eval == "syn"):
    ubma_term_corr_bma_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
elif (correction_strategy == "none" and data_to_eval == "test"):
    bma_biased(data_score, probs_path, out_path, dataset)
elif (correction_strategy == "none" and data_to_eval == "syn"):
    bma_biased_sintest(data_score, probs_path, out_path, dataset, syn_folds, key_of_folds)
#dataset = pd.read_csv(data_path, sep="\t")


    




