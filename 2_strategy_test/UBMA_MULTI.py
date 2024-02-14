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
import spacy_stanza
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
#parser.add_argument("-s", "--score_path_text",  default="../data/result_texts2strategy/text/sintest/score_sin_test_text10fold.csv", type=str, help="path of output models")
#parser.add_argument("-p", "--probs_path_text",  default="../data/result_texts2strategy/text/sintest/probs_sin_test_fold_", type=str, help="path where prob result_texts are located")
#parser.add_argument("-r", "--result_path",  default='result_texts/text_bma_neu/sintest/probs_sintest_', type=str, help="syn result_texts path")

parser.add_argument("-d", "--data_to_eval",  default='syn', type=str, help="dataset test o syntest")
parser.add_argument("-c", "--correction_strategy",  default='neu', type=str, help="test")
parser.add_argument("-m", "--modality",  default='multi', type=str, help="test")
parser.add_argument('--mitigation', type=str_to_bool, nargs='?', const=True, default=False)




args = parser.parse_args()
config_arg= vars(args)
print(config_arg)
data_to_eval = args.data_to_eval
correction_strategy = args.correction_strategy
modality_correction = args.modality
data_path = ""
dataset = pd.DataFrame()
result_path = ""
score_path_text = ""
out_path = ""
fold_mitigation = args.mitigation
if fold_mitigation:
    key_of_folds = "mitigation"
else:
    key_of_folds = "measure"


##################BMA####################

def init_dataset(dataset, text_column, tag_column):
    # ____________________________________________________Laod Data______________________________________________

    #identity_terms = [['dishwasher', 'chick', 'whore', 'demotivational', 'diy', 'promotion', 'bestdemotivationalposters', 'motivateusnot', 'imgur', 'motifake'], ['memeshappen', 'mcdonald', 'ambulance', 'developer', 'template', 'anti', 'valentine', 'communism', 'weak', 'memecrunch']]
    identity_terms = [['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong', "blonde"], ['mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']]
    identity_text = identity_terms[0] + identity_terms[1]
    identity_tags_mis = id_tags_mis[:10]
    identity_tags_notmis = id_tags_notmis[:10]
    identity_tags = identity_tags_mis + identity_tags_notmis

    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")

    
    # _______________________________________________UTILS_______________________________________________________


    for index, row in tqdm(dataset.iterrows()):
        dataset.loc[index, 'clear text'] = str(preprocessing.text_preprocessing(row[text_column],nlp)).replace("'", '').replace(",", '').replace("[",
                                                                                                                '').replace(
            "]", '').replace("\"", '')
                                                                                                             
    for identity in identity_text:
        temp = []
        for index, row in dataset.iterrows():
            tokens = row["clear text"].lower().split(" ")
            #print(tokens)
            if identity in tokens:
                temp.append(1)
            else:
                temp.append(0)
        #if not check_integrity(temp):
        #    print("OOOOOOOOOOOOOOOOOOOO")
        dataset[identity] = temp


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
    #taggy = []
    #for i in range(0, len(dataset)):
    #    img_tag = []
#
    #    lista_tags_row=dataset["tag list"][i]
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

def bma_biased(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset):
    
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
    #data_score_text =  pd.read_csv(project_paths.csv_uni_text_syn_scores, sep="\t")
    #data_score_tags =  pd.read_csv(project_paths.csv_uni_tags_test_scores, sep="\t")
    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        print("DATA TEXT LEN",len(y_test))
        print("data len y test tag ", len(y_test_tag))
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)

            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
              labels_bma.append(0)
            else:
              labels_bma.append(1)
              
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        print("result_text path ", result_path)
        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")

        result = evaluation_metrics.compute_evaluation_metrics(y_test, labels_bma)

        rec_pos.append(result['recall'][0]) 
        rec_neg.append(result['recall'][1]) 
        f1_pos.append(result['f1'][0])   
        f1_neg.append(result['f1'][1])
        prec_pos.append(result['precision'][0])
        prec_neg.append(result['precision'][1])

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
    
def bma_biased_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, syn_folds, keyfold):
    
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
    #data_score_text =  pd.read_csv(project_paths.csv_uni_text_syn_scores, sep="\t")
    #data_score_tags =  pd.read_csv(project_paths.csv_uni_tags_test_scores, sep="\t")
    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        print("DATA TEXT LEN",len(y_test))
        print("data len y test tag ", len(y_test_tag))
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            sum_prob0_bma.append(label_norm_0)
            sum_prob1_bma.append(label_norm_1)

            y_prob_auc.append(marginale_1_)
            if label_norm_0 > label_norm_1:
              labels_bma.append(0)
            else:
              labels_bma.append(1)
              
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        print("result_text path ", result_path)
        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")

        result = evaluation_metrics.compute_evaluation_metrics(y_test, labels_bma)

        rec_pos.append(result['recall'][0]) 
        rec_neg.append(result['recall'][1]) 
        f1_pos.append(result['f1'][0])   
        f1_neg.append(result['f1'][1])
        prec_pos.append(result['precision'][0])
        prec_neg.append(result['precision'][1])

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

def ubma_dyn_sub_models(data_score_text, data_score_tags,probs_path_text, probs_path_tags, result_path, dataset, modality):
    #y_bma_pred = []
    print(modality)
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print("###############################mis#############################################")
    print(identity_terms_mis)
    print("###############################non mis#############################################")
    print(identity_terms_notmis)
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
    bbma_preds = []
    ubma_preds = []
    ground_truth = []
    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        correzione_tag = []
        for i in range(len(dataset)):
            pres_mis_text = False
            pres_not_mis_text = False
            for id_term in identity_terms_mis:
                if data_probs_text[id_term][i] ==1:
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_not_mis_tags = True            
            if modality == "multi":
                if pres_mis_text and pres_not_mis_text:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j] * bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] * bias_text_dict["mlp_neg"])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] * bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] * bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] * bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                  correzione.append("neu")
                elif pres_mis_text and not(pres_not_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                  correzione.append("pos")
                elif pres_not_mis_text and not(pres_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                  correzione.append("neg")
                elif not (pres_not_mis_text) and not (pres_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                  correzione.append("nan")
                if pres_mis_tags and pres_not_mis_tags:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("neu")
                elif pres_mis_tags and not(pres_not_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("pos")
                elif pres_not_mis_tags and not(pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("neg")
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("nan")
                       
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j] * bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] * bias_text_dict["mlp_neg"])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] * bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] * bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] * bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                    correzione.append("neu")
                elif pres_mis_text and not(pres_not_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                    correzione.append("pos")
                elif pres_not_mis_text and not(pres_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                    correzione.append("neg")
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                    correzione.append("nan")
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                correzione_tag.append("nan")
            
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                correzione.append("nan")
                if pres_mis_tags and pres_not_mis_tags:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("neu")
                elif pres_mis_tags and not(pres_not_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("pos")
                elif pres_not_mis_tags and not(pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("neg")
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("nan")
                                
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            
            #print("MARGINALE 0 ", marginale_0_text)
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
        #####################################################################àààà   
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        data_probs_text["CORREZIONE USATA"] = correzione
        data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        bma_non_corretto_pred = data_probs_text["BMA LABELS"]
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        
        bma_corretto_pred = labels_bma
        bbma_preds.append(bma_non_corretto_pred)
        ubma_preds.append(bma_corretto_pred)
        print("result_text path ", result_path)
        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        ground_truth.append(y_test)
        #######mcnemar stats#############
        #pospos = 0
        #posneg = 0
        #negpos = 0
        #negneg = 0
        #for k in range(len(bma_non_corretto_pred)):
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 1 and y_test[k] == 1:
        #        pospos+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 0 and y_test[k] == 0:
        #        pospos+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 0 and y_test[k] == 1:
        #        posneg+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 1 and y_test[k] == 0:
        #        posneg+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 1 and y_test[k] == 1:
        #        negpos+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 0 and y_test[k] == 0:
        #        negpos+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 0 and y_test[k] == 1:
        #        negneg+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 1 and y_test[k] == 0:
        #        negneg+=1
        #print("fold ", j+1, " ", pospos, posneg, negpos, negneg)
        
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)
    #pospos = 0
    #posneg = 0
    #negpos = 0
    #negneg = 0
    #bbma_preds = [item for sublist in bbma_preds for item in sublist]
    #print(len(bbma_preds))
    #ubma_preds = [item for sublist in ubma_preds for item in sublist]
    #verita = [item for sublist in ground_truth for item in sublist]
    ##auc_score_bma = roc_auc_score(y_test, y_prob_auc)
    #for k in range(len(bbma_preds)):
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 1 and verita[k] == 1:
    #        pospos+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 0 and verita[k] == 0:
    #        pospos+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 0 and verita[k] == 1:
    #        posneg+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 1 and verita[k] == 0:
    #        posneg+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 1 and verita[k] == 1:
    #        negpos+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 0 and verita[k] == 0:
    #        negpos+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 0 and verita[k] == 1:
    #        negneg+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 1 and verita[k] == 0:
    #        negneg+=1
    #print("totale ", pospos, posneg, negpos, negneg)
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

def ubma_dyn_sub_models_sintest(data_score_text, data_score_tags,probs_path_text, probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
    #y_bma_pred = []
    print(modality)
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print("###############################mis#############################################")
    print(identity_terms_mis)
    print("###############################non mis#############################################")
    print(identity_terms_notmis)
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
    bbma_preds = []
    ubma_preds = []
    ground_truth = []
    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(keyfold)
        #print(data_probs_all.info())
        foldsizes = list(val[keyfold])
        
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        correzione_tag = []
        for i in range(len(foldsizes)):
            pres_mis_text = False
            pres_not_mis_text = False
            for id_term in identity_terms_mis:
                if data_probs_text[id_term][i] ==1:
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_not_mis_tags = True            
            if modality == "multi":
                if pres_mis_text and pres_not_mis_text:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j] * bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] * bias_text_dict["mlp_neg"])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] * bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] * bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] * bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                  correzione.append("neu")
                elif pres_mis_text and not(pres_not_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                  correzione.append("pos")
                elif pres_not_mis_text and not(pres_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                  correzione.append("neg")
                elif not (pres_not_mis_text) and not (pres_mis_text):
                  marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                  marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                  correzione.append("nan")
                if pres_mis_tags and pres_not_mis_tags:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("neu")
                elif pres_mis_tags and not(pres_not_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("pos")
                elif pres_not_mis_tags and not(pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("neg")
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("nan")
                       
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j] * bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] * bias_text_dict["mlp_neg"])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] * bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] * bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] * bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                    correzione.append("neu")
                elif pres_mis_text and not(pres_not_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                    correzione.append("pos")
                elif pres_not_mis_text and not(pres_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                    correzione.append("neg")
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                    marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                    correzione.append("nan")
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                correzione_tag.append("nan")
            
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                correzione.append("nan")
                if pres_mis_tags and pres_not_mis_tags:
                  print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte nel syn")
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j] * bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] * bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] * bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] * bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] * bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("neu")
                elif pres_mis_tags and not(pres_not_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
                  correzione_tag.append("pos")
                elif pres_not_mis_tags and not(pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("neg")
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                  marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                  marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
                  correzione_tag.append("nan")
                                
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            
            #print("MARGINALE 0 ", marginale_0_text)
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
        #####################################################################àààà   
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)
        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        bma_non_corretto_pred = data_probs_text["BMA LABELS"]
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        
        bma_corretto_pred = labels_bma
        bbma_preds.append(bma_non_corretto_pred)
        ubma_preds.append(bma_corretto_pred)
        print("result_text path ", result_path)
        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
        auc_bma_list.append(roc_auc_bma)
        acc_bma = accuracy_score(y_test,labels_bma)
        print("ACC BMA ", acc_bma)
        print("AUC BMA ", roc_auc_bma)
        acc_bma_list.append(acc_bma)
        predictions_bma.append(labels_bma)
        ground_truth.append(y_test)
        j+=1
        #######mcnemar stats#############
        #pospos = 0
        #posneg = 0
        #negpos = 0
        #negneg = 0
        #for k in range(len(bma_non_corretto_pred)):
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 1 and y_test[k] == 1:
        #        pospos+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 0 and y_test[k] == 0:
        #        pospos+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 0 and y_test[k] == 1:
        #        posneg+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 1 and y_test[k] == 0:
        #        posneg+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 1 and y_test[k] == 1:
        #        negpos+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 0 and y_test[k] == 0:
        #        negpos+=1
        #    if bma_non_corretto_pred[k] == 0 and bma_corretto_pred[k] == 0 and y_test[k] == 1:
        #        negneg+=1
        #    if bma_non_corretto_pred[k] == 1 and bma_corretto_pred[k] == 1 and y_test[k] == 0:
        #        negneg+=1
        #print("fold ", j+1, " ", pospos, posneg, negpos, negneg)
        
    #print("PROBS POS PER AUC ", y_prob_auc)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_bma_pred).ravel()
    #print(tn,fp,fn,tp)
    #true_postives_bma.append(tp)
    #true_negatives_bma.append(tn)
    #false_negative_bma.append(fn)
    #false_positives_bma.append(fp)
    #pospos = 0
    #posneg = 0
    #negpos = 0
    #negneg = 0
    #bbma_preds = [item for sublist in bbma_preds for item in sublist]
    #print(len(bbma_preds))
    #ubma_preds = [item for sublist in ubma_preds for item in sublist]
    #verita = [item for sublist in ground_truth for item in sublist]
    ##auc_score_bma = roc_auc_score(y_test, y_prob_auc)
    #for k in range(len(bbma_preds)):
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 1 and verita[k] == 1:
    #        pospos+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 0 and verita[k] == 0:
    #        pospos+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 0 and verita[k] == 1:
    #        posneg+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 1 and verita[k] == 0:
    #        posneg+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 1 and verita[k] == 1:
    #        negpos+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 0 and verita[k] == 0:
    #        negpos+=1
    #    if bbma_preds[k] == 0 and ubma_preds[k] == 0 and verita[k] == 1:
    #        negneg+=1
    #    if bbma_preds[k] == 1 and ubma_preds[k] == 1 and verita[k] == 0:
    #        negneg+=1
    #print("totale ", pospos, posneg, negpos, negneg)
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

def ubma_pos_corr(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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

def ubma_pos_corr_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(foldsizes)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)
        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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

def ubma_neg_corr(data_score_text, data_score_tags, probs_path_text,probs_path_tags, result_path, dataset, modality):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])            
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test
        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
    
def ubma_neg_corr_sintest(data_score_text, data_score_tags, probs_path_text,probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])            
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)
        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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

def ubma_neu(data_score_text,data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(dataset)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])     
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN
        data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
    
def ubma_neu_sintest(data_score_text,data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
    print(modality)
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        for i in range(len(foldsizes)):
            if modality == "multi":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            elif modality == "text":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j] * bias_text_dict["svm_neg"]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j] * bias_text_dict["knn_neg"]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j] * bias_text_dict["nby_neg"]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]* bias_text_dict["dtr_neg"]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j] *bias_text_dict["mlp_neg"])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] *bias_text_dict["svm_pos"]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j] *bias_text_dict["knn_pos"]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j] *bias_text_dict["nby_pos"]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j] * bias_text_dict["dtr_pos"]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j] * bias_text_dict["mlp_pos"])
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            elif modality == "tags":
                marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
                marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])     
                marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j] * bias_tags_dict["svm_neg"]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j] * bias_tags_dict["knn_neg"]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j] * bias_tags_dict["nby_neg"]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]* bias_tags_dict["dtr_neg"]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j] *bias_tags_dict["mlp_neg"])
                marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] *bias_tags_dict["svm_pos"]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j] *bias_tags_dict["knn_pos"]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j] *bias_tags_dict["nby_pos"]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j] * bias_tags_dict["dtr_pos"]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j] * bias_tags_dict["mlp_pos"])
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
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
        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)
        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path +f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        data_probs_multi.to_csv(probs_name, sep="\t")
        #data_probs_text.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
    
def ubma_dyn_corr_bma(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality):
        #y_bma_pred = []
    print(modality)
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]


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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        for i in range(len(dataset)):
            pres_mis_text = False
            pres_not_mis_text = False
            for id_term in identity_terms_mis:
                if data_probs_text[id_term][i] ==1:
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    pres_not_mis_text = True
                    
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_not_mis_tags = True 

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] ) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            if modality == "multi":
            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
                if pres_mis_text and pres_not_mis_text:
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text

                if pres_mis_tags and pres_not_mis_tags:
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                    
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text
                label_1_corr_tags = marginale_1_tags
                label_0_corr_tags = marginale_0_tags
            elif modality == "tags":
                if pres_mis_tags and pres_not_mis_tags:
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                    
                label_1_corr_text = marginale_1_text
                label_0_corr_text = marginale_0_text
            label_1_corr = label_1_corr_text + label_1_corr_tags
            label_0_corr = label_0_corr_text + label_0_corr_tags
            #print("MARGINALE 0 ", marginale_0_text)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
    
def ubma_dyn_corr_bma_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
        #y_bma_pred = []
    print(modality)
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]


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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        for i in range(len(foldsizes)):
            pres_mis_text = False
            pres_not_mis_text = False
            for id_term in identity_terms_mis:
                if data_probs_text[id_term][i] ==1:
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    pres_not_mis_text = True
                    
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    pres_not_mis_tags = True 

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j]) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j] ) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])
            if modality == "multi":
            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
                if pres_mis_text and pres_not_mis_text:
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text    
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text

                if pres_mis_tags and pres_not_mis_tags:
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                    
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    label_1_corr_text = marginale_1_text * BMA_BIAS_POS_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text * BMA_BIAS_NEG_text
                elif not (pres_not_mis_text) and not (pres_mis_text):
                    label_1_corr_text = marginale_1_text
                    label_0_corr_text = marginale_0_text
                label_1_corr_tags = marginale_1_tags
                label_0_corr_tags = marginale_0_tags
            elif modality == "tags":
                if pres_mis_tags and pres_not_mis_tags:
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    label_1_corr_tags = marginale_1_tags * BMA_BIAS_POS_tag
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * BMA_BIAS_NEG_tag
                elif not (pres_not_mis_tags) and not (pres_mis_tags):
                    label_1_corr_tags = marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                    
                label_1_corr_text = marginale_1_text
                label_0_corr_text = marginale_0_text
            label_1_corr = label_1_corr_text + label_1_corr_tags
            label_0_corr = label_0_corr_text + label_0_corr_tags
            #print("MARGINALE 0 ", marginale_0_text)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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

def ubma_term_corr_bma(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality):
    print(modality)
    #y_bma_pred = []
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print(identity_terms)

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
    id_dict_text = {
        "pos": {
        "dishwasher": 0.3807,
        "chick": 0.3440,
        "whore": 0.289223,
        "demotivational": 0.3935,
        "diy": 0.281618,
        "promotion": 0.3541,
        "motivate":0.33222,
        "chloroform":0.30898810077639666,
        "blond": 0.30336300854129217,
        "diy":0.3012319342817566,
        "belong":0.28922266696217547,
        "blonde":0.28518423774286483
        },
        "neg":{
        "identify": 0.167053,
        "mcdonald": 0.2634,
        "ambulance": 0.2389,
        "developer": 0.2,
        "template": 0.1998,
        "anti":0.2077,
        "valentine": 0.20125,
        "communism": 0.2274,
        "weak": 0.1858,
        "zipmeme": 0.17642
        }
    
    }
    id_dict_tags = {
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




    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni_text = []
        penalizzazioni_tags = []
        print(data_probs_text.info())
        for i in range(len(dataset)):
            pres_mis_text = False
            pres_not_mis_text = False
            
            terms_pos_to_correct = []
            terms_neg_to_correct = []
            tags_pos_to_correct = []
            tags_neg_to_correct = []
            terms_scores_neg = []
            terms_scores_pos =  []
            tags_scores_neg = []
            tags_scores_pos =  []
            for id_term in identity_terms_mis:
                #print(data_probs_text[id_term][i])
                if data_probs_text[id_term][i] ==1:
                    terms_pos_to_correct.append(id_term)
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    terms_neg_to_correct.append(id_term)
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                #print(data_probs_tags[id_tag][i])
                if data_probs_tags[id_tag][i] ==1:
                    tags_pos_to_correct.append(id_tag)
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    tags_neg_to_correct.append(id_tag)
                    pres_not_mis_tags = True

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] ) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])


            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
            
            if modality == "multi":
                if pres_mis_text and pres_not_mis_text:
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text_pos = mean(terms_scores_pos)
                    penalization_text_neg = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                    label_0_corr_text = marginale_0_text * penalization_text_neg
                    label_1_corr_text= marginale_1_text * penalization_text_pos
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    penalization_text = mean(terms_scores_pos)
                    penalizzazioni_text.append((0, penalization_text))
                    label_1_corr_text= marginale_1_text * penalization_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text, 1))
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text * penalization_text
                elif not(pres_not_mis_text) and not(pres_mis_text):
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text
                if pres_mis_tags and pres_not_mis_tags:
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags_pos = mean(tags_scores_pos)
                    penalization_tags_neg = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                    label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                    label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                    print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    penalization_tags = mean(tags_scores_pos)
                    penalizzazioni_tags.append((0, penalization_tags))
                    label_1_corr_tags= marginale_1_tags * penalization_tags
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags, 1))
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * penalization_tags
                elif not(pres_not_mis_tags) and not(pres_mis_tags):
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text_pos = mean(terms_scores_pos)
                    penalization_text_neg = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                    label_0_corr_text = marginale_0_text * penalization_text_neg
                    label_1_corr_text= marginale_1_text * penalization_text_pos
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    penalization_text = mean(terms_scores_pos)
                    penalizzazioni_text.append((0, penalization_text))
                    label_1_corr_text= marginale_1_text * penalization_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text, 1))
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text * penalization_text
                elif not(pres_not_mis_text) and not(pres_mis_text):
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text
                    
                label_1_corr_tags = marginale_1_tags
                label_0_corr_tags = marginale_0_tags  
            elif modality == "tags":
                if pres_mis_tags and pres_not_mis_tags:
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags_pos = mean(tags_scores_pos)
                    penalization_tags_neg = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                    label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                    label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                    print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    penalization_tags = mean(tags_scores_pos)
                    penalizzazioni_tags.append((0, penalization_tags))
                    label_1_corr_tags= marginale_1_tags * penalization_tags
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags, 1))
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * penalization_tags
                elif not(pres_not_mis_tags) and not(pres_mis_tags):
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                label_1_corr_text = marginale_1_text
                label_0_corr_text = marginale_0_text
                
            marginale_1_ = label_1_corr_text + label_1_corr_tags
            marginale_0_ = label_0_corr_text + label_0_corr_tags
            #if pres_mis_text and pres_not_mis_text:
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization_pos = mean(terms_scores_pos)
            #    penalization_neg = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization_neg, penalization_pos))
            #    label_0_corr = label_0 * penalization_neg
            #    label_1_corr = label_1 * penalization_pos
            #    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            #elif pres_mis_text and not(pres_not_mis_text):
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    penalization = mean(terms_scores_pos)
            #    penalizzazioni.append((0, penalization))
            #    label_1_corr = label_1 * penalization
            #    label_0_corr = label_0
            #elif pres_not_mis_text and not(pres_mis_text):
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization, 1))
            #    label_1_corr = label_1
            #    label_0_corr = label_0 * penalization
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr = label_1
            #    label_0_corr = label_0
            label_norm_0, label_norm_1= evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
    
def ubma_term_corr_bma_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, modality, syn_folds, keyfold):
    print(modality)
    #y_bma_pred = []
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print(identity_terms)

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
    id_dict_text = {
        "pos": {
        "dishwasher": 0.3807,
        "chick": 0.3440,
        "whore": 0.289223,
        "demotivational": 0.3935,
        "diy": 0.281618,
        "promotion": 0.3541,
        "motivate":0.33222,
        "chloroform":0.30898810077639666,
        "blond": 0.30336300854129217,
        "diy":0.3012319342817566,
        "belong":0.28922266696217547,
        "blonde":0.28518423774286483
        },
        "neg":{
        "identify": 0.167053,
        "mcdonald": 0.2634,
        "ambulance": 0.2389,
        "developer": 0.2,
        "template": 0.1998,
        "anti":0.2077,
        "valentine": 0.20125,
        "communism": 0.2274,
        "weak": 0.1858,
        "zipmeme": 0.17642
        }
    
    }
    id_dict_tags = {
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni_text = []
        penalizzazioni_tags = []
        print(data_probs_text.info())
        for i in range(len(foldsizes)):
            pres_mis_text = False
            pres_not_mis_text = False
            
            terms_pos_to_correct = []
            terms_neg_to_correct = []
            tags_pos_to_correct = []
            tags_neg_to_correct = []
            terms_scores_neg = []
            terms_scores_pos =  []
            tags_scores_neg = []
            tags_scores_pos =  []
            for id_term in identity_terms_mis:
                #print(data_probs_text[id_term][i])
                if data_probs_text[id_term][i] ==1:
                    terms_pos_to_correct.append(id_term)
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    terms_neg_to_correct.append(id_term)
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                #print(data_probs_tags[id_tag][i])
                if data_probs_tags[id_tag][i] ==1:
                    tags_pos_to_correct.append(id_tag)
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    tags_neg_to_correct.append(id_tag)
                    pres_not_mis_tags = True

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] ) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])


            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
            
            if modality == "multi":
                if pres_mis_text and pres_not_mis_text:
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text_pos = mean(terms_scores_pos)
                    penalization_text_neg = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                    label_0_corr_text = marginale_0_text * penalization_text_neg
                    label_1_corr_text= marginale_1_text * penalization_text_pos
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    penalization_text = mean(terms_scores_pos)
                    penalizzazioni_text.append((0, penalization_text))
                    label_1_corr_text= marginale_1_text * penalization_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text, 1))
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text * penalization_text
                elif not(pres_not_mis_text) and not(pres_mis_text):
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text
                if pres_mis_tags and pres_not_mis_tags:
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags_pos = mean(tags_scores_pos)
                    penalization_tags_neg = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                    label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                    label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                    print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    penalization_tags = mean(tags_scores_pos)
                    penalizzazioni_tags.append((0, penalization_tags))
                    label_1_corr_tags= marginale_1_tags * penalization_tags
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags, 1))
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * penalization_tags
                elif not(pres_not_mis_tags) and not(pres_mis_tags):
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
            elif modality == "text":
                if pres_mis_text and pres_not_mis_text:
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text_pos = mean(terms_scores_pos)
                    penalization_text_neg = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                    label_0_corr_text = marginale_0_text * penalization_text_neg
                    label_1_corr_text= marginale_1_text * penalization_text_pos
                    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_text and not(pres_not_mis_text):
                    for term in terms_pos_to_correct:
                        score_term = id_dict_text["pos"][term]
                        terms_scores_pos.append(score_term)
                    penalization_text = mean(terms_scores_pos)
                    penalizzazioni_text.append((0, penalization_text))
                    label_1_corr_text= marginale_1_text * penalization_text
                    label_0_corr_text = marginale_0_text
                elif pres_not_mis_text and not(pres_mis_text):
                    for term in terms_neg_to_correct:
                        score_term = id_dict_text["neg"][term]
                        terms_scores_neg.append(score_term)
                    penalization_text = mean(terms_scores_neg)
                    penalizzazioni_text.append((penalization_text, 1))
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text * penalization_text
                elif not(pres_not_mis_text) and not(pres_mis_text):
                    label_1_corr_text= marginale_1_text
                    label_0_corr_text = marginale_0_text
                    
                label_1_corr_tags = marginale_1_tags
                label_0_corr_tags = marginale_0_tags  
            elif modality == "tags":
                if pres_mis_tags and pres_not_mis_tags:
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags_pos = mean(tags_scores_pos)
                    penalization_tags_neg = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                    label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                    label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                    print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
                elif pres_mis_tags and not(pres_not_mis_tags):
                    for tag in tags_pos_to_correct:
                        score_tag = id_dict_tags["pos"][tag]
                        tags_scores_pos.append(score_tag)
                    penalization_tags = mean(tags_scores_pos)
                    penalizzazioni_tags.append((0, penalization_tags))
                    label_1_corr_tags= marginale_1_tags * penalization_tags
                    label_0_corr_tags = marginale_0_tags
                elif pres_not_mis_tags and not(pres_mis_tags):
                    for tag in tags_neg_to_correct:
                        score_tag = id_dict_tags["neg"][tag]
                        tags_scores_neg.append(score_tag)
                    penalization_tags = mean(tags_scores_neg)
                    penalizzazioni_tags.append((penalization_tags, 1))
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags * penalization_tags
                elif not(pres_not_mis_tags) and not(pres_mis_tags):
                    label_1_corr_tags= marginale_1_tags
                    label_0_corr_tags = marginale_0_tags
                label_1_corr_text = marginale_1_text
                label_0_corr_text = marginale_0_text
                
            marginale_1_ = label_1_corr_text + label_1_corr_tags
            marginale_0_ = label_0_corr_text + label_0_corr_tags
            #if pres_mis_text and pres_not_mis_text:
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization_pos = mean(terms_scores_pos)
            #    penalization_neg = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization_neg, penalization_pos))
            #    label_0_corr = label_0 * penalization_neg
            #    label_1_corr = label_1 * penalization_pos
            #    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            #elif pres_mis_text and not(pres_not_mis_text):
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    penalization = mean(terms_scores_pos)
            #    penalizzazioni.append((0, penalization))
            #    label_1_corr = label_1 * penalization
            #    label_0_corr = label_0
            #elif pres_not_mis_text and not(pres_mis_text):
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization, 1))
            #    label_1_corr = label_1
            #    label_0_corr = label_0 * penalization
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr = label_1
            #    label_0_corr = label_0
            label_norm_0, label_norm_1= evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]* len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]* len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]* len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]* len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]* len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]* len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]* len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]* len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]* len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]* len(foldsizes)

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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


def ubma_term_tag_corr(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset):
    
    #y_bma_pred = []
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print(identity_terms)

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
    id_dict_text = {
        "pos": {
        "dishwasher": 0.3807,
        "chick": 0.3440,
        "whore": 0.289223,
        "demotivational": 0.3935,
        "diy": 0.281618,
        "promotion": 0.3541,
        "motivate":0.33222,
        "chloroform":0.30898810077639666,
        "blond": 0.30336300854129217,
        "diy":0.3012319342817566,
        "belong":0.28922266696217547,
        "blonde":0.28518423774286483
        },
        "neg":{
        "identify": 0.167053,
        "mcdonald": 0.2634,
        "ambulance": 0.2389,
        "developer": 0.2,
        "template": 0.1998,
        "anti":0.2077,
        "valentine": 0.20125,
        "communism": 0.2274,
        "weak": 0.1858,
        "zipmeme": 0.17642
        }
    
    }
    id_dict_tags = {
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




    for j in range(0, 10):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni_text = []
        penalizzazioni_tags = []
        print(data_probs_text.info())
        for i in range(len(dataset)):
            pres_mis_text = False
            pres_not_mis_text = False
            
            terms_pos_to_correct = []
            terms_neg_to_correct = []
            tags_pos_to_correct = []
            tags_neg_to_correct = []
            terms_scores_neg = []
            terms_scores_pos =  []
            tags_scores_neg = []
            tags_scores_pos =  []
            for id_term in identity_terms_mis:
                #print(data_probs_text[id_term][i])
                if data_probs_text[id_term][i] ==1:
                    terms_pos_to_correct.append(id_term)
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    terms_neg_to_correct.append(id_term)
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                #print(data_probs_tags[id_tag][i])
                if data_probs_tags[id_tag][i] ==1:
                    tags_pos_to_correct.append(id_tag)
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    tags_neg_to_correct.append(id_tag)
                    pres_not_mis_tags = True

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] ) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])

            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
            
            
            if pres_mis_text and pres_not_mis_text:
                for term in terms_pos_to_correct:
                    score_term = id_dict_text["pos"][term]
                    terms_scores_pos.append(score_term)
                for term in terms_neg_to_correct:
                    score_term = id_dict_text["neg"][term]
                    terms_scores_neg.append(score_term)
                for pen in terms_scores_pos:
                    marginale_1_text= marginale_1_text * pen
                for pen in terms_scores_neg:
                    marginale_0_text= marginale_0_text * pen
                #penalization_text_pos = mean(terms_scores_pos)
                #penalization_text_neg = mean(terms_scores_neg)
                #penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                #label_0_corr_text = marginale_0_text * penalization_text_neg
                
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis_text and not(pres_not_mis_text):
                for term in terms_pos_to_correct:
                    score_term = id_dict_text["pos"][term]
                    terms_scores_pos.append(score_term)
                for pen in terms_scores_pos:
                    marginale_1_text= marginale_1_text * pen
                #penalization_text = mean(terms_scores_pos)
                #penalizzazioni_text.append((0, penalization_text))
                #label_1_corr_text= marginale_1_text * penalization_text
                #label_0_corr_text = marginale_0_text
            elif pres_not_mis_text and not(pres_mis_text):
                for term in terms_neg_to_correct:
                    score_term = id_dict_text["neg"][term]
                    terms_scores_neg.append(score_term)
                for pen in terms_scores_neg:
                    marginale_0_text= marginale_0_text * pen
                #penalization_text = mean(terms_scores_neg)
                #penalizzazioni_text.append((penalization_text, 1))
                #label_1_corr_text= marginale_1_text
                #label_0_corr_text = marginale_0_text * penalization_text
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr_text= marginale_1_text
            #    label_0_corr_text = marginale_0_text
            if pres_mis_tags and pres_not_mis_tags:
                for tag in tags_pos_to_correct:
                    score_tag = id_dict_tags["pos"][tag]
                    tags_scores_pos.append(score_tag)
                for tag in tags_neg_to_correct:
                    score_tag = id_dict_tags["neg"][tag]
                    tags_scores_neg.append(score_tag)
                for pen in tags_scores_pos:
                    marginale_1_tags = marginale_1_tags * pen
                for pen in tags_scores_neg:
                    marginale_0_tags = marginale_0_tags * pen
                #penalization_tags_pos = mean(tags_scores_pos)
                #penalization_tags_neg = mean(tags_scores_neg)
                #penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                #label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                #label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis_tags and not(pres_not_mis_tags):
                for tag in tags_pos_to_correct:
                    score_tag = id_dict_tags["pos"][tag]
                    tags_scores_pos.append(score_tag)
                for pen in tags_scores_pos:
                    marginale_1_tags = marginale_1_tags * pen
                #penalization_tags = mean(tags_scores_pos)
                #penalizzazioni_tags.append((0, penalization_tags))
                #label_1_corr_tags= marginale_1_tags * penalization_tags
                #label_0_corr_tags = marginale_0_tags
            elif pres_not_mis_tags and not(pres_mis_tags):
                for tag in tags_neg_to_correct:
                    score_tag = id_dict_tags["neg"][tag]
                    tags_scores_neg.append(score_tag)
                for pen in tags_scores_neg:
                    marginale_0_tags = marginale_0_tags * pen
                #penalization_tags = mean(tags_scores_neg)
                #penalizzazioni_tags.append((penalization_tags, 1))
                #label_1_corr_tags= marginale_1_tags
                #label_0_corr_tags = marginale_0_tags * penalization_tags
            #elif not(pres_not_mis_tags) and not(pres_mis_tags):
            #    label_1_corr_tags= marginale_1_tags
            #    label_0_corr_tags = marginale_0_tags
            #marginale_1_ = label_1_corr_text + label_1_corr_tags
            #marginale_0_ = label_0_corr_text + label_0_corr_tags
            #if pres_mis_text and pres_not_mis_text:
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization_pos = mean(terms_scores_pos)
            #    penalization_neg = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization_neg, penalization_pos))
            #    label_0_corr = label_0 * penalization_neg
            #    label_1_corr = label_1 * penalization_pos
            #    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            #elif pres_mis_text and not(pres_not_mis_text):
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    penalization = mean(terms_scores_pos)
            #    penalizzazioni.append((0, penalization))
            #    label_1_corr = label_1 * penalization
            #    label_0_corr = label_0
            #elif pres_not_mis_text and not(pres_mis_text):
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization, 1))
            #    label_1_corr = label_1
            #    label_0_corr = label_0 * penalization
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr = label_1
            #    label_0_corr = label_0
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1= evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*DATA_LEN
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*DATA_LEN
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*DATA_LEN
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*DATA_LEN
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*DATA_LEN
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*DATA_LEN

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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
############END BMA######################

def ubma_term_tag_corr_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, result_path, dataset, syn_folds, keyfold):
    
    #y_bma_pred = []
    identity_terms_mis = identity_terms[0]
    identity_terms_notmis = identity_terms[1]
    print(identity_terms)

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
    id_dict_text = {
        "pos": {
        "dishwasher": 0.3807,
        "chick": 0.3440,
        "whore": 0.289223,
        "demotivational": 0.3935,
        "diy": 0.281618,
        "promotion": 0.3541,
        "motivate":0.33222,
        "chloroform":0.30898810077639666,
        "blond": 0.30336300854129217,
        "diy":0.3012319342817566,
        "belong":0.28922266696217547,
        "blonde":0.28518423774286483
        },
        "neg":{
        "identify": 0.167053,
        "mcdonald": 0.2634,
        "ambulance": 0.2389,
        "developer": 0.2,
        "template": 0.1998,
        "anti":0.2077,
        "valentine": 0.20125,
        "communism": 0.2274,
        "weak": 0.1858,
        "zipmeme": 0.17642
        }
    
    }
    id_dict_tags = {
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
        n_fold_text = probs_path_text+f"{j+1}.csv"
        n_fold_tags = probs_path_tags+f"{j+1}.csv"
        #print(n_fold_text)
        result_text = pd.read_csv(n_fold_text, sep="\t")
        result_tags = pd.read_csv(n_fold_tags, sep="\t")
        #print()
        #data_probs_text = pd.read_csv(project_paths.csv_uni_text_syn_probs, sep="\t")[j*160: j*160+160].reset_index()
        #print()
        data_probs_text = pd.merge(dataset,result_text,on='file_name')
        data_probs_tags = pd.merge(dataset,result_tags,on='file_name')
        print(data_probs_text.info())
        print(data_probs_tags.info())
        
        y_test = data_probs_text["ground_truth"]
        y_test_tag = data_probs_tags["ground_truth"]
        for i in range(len(y_test)):
            assert(y_test[i] == y_test_tag[i])
        labels_bma = []
        y_prob_auc = []
        correzione = []
        penalizzazioni_text = []
        penalizzazioni_tags = []
        print(data_probs_text.info())
        for i in range(len(foldsizes)):
            pres_mis_text = False
            pres_not_mis_text = False
            
            terms_pos_to_correct = []
            terms_neg_to_correct = []
            tags_pos_to_correct = []
            tags_neg_to_correct = []
            terms_scores_neg = []
            terms_scores_pos =  []
            tags_scores_neg = []
            tags_scores_pos =  []
            for id_term in identity_terms_mis:
                #print(data_probs_text[id_term][i])
                if data_probs_text[id_term][i] ==1:
                    terms_pos_to_correct.append(id_term)
                    pres_mis_text = True
            for id_term in identity_terms_notmis:
                if data_probs_text[id_term][i] ==1:
                    terms_neg_to_correct.append(id_term)
                    pres_not_mis_text = True
            pres_mis_tags = False
            pres_not_mis_tags = False
            for id_tag in identity_tags_mis:
                #print(data_probs_tags[id_tag][i])
                if data_probs_tags[id_tag][i] ==1:
                    tags_pos_to_correct.append(id_tag)
                    pres_mis_tags = True
            for id_tag in identity_tags_notmis:
                if data_probs_tags[id_tag][i] ==1:
                    tags_neg_to_correct.append(id_tag)
                    pres_not_mis_tags = True

            marginale_0_text = (data_probs_text["SVM PROB 0"][i]* data_score_text["SCORE 0 SVM"][j]) + (data_probs_text["KNN PROB 0"][i]* data_score_text["SCORE 0 KNN"][j]) + (data_probs_text["NB PROB 0"][i]* data_score_text["SCORE 0 NB"][j]) +  (data_probs_text["DT PROB 0"][i]* data_score_text["SCORE 0 DT"][j]) +  (data_probs_text["MLP PROB 0"][i]* data_score_text["SCORE 0 MLP"][j])
            marginale_1_text = (data_probs_text["SVM PROB 1"][i]* data_score_text["SCORE 1 SVM"][j] ) + (data_probs_text["KNN PROB 1"][i]* data_score_text["SCORE 1 KNN"][j]) + (data_probs_text["NB PROB 1"][i]* data_score_text["SCORE 1 NB"][j]) +  (data_probs_text["DT PROB 1"][i]* data_score_text["SCORE 1 DT"][j]) +  (data_probs_text["MLP PROB 1"][i]* data_score_text["SCORE 1 MLP"][j])
            marginale_0_tags = (data_probs_tags["SVM PROB 0"][i]* data_score_tags["SCORE 0 SVM"][j]) + (data_probs_tags["KNN PROB 0"][i]* data_score_tags["SCORE 0 KNN"][j]) + (data_probs_tags["NB PROB 0"][i]* data_score_tags["SCORE 0 NB"][j]) +  (data_probs_tags["DT PROB 0"][i]* data_score_tags["SCORE 0 DT"][j]) +  (data_probs_tags["MLP PROB 0"][i]* data_score_tags["SCORE 0 MLP"][j])
            marginale_1_tags = (data_probs_tags["SVM PROB 1"][i]* data_score_tags["SCORE 1 SVM"][j]) + (data_probs_tags["KNN PROB 1"][i]* data_score_tags["SCORE 1 KNN"][j]) + (data_probs_tags["NB PROB 1"][i]* data_score_tags["SCORE 1 NB"][j]) +  (data_probs_tags["DT PROB 1"][i]* data_score_tags["SCORE 1 DT"][j]) +  (data_probs_tags["MLP PROB 1"][i]* data_score_tags["SCORE 1 MLP"][j])

            #label_0, label_1 = evaluation_metrics.normalize(marginale_0_text,marginale_1_text)
            
            
            if pres_mis_text and pres_not_mis_text:
                for term in terms_pos_to_correct:
                    score_term = id_dict_text["pos"][term]
                    terms_scores_pos.append(score_term)
                for term in terms_neg_to_correct:
                    score_term = id_dict_text["neg"][term]
                    terms_scores_neg.append(score_term)
                for pen in terms_scores_pos:
                    marginale_1_text= marginale_1_text * pen
                for pen in terms_scores_neg:
                    marginale_0_text= marginale_0_text * pen
                #penalization_text_pos = mean(terms_scores_pos)
                #penalization_text_neg = mean(terms_scores_neg)
                #penalizzazioni_text.append((penalization_text_neg, penalization_text_pos))
                #label_0_corr_text = marginale_0_text * penalization_text_neg
                
                print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis_text and not(pres_not_mis_text):
                for term in terms_pos_to_correct:
                    score_term = id_dict_text["pos"][term]
                    terms_scores_pos.append(score_term)
                for pen in terms_scores_pos:
                    marginale_1_text= marginale_1_text * pen
                #penalization_text = mean(terms_scores_pos)
                #penalizzazioni_text.append((0, penalization_text))
                #label_1_corr_text= marginale_1_text * penalization_text
                #label_0_corr_text = marginale_0_text
            elif pres_not_mis_text and not(pres_mis_text):
                for term in terms_neg_to_correct:
                    score_term = id_dict_text["neg"][term]
                    terms_scores_neg.append(score_term)
                for pen in terms_scores_neg:
                    marginale_0_text= marginale_0_text * pen
                #penalization_text = mean(terms_scores_neg)
                #penalizzazioni_text.append((penalization_text, 1))
                #label_1_corr_text= marginale_1_text
                #label_0_corr_text = marginale_0_text * penalization_text
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr_text= marginale_1_text
            #    label_0_corr_text = marginale_0_text
            if pres_mis_tags and pres_not_mis_tags:
                for tag in tags_pos_to_correct:
                    score_tag = id_dict_tags["pos"][tag]
                    tags_scores_pos.append(score_tag)
                for tag in tags_neg_to_correct:
                    score_tag = id_dict_tags["neg"][tag]
                    tags_scores_neg.append(score_tag)
                for pen in tags_scores_pos:
                    marginale_1_tags = marginale_1_tags * pen
                for pen in tags_scores_neg:
                    marginale_0_tags = marginale_0_tags * pen
                #penalization_tags_pos = mean(tags_scores_pos)
                #penalization_tags_neg = mean(tags_scores_neg)
                #penalizzazioni_tags.append((penalization_tags_neg, penalization_tags_pos))
                #label_0_corr_tags = marginale_0_tags * penalization_tags_neg
                #label_1_corr_tags= marginale_1_tags * penalization_tags_pos
                print("CORR neutrale ovvero ho 2 tagini di classi diverse, dovrebbe capitare solo 3 volte")
            elif pres_mis_tags and not(pres_not_mis_tags):
                for tag in tags_pos_to_correct:
                    score_tag = id_dict_tags["pos"][tag]
                    tags_scores_pos.append(score_tag)
                for pen in tags_scores_pos:
                    marginale_1_tags = marginale_1_tags * pen
                #penalization_tags = mean(tags_scores_pos)
                #penalizzazioni_tags.append((0, penalization_tags))
                #label_1_corr_tags= marginale_1_tags * penalization_tags
                #label_0_corr_tags = marginale_0_tags
            elif pres_not_mis_tags and not(pres_mis_tags):
                for tag in tags_neg_to_correct:
                    score_tag = id_dict_tags["neg"][tag]
                    tags_scores_neg.append(score_tag)
                for pen in tags_scores_neg:
                    marginale_0_tags = marginale_0_tags * pen
                #penalization_tags = mean(tags_scores_neg)
                #penalizzazioni_tags.append((penalization_tags, 1))
                #label_1_corr_tags= marginale_1_tags
                #label_0_corr_tags = marginale_0_tags * penalization_tags
            #elif not(pres_not_mis_tags) and not(pres_mis_tags):
            #    label_1_corr_tags= marginale_1_tags
            #    label_0_corr_tags = marginale_0_tags
            #marginale_1_ = label_1_corr_text + label_1_corr_tags
            #marginale_0_ = label_0_corr_text + label_0_corr_tags
            #if pres_mis_text and pres_not_mis_text:
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization_pos = mean(terms_scores_pos)
            #    penalization_neg = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization_neg, penalization_pos))
            #    label_0_corr = label_0 * penalization_neg
            #    label_1_corr = label_1 * penalization_pos
            #    print("CORR neutrale ovvero ho 2 termini di classi diverse, dovrebbe capitare solo 3 volte")
            #elif pres_mis_text and not(pres_not_mis_text):
            #    for term in terms_pos_to_correct:
            #        score_term = id_dict["pos"][term]
            #        terms_scores_pos.append(score_term)
            #    penalization = mean(terms_scores_pos)
            #    penalizzazioni.append((0, penalization))
            #    label_1_corr = label_1 * penalization
            #    label_0_corr = label_0
            #elif pres_not_mis_text and not(pres_mis_text):
            #    for term in terms_neg_to_correct:
            #        score_term = id_dict["neg"][term]
            #        terms_scores_neg.append(score_term)
            #    penalization = mean(terms_scores_neg)
            #    penalizzazioni.append((penalization, 1))
            #    label_1_corr = label_1
            #    label_0_corr = label_0 * penalization
            #elif not(pres_not_mis_text) and not(pres_mis_text):
            #    label_1_corr = label_1
            #    label_0_corr = label_0
            marginale_1_ = marginale_1_text + marginale_1_tags
            marginale_0_ = marginale_0_text + marginale_0_tags
            label_norm_0, label_norm_1 = evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1= evaluation_metrics.normalize(marginale_0_,marginale_1_)
            #label_norm_0, label_norm_1 = evaluation_metrics.normalize(label_0_corr,label_1_corr) 
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

        data_probs_multi = pd.merge(data_probs_text,data_probs_tags,on='file_name')
        data_probs_text["SCORE 1 SVM"] = [data_score_text["SCORE 1 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 SVM"] = [data_score_text["SCORE 0 SVM"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 KNN"] = [data_score_text["SCORE 0 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 KNN"] = [data_score_text["SCORE 1 KNN"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 NB"] =  [data_score_text["SCORE 0 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 NB"] =  [data_score_text["SCORE 1 NB"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 DT"] =  [data_score_text["SCORE 0 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 DT"] =  [data_score_text["SCORE 1 DT"][j]]*len(foldsizes)
        data_probs_text["SCORE 0 MLP"] = [data_score_text["SCORE 0 MLP"][j]]*len(foldsizes)
        data_probs_text["SCORE 1 MLP"] = [data_score_text["SCORE 1 MLP"][j]]*len(foldsizes)

        #data_probs_text["AUC_FINAL POS SVM TEXT"]= bias_pos_svm_text
        #data_probs_text["AUC_FINAL NEG SVM TEXT"]= bias_neg_svm_text
        #data_probs_text["CORREZIONE USATA"] = correzione
        #data_probs_text["AUC_FINAL POS KNN TEXT"]= bias_pos_KNN_text
        #data_probs_text["AUC_FINAL NEG KNN TEXT"]= bias_neg_KNN_text
        #data_probs_text["AUC_FINAL POS NBY TEXT"]= bias_pos_NBY_text
        #data_probs_text["AUC_FINAL NEG NBY TEXT"]= bias_neg_NBY_text
        #data_probs_text["AUC_FINAL POS DTR TEXT"]= bias_pos_DTR_text
        #data_probs_text["AUC_FINAL NEG DTR TEXT"]= bias_neg_DTR_text
        #data_probs_text["AUC_FINAL POS MLP TEXT"]= bias_pos_MLP_text
        #data_probs_text["AUC_FINAL NEG MLP TEXT"]= bias_neg_MLP_text
        data_probs_multi["BMA PROB 0"] = sum_prob0_bma
        data_probs_multi["BMA PROB 1"] = sum_prob1_bma
        data_probs_multi["BMA LABELS"] = labels_bma
        data_probs_multi["true_labels"] =y_test

        probs_name = result_path+f'{j+1}.csv'
        #result_text = pd.merge(dataset,data_probs_text,on='file_name')
        #result_text.to_csv(probs_name, sep="\t")
        data_probs_multi.to_csv(probs_name, sep="\t")
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
        #printResult_text(labels_bma, y_prob_auc, y_test)
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




if (correction_strategy == "neu" and modality_correction == "multi"):
    result_path = project_paths.out_corr_bma_neu_multi
elif (correction_strategy == "neg" and modality_correction == "multi"):
    result_path = project_paths.out_corr_bma_neg_multi
elif (correction_strategy == "pos" and modality_correction == "multi"):
    result_path = project_paths.out_corr_bma_pos_multi
elif (correction_strategy == "dyn_base" and modality_correction == "multi"):
    result_path = project_paths.out_corr_dyn_base_multi
elif (correction_strategy == "dyn_bma" and modality_correction == "multi"):
    result_path = project_paths.out_corr_dyn_bma_multi
elif (correction_strategy == "terms_base" and modality_correction == "multi"):
    result_path = project_paths.out_corr_terms_base_multi
elif (correction_strategy == "terms_bma" and modality_correction == "multi"):
    result_path = project_paths.out_corr_terms_bma_multi
elif (correction_strategy == "terms_tags" and modality_correction == "multi"):
    result_path = project_paths.out_corr_terms_tags_bma_multi
elif (correction_strategy == "masked" and modality_correction == "multi"):
    result_path = project_paths.out_corr_masked
elif (correction_strategy == "masked_count" and modality_correction == "multi"):
    result_path = project_paths.out_corr_masked_count
elif (correction_strategy == "censored" and modality_correction == "multi"):
    result_path = project_paths.out_corr_censored
elif (correction_strategy == "rank_cls" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_rank_cls
elif (correction_strategy == "rank" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_rank
elif (correction_strategy == "uniform" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_uniform
elif (correction_strategy == "treshold" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_treshold
elif (correction_strategy == "sample" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_sample
elif (correction_strategy == "censored_uniform" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_censored_uniform
elif (correction_strategy == "masked_uniform" and modality_correction == "multi"):
    result_path = project_paths.out_corr_repair_masked_uniform
elif(correction_strategy == "none"):
    result_path = project_paths.out_bma_biased_multi

    
    
if (correction_strategy == "neu" and modality_correction == "text"):
    result_path = project_paths.out_corr_bma_neu_multi_text
elif (correction_strategy == "neg" and modality_correction == "text"):
    result_path = project_paths.out_corr_bma_neg_multi_text
elif (correction_strategy == "pos" and modality_correction == "text"):
    result_path = project_paths.out_corr_bma_pos_multi_text
elif (correction_strategy == "dyn_base" and modality_correction == "text"):
    result_path = project_paths.out_corr_dyn_base_multi_text
elif (correction_strategy == "dyn_bma" and modality_correction == "text"):
    result_path = project_paths.out_corr_dyn_bma_multi_text
elif (correction_strategy == "terms_base" and modality_correction == "text"):
    result_path = project_paths.out_corr_terms_base_multi_text
elif (correction_strategy == "terms_bma" and modality_correction == "text"):
    result_path = project_paths.out_corr_terms_bma_multi_text
elif (correction_strategy == "terms_tags" and modality_correction == "text"):
    result_path = project_paths.out_corr_terms_tags_bma_multi_text
elif (correction_strategy == "masked" and modality_correction == "text"):
    result_path = project_paths.out_corr_masked_text
elif (correction_strategy == "censored" and modality_correction == "text"):
    result_path = project_paths.out_corr_censored_text
elif (correction_strategy == "rank_cls" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_rank_cls_text
elif (correction_strategy == "rank" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_rank_text
elif (correction_strategy == "uniform" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_uniform_text
elif (correction_strategy == "treshold" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_treshold_text
elif (correction_strategy == "sample" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_sample_text
elif (correction_strategy == "censored_uniform" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_censored_uniform_text
elif (correction_strategy == "masked_uniform" and modality_correction == "text"):
    result_path = project_paths.out_corr_repair_masked_uniform_text
       
#############################################################################

if (correction_strategy == "neu" and modality_correction == "tags"):
    result_path = project_paths.out_corr_bma_neu_multi_tags
elif (correction_strategy == "neg" and modality_correction == "tags"):
    result_path = project_paths.out_corr_bma_neg_multi_tags
elif (correction_strategy == "pos" and modality_correction == "tags"):
    result_path = project_paths.out_corr_bma_pos_multi_tags
elif (correction_strategy == "dyn_base" and modality_correction == "tags"):
    result_path = project_paths.out_corr_dyn_base_multi_tags
elif (correction_strategy == "dyn_bma" and modality_correction == "tags"):
    result_path = project_paths.out_corr_dyn_bma_multi_tags
elif (correction_strategy == "terms_base" and modality_correction == "tags"):
    result_path = project_paths.out_corr_terms_base_multi_tags
elif (correction_strategy == "terms_bma" and modality_correction == "tags"):
    result_path = project_paths.out_corr_terms_bma_multi_tags
elif (correction_strategy == "terms_tags" and modality_correction == "tags"):
    result_path = project_paths.out_corr_terms_tags_bma_multi_tags
elif (correction_strategy == "masked" and modality_correction == "tags"):
    result_path = project_paths.out_corr_masked_tags
elif (correction_strategy == "masked_count" and modality_correction == "tags"):
    result_path = project_paths.out_corr_masked_count_tags
elif (correction_strategy == "censored" and modality_correction == "tags"):
    result_path = project_paths.out_corr_censored_tags
elif (correction_strategy == "rank_cls" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_rank_cls_tags
elif (correction_strategy == "rank" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_rank_tags
elif (correction_strategy == "uniform" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_uniform_tags
elif (correction_strategy == "treshold" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_treshold_tags
elif (correction_strategy == "sample" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_sample_tags
elif (correction_strategy == "censored_uniform" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_censored_uniform_tags
elif (correction_strategy == "masked_uniform" and modality_correction == "tags"):
    result_path = project_paths.out_corr_repair_masked_uniform_tags

import pickle
with open('identity_tags_mis', 'rb') as f:
    id_tags_mis = pickle.load(f)    
with open('identity_tags_notmis', 'rb') as f:
    id_tags_notmis = pickle.load(f)

identity_tags_mis = id_tags_mis[:10]
identity_tags_notmis = id_tags_notmis[:10]

if (data_to_eval =="syn"): 
    dataset = load_data.load_syn_data_tag()
    text_column = "text"
    tag_column = "1"
    dataset.drop(columns=dataset.columns[8:], axis=1,inplace=True)
    dataset.drop(columns = ["Unnamed: 0.1", "cleaned", "lemmas", "tag list"], axis=1, inplace=True)     
    dataset.rename(columns={'0': 'file_name'}, inplace= True)
    print(dataset.info())
    print(dataset["file_name"][0])
    #########correzione con text e masked count è uguale a masked perchè solo i tag cambia
    ######### l'esperimento da masked e mask pos e neg pos
    if correction_strategy == "masked" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_syn_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"    
    elif correction_strategy == "masked" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"    
    #elif correction_strategy == "masked" and modality_correction == "tags":
    #    probs_path_text = project_paths.csv_uni_text_new_syn_probs#########################occhio ai nomi delle probs
    #    score_path_text = project_paths.csv_uni_text_syn_scores
    #    probs_path_tags = project_paths.csv_uni_tags_syn_masked_probs_folds
    #    score_path_tags = project_paths.csv_uni_tags_syn_masked_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "masked_count" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_count_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_count_scores
    elif correction_strategy == "censored" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_censored_probs_folds######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_censored_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "censored" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_censored_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_censored_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "masked" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_syn_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_scores
    elif correction_strategy == "masked_count" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_syn_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_count_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_count_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"         
    elif correction_strategy == "censored" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_censored_probs_folds######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_censored_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_censored_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_censored_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
################################REPAIR RANK CLS CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "rank_cls" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_rank_cls_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_rank_cls_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_rank_cls_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_rank_cls_scores
    elif correction_strategy == "rank_cls" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_rank_cls_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_rank_cls_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "rank_cls" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_rank_cls_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_rank_cls_scores
################################REPAIR RANK CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "rank" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_rank_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_rank_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_rank_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_rank_scores
    elif correction_strategy == "rank" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_rank_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_rank_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "rank" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_rank_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_rank_scores
################################REPAIR UNIFORM CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_uniform_scores
    elif correction_strategy == "uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_uniform_scores
################################REPAIR TRESHOLD CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "treshold" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_treshold_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_treshold_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_treshold_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_treshold_scores
    elif correction_strategy == "treshold" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_treshold_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_treshold_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "treshold" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_treshold_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_treshold_scores
################################REPAIR SAMPLE CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "sample" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_sample_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_sample_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_sample_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_sample_scores
    elif correction_strategy == "sample" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_sample_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_sample_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "sample" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_sample_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_sample_scores
################################REPAIR censored uniform CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "censored_uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_censored_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_censored_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_censored_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_censored_uniform_scores
    elif correction_strategy == "censored_uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_censored_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_censored_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "censored_uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_censored_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_censored_uniform_scores
################################REPAIR maksed uniform CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "masked_uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_syn_masked_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_masked_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_uniform_scores
    elif correction_strategy == "masked_uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_syn_masked_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_syn_masked_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_probs
        score_path_tags = project_paths.csv_uni_tags_syn_scores
    elif correction_strategy == "masked_uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_new_syn_probs######################################occhio!
        score_path_text = project_paths.csv_uni_text_syn_scores
        probs_path_tags = project_paths.csv_uni_tags_syn_masked_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_syn_masked_uniform_scores
    else:
        if key_of_folds == "mitigation":
            probs_path_text = project_paths.csv_uni_text_new_syn_probs
            score_path_text = project_paths.csv_uni_text_syn_scores
            probs_path_tags = project_paths.csv_uni_tags_syn_probs
            score_path_tags = project_paths.csv_uni_tags_syn_scores
        if key_of_folds == "measure":
            print("OOOOOOOOOOOOOOOOOO")
            probs_path_tags = "../data/results2strategy/tags/new_sintest/measure/probs_sin_test_fold_"
            score_path_tags =  project_paths.csv_uni_tags_syn_scores
            probs_path_text = "../data/results2strategy/text/new_sintest/measure/probs_sin_test_fold_"
            score_path_text =  project_paths.csv_uni_text_syn_scores
    if key_of_folds == "mitigation":
        out_path = result_path + "new_sintest/probs_sin_test_fold_"
    if key_of_folds == "measure":
        out_path = result_path + "new_sintest/measure/probs_sin_test_fold_"
elif(data_to_eval =="test"):
    data_text = load_data.load_test_data()
    data_tags = load_data.load_test_data_tag()
    data_tags.drop(columns=data_tags.columns[3:], axis=1,inplace=True)
    dataset = pd.merge(data_text,data_tags,on='file_name')
    text_column = "Text Transcription"
    tag_column = "1_y"
    print(dataset.info())
    if correction_strategy == "masked" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_test_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"    
    elif correction_strategy == "masked" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_scores
    elif correction_strategy == "masked_count" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_count_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_count_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "censored" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_censored_probs_folds######################################occhio!
        score_path_text = project_paths.csv_uni_text_test_censored_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "censored" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_censored_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_censored_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
    elif correction_strategy == "masked" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_test_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"  
    elif correction_strategy == "masked_count" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_masked_probs_folds#########################occhio ai nomi delle probs
        score_path_text = project_paths.csv_uni_text_test_masked_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_count_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_count_scores       
        
    elif correction_strategy == "censored" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_censored_probs_folds######################################occhio!
        score_path_text = project_paths.csv_uni_text_test_censored_scores
        probs_path_tags = project_paths.csv_uni_tags_test_censored_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_censored_scores
        #out_path = result_path + "new_sintest/probs_sin_test_fold_"
################################REPAIR RANK CLS CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "rank_cls" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_rank_cls_probs_folds
        score_path_text = project_paths.csv_uni_text_test_rank_cls_scores
        probs_path_tags = project_paths.csv_uni_tags_test_rank_cls_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_rank_cls_scores
    elif correction_strategy == "rank_cls" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_rank_cls_probs_folds
        score_path_text = project_paths.csv_uni_text_test_rank_cls_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "rank_cls" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_rank_cls_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_rank_cls_scores
################################REPAIR RANK CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "rank" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_rank_probs_folds
        score_path_text = project_paths.csv_uni_text_test_rank_scores
        probs_path_tags = project_paths.csv_uni_tags_test_rank_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_rank_scores
    elif correction_strategy == "rank" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_rank_probs_folds
        score_path_text = project_paths.csv_uni_text_test_rank_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "rank" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_rank_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_rank_scores
################################REPAIR UNIFORM CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_uniform_scores
    elif correction_strategy == "uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_uniform_scores
################################REPAIR TRESHOLD CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "treshold" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_treshold_probs_folds
        score_path_text = project_paths.csv_uni_text_test_treshold_scores
        probs_path_tags = project_paths.csv_uni_tags_test_treshold_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_treshold_scores
    elif correction_strategy == "treshold" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_treshold_probs_folds
        score_path_text = project_paths.csv_uni_text_test_treshold_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "treshold" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_treshold_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_treshold_scores
################################REPAIR SAMPLE CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "sample" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_sample_probs_folds
        score_path_text = project_paths.csv_uni_text_test_sample_scores
        probs_path_tags = project_paths.csv_uni_tags_test_sample_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_sample_scores
    elif correction_strategy == "sample" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_sample_probs_folds
        score_path_text = project_paths.csv_uni_text_test_sample_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "sample" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_sample_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_sample_scores
################################REPAIR CENSORED UNIFORM CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "censored_uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_censored_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_censored_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_censored_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_censored_uniform_scores
    elif correction_strategy == "censored_uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_censored_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_censored_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "censored_uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_censored_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_censored_uniform_scores
################################REPAIR CENSORED UNIFORM CORRECTION MULTI POI SOLO TAGS E SOLO TEXT#####################################
    elif correction_strategy == "masked_uniform" and modality_correction == "multi":
        probs_path_text = project_paths.csv_uni_text_test_masked_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_masked_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_uniform_scores
    elif correction_strategy == "masked_uniform" and modality_correction == "text":
        probs_path_text = project_paths.csv_uni_text_test_masked_uniform_probs_folds
        score_path_text = project_paths.csv_uni_text_test_masked_uniform_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    elif correction_strategy == "masked_uniform" and modality_correction == "tags":
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_masked_uniform_probs_folds
        score_path_tags = project_paths.csv_uni_tags_test_masked_uniform_scores
    else:
        probs_path_text = project_paths.csv_uni_text_test_probs
        score_path_text = project_paths.csv_uni_text_test_scores
        probs_path_tags = project_paths.csv_uni_tags_test_probs
        score_path_tags = project_paths.csv_uni_tags_test_scores
    out_path = result_path + "test/probs_test_fold_"
    
if (correction_strategy != "neu" and correction_strategy != "none" and correction_strategy != "pos" and
    correction_strategy != "neg" and correction_strategy != "masked" and correction_strategy != "censored" and
    correction_strategy != "masked_count" and correction_strategy != "rank_cls" and correction_strategy != "rank"
    and correction_strategy != "uniform" and correction_strategy != "treshold" and correction_strategy != "sample" 
    and correction_strategy != "censored_uniform" and correction_strategy != "masked_uniform"):
    dataset = init_dataset(dataset, text_column, tag_column)
    print(dataset.info())

DATA_LEN = len(dataset)
print(DATA_LEN)

with open('../data/datasets/synthetic_folds.pkl', 'rb') as f:
    syn_folds = pickle.load(f)
    

    
with open('../data/datasets/bias_mitigation_tags.pkl', 'rb') as f:
    bias_tags_dict = pickle.load(f)

with open('../data/datasets/bias_mitigation_text.pkl', 'rb') as f:
    bias_text_dict = pickle.load(f)
#bias_text_dict["svm_neg"] = 0.7168
#bias_text_dict["knn_neg"] = 0.6843
#bias_text_dict["nby_neg"] = 0.6863
#bias_text_dict["dtr_neg"] = 0.6302
#bias_text_dict["mlp_neg"] = 0.7130
#
#bias_text_dict["svm_pos"] = 0.6730 
#bias_text_dict["knn_pos"] = 0.6227
#bias_text_dict["nby_pos"] = 0.6435
#bias_text_dict["dtr_pos"] = 0.6373
#bias_text_dict["mlp_pos"] = 0.6182

#################new sin##################
#bias_text_dict["svm_neg"] = 0.7329
#bias_text_dict["knn_neg"] = 0.718
#bias_text_dict["nby_neg"] = 0.7232
#bias_text_dict["dtr_neg"] = 0.6654
#bias_text_dict["mlp_neg"] = 0.7328
#
#bias_text_dict["svm_pos"] = 0.7473 
#bias_text_dict["knn_pos"] = 0.6721
#bias_text_dict["nby_pos"] = 0.7095
#bias_text_dict["dtr_pos"] = 0.6550
#bias_text_dict["mlp_pos"] = 0.7203
###################new sin#########################
#bias_text_dict["svm_neg"] = 0.7326
#bias_text_dict["knn_neg"] = 0.7248
#bias_text_dict["nby_neg"] = 0.7279
#bias_text_dict["dtr_neg"] = 0.6739
#bias_text_dict["mlp_neg"] = 0.7387
#
#bias_text_dict["svm_pos"] = 0.7583
#bias_text_dict["knn_pos"] = 0.678
#bias_text_dict["nby_pos"] = 0.7163
#bias_text_dict["dtr_pos"] = 0.6652
#bias_text_dict["mlp_pos"] = 0.7277



bias_pos_svm_text = [bias_text_dict["svm_pos"]] * DATA_LEN
bias_neg_svm_text = [bias_text_dict["svm_neg"]] * DATA_LEN
bias_pos_KNN_text = [bias_text_dict["knn_pos"]] * DATA_LEN
bias_neg_KNN_text = [bias_text_dict["knn_neg"]] * DATA_LEN
bias_pos_NBY_text = [bias_text_dict["nby_pos"]] * DATA_LEN
bias_neg_NBY_text = [bias_text_dict["nby_neg"]] * DATA_LEN
bias_pos_DTR_text = [bias_text_dict["dtr_pos"]] * DATA_LEN
bias_neg_DTR_text = [bias_text_dict["dtr_neg"]] * DATA_LEN
bias_pos_MLP_text = [bias_text_dict["mlp_pos"]] * DATA_LEN
bias_neg_MLP_text = [bias_text_dict["mlp_neg"]] * DATA_LEN


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

#identity_terms = [['dishwasher', 'chick', 'whore', 'demotivational', 'diy', 'promotion', 'bestdemotivationalposters', 'motivateusnot', 'imgur', 'motifake'], ['memeshappen', 'mcdonald', 'ambulance', 'developer', 'template', 'anti', 'valentine', 'communism', 'weak', 'memecrunch']]
identity_terms = [['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong', "blonde"], ['mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']]

BMA_BIAS_POS_text = 0.7435
BMA_BIAS_NEG_text = 0.7580

BMA_BIAS_POS_tag = 0.6171
BMA_BIAS_NEG_tag = 0.5868




data_score_text =  load_data.load_data_score(score_path_text)
data_score_tags =  load_data.load_data_score(score_path_tags)
#### per le correzioni in training runno il "biased bma" perchè le prob sono già corrette
training_corrections = ["none", "masked", "masked_count", "censored", "rank_cls", "rank", "uniform", 
                        "treshold", "sample", "censored_uniform", "masked_uniform"]

if (correction_strategy == "neu" and data_to_eval == "test"):
    ubma_neu(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "neu" and data_to_eval == "syn"):
    ubma_neu_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction, syn_folds, key_of_folds)
elif(correction_strategy == "neg" and data_to_eval == "test"):
    ubma_neg_corr(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif(correction_strategy == "neg" and data_to_eval == "syn"):
    ubma_neg_corr_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction, syn_folds, key_of_folds)
elif(correction_strategy == "pos"  and data_to_eval == "test"):
    ubma_pos_corr(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif(correction_strategy == "pos"  and data_to_eval == "syn"):
    ubma_pos_corr_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction,  syn_folds, key_of_folds)
elif (correction_strategy == "dyn_base"  and data_to_eval == "test"):
    ubma_dyn_sub_models(data_score_text, data_score_tags,probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "dyn_base"  and data_to_eval == "syn"):
    ubma_dyn_sub_models_sintest(data_score_text, data_score_tags,probs_path_text, probs_path_tags, out_path, dataset, modality_correction, syn_folds, key_of_folds)
elif (correction_strategy == "dyn_bma"  and data_to_eval == "test"):
    ubma_dyn_corr_bma(data_score_text,data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "dyn_bma"  and data_to_eval == "syn"):
    ubma_dyn_corr_bma_sintest(data_score_text,data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction, syn_folds, key_of_folds)
elif (correction_strategy == "terms_base"  and data_to_eval == "test"):
    ubma_term_corr_bma(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "terms_base"  and data_to_eval == "syn"):
    ubma_term_corr_bma_sintest(data_score_text, data_score_tags, probs_path_text, probs_path_tags, out_path, dataset, modality_correction,  syn_folds, key_of_folds)
elif (correction_strategy == "terms_bma"  and data_to_eval == "test"):
    ubma_term_corr_bma(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "terms_bma"  and data_to_eval == "syn"):
    ubma_term_corr_bma_sintest(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset, modality_correction, syn_folds, key_of_folds)
elif (correction_strategy == "terms_tags"  and data_to_eval == "test"):
    ubma_term_tag_corr(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset, modality_correction)
elif (correction_strategy == "terms_tags"  and data_to_eval == "syn"):
    ubma_term_tag_corr_sintest(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset, modality_correction,  syn_folds, key_of_folds)
elif (correction_strategy  in training_corrections  and data_to_eval == "test"):
    bma_biased(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset)
elif (correction_strategy  in training_corrections  and data_to_eval == "syn"):
    bma_biased_sintest(data_score_text, data_score_tags,  probs_path_text, probs_path_tags, out_path, dataset,  syn_folds, key_of_folds)
    




#dataset = pd.read_csv(data_path, sep="\t")


    




