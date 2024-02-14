import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from statistics import mean
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from statistics import mean
from tqdm import tqdm

from Utils import load_data, project_paths, evaluation_metrics, preprocessing
import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-n", "--name",  default="spancat_secondo", type=str, help="name of model")
parser.add_argument('--masked', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument("-m", "--model", default="none", type=str, help="name of huggingface model")
parser.add_argument('--noprepro', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--train', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--test', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--syn', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--pred', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--mitigation', type=str_to_bool, nargs='?', const=True, default=False)

model_list = ["SVM", "KNN", "NB","DT", "MLP"]
# Create a folder to store results
data_folder = project_paths.folder_results
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    

import pickle

def write_pickle(data,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

args = parser.parse_args()
config_arg= vars(args)
print(config_arg)
#model_name = args.name
masked = args.masked
no_prepro = args.noprepro
train = args.train
test = args.test
syn = args.syn
only_pred = args.pred
fold_mitigation = args.mitigation
if fold_mitigation:
    key_of_folds = "mitigation"
else:
    key_of_folds = "measure"
print(fold_mitigation)
print(key_of_folds)

kfold = load_data.load_folds()



def test_cross_validaton(dataset, kfold):


    keys = ['SCORE ' + str(num) + ' ' + model.upper()  for model in model_list for num in [0,1]]
    scores = dict((el,[]) for el in keys)
    del keys
    
    test_complete = []
    y_pred_svm = []
    y_pred_knn = []
    y_pred_nb = []
    y_pred_dt = []
    y_pred_mlp = []
    k_folds = 10
    num_folds = 10
    for key, val in tqdm(kfold.items()):
        train_ids = val["train"]
        test_ids = val["test"]
        # Print
        #print(f'FOLD BMA {key}')
        # Define the K-fold Cross Validaton 
        train_fold = dataset.iloc[train_ids,2:]
        test_fold = dataset.iloc[test_ids,2:]
        train_f = train_fold
        test_f = test_fold
        targets_train =  dataset.iloc[train_ids][['misogynous']]
        targets_test = dataset.iloc[test_ids][['misogynous']]  
        target_train = list(targets_train["misogynous"])
        target_test = list(targets_test["misogynous"])

        test_complete = test_complete + target_test 
        #_______________________________ SVM_______________________________
        svc = LinearSVC(C = 1,  dual=True, penalty = "l2", loss="hinge")
        svc = CalibratedClassifierCV(svc)
        svc.fit(train_f, target_train)
        y_pred_svm = y_pred_svm + svc.predict(test_f).tolist()
        write_pickle(svc, "tags_models_checkpoints/svc_"+key+".pickle")
        
        #_______________________________ KNN_______________________________
        knn = KNeighborsClassifier(n_neighbors=23)
        clf = knn.fit(train_f, target_train)
        y_pred_knn = y_pred_knn +  clf.predict(test_f).tolist()
        write_pickle(clf, "tags_models_checkpoints/knn_"+key+".pickle")
        #_______________________________ NB_______________________________
        naive_bayes= MultinomialNB()
        naive_bayes.fit(train_f, target_train)
        y_pred_nb = y_pred_nb + naive_bayes.predict(test_f).tolist() 
        write_pickle(naive_bayes, "tags_models_checkpoints/nby_"+key+".pickle")
        #_______________________________ DT_______________________________
        dtr = DecisionTreeClassifier(max_depth=7)
        dtr.fit(train_f, target_train)
        y_pred_dt = y_pred_dt + dtr.predict(test_f).tolist()
        write_pickle(dtr, "tags_models_checkpoints/dtr_"+key+".pickle")
        #_______________________________ MLP_______________________________
        mlp = MLPClassifier(max_iter=400, activation="relu", alpha=0.05, hidden_layer_sizes=(100,), solver="adam", learning_rate="constant")
        mlp.fit(train_f, target_train)
        y_pred_mlp = y_pred_mlp + mlp.predict(test_f).tolist()
        write_pickle(mlp, "tags_models_checkpoints/mlp_"+key+".pickle")
        
        #_______________________________ RESULTS_______________________________
        svm_results = evaluation_metrics.compute_evaluation_metrics(y_pred_svm, test_complete)
        knn_results = evaluation_metrics.compute_evaluation_metrics(y_pred_knn, test_complete)
        nb_results = evaluation_metrics.compute_evaluation_metrics(y_pred_nb, test_complete)
        dt_results = evaluation_metrics.compute_evaluation_metrics(y_pred_dt, test_complete)
        mlp_results = evaluation_metrics.compute_evaluation_metrics(y_pred_mlp, test_complete)


        for key, value in scores.items():
            if key != 'FOLD':
                num = key.split(' ')[1]
                model = key.split(' ')[2].lower()
                value.append(locals()[model + '_results']['f1'][int(num)])

    return scores


def test_kfold_preds(test_set, kfold):
    
    keys = [model.upper() + ' PROB ' + str(num)  for model in model_list for num in [0,1]]
    keys.append("ground_truth")
    keys.insert(0, "file_name")
    probs = dict((el,[]) for el in keys)
    del keys
    acc_dict = dict((el,[]) for el in [model.upper() + ' ACC' for model in model_list])
    auc_dict = dict((el,[]) for el in [model.upper() + ' AUC' for model in model_list])
    svm_probs_0 = []
    svm_probs_1 = []
    knn_probs_0 = []
    knn_probs_1 = []
    nb_probs_0 = []
    nb_probs_1 = []
    dt_probs_0 = []
    dt_probs_1 = []
    mlp_probs_0 = []
    mlp_probs_1 = []
    file_name = []
    
    ground_truth = []
    
    label_nb_complete=[]
    label_svm_complete=[]
    label_knn_complete=[]
    label_dt_complete=[]
    label_mlp_complete=[]
    
    num_folds = 10
    test = test_set.iloc[:,2:]
    y_test = list(test_set["misogynous"])
    file_names = list(test_set["file_name"])
    ground_truth = y_test

    for key, val in tqdm(kfold.items()):
  # Print
        #print(f'FOLD BMA {key}')

        # Define the K-fold Cross Validaton
        #dataset = dataset.sample(frac=1).reset_index(drop=True)  #return all rows (in random order)
        #train_fold = dataset.iloc[train_ids][["use"]]
        #train =  list(train_fold["use"])#list(dataset["use"])
        #y_train =# list(dataset["misogynous"])
        probs['ground_truth'].extend(y_test)
        probs['file_name'].extend(file_names)

        # _____________________Naive Bayes_____________________
        #naive_bayes= GaussianNB()
        #naive_bayes.fit(train, y_train)
        naive_bayes = load_pickle("tags_models_checkpoints/nby_"+key+".pickle")
        labels_nb = naive_bayes.predict(test)
        label_prob_nb = naive_bayes.predict_proba(test)
        label_nb_complete = label_nb_complete + labels_nb.tolist()      
        fpr_n, tpr_n, thresholds_n = roc_curve(y_test, label_prob_nb[:,1])

        acc_dict['NB ACC'].append(accuracy_score(y_test,labels_nb))
        auc_dict['NB AUC'].append(auc(fpr_n, tpr_n))

        nb_probs_0.extend(label_prob_nb[:,0])
        nb_probs_1.extend(label_prob_nb[:,1])       
        #_____________________ Support Vector Machine_____________________
        svc = load_pickle("tags_models_checkpoints/svc_"+key+".pickle")
        labels_svm = svc.predict(test)
        label_prob_svm = svc.predict_proba(test)
        label_svm_complete = label_svm_complete + labels_svm.tolist()

        fpr_s, tpr_s, thresholds_s = roc_curve(y_test, label_prob_svm[:,1])     
        acc_dict['SVM ACC'].append(accuracy_score(y_test,labels_svm))
        auc_dict['SVM AUC'].append(auc(fpr_s, tpr_s))       
        svm_probs_0.extend(label_prob_svm[:,0])
        svm_probs_1.extend(label_prob_svm[:,1])     
        #_____________________K-Nearest Neighbor_____________________
        clf = load_pickle("tags_models_checkpoints/knn_"+key+".pickle")
        labels_knn = clf.predict(test)
        label_prob_knn = clf.predict_proba(test)
        label_knn_complete = label_knn_complete + labels_knn.tolist()       
        fpr_k, tpr_k, thresholds_k = roc_curve(y_test, label_prob_knn[:,1])

        acc_dict['KNN ACC'].append(accuracy_score(y_test,labels_knn))
        auc_dict['KNN AUC'].append(auc(fpr_k, tpr_k))       
        knn_probs_0.extend(label_prob_knn[:, 0])
        knn_probs_1.extend(label_prob_knn[:, 1])        
        #_____________________Decisioon Tree_____________________
        dtr = load_pickle("tags_models_checkpoints/dtr_"+key+".pickle")
        labels_dt = dtr.predict(test)
        label_prob_dt = dtr.predict_proba(test)
        label_dt_complete = label_dt_complete + labels_dt.tolist()      
        fpr_d, tpr_d, thresholds_d = roc_curve(y_test, label_prob_dt[:,1])      
        acc_dict['DT ACC'].append(accuracy_score(y_test,labels_dt))
        auc_dict['DT AUC'].append(auc(fpr_d, tpr_d))        
        dt_probs_0.extend(label_prob_dt[:, 0])
        dt_probs_1.extend(label_prob_dt[:, 1])      
        #_______________________________ Multilayer Percepron _______________________________
        mlp = load_pickle("tags_models_checkpoints/mlp_"+key+".pickle")
        labels_mlp = mlp.predict(test)
        label_prob_mlp = mlp.predict_proba(test)
        label_mlp_complete = label_mlp_complete + labels_mlp.tolist()       
        fpr_m, tpr_m, thresholds_m = roc_curve(y_test, label_prob_mlp[:,1])     
        acc_dict['MLP ACC'].append(accuracy_score(y_test,labels_mlp))
        auc_dict['MLP AUC'].append(auc(fpr_m, tpr_m))       
        mlp_probs_0.extend(label_prob_mlp[:, 0])
        mlp_probs_1.extend(label_prob_mlp[:, 1])

    for model in model_list:
        locals()['avg_auc_score_' + model.lower()] = mean(auc_dict[str(model) + ' AUC'])
        locals()['avg_acc_score_' + model.lower()] = mean(acc_dict[str(model) + ' ACC'])

    results_svm = evaluation_metrics.compute_evaluation_metrics(probs['ground_truth'], label_svm_complete)
    results_knn = evaluation_metrics.compute_evaluation_metrics(probs['ground_truth'], label_knn_complete)
    results_nb = evaluation_metrics.compute_evaluation_metrics(probs['ground_truth'], label_nb_complete)
    results_dt = evaluation_metrics.compute_evaluation_metrics(probs['ground_truth'], label_dt_complete)
    results_mlp = evaluation_metrics.compute_evaluation_metrics(probs['ground_truth'], label_mlp_complete)
    for key, value in probs.items():
        if key != 'file_name' and key != 'ground_truth':
            num = key.split(' ')[2]
            model = key.split(' ')[0].lower()
            probs[key] = locals()[model + '_probs_'+ str(num)]
 
    global_results = dict((el,[]) for el in [model for model in model_list])
    for model in model_list:
      global_results[model] = [
        locals()['results_' + model.lower()]['precision'],
        locals()['results_' + model.lower()]['recall'],
        locals()['results_' + model.lower()]['f1'],
        [locals()['avg_acc_score_' + model.lower()]],
        [locals()['avg_auc_score_' + model.lower()]]
      ]
    return probs, global_results

def bma_test(data_score, data_probs_all, result_path):
    
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
   # data_score =  pd.read_csv(data_score_path, sep="\t")
    for j in tqdm(range(0, 10)):
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        data_probs = data_probs_all.iloc[j*1000: j*1000+1000].reset_index()
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(0,1000):
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
        data_probs.to_csv(result_path+str(j+1)+".csv", sep="\t")
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
    #j = 0
    #for i in range(0,10):
    #    
    #    j+=len(test_set) 
    #print(verit_assoluta)
    #print([mean(prec_pos),  mean(prec_neg), mean(rec_pos), mean(rec_neg), mean(f1_pos), mean(f1_neg), sum(acc_bma_list)/10,  sum(auc_bma_list)/10])
    #print(predictions_bma)
    return [mean(prec_pos),  mean(prec_neg), mean(rec_pos), mean(rec_neg), mean(f1_pos), mean(f1_neg), sum(acc_bma_list)/10,  sum(auc_bma_list)/10], verit_assoluta, predictions_bma    


def syntest_kfold(test_set, kfold, keyfolds):
    
    keys = [model.upper() + ' PROB ' + str(num)  for model in model_list for num in [0,1]]
    keys.append("ground_truth")
    keys.insert(0, "file_name")
    probs = dict((el,[]) for el in keys)
    #probs
    del keys
    acc_dict = dict((el,[]) for el in [model.upper() + ' ACC' for model in model_list])
    auc_dict = dict((el,[]) for el in [model.upper() + ' AUC' for model in model_list])
    svm_probs_0 = []
    svm_probs_1 = []
    knn_probs_0 = []
    knn_probs_1 = []
    nb_probs_0 = []
    nb_probs_1 = []
    dt_probs_0 = []
    dt_probs_1 = []
    mlp_probs_0 = []
    mlp_probs_1 = []
    file_name = []

    ground_truth = []

    label_nb_complete=[]
    label_svm_complete=[]
    label_knn_complete=[]
    label_dt_complete=[]
    label_mlp_complete=[]

    num_folds = 10

    
    ground_truth = []
    
    sizes = []
    for key, val in tqdm(kfold.items()):
  # Print
        print(f'FOLD BMA {key}')
        print(len(val[keyfolds]))
        sizes.append(len(val[keyfolds]))
        test_ids = val[keyfolds]
        #train_fold = dataset.iloc[train_ids,2:]
        #test_fold = dataset.iloc[test_ids,2:]
        test = test_set.iloc[test_ids,2:]
        #print(test.columns[2:])
        y_test = list(test_set.iloc[test_ids]["misogynous"])
        print("TEST LEN SIZE INDICI FOLD",len(val[keyfolds]))
        print("TEST LEN CON ILOC[IDS] ", len(test))
        # Define the K-fold Cross Validaton
        #dataset = dataset.sample(frac=1).reset_index(drop=True)  #return all rows (in random order)
        #train_fold = dataset.iloc[train_ids][["use"]]
        #train =  list(train_fold["use"])#list(dataset["use"])
        #y_train =# list(dataset["misogynous"])
        file_names = list(test_set.iloc[test_ids]["file_name"])
        probs['ground_truth'].extend(y_test)
        probs['file_name'].extend(file_names)

        # _____________________Naive Bayes_____________________
        #naive_bayes= GaussianNB()
        #naive_bayes.fit(train, y_train)
        naive_bayes = load_pickle("tags_models_checkpoints/nby_"+key+".pickle")
        labels_nb = naive_bayes.predict(test)
        label_prob_nb = naive_bayes.predict_proba(test)
        label_nb_complete = label_nb_complete + labels_nb.tolist()      
        fpr_n, tpr_n, thresholds_n = roc_curve(y_test, label_prob_nb[:,1])

        acc_dict['NB ACC'].append(accuracy_score(y_test,labels_nb))
        auc_dict['NB AUC'].append(auc(fpr_n, tpr_n))

        nb_probs_0.extend(label_prob_nb[:,0])
        nb_probs_1.extend(label_prob_nb[:,1])       
        #_____________________ Support Vector Machine_____________________
        svc = load_pickle("tags_models_checkpoints/svc_"+key+".pickle")
        labels_svm = svc.predict(test)
        label_prob_svm = svc.predict_proba(test)
        label_svm_complete = label_svm_complete + labels_svm.tolist()

        fpr_s, tpr_s, thresholds_s = roc_curve(y_test, label_prob_svm[:,1])     
        acc_dict['SVM ACC'].append(accuracy_score(y_test,labels_svm))
        auc_dict['SVM AUC'].append(auc(fpr_s, tpr_s))       
        svm_probs_0.extend(label_prob_svm[:,0])
        svm_probs_1.extend(label_prob_svm[:,1])     
        #_____________________K-Nearest Neighbor_____________________
        clf = load_pickle("tags_models_checkpoints/knn_"+key+".pickle")
        labels_knn = clf.predict(test)
        label_prob_knn = clf.predict_proba(test)
        label_knn_complete = label_knn_complete + labels_knn.tolist()       
        fpr_k, tpr_k, thresholds_k = roc_curve(y_test, label_prob_knn[:,1])

        acc_dict['KNN ACC'].append(accuracy_score(y_test,labels_knn))
        auc_dict['KNN AUC'].append(auc(fpr_k, tpr_k))       
        knn_probs_0.extend(label_prob_knn[:, 0])
        knn_probs_1.extend(label_prob_knn[:, 1])        
        #_____________________Decisioon Tree_____________________
        dtr = load_pickle("tags_models_checkpoints/dtr_"+key+".pickle")
        labels_dt = dtr.predict(test)
        label_prob_dt = dtr.predict_proba(test)
        label_dt_complete = label_dt_complete + labels_dt.tolist()      
        fpr_d, tpr_d, thresholds_d = roc_curve(y_test, label_prob_dt[:,1])      
        acc_dict['DT ACC'].append(accuracy_score(y_test,labels_dt))
        auc_dict['DT AUC'].append(auc(fpr_d, tpr_d))        
        dt_probs_0.extend(label_prob_dt[:, 0])
        dt_probs_1.extend(label_prob_dt[:, 1])      
        #_______________________________ Multilayer Percepron _______________________________
        mlp = load_pickle("tags_models_checkpoints/mlp_"+key+".pickle")
        labels_mlp = mlp.predict(test)
        label_prob_mlp = mlp.predict_proba(test)
        label_mlp_complete = label_mlp_complete + labels_mlp.tolist()       
        fpr_m, tpr_m, thresholds_m = roc_curve(y_test, label_prob_mlp[:,1])     
        acc_dict['MLP ACC'].append(accuracy_score(y_test,labels_mlp))
        auc_dict['MLP AUC'].append(auc(fpr_m, tpr_m))       
        mlp_probs_0.extend(label_prob_mlp[:, 0])
        mlp_probs_1.extend(label_prob_mlp[:, 1])
    
    #print(len(probs['ground_truth']), len(label_svm_complete))
    for model in model_list:
        locals()['avg_auc_score_' + model.lower()] = mean(auc_dict[str(model) + ' AUC'])
        locals()['avg_acc_score_' + model.lower()] = mean(acc_dict[str(model) + ' ACC'])
    results_svm =evaluation_metrics.compute_evaluation_metrics_mean(probs['ground_truth'], label_svm_complete,sizes)
    results_knn =evaluation_metrics.compute_evaluation_metrics_mean(probs['ground_truth'], label_knn_complete,sizes)
    results_nb = evaluation_metrics.compute_evaluation_metrics_mean(probs['ground_truth'], label_nb_complete, sizes)
    results_dt = evaluation_metrics.compute_evaluation_metrics_mean(probs['ground_truth'], label_dt_complete, sizes)
    results_mlp =evaluation_metrics.compute_evaluation_metrics_mean(probs['ground_truth'], label_mlp_complete,sizes)
    print("TEST LEN CON ILOC[IDS] ", len(test))
    for key, value in probs.items():
        if key != 'file_name' and key != 'ground_truth':
            num = key.split(' ')[2]
            model = key.split(' ')[0].lower()
            probs[key] = locals()[model + '_probs_'+ str(num)]


    global_results = dict((el,[]) for el in [model for model in model_list])
    for model in model_list:
      global_results[model] = [
        locals()['results_' + model.lower()]['precision'],
        locals()['results_' + model.lower()]['recall'],
        locals()['results_' + model.lower()]['f1'],
        [locals()['avg_acc_score_' + model.lower()]],
        [locals()['avg_auc_score_' + model.lower()]],
      ] 
      

        
    return probs, global_results


def bma_sintest(data_score, data_probs_all, result_path, syn_folds, keyfolds):
  
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
    #data_score =  pd.read_csv(data_score_path, sep="\t")
    j = 0
    index_sum = 0
    for key, val in syn_folds.items():
        print(data_probs_all.info())
        foldsizes = list(val[keyfolds])
        sum_prob0_bma =[]
        sum_prob1_bma =[]
        print("indexess : ", index_sum, " e ", index_sum +len(foldsizes))
        print(index_sum + len(foldsizes))
        data_probs = data_probs_all.iloc[index_sum: index_sum + len(foldsizes)].reset_index()
        index_sum = index_sum + len(foldsizes)
        print(data_probs.info())
        print(index_sum + len(foldsizes))
        print(len(data_probs))
        print(len(foldsizes))
        y_test = data_probs["ground_truth"]
        labels_bma = []
        y_prob_auc = []
        for i in range(0,len(foldsizes)):
            #print(len(foldsizes))
            #print(i)
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
        data_probs.to_csv(result_path+str(j+1)+".csv", sep="\t")
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
    #j = 0
    #for i in range(0,10):
    #    data_probs.iloc[j:j+len(test_set)].to_csv(result_path+str(i+1)+".csv", sep="\t")
    #    j+=len(test_set)
    
    #print(verit_assoluta)
    #print([mean(prec_pos),  mean(prec_neg), mean(rec_pos), mean(rec_neg), mean(f1_pos), mean(f1_neg), sum(acc_bma_list)/10,  sum(auc_bma_list)/10])
    #print(predictions_bma)
    return [mean(prec_pos),  mean(prec_neg), mean(rec_pos), mean(rec_neg), mean(f1_pos), mean(f1_neg), sum(acc_bma_list)/10,  sum(auc_bma_list)/10], verit_assoluta, predictions_bma

if masked:
    dataset = load_data.load_training_data_tag_masked()
    dataset.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "1_y"], axis=1, inplace=True)
else:
    dataset = load_data.load_training_data_tag()
    #print(dataset.columns[:13])
    #dataset.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "1_y"], axis=1, inplace=True)
    #dataset.drop(columns = ["1_y", "Unnamed: 0"], inplace=True)
    print("per train controlla che dalla terza colonna in poi ci sono i tag")
    print(dataset.columns[:20])
    
if test:
    if masked:
        test_set = load_data.load_test_data_tag_masked()
        test_set.drop(columns=["1_y", "Unnamed: 0"], inplace=True)
    else:
        test_set = load_data.load_test_data_tag()
        print("per test controlla che dalla terza colonna in poi ci sono i tag")
        test_set.drop(columns=["1_y"], inplace=True)
        print(test_set.columns[:20])
        print(test_set.info())
        #test_set.drop(columns = ["Unnamed: 0"], inplace=True)
    test_scores = test_cross_validaton(dataset, kfold)
    data_score= pd.DataFrame(test_scores)
    data_score.to_csv(project_paths.csv_uni_tags_test_scores, sep="\t", index=False)
    print("k fold retrive preds")    
    probs, global_results = test_kfold_preds(test_set, kfold)
    
    test_probs = pd.DataFrame(probs)
    test_probs.to_csv('../data/results2strategy/tags/test/probs_test_all_fold.csv', sep="\t", index=False)  
    output_path = "../data/results2strategy/tags/test/probs_test_fold_"
    print("bma")
    bma_res_test, verit_assoluta_test, predictions_bma_test = bma_test(data_score, test_probs, output_path)
    measures = ["Prec pos","Prec neg","Rec pos","Rec neg","F1 pos","F1 neg", "ACC", "AUC"]

    res_bma = {"measures": measures, 
           "SVM": [item for sublist in global_results['SVM'] for item in sublist] ,
           "KNN":  [item for sublist in global_results['KNN'] for item in sublist] ,
           "NB":  [item for sublist in global_results['NB'] for item in sublist] ,
           "DT":  [item for sublist in global_results['DT'] for item in sublist] ,
           "MLP":  [item for sublist in global_results['MLP'] for item in sublist] ,
           "BMA": bma_res_test
           }
    risultati = pd.DataFrame(res_bma)
    # Crate a csv file with averages model performances on test data
    risultati.to_csv(project_paths.csv_uni_tags_test_res, sep="\t", index= False)
elif syn:
    print("syn")
    with open('../data/datasets/synthetic_folds.pkl', 'rb') as f:
        syn_folds = pickle.load(f)
    test_set = load_data.load_syn_data_tag()
    test_set.drop(columns = ["Unnamed: 0.1", "tag list", "lemmas", "cleaned", "text","Unnamed: 0.1.1.1", "1" ], inplace=True)
    test_set.rename(columns={'0': 'file_name'}, inplace= True)
    print(test_set.columns[:20])
    if only_pred:
        data_score = pd.read_csv(project_paths.csv_uni_tags_syn_scores, sep="\t")
    else:
        syn_scores = test_cross_validaton(dataset, kfold)
        data_score= pd.DataFrame(syn_scores)
        data_score.to_csv(project_paths.csv_uni_tags_syn_scores, sep="\t", index=False)
    print("k fold retrive preds") 
    syn_probs, global_results = syntest_kfold(test_set, syn_folds, key_of_folds)
    
    syn_probs_df = pd.DataFrame(syn_probs)
    if key_of_folds == "mitigation":
        start_dir = "../data/results2strategy/tags/new_sintest/"
    else:
        start_dir = "../data/results2strategy/tags/new_sintest/measure/"        
    syn_probs_df.to_csv(start_dir+'probs_sin_test_all_fold.csv', sep="\t", index=False)  
    output_path =start_dir+"probs_sin_test_fold_"
    print("bma ")
    bma_res_test, verit_assoluta_test, predictions_bma_test = bma_sintest(data_score, syn_probs_df, output_path, syn_folds, key_of_folds)
    measures = ["Prec pos","Prec neg","Rec pos","Rec neg","F1 pos","F1 neg", "ACC", "AUC"]

    res_bma = {"measures": measures, 
           "SVM": [item for sublist in global_results['SVM'] for item in sublist] ,
           "KNN":  [item for sublist in global_results['KNN'] for item in sublist] ,
           "NB":  [item for sublist in global_results['NB'] for item in sublist] ,
           "DT":  [item for sublist in global_results['DT'] for item in sublist] ,
           "MLP":  [item for sublist in global_results['MLP'] for item in sublist] ,
           "BMA": bma_res_test
           }
    risultati = pd.DataFrame(res_bma)
    # Crate a csv file with averages model performances on test data
    risultati.to_csv(start_dir+'tags_bma_newsyn.csv', sep="\t", index= False)