import pandas as pd
import glob
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report, roc_auc_score
import collections
import pickle

import argparse
import ast
import models_performances_bias

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



parser.add_argument("-m", "--model",  default="bma", type=str, help="uni_tag, uni_text, multi")
#parser.add_argument("-t", "--test_path",  default='/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/text_csv/test_Spacy.csv', type=str, help="test path")
#parser.add_argument("-s", "--syn_path",  default='/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/new_synthetic_text_tag.csv', type=str, help="syn path")
#parser.add_argument("-c", "--correction_strategy",  default='none', type=str, help="test")
#parser.add_argument("-g", "--subgroup",  default='both', type=str, help="pos, neg, both")
parser.add_argument("-b", "--bias", default="text", type=str, help="text, multi, tags")



args = parser.parse_args()
config_arg= vars(args)
print(config_arg)

bias_type = args.bias

#sottogruppo = args.subgroup
model = args.model

BASE_DIR = "../../2_strategy_test/results/bias/mebs_for_ttest/"
CORREZIONI_testo = ["neu", "pos", "neg", "dyn_base", "dyn_bma", "terms_bma"]
CORREZIONI_tags = ["neu", "pos", "neg", "dyn_base", "dyn_bma", "tags_bma"]

path_models = "results/"+ model +"/"

if model == "uni_text":
    model_file = "text"
elif model == "uni_tags":
    model_file = "tags"
elif model == "multi":
    model_file = "multi"
    
    
if bias_type == "text" and model ==" multi":
    corr_file = "text"
elif bias_type == "tags" and model ==" multi":
    corr_file = "tags"
elif bias_type == "multi" and model ==" multi":
    corr_file = "multi"  
    
    

if model != "multi":
    baseline = BASE_DIR+ model_file +  "_bma_none.pkl"
    with open(baseline, 'rb') as f:
        baseline_dict = pickle.load(f)
    if model == "uni_text":
        for corr in CORREZIONI_testo:
            print("#### CORREZIONE",corr ,"######")
            filename_bias = BASE_DIR+ model_file+"_bma_"+ corr + ".pkl"
            print(filename_bias)
            with open(filename_bias, 'rb') as f:
                bias_dict = pickle.load(f)
            #print("######### AUC RAW #############")
            #test = ttest_rel(bias_dict["auc_raw"], baseline_dict["auc_raw"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #print("######### BIAS VALUE #############")
            #test = ttest_rel(bias_dict["bias_value"], baseline_dict["bias_value"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #print("######### AUC SYN #############")
            #test = ttest_rel(bias_dict["auc_sin"], baseline_dict["auc_sin"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            print("######### MEB #############")
            print(bias_dict["meb"], baseline_dict["meb"])
            test = ttest_rel(bias_dict["meb"], baseline_dict["meb"])
            print(test.pvalue)
            matrix = test.pvalue
            alpha = 0.1
            print("T-TEST alpha = ", alpha)
            print(matrix < alpha )
            alpha = 0.05
            print("T-TEST alpha = ", alpha)
            print(matrix < alpha )          
    elif model == "uni_tags":
        for corr in CORREZIONI_tags:
            print("#### CORREZIONE",corr ,"######")
            filename_bias = BASE_DIR+ model_file+"_bma_"+ corr + ".pkl"
            print(filename_bias)
            with open(filename_bias, 'rb') as f:
                bias_dict = pickle.load(f)
            #print("######### AUC RAW #############")
            #test = ttest_rel(bias_dict["auc_raw"], baseline_dict["auc_raw"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #print("######### BIAS VALUE #############")
            #test = ttest_rel(bias_dict["bias_value"], baseline_dict["bias_value"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #print("######### AUC SYN #############")
            #test = ttest_rel(bias_dict["auc_sin"], baseline_dict["auc_sin"])
            #print(test.pvalue)
            #matrix = test.pvalue
            #alpha = 0.1
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            #alpha = 0.05
            #print("T-TEST alpha = ", alpha)
            #print(matrix < alpha )
            print("######### MEB #############")
            print(bias_dict["meb"], baseline_dict["meb"])
            test = ttest_rel(bias_dict["meb"], baseline_dict["meb"])
            print(test.pvalue)
            matrix = test.pvalue
            alpha = 0.1
            print("T-TEST alpha = ", alpha)
            print(matrix < alpha )
            alpha = 0.05
            print("T-TEST alpha = ", alpha)
            print(matrix < alpha )
else:
    baseline = BASE_DIR+model_file + "_" + bias_type+  "_bma_none.pkl"
    with open(baseline, 'rb') as f:
        baseline_dict = pickle.load(f)
    for corr in CORREZIONI_testo:
        print("#### CORREZIONE",corr ,"######")
        filename_bias = BASE_DIR+ model_file+ "_"+ bias_type+"_bma_"+ corr + ".pkl"
        print(filename_bias)
        with open(filename_bias, 'rb') as f:
            bias_dict = pickle.load(f)
        #print("######### AUC RAW #############")
        #test = ttest_rel(bias_dict["auc_raw"], baseline_dict["auc_raw"])
        #print(test.pvalue)
        #matrix = test.pvalue
        #alpha = 0.1
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        #alpha = 0.05
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        #print("######### BIAS VALUE #############")
        #test = ttest_rel(bias_dict["bias_value"], baseline_dict["bias_value"])
        #print(test.pvalue)
        #matrix = test.pvalue
        #alpha = 0.1
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        #alpha = 0.05
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        #print("######### AUC SYN #############")
        #test = ttest_rel(bias_dict["auc_sin"], baseline_dict["auc_sin"])
        #print(test.pvalue)
        #matrix = test.pvalue
        #alpha = 0.1
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        #alpha = 0.05
        #print("T-TEST alpha = ", alpha)
        #print(matrix < alpha )
        print("######### MEB #############")
        print(bias_dict["meb"], baseline_dict["meb"])
        test = ttest_rel(bias_dict["meb"], baseline_dict["meb"])
        print(test.pvalue)
        matrix = test.pvalue
        alpha = 0.1
        print("T-TEST alpha = ", alpha)
        print(matrix < alpha )
        alpha = 0.05
        print("T-TEST alpha = ", alpha)
        print(matrix < alpha )
            