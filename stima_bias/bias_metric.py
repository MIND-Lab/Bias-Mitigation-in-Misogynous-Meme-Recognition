import sys
sys.path.append('/home/jimmy/Documenti/tesi/bma/bias/')
#import load_data
import model_bias_analysis
#import preprocessing
import pandas as pd
import os
from collections import Counter
import numpy as np
import pickle
import stanza
import spacy_stanza
import string
import re
import gc
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
parser.add_argument("-m", "--model",  default="bma", type=str, help="path of output models")
parser.add_argument("-t", "--test_path",  default='/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/text_csv/test_Spacy.csv', type=str, help="test path")
parser.add_argument("-s", "--syn_path",  default='/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/new_synthetic_text_tag.csv', type=str, help="syn path")
parser.add_argument("-c", "--correction_strategy",  default='none', type=str, help="test")
parser.add_argument("-g", "--subgroup",  default='both', type=str, help="pos, neg, both")
parser.add_argument("-b", "--bias", default="text", type=str, help="text, multi")
parser.add_argument('--mitigation', type=str_to_bool, nargs='?', const=True, default=False)

##### bias per bias type su unimodali ovvero: text faccio metrica unimodale
##### multi applico metrica multimodale al modello unimodale

args = parser.parse_args()
config_arg= vars(args)
print(config_arg)

bias_type = args.bias
csv_path_test = args.test_path
csv_path_syntest = args.syn_path
sottogruppo = args.subgroup
model = args.model
path_models = "results/"+ model +"/"
correction_strategy = args.correction_strategy
if args.mitigation:
    key_of_folds = "mitigation"
else:
    key_of_folds = "measure"

if model == "bma":
    MODELNAMES = ['probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "BMA PROB 1"
elif model == "svm":
    MODELNAMES = ['svm_probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "SVM PROB 1"
elif model == "mlp":
    MODELNAMES = ['mlp_probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "MLP PROB 1"
elif model == "knn":
    MODELNAMES = ['knn_probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "KNN PROB 1"
elif model == "nby":
    MODELNAMES = ['nby_probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "NB PROB 1"
elif model == "dtr":
    MODELNAMES = ['dtr_probs_test_{}'.format(i+1) for i in range(10)]
    column_prob = "DT PROB 1"

assert(not (model != "bma" and correction_strategy != "none"))
#path2TEST= 'visualBERT'
label_column_test = "misogynous"
label_column_syn='ground_truth'

if (correction_strategy == "neu" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neu/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neu/test/"
elif (correction_strategy == "neu" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neu/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neu/test/"
    
elif (correction_strategy == "neg" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neg/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neg/test/"
elif (correction_strategy == "neg" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neg/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_neg/test/"
    
elif (correction_strategy == "pos" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_pos/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_pos/test/"
elif (correction_strategy == "pos" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_pos/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_bma_pos/test/"
    
elif (correction_strategy == "dyn_base" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr/test/"
elif (correction_strategy == "dyn_base" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr/test/"
    
elif (correction_strategy == "dyn_bma" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr_only_bma/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr_only_bma/test/"
elif (correction_strategy == "dyn_bma" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr_only_bma/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_dyn_corr_only_bma/test/"
    
elif (correction_strategy == "terms_base" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/test/"
elif (correction_strategy == "terms_base" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/test/"
    
elif (correction_strategy == "terms_bma" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/test/"
elif (correction_strategy == "terms_bma" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_terms_corr_only_bma/test/"
    
elif (correction_strategy == "none" and key_of_folds == "mitigation"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_biased_bma/new_sintest/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_biased_bma/test/"
elif (correction_strategy == "none" and key_of_folds == "measure"):
    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_biased_bma/new_sintest/measure/"
    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/text_biased_bma/test/"
elif (correction_strategy == "masked"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/masked_sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/masked_test/"
    #elif (correction_strategy == "masked_count"):
    #    path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/multi/multi_bma_masked_count_only_text/new_sintest/"
    #    path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/2_strategy_test/results/multi/multi_bma_masked_count_only_text/test/"
elif (correction_strategy == "censored"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/censored_sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/censored_test/"
elif (correction_strategy == "ranked_cls"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/rank_cls/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/rank_cls/test/"
elif (correction_strategy == "ranked"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/rank/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/rank/test/"
elif (correction_strategy == "uniform"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/uniform/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/uniform/test/"
elif (correction_strategy == "treshold"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/tresholding/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/tresholding/test/"
elif (correction_strategy == "sample"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/sample/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/sample/test/"
elif (correction_strategy == "censored_uniform"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/censored_uniform/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/censored_uniform/test/"
elif (correction_strategy == "masked_uniform"):
        path_results_sin  = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/masked_uniform/sintest/"
        path_results_test = "/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/masked_uniform/test/"
if not os.path.exists(path_models):
    os.makedirs(path_models)
  
stopwords = ["a", "about", "above", "across", "afterwards", "again", "against", 
    "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst",
    "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", 
    "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been",  "beforehand", 
    "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", 
    "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough",
    "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", 
    "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", 
    "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter", "hereby",
    "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is",
    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may",  "meanwhile", "might", "mill",
    "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never", "nevertheless", 
    "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "now", "nowhere", "of", "off", "often", "on", "once", 
    "one", "only", "onto", "or", "other", "others", "otherwise",  "out", "over", "part", "per", "perhaps", "please", 
    "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should", "show",
    "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the",  "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "thick", "thin", "third", "this", "those", "though", "three", "through",
    "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", 
    "under", "until", "up", "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence", "whenever",
    "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "the",
    "ve", "re", "ll", "10", "11","18", "oh","s", "t","m", "did","don", "got"]

START_OF_LINE = r"^"
OPTIONAL = "?"
ANYTHING = "."
ZERO_OR_MORE = "*"
ONE_OR_MORE = "+"

SPACE = "\s"
SPACES = SPACE + ONE_OR_MORE
NOT_SPACE = "[^\s]" + ONE_OR_MORE
EVERYTHING_OR_NOTHING = ANYTHING + ZERO_OR_MORE

ERASE = ""
FORWARD_SLASH = "\/"
NEWLINES = r"[\r\n]"


HYPERLINKS = ("http" + "s" + OPTIONAL + ":" + FORWARD_SLASH + FORWARD_SLASH
              + NOT_SPACE + NEWLINES + ZERO_OR_MORE)

def lemmatization(text, nlp):
    meme = []
    #print(text)
    doc = nlp(text)
    for token in doc:
      #print("TOKEN ", token, " LEMMA ", token.lemma_, " POS ", token.pos_, " TAG ", token.tag_)
      meme.append(token.lemma_)

    review = ' '.join(meme)
    return review 
    
def data_pre_processing(data):
    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    processed_text = []
    gc.collect()
    for item in data:
        #print(item)
        text = re.sub("@[A-Za-z0-9_]+","", item) ##rimouvo menzioni
        text = re.sub(HYPERLINKS, "", text) #url
        text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text) ##domini
        text = re.sub(r'\d+', '', text) ### numeri
        text = text.lower()
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) ###punteggiatura
        text = lemmatization(text, nlp)
        #text = NormalizeWithPOS(text) ## lemmatization
        text = re.sub('[^A-Za-z0-9 ]+', '', text) ###char speciali
        text = text.split()
        text = [word.lower() for word in text if not word.lower() in stopwords] ##stopword
        #stem_text = [porter_stemmer.stem(word) for word in text]
        text = ' '.join(text)
        #print(text)
        processed_text.append(text)
    #text = NormalizeWithPOS(text)

    return processed_text

def confusion_rates_on_file_10Fold_syn(syn_folds, data, model_names, threshold, keyfold):
    """
    Compute mean value for the above-listed metrics for each model in model-names.

    :param txt_path: path to txt file to store results
    :param data: dataframe containing, for each execution, a column with real values (label_column called 'label_'+modelname) 
        and a column for each model with the predicted labels (called model_name)
    :param model_names: list of model names
    :param threshold: threshold value to use during predicted probability analysis. Default at 0.5
    :return: /
    """
    # Mean of score confusion_rates for models
    score = {'tpr': 0,
             'tnr': 0,
             'fpr': 0,
             'fnr': 0,
             'precision': 0,
             'recall': 0,
             'accuracy': 0,
             'f1': 0,
             'auc': 0,
             }
    real_values = np.array([])
    predict_values = np.array([])
    keys = list(syn_folds.keys())
    print(keys)
    sizes = []
    i = 0
    for model_name in model_names:
        #print(len(syn_folds[keys[i]]["mitigation"]))
        label_column = 'label_'+model_name
        predict_values = np.append(predict_values, data[model_name].tolist())
        real_values = np.append(real_values, data[label_column].tolist())
        score = dict(Counter(score) + Counter(models_performances_bias.compute_confusion_rates(data, model_name, label_column, threshold, len(syn_folds[keys[i]][keyfold]))))
        sizes.append(len(syn_folds[keys[i]][keyfold]))
        i+=1
    score = {key: value / len(model_names) for key, value in score.items()}
    #print(score)
    """NB: the included measure of AUC is not accurate because it's performed on labels (therefore on thresholded data). 
    The follow row, correct that"""
    score['auc'] = models_performances_bias.model_family_auc_10Fold_Syn(data, model_names,sizes)['mean']
    print("compute score ", score)
    return score

def compute_score(test_df, MODELNAMES, label_column, treshold):

    #Mean of score confusion_rates for models
    score={ 'tpr': 0,
            'tnr': 0,
            'fpr': 0,
            'fnr': 0,
            'precision': 0,
            'recall': 0,
            'accuracy': 0,
            'f1': 0,
            'auc': 0,
        }
    for MODELNAME in MODELNAMES:
        score = dict(Counter(score) + Counter(model_bias_analysis.compute_confusion_rates(test_df, MODELNAME, label_column, treshold)))


    score = {key: value / len(MODELNAMES) for key, value in score.items()}

    """NB: the included measure of AUC is not accurate because it's performed on labels (therefore on thresholded data). The follow row, correct that"""
    score['auc']= model_bias_analysis.model_family_auc(test_df, MODELNAMES, label_column)['mean']
    print("compute score ", score)
    return score



def compute_bias_metrics_on_syn_10_text(modelnames, syn_10_df, test_df, identity_terms, label_column, syn_folds, keyfold):
    folder_folds_id = "fold_id_presence/"+keyfold+"/"
    #final_multimodal_scores = {}
    final_scores_unimodal = {}
    bias_metrics_text = {}
    #bias_metrics_image = {}
   # bias_value_multimodal_metrics = {}
    bias_value_metrics = {}
    overall_auc_metrics = {}
    syn_data=syn_10_df.copy()
    text={}
    image={}
    overall_auc_metrics_syn = {}
    i = 0
    for key, value in syn_folds.items(): #range(len([MODELNAMES[1]])): #range(len(MODELNAMES)):
        syn_10_df=syn_data
       
        syn_10_df['file_name'] = syn_10_df['file_name_'+str(i+1)]
        #print("file names syndf", syn_10_df["file_name"])
        
        ide_pres = pd.read_csv(folder_folds_id+ key+ "_"+ keyfold+"_tag_term_presence.csv", sep="\t")
        ide_pres.rename(columns={'0': 'file_name'}, inplace= True)
        #print("file name idepres ", ide_pres["file_name"])
        #ide_pres.rename({"0": "file_name"}, inplace=True)
        #print("ken id pres", len(ide_pres))
        syn_10_df=syn_10_df.merge(ide_pres.drop(columns=['misogynous', 'text']),
                        how='inner', on='file_name')

        syn_10_df['misogynous'] = syn_10_df['label_'+modelnames[i]]
        #print(syn_10_df.columns[:100])
        
       # print("len data ", len(syn_10_df))
        #print(syn_10_df.info())
        #not syn_10_df[syn_10_df["Bear"] & syn_10_df['misogynous']].empty and not syn_10_df[syn_10_df["Bear"] & ~syn_10_df[label_column]].empty
        #print(syn_10_df["misogynous"])
        #print(syn_10_df["Bear"])
        #print(~syn_10_df["misogynous"])
        #print(~syn_10_df["Bear"])
        #print(syn_10_df[syn_10_df["Bear"]])
        #print(syn_10_df[syn_10_df['misogynous']])
        #print(syn_10_df[syn_10_df["Bear"] & syn_10_df['misogynous']].empty)
        ##print(syn_10_df['misogynous']].empty)
        #print(syn_10_df[syn_10_df["Bear"] & ~syn_10_df["misogynous"]].empty)
        
        identity_terms_present = models_performances_bias.identity_element_presence(syn_10_df, identity_terms, 'misogynous')
        #tags_present = models_performances_bias.identity_element_presence_OR(syn_10_df, identity_tags, label_column)
        print("GUARDA QUI ", len(identity_terms_present))


        bias_metrics_text[i] = models_performances_bias.compute_bias_metrics_for_model(syn_10_df, identity_terms_present, modelnames[i], "misogynous")
        overall_auc_metrics[i]=models_performances_bias.calculate_overall_auc(test_df, modelnames[i])
        overall_auc_metrics_syn[i]=models_performances_bias.calculate_overall_auc(syn_10_df, modelnames[i])
        final_scores_unimodal[i] = models_performances_bias.get_final_metric(bias_metrics_text[i],overall_auc_metrics[i], modelnames[i])
        

        bias_value_metrics[i]=np.nanmean([
                  bias_metrics_text[i][MODELNAMES[i] + '_subgroup_auc'],
                  bias_metrics_text[i][MODELNAMES[i] + '_bpsn_auc'],
                  bias_metrics_text[i][MODELNAMES[i] + '_bnsp_auc']
              ])
        text[i] = np.nanmean([
                bias_metrics_text[i][modelnames[i] + '_subgroup_auc'],
                bias_metrics_text[i][modelnames[i] + '_bpsn_auc'],
                bias_metrics_text[i][modelnames[i] + '_bnsp_auc']
        ])
        i+=1
        #bias_metrics_text[i] = models_performances_bias.compute_bias_metrics_for_model(syn_10_df, identity_terms_present, modelnames[i],
        #                                                    label_column)
        #bias_metrics_image[i] = models_performances_bias.compute_bias_metrics_for_model(syn_10_df, tags_present, modelnames[i],
        #                                                    label_column)
        #overall_auc_metrics[i] = models_performances_bias.calculate_overall_auc(test_df, modelnames[i])
        
        
       # overall_auc_metrics_syn[i] = models_performances_bias.calculate_overall_auc(syn_10_df, modelnames[i])
        

        #final_multimodal_scores[i] = models_performances_bias.get_final_multimodal_metric_nan(bias_metrics_text[i],
                                                                #bias_metrics_image[i],
                                                                #overall_auc_metrics[i],
                                                                #modelnames[i])

        #bias_value_multimodal_metrics[i] = np.nanmean([
        #    np.nanmean([
        #        bias_metrics_text[i][modelnames[i] + '_subgroup_auc'],
        #        bias_metrics_text[i][modelnames[i] + '_bpsn_auc'],
        #        bias_metrics_text[i][modelnames[i] + '_bnsp_auc']
        #    ]),
        #    np.nanmean([
        #        bias_metrics_image[i][modelnames[i] + '_subgroup_auc'],
        #        bias_metrics_image[i][modelnames[i] + '_bpsn_auc'],
        #        bias_metrics_image[i][modelnames[i] + '_bnsp_auc']
        #    ])
        #])
        #
        #text[i] = np.nanmean([
        #        bias_metrics_text[i][modelnames[i] + '_subgroup_auc'],
        #        bias_metrics_text[i][modelnames[i] + '_bpsn_auc'],
        #        bias_metrics_text[i][modelnames[i] + '_bnsp_auc']
        #    ])
        #image[i] = np.nanmean([
        #        bias_metrics_image[i][modelnames[i] + '_subgroup_auc'],
        #        bias_metrics_image[i][modelnames[i] + '_bpsn_auc'],
        #        bias_metrics_image[i][modelnames[i] + '_bnsp_auc']
        #    ])
    return final_scores_unimodal, overall_auc_metrics, bias_value_metrics, text, overall_auc_metrics_syn


def add_tag_string(df):
    taggy = []
    for i in range(0, len(df)):
        #img_tag = []

        lista_tags_row=ast.literal_eval(df["tag list"][i])
        #print(lista_tags_row)
        #print("lista tags riga ", i, " : ", lista_tags_row
        #conf > 0.5 aPPEND(0 1)
        stringa = ' '.join(lista_tags_row)
        #print(stringa)
        #img_tag.append(item["class"])
        #unique_tags = list(set(img_tag))
        taggy.append(stringa)
        #max_conf = max(img_tag)
        #print("max conf ", max_conf)    
    df["tag_string"] = taggy
    print(df.info())
    return df



def compute_bias_metrics_on_syn_10_multi(modelnames, syn_10_df, test_df, identity_terms, identity_tags, label_column, syn_folds, keyfold):
    
    folder_folds_id = "fold_id_presence/"+keyfold+"/"
    final_multimodal_scores = {}
    bias_metrics_text = {}
    bias_metrics_image = {}
    bias_value_multimodal_metrics = {}
    bias_value_metrics = {}
    overall_auc_metrics = {}
    syn_data=syn_10_df.copy()
    text={}
    image={}
    overall_auc_metrics_syn = {}
    i = 0
    for key, value in syn_folds.items(): #range(len([MODELNAMES[1]])): #range(len(MODELNAMES)):
        syn_10_df=syn_data
        syn_10_df['file_name'] = syn_10_df['file_name_'+str(i+1)]
        ide_pres = pd.read_csv(folder_folds_id+ key+ "_"+keyfold+"_tag_term_presence.csv", sep="\t")
        ide_pres.rename(columns={'0': 'file_name'}, inplace= True)
        syn_10_df=syn_10_df.merge(ide_pres.drop(columns=['misogynous', 'text']),
                        how='inner', on='file_name')

        syn_10_df['misogynous'] = syn_10_df['label_'+modelnames[i]]
        
        identity_terms_present = models_performances_bias.identity_element_presence(syn_10_df, identity_terms, 'misogynous')
        tags_present = models_performances_bias.identity_element_presence_OR(syn_10_df, identity_tags, "misogynous")
        print("LEN ID TERM ", len(identity_terms))
        print("LEN ID PRES ", len(identity_terms_present))
        print(len(syn_10_df))



        bias_metrics_text[i] = models_performances_bias.compute_bias_metrics_for_model(syn_10_df, identity_terms_present, modelnames[i],
                                                            "misogynous")
        bias_metrics_image[i] = models_performances_bias.compute_bias_metrics_for_model(syn_10_df, tags_present, modelnames[i],
                                                            "misogynous")
        overall_auc_metrics[i] = models_performances_bias.calculate_overall_auc(test_df, modelnames[i])
        
        
        overall_auc_metrics_syn[i] = models_performances_bias.calculate_overall_auc(syn_10_df, modelnames[i])
        

        final_multimodal_scores[i] = models_performances_bias.get_final_multimodal_metric_nan(bias_metrics_text[i],
                                                                bias_metrics_image[i],
                                                                overall_auc_metrics[i],
                                                                modelnames[i])

        bias_value_multimodal_metrics[i] = np.nanmean([
            np.nanmean([
                bias_metrics_text[i][modelnames[i] + '_subgroup_auc'],
                bias_metrics_text[i][modelnames[i] + '_bpsn_auc'],
                bias_metrics_text[i][modelnames[i] + '_bnsp_auc']
            ]),
            np.nanmean([
                bias_metrics_image[i][modelnames[i] + '_subgroup_auc'],
                bias_metrics_image[i][modelnames[i] + '_bpsn_auc'],
                bias_metrics_image[i][modelnames[i] + '_bnsp_auc']
            ])
        ])
        
        text[i] = np.nanmean([
                bias_metrics_text[i][modelnames[i] + '_subgroup_auc'],
                bias_metrics_text[i][modelnames[i] + '_bpsn_auc'],
                bias_metrics_text[i][modelnames[i] + '_bnsp_auc']
            ])
        image[i] = np.nanmean([
                bias_metrics_image[i][modelnames[i] + '_subgroup_auc'],
                bias_metrics_image[i][modelnames[i] + '_bpsn_auc'],
                bias_metrics_image[i][modelnames[i] + '_bnsp_auc']
            ])
        
        i+=1
    return final_multimodal_scores, overall_auc_metrics, bias_value_multimodal_metrics, text, image, overall_auc_metrics_syn

print(column_prob)
# Load predictions on Test dataset


test_df = pd.read_csv(csv_path_test,sep='\t')
#test_df.rename(columns={'c': 'text'}, inplace=True)
test_df["clean"]= test_df.clean.fillna("")
#test_df.drop(columns = ["Unnamed: 0"], inplace=True)
print(test_df.info())

for i in range(1,11):
    pred_csv = path_results_test+'probs_test_fold_'+str(i)+'.csv'
    pred = pd.read_csv(pred_csv, sep="\t")
    #pred.columns[33] = MODELNAMES[i-1]
    pred.rename(columns={column_prob: MODELNAMES[i-1]}, inplace=True)
    #pred['file_name'] = pred['file_name'].astype(str) + '.jpg'
    #pred[MODELNAMES[i]+'_label']=pred[MODELNAMES[i]+'_label'].astype(int)
    test_df = pd.merge(test_df, pred[["file_name", MODELNAMES[i-1]]], on="file_name")
test_df.to_csv(path_models + "baseline_test_bias_"+model+"_text_28_07.csv", sep="\t", index=False)
#model_bias_analysis.plot_model_family_auc(test_df, MODELNAMES, label_column)


score_test = compute_score(test_df, MODELNAMES, label_column_test, 0.5)
# Load predictions on Syn dataset
print("score test calculated")

#syn_df = pd.read_csv(csv_path_syntest,sep='\t')
#syn_df.rename(columns={'cleaned': 'clean'}, inplace=True)
#
##syn_df["clean"]= syn_df.clean.fillna("")
#syn_df.drop(columns = ["Unnamed: 0",  "Unnamed: 0.1.1.1" ], inplace=True)
#syn_df.rename(columns={'0': 'file_name'}, inplace= True)
#print(syn_df.info())
with open('../../data/datasets/synthetic_folds.pkl', 'rb') as f:
    syn_folds = pickle.load(f)
    
print(syn_folds.keys())
i = 1
syn_df = pd.DataFrame()
syn_completo = pd.read_csv(csv_path_syntest,sep='\t')
syn_completo.rename(columns={'cleaned': 'clean'}, inplace=True)
add_tag_string(syn_completo)
#syn_df["clean"]= syn_df.clean.fillna("")
syn_completo.drop(columns = ["Unnamed: 0",  "Unnamed: 0.1.1.1" ], inplace=True)
syn_completo.rename(columns={'0': 'file_name'}, inplace= True)
#print(list(syn_completo["file_name"])[0])
#print(syn_completo.info())

for key,value in syn_folds.items():

    syn_indexes = value[key_of_folds]
    ##dividere il pred syn per gli indici dei fold se no non combaciano le dimensioni
    #print(syn_indexes)
    syn_df_fold = syn_completo.loc[syn_indexes]
    pred_csv = path_results_sin+'probs_sin_test_fold_'+str(i)+'.csv'
    pred = pd.read_csv(pred_csv, sep="\t")
    pred.rename(columns={column_prob: MODELNAMES[i-1]}, inplace=True)
    #print("##############coincidono?")
    #print(list(syn_df_fold["file_name"])[300])
    #print(list(pred["file_name"])[300])
    fold_file_names = list(pred["file_name"])
    fold_probs = pred[MODELNAMES[i-1]]
    #print(type(pred["ground_truth"]))
    fold_labels = [item for item in list(pred["ground_truth"])]
    #fold_labels = pred["ground_truth"]
    name_label = "label_"+ MODELNAMES[i-1]
    file_name = "file_name_"+key.split("_")[1]
    assert(len(fold_file_names) == len(fold_probs) == len(fold_labels) == len(list(syn_df_fold["clean"])) == len(list(syn_df_fold["tag_string"])))
    text_name = "text_"+ key.split("_")[1]
    tag_name = "tag_string_"+ key.split("_")[1]
    #print("############################ ",len(pd.Series(fold_probs)))
    #print("############################ ",len(pred[MODELNAMES[i-1]]))
    
    new_df = pd.DataFrame({file_name: fold_file_names, MODELNAMES[i-1]: fold_probs, name_label: fold_labels, text_name: list(syn_df_fold["clean"]), tag_name: list(syn_df_fold["tag_string"])})
    syn_df = pd.concat([syn_df, new_df], axis=1)
    syn_df[name_label] = syn_df[name_label].astype(bool)
    #print("############################ ",len(syn_df[MODELNAMES[i-1]]))
    #syn_df[file_name] = pd.Series(fold_file_names)
    #syn_df[MODELNAMES[i-1]] = pd.Series(fold_probs)
    #syn_df[name_label] = pd.Series(fold_labels)
    #syn_df["text"] = pd.Series(syn_df_fold["text"])
    #syn_df["tag list"] = pd.Series(syn_df_fold["tag list"])

    #print(list(pred))
    #pred['file_name'] = 'SIN_'+ pred['file_name'].astype(str) + '.jpg'
    #pred[MODELNAMES[i-1]+'_label']=pred[MODELNAMES[i-1]+'_label'].astype(int)
    #syn_df_fold = pd.merge(syn_df_fold, pred[["file_name", MODELNAMES[i-1]]], on="file_name")
    i+=1
print(syn_df.info())
syn_df.to_csv(path_models + "/baseline_SYN_bias_"+model+"_text_28_07.csv", sep="\t", index=False)

print("PATH  ", path_results_sin)
score_syn = confusion_rates_on_file_10Fold_syn(syn_folds,syn_df, MODELNAMES, 0.5, key_of_folds)
print("score syn calculated")




def apply_lemmatization_stanza(texts):
    print("preprocessing lemmatization")
    """ Apply lemmatizaion with post tagging operatio through Stanza.
    Remove stopwords and puntuation.
    Lower case """
    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    
    processed_text = []
    gc.collect()
    for testo in texts:
        rev = []
        testo = testo.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        testo = testo.lower()
        testo = re.sub(r'\d+', '', testo)  # remove numbers
        testo = re.sub('[^A-Za-z0-9 ]+', '', testo)  # remove special character
        testo = " ".join(testo.split())  # single_spaces

        doc = nlp(testo)
        for token in doc:
            rev.append(token.lemma_)
            # print(token.lemma_)

        for word in list(rev):  # iterating on a copy since removing will mess things up
            if word in model_bias_analysis.stopwords:
                rev.remove(word)
        pro_tokens = " ".join(rev)
        #rev = str(rev).replace("'", '').replace(",", '').replace("[", '').replace("]", '').replace("\"", '')
        processed_text.append(pro_tokens)
    return (processed_text)

#res syn have meme in the seme order od syn_df
#syn_df['clear_text']= data_pre_processing(syn_df['text'])
print(syn_df.info())

print("bias metric")

if sottogruppo == "pos":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'tag_string', model_bias_analysis.identity_tags_pos)
    subgroup_tag= model_bias_analysis.identity_tags_pos
elif sottogruppo == "neg":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'tag_string', model_bias_analysis.identity_tags_neg)
    subgroup_tag = model_bias_analysis.identity_tags_neg
elif sottogruppo == "both":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'tag_string', model_bias_analysis.identity_tags)
    #Computes per-subgroup metrics for all subgroups and a list of models.
    subgroup_tag = model_bias_analysis.identity_tags
#print(syn_df.info())
#Add labels indicating the term presence

print(sottogruppo)
if sottogruppo == "pos":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'clean', model_bias_analysis.identity_terms_pos)
    subgroup_text = model_bias_analysis.identity_terms_pos
elif sottogruppo == "neg":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'clean', model_bias_analysis.identity_terms_neg)
    subgroup_text = model_bias_analysis.identity_terms_neg
elif sottogruppo == "both":
    #model_bias_analysis.add_subgroup_columns_from_text(syn_df, 'clean', model_bias_analysis.identity_terms)
    #Computes per-subgroup metrics for all subgroups and a list of models.
    subgroup_text = model_bias_analysis.identity_terms



#subgroups = subgroup_text + subgroup_tag
#print("sottogruppi ",subgroups)
#print(syn_df)
#print(subgroups)
#print(MODELNAMES)
#print(label_column)

#def compute_bias_metrics_on_syn_10(modelnames, syn_10_df, test_df, identity_terms, identity_tags, label_column):
if bias_type == "text":
    #return final_multimodal_scores, overall_auc_metrics, bias_value_multimodal_metrics, text, image, overall_auc_metrics_syn

    final_scores, overall_auc_metrics, bias_value_metrics, text, auc_sin = compute_bias_metrics_on_syn_10_text(MODELNAMES, syn_df, test_df, subgroup_text, label_column_syn, syn_folds, key_of_folds)
    
    #model_bias_analysis.compute_bias_metrics_for_models(syn_df,
    #                                    subgroup_text,
    #                                    MODELNAMES,
    #                                    label_column_syn)
    #final_scores = {}
    #bias_metrics = {}
    #bias_value_metrics={}
    #overall_auc_metrics = {}
    ##bias_metrics = model_bias_analysis.compute_bias_metrics_for_models(syn_df, subgroups, MODELNAMES, label_column)
    #auc_sin = {}


    #for i in range(10):
    #  bias_metrics[i] = model_bias_analysis.compute_bias_metrics_for_model(syn_df, subgroup_text, MODELNAMES[i], label_column_syn)
    #  overall_auc_metrics[i]=model_bias_analysis.calculate_overall_auc(test_df, MODELNAMES[i])
    #  final_scores[i], auc_sin[i] = model_bias_analysis.get_final_metric(bias_metrics[i],overall_auc_metrics[i], MODELNAMES[i])
#
    #  bias_value_metrics[i]=np.average([
    #        bias_metrics[i][MODELNAMES[i] + '_subgroup_auc'],
    #        bias_metrics[i][MODELNAMES[i] + '_bpsn_auc'],
    #        bias_metrics[i][MODELNAMES[i] + '_bnsp_auc']
    #    ])


    print('max overall auc (AUC raw): ',max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys())))
    print('average overall auc (AUC raw): ', np.average(list(overall_auc_metrics.values())))

    print('max bias value: ',max(zip(bias_value_metrics.values(), bias_value_metrics.keys())))
    print('average bias value: ', np.average(list(bias_value_metrics.values())))

    print('max auc_sin: ',max(zip(auc_sin.values(), auc_sin.keys())))
    print('average auc_sin: ', np.average(list(auc_sin.values())))

    print('max AUC final: ',max(zip(final_scores.values(), final_scores.keys())))
    print('average AUC final: ', np.average(list(final_scores.values())))
    
    print("#################ALL AUC RAW FOLDS FOR T-TEST############################")
    print(overall_auc_metrics.values())
    print("#################ALL AUC SYN FOLDS FOR T-TEST#####################")
    print(auc_sin.values())
    print("#################ALL MEB FOLDS FOR T-TEST########################")
    print(final_scores.values())
    
elif bias_type == "multi":
    
    final_multimodal_scores, overall_auc_metrics, bias_value_multimodal_metrics, bias_metrics_text, bias_metrics_image, auc_sin =  compute_bias_metrics_on_syn_10_multi(MODELNAMES, syn_df, test_df, subgroup_text, subgroup_tag, label_column_syn, syn_folds, key_of_folds)
    print("METRICA MULTIMODALE")
    #final_multimodal_scores = {}
    #bias_metrics_text = {}
    #bias_metrics_image = {}
    #bias_value_multimodal_metrics={}
    #overall_auc_metrics = {}
    #auc_sin = {}
    #for i in range(10):
    #  bias_metrics_text[i] = model_bias_analysis.compute_bias_metrics_for_model(syn_df, model_bias_analysis.identity_terms, MODELNAMES[i], label_column_syn)
    #  bias_metrics_image[i] = model_bias_analysis.compute_bias_metrics_for_model(syn_df,model_bias_analysis.identity_tags, MODELNAMES[i], label_column_syn)
    #  overall_auc_metrics[i]=model_bias_analysis.calculate_overall_auc(test_df, MODELNAMES[i])
#
    #  final_multimodal_scores[i], auc_sin[i]=model_bias_analysis.get_final_multimodal_metric(bias_metrics_text[i], bias_metrics_image[i], overall_auc_metrics[i], MODELNAMES[i])
#
    #  bias_value_multimodal_metrics[i]=np.average([
    #      np.average([
    #        bias_metrics_text[i][MODELNAMES[i] + '_subgroup_auc'],
    #        bias_metrics_text[i][MODELNAMES[i] + '_bpsn_auc'],
    #        bias_metrics_text[i][MODELNAMES[i] + '_bnsp_auc']
    #      ]),
    #      np.average([
    #        bias_metrics_image[i][MODELNAMES[i] + '_subgroup_auc'],
    #        bias_metrics_image[i][MODELNAMES[i] + '_bpsn_auc'],
    #        bias_metrics_image[i][MODELNAMES[i] + '_bnsp_auc']
    #      ])
    #      ])
    
    print('max overall auc (AUC raw): ',max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys())))
    print('average overall auc (AUC raw): ', np.average(list(overall_auc_metrics.values())))

    print('max multimodal bias: ',max(zip(bias_value_multimodal_metrics.values(), bias_value_multimodal_metrics.keys())))
    print('average multimodal bias: ', np.average(list(bias_value_multimodal_metrics.values())))

    print('max auc_sin: ',max(zip(auc_sin.values(), auc_sin.keys())))
    print('average auc_sin: ', np.average(list(auc_sin.values())))
    print('max AUC multimodal final: ',max(zip(final_multimodal_scores.values(), final_multimodal_scores.keys())))
    print('average AUC multimodal final: ', np.average(list(final_multimodal_scores.values())))
    
    print("#################ALL AUC RAW FOLDS FOR T-TEST############################")
    print(overall_auc_metrics.values())
    print("#################ALL AUC SYN FOLDS FOR T-TEST#####################")
    print(auc_sin.values())
    print("#################ALL AUC SYN FOLDS FOR T-TEST#####################")
    print(bias_value_multimodal_metrics.values())
    print("#################ALL MEB FOLDS FOR T-TEST########################")
    print(final_multimodal_scores.values())
    bias_dict = {
        "auc_raw": list(overall_auc_metrics.values()),
        "auc_sin": list(auc_sin.values()),
        "bias_value": list(bias_value_multimodal_metrics.values()),
        "meb":  list(final_multimodal_scores.values())
    }
    filename_bias = "text_" + model + "_"+ correction_strategy+ ".pkl"
    with open('../../2_strategy_test/results/bias/mebs_for_ttest/'+filename_bias, 'wb') as f:
        pickle.dump(bias_dict, f)
    
    #with open('../../2_strategy_test/results/bias/mebs_for_ttest/'+filename_bias, 'rb') as f:
    #    bias_text_dict = pickle.load(f)
    #print(bias_text_dict)
        