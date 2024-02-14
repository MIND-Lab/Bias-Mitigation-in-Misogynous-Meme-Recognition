import os
import pandas as pd
import re
import numpy as np
from sklearn import metrics
import sys
import matplotlib.pyplot as plt



stopwords = ["a", "about", "above", "above", "across", "afterwards", "again", "against",
             "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
             "amoungst",
             "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
             "around",
             "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "beforehand",
             "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
             "by",
             "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
             "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
             "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify",
             "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from",
             "front",
             "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
             "hereby",
             "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into",
             "is",
             "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "meanwhile", "might",
             "mill",
             "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never",
             "nevertheless",
             "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "now", "nowhere", "of", "off", "often",
             "on", "once",
             "one", "only", "onto", "or", "other", "others", "otherwise", "out", "over", "part", "per", "perhaps",
             "please",
             "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should",
             "show",
             "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "sometime", "sometimes",
             "somewhere",
             "still", "such", "system", "take", "ten", "than", "that", "the", "then", "thence", "there", "thereafter",
             "thereby",
             "therefore", "therein", "thereupon", "these", "thick", "thin", "third", "this", "those", "though", "three",
             "through",
             "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
             "two", "un",
             "under", "until", "up", "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence",
             "whenever",
             "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
             "while", "whither",
             "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
             "the",
             "ve", "re", "ll", "10", "11", "18", "oh", "s", "t", "m", "did", "don", "got"]

SUBGROUP = 'subgroup'
SUBSET_SIZE = 'subset_size'
SUBGROUP_AUC = 'subgroup_auc'
NEGATIVE_CROSS_AUC = 'bpsn_auc'
POSITIVE_CROSS_AUC = 'bnsp_auc'
#identity_terms = ['dishwasher', 'chick', 'whore', 'demotivational', 'diy', 'promotion', 'bestdemotivationalposters', 'motivateusnot', 'imgur', 'motifake','memeshappen', 'mcdonald', 'ambulance', 'developer', 'template', 'anti', 'valentine', 'communism', 'weak', 'memecrunch']
identity_terms = ['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong','mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']
identity_terms_pos = ['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong']
identity_terms_neg = ['mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']
#identity_terms_pos = ['dishwasher', 'chick', 'whore', 'demotivational', 'diy', 'promotion', 'bestdemotivationalposters', 'motivateusnot', 'imgur', 'motifake']
#identity_terms_neg = ['memeshappen', 'mcdonald', 'ambulance', 'developer', 'template', 'anti', 'valentine', 'communism', 'weak', 'memecrunch']

import pickle
with open('identity_tags_mis', 'rb') as f:
    id_tags_mis = pickle.load(f)    
with open('identity_tags_notmis', 'rb') as f:
    id_tags_notmis = pickle.load(f)

identity_tags_pos = id_tags_mis[:10]
identity_tags_neg = id_tags_notmis[:10]
identity_tags = identity_tags_pos+ identity_tags_neg
print(identity_tags_pos)
print(identity_tags_neg)
print(identity_tags)
def add_subgroup_columns_from_text(df, text_column, subgroups):
    """Adds a boolean column for each subgroup to the data frame.

      New column contains True if the text contains that subgroup term.
      """
    for term in subgroups:
        if term == "blond":
            df[term] = df[text_column].apply(lambda x: bool(
                re.search('\\b' + term + '\\b | \\b'+ term+ 'e\\b', str(x), flags=re.UNICODE | re.IGNORECASE)))
        # pylint: disable=cell-var-from-loop
        else:
            df[term] = df[text_column].apply(lambda x: bool(
                re.search('\\b' + term + '\\b', str(x), flags=re.UNICODE | re.IGNORECASE)))

#________________________________________________COMPUTE METRICS__________________________________________________
def compute_auc(y_true, y_pred):        
     #print(y_true)
    #print(y_pred)
    try:
        #print(len(y_true))
        #print(len(y_pred))
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError as e:
  
        #print("TRUE", y_true)
        #print("PRED ", y_pred)
        return np.nan

def model_family_auc(dataset, model_names, label_col):
    
    aucs = [
        compute_auc(dataset[label_col], dataset[model_name])
        for model_name in model_names
    ]
    return {
        'aucs': aucs,
        'mean': np.mean(aucs),
        'median': np.median(aucs),
        'std': np.std(aucs),
    }

def compute_subgroup_auc(df, subgroup, label, model_name):
    #print(subgroup)
    subgroup_examples = df[df[subgroup]]
    #print(subgroup_examples)
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def confusion_matrix_counts(df, score_col, label_col, threshold=0):
    """compute confusion rates _
    if threshold is not passed (=0), it computes matrix using predicted labels (from boolean values)"""
    if threshold:
        return {
            'tp': len(df[(df[score_col] >= threshold) & df[label_col]]),
            'tn': len(df[(df[score_col] < threshold) & ~(df[label_col])]),
            'fp': len(df[(df[score_col] >= threshold) & ~df[label_col]]),
            'fn': len(df[(df[score_col] < threshold) & df[label_col]]),
        }
    else:
        return {
            'tp': len(df[(df[score_col]==1) & df[label_col]]),
            'tn': len(df[(df[score_col]==0) & ~(df[label_col])]),
            'fp': len(df[(df[score_col]==1) & ~df[label_col]]),
            'fn': len(df[(df[score_col]==0) & df[label_col]]),
        }


# https://en.wikipedia.org/wiki/Confusion_matrix
def compute_confusion_rates(df, score_col, label_col, threshold=0):
    confusion = confusion_matrix_counts(df, score_col, label_col, threshold)
    #print(confusion)
    #print(score_col)
    actual_positives = confusion['tp'] + confusion['fn']
    actual_negatives = confusion['tn'] + confusion['fp']
    # True positive rate, sensitivity, recall.
    if (actual_positives > 0):
        tpr = confusion['tp'] / actual_positives
    else:
        tpr = 0
    if (actual_negatives > 0):
        # True negative rate, specificity.
        tnr = confusion['tn'] / actual_negatives
    else:
        tnr = 0

    # False positive rate, fall-out.
    fpr = 1 - tnr
    # False negative rate, miss rate.
    fnr = 1 - tpr
    if ((confusion['tp'] + confusion['fp']) > 0):
        # Precision, positive predictive value.
        precision = confusion['tp'] / (confusion['tp'] + confusion['fp'])
    else:
        precision = 0

    accuracy = (confusion['tp'] + confusion['tn'])/(actual_positives + actual_negatives)
    f1=2*(precision*tpr)/(precision+tpr)
    auc = compute_auc(df[label_col], df[score_col])

    return {
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'precision': precision,
        'recall': tpr,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,

    }

def compute_bias_metrics_for_subgroup_and_model(dataset,
                                                subgroup,
                                                model,
                                                label_col):
    """Computes per-subgroup metrics for one model and subgroup."""
    record = {
        SUBGROUP: subgroup,
        SUBSET_SIZE: len(dataset[dataset[subgroup]])
    }
    record[column_name(model, SUBGROUP_AUC)] = compute_subgroup_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, NEGATIVE_CROSS_AUC)] = compute_negative_cross_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, POSITIVE_CROSS_AUC)] = compute_positive_cross_auc(
        dataset, subgroup, label_col, model)
    #print("RECORD ",record)
    return record


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        subgroup_record = compute_bias_metrics_for_subgroup_and_model(
            dataset, subgroup, model, label_col)
        #records.append(subgroup_record) #append function is deprecated
        records = [*records, subgroup_record]
    return pd.DataFrame(records)


def compute_bias_metrics_for_models(dataset,
                                    subgroups,
                                    models,
                                    label_col):
    """Computes per-subgroup metrics for all subgroups and a list of models."""
    output = None

    for model in models:
        model_results = compute_bias_metrics_for_model(dataset, subgroups, model,
                                                       label_col)
        if output is None:
            output = model_results
        else:
            output = output.merge(model_results, on=[SUBGROUP, SUBSET_SIZE])
    return output


#___________________________________________OTHER________________________________________________________


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def column_name(model, metric):
    return model + '_' + metric


def compute_negative_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    #examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    examples=pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])
    print(subgroup)
    return compute_auc(examples[label], examples[model_name])


def compute_positive_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    #examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples])
    print(subgroup)
    return compute_auc(examples[label], examples[model_name])


def calculate_overall_auc(df, model_name):
    true_labels = df['misogynous']
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc_test, model_name):
    bias_score = np.average([
        bias_df[model_name + '_subgroup_auc'],
        bias_df[model_name + '_bpsn_auc'],
        bias_df[model_name + '_bnsp_auc']
    ])
    return np.mean([overall_auc_test, bias_score]), bias_score

def get_final_multimodal_metric(bias_df_text, bias_df_image, overall_auc_test, model_name):
    """compute AUC Final Meme _ a metric bias proposed to compute multimodal bias in memes
    it considers bias in text and in image """
    bias_score_text = np.average([
        bias_df_text[model_name + '_subgroup_auc'],
        bias_df_text[model_name + '_bpsn_auc'],
        bias_df_text[model_name + '_bnsp_auc']
    ])
    bias_score_image = np.average([
        bias_df_image[model_name + '_subgroup_auc'],
        bias_df_image[model_name + '_bpsn_auc'],
        bias_df_image[model_name + '_bnsp_auc']
    ])
    bias_score=np.mean([bias_score_text, bias_score_image])
    return np.mean([overall_auc_test, bias_score]), bias_score

#____________________________________________PLOT________________________________________________________________
def plot_model_family_auc(dataset, model_names, label_col, min_auc=0.7, max_auc = 1.0):
    result = model_family_auc(dataset, model_names, label_col)
    print('mean AUC:', result['mean'])
    print('median:', result['median'])
    print('stddev:', result['std'])
    plt.hist(result['aucs'])
    plt.gca().set_xlim([min_auc, max_auc])
    plt.show()
    return result