import pandas as pd
from . import project_paths
import json

def load_training_data():
    dataset = pd.read_csv(project_paths.csv_path_train, sep='\t')
    dataset.drop(columns=["shaming", "stereotype", "objectification", "violence"], inplace=True)
    return dataset

def load_training_data_Spacy():
    dataset = pd.read_csv(project_paths.csv_path_train_Spacy, sep='\t')
    return dataset

def load_test_data():
    dataset = pd.read_csv(project_paths.csv_path_test, sep='\t')
    dataset.drop(columns=["shaming", "stereotype", "objectification", "violence"], inplace=True)
    return dataset

def load_test_data_Spacy():
    dataset = pd.read_csv(project_paths.csv_path_test_Spacy, sep='\t')
    return dataset

def load_folds():
    """ it loads folds store in the folds.json file to ensure that every
    model uses the same folds"""
    with open(project_paths.json_path_fold) as f:
        kfold = json.load(f)
    return kfold

def load_syn_data():
    dataset = pd.read_csv(project_paths.csv_path_syn, sep='\t')
    return dataset

def load_new_syn_data():
    dataset = pd.read_csv(project_paths.csv_path_new_syn, sep='\t')
    return dataset

def load_syn_data_Spacy():
    dataset = pd.read_csv(project_paths.csv_path_syn_Spacy, sep='\t')
    return dataset

def load_data_score(score_path):
    dataset = pd.read_csv(score_path, sep='\t')
    return dataset
# _______________________ Captions ___________________________________
def load_training_data_capt():
    dataset = pd.read_csv(project_paths.csv_path_train_caps, sep='\t')
    dataset["caption"]= dataset.caption.fillna('')
    dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_test_data_capt():
    dataset = pd.read_csv(project_paths.csv_path_test_caps, sep='\t')
    dataset["caption"]= dataset.caption.fillna('')
    dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset
    
def load_syn_data_capt():
    dataset = pd.read_csv(project_paths.csv_path_test_caps, sep='\t')
    dataset["caption"]= dataset.caption.fillna('')
    dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

# ____________________________ Tags ___________________________________
def load_training_data_tag():
    dataset = pd.read_csv(project_paths.csv_path_train_tags, sep='\t')
    dataset.drop(columns = ["Unnamed: 0", "1_y"], inplace=True)
    return dataset

def load_training_data_tag_masked():
    dataset = pd.read_csv(project_paths.csv_path_train_masked_tag)
   # dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_training_data_tag_masked_count():
    dataset = pd.read_csv(project_paths.csv_path_train_count_masked_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_training_data_tag_censored():
    dataset = pd.read_csv(project_paths.csv_path_train_censored_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_test_data_tag_masked():
    dataset = pd.read_csv(project_paths.csv_path_test_masked_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_test_data_tag_masked_count():
    dataset = pd.read_csv(project_paths.csv_path_test_count_masked_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_test_data_tag_censored():
    dataset = pd.read_csv(project_paths.csv_path_test_censored_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset


def load_sintest_data_tag_masked():
    dataset = pd.read_csv(project_paths.csv_path_sintest_masked_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_sintest_data_tag_masked_count():
    dataset = pd.read_csv(project_paths.csv_path_sintest_count_masked_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def load_sintest_data_tag_censored():
    dataset = pd.read_csv(project_paths.csv_path_sintest_censored_tag)
    #dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset


def load_test_data_tag():
    dataset = pd.read_csv(project_paths.csv_path_test_tags, sep='\t')
    dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset
    
def load_syn_data_tag():
    dataset = pd.read_csv(project_paths.csv_path_new_syn_tags, sep='\t')
    dataset.drop(columns = ["Unnamed: 0"], inplace=True)
    return dataset

def data_tag_selection(dataset, tags_impo):
    for tag in dataset.columns[2:]:
        if tag not in tags_impo:
            dataset.drop(columns = [tag], inplace=True)
    dataset.iloc[:,2:].round
    return dataset