# ___________________________________ Data ____________________________________________
csv_path_train = '../data/datasets/training.csv'
csv_path_test = '../data/datasets/test_label.csv'
csv_path_syn = '../data/datasets/synthetic.csv'
csv_path_new_syn = '../data/datasets/new_synthetic_text_tag.csv'



""" Json file containing information to perform 10 fold dataset division"""
json_path_fold = '../data/folds.json'

""" 'clear' obtained via preprocessingtrain data with a column  operations, including
 Spacy Stanza lemmatization, performed on OCR text.
 csv file produced by 'unimodal_text_bma.ipynb' """
csv_path_train_Spacy = '/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/text_csv/training_Spacy.csv'
csv_path_test_Spacy = '../data/datasets/text_csv/test_Spacy.csv'
csv_path_syn_Spacy = '../data/datasets/text_csv/synthetic_Spacy.csv'
csv_path_new_syn_spacy = '../data/datasets/text_csv/new_synthetic_spacy.csv'

######################REPAIR##################################

csv_path_train_repair_rank_cls = "../data/datasets/repair/cls_rank.csv"
csv_path_train_repair_rank = "../data/datasets/repair/rank.csv"
csv_path_train_repair_sample = "../data/datasets/repair/sample.csv"
csv_path_train_repair_uniform = "../data/datasets/repair/uniform.csv"
csv_path_train_repair_tresh = "../data/datasets/repair/treshold.csv"


######################MASKED##########################
csv_path_train_Spacy_masked = '/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/text_csv/training_Spacy_masked.csv'
csv_path_test_Spacy_masked = '../data/datasets/text_csv/test_Spacy_masked.csv'
csv_path_new_syn_spacy_masked = '../data/datasets/text_csv/new_synthetic_spacy_masked.csv'
#####################CENSORED################################
csv_path_train_Spacy_censored = '/home/jimmy/Documenti/PhD/Project_BMA/data/datasets/text_csv/training_Spacy_censored.csv'
csv_path_test_Spacy_censored = '../data/datasets/text_csv/test_Spacy_censored.csv'
csv_path_new_syn_spacy_censored = '../data/datasets/text_csv/new_synthetic_spacy_censored.csv'

# Captions
csv_path_train_caps = '../data/datasets/cap_csv/train_caps.csv'
csv_path_test_caps = '../data/datasets/cap_csv/test_caps.csv'
csv_path_syn_caps = '../data/datasets/cap_csv/syn_caps.csv'

# Tags
csv_path_train_tags = '../data/datasets/tags_csv/train_tags.csv'
csv_path_test_tags = '../data/datasets/tags_csv/test_tags.csv'
csv_path_syn_tags = '../data/datasets/tags_csv/syn_tags.csv'
csv_path_new_syn_tags = '../data/datasets/tags_csv/tags_new_sintest.csv'


csv_path_train_masked_tag = "../data/datasets/tags_csv/tags_masked_train.csv"
csv_path_train_count_masked_tag = "../data/datasets/tags_csv/tags_masked_count_train.csv"
csv_path_train_censored_tag =  "../data/datasets/tags_csv/tags_censored_train.csv"

csv_path_test_masked_tag = "../data/datasets/tags_csv/tags_masked_test.csv"
csv_path_test_count_masked_tag = "../data/datasets/tags_csv/tags_masked_count_test.csv"
csv_path_test_censored_tag =  "../data/datasets/tags_csv/tags_censored_test.csv"

csv_path_sintest_masked_tag = "../data/datasets/tags_csv/tags_masked_sintest.csv"
csv_path_sintest_count_masked_tag = "../data/datasets/tags_csv/tags_masked_count_sintest.csv"
csv_path_sintest_censored_tag =  "../data/datasets/tags_csv/tags_censored_sintest.csv"

# ___________________________________ Folders ____________________________________________
""" Results folder in which store Unimodal models results """
folder_results = './results/'

folder_res_train = folder_results+ 'train/'
folder_res_test = folder_results+'test/'
folder_res_syn = folder_results+'syn/'




# csv with models F1 scores coputed on training data (9Fold_8+1), used by BMA 
csv_uni_text_train_scores = folder_res_train + 'unimodal_text_scores.csv'

# csv with models predictions (probability) on training data (10Fold approach)
csv_uni_text_train_probs = folder_res_train + 'unimodal_text_probs.csv'

# csv with averaged model performance on training data
csv_uni_text_train_res = folder_res_train + 'unimodal_text_res.csv'
# ____________________________________Results storage file________________________________
""" _______________________ Unimodal text BMA _______________________"""
""" Unimodal text BMA _Training data """

# csv with models F1 scores coputed on training data (9Fold_8+1), used by BMA 
csv_uni_text_train_scores = folder_res_train + 'unimodal_text_scores.csv'

# csv with models predictions (probability) on training data (10Fold approach)
csv_uni_text_train_probs = folder_res_train + 'unimodal_text_probs.csv'

# csv with averaged model performance on training data
csv_uni_text_train_res = folder_res_train + 'unimodal_text_res.csv'

""" Unimodal text BMA _ Test data """

########################JIMMY RULES##### text paths###########################################################################################
####jimmy's rules#############
""" Results folder in which store Unimodal models results """
folder_results2 = '../data/results2strategy/'

folder_res_text = folder_results2+ 'text/'
folder_res_tags = folder_results2+'tags/'
folder_res_caps = folder_results2+'caps/'

csv_uni_text_test_probs_all = folder_res_text + 'test/probs_test_all_fold.csv'
csv_uni_text_test_probs = folder_res_text + 'test/probs_test_fold_'
csv_uni_text_test_scores = folder_res_text + 'test/score_test_text10fold.csv'
csv_uni_text_test_res = folder_res_text + 'test/text_res_bma_test2202.csv'

########################MASKED####################################################
csv_uni_text_test_masked_probs = folder_res_text + 'masked_test/probs_test_all_fold.csv'
csv_uni_text_test_masked_probs_folds = folder_res_text + 'masked_test/probs_test_fold_'
csv_uni_text_test_masked_scores = folder_res_text + 'masked_test/score_test_text10fold.csv'
csv_uni_text_test_masked_res = folder_res_text + 'masked_test/text_res_bma_test2202.csv'
#########################MASKED COUNT##########################################
csv_uni_text_test_masked_count_probs = folder_res_text + 'masked_count_test/probs_test_all_fold.csv'
csv_uni_text_test_masked_count_probs_folds = folder_res_text + 'masked_count_test/probs_test_fold_'
csv_uni_text_test_masked_count_scores = folder_res_text + 'masked_count_test/score_test_text10fold.csv'
csv_uni_text_test_masked_count_res = folder_res_text + 'masked_count_test/text_res_bma_test2202.csv'
###########################CENSORED######################################
csv_uni_text_test_censored_probs = folder_res_text + 'censored_test/probs_test_all_fold.csv'
csv_uni_text_test_censored_probs_folds = folder_res_text + 'censored_test/probs_test_fold_'
csv_uni_text_test_censored_scores = folder_res_text + 'censored_test/score_test_text10fold.csv'
csv_uni_text_test_censored_res = folder_res_text + 'censored_test/text_res_bma_test2202.csv'
################################REPAIR RANK CLS####################################################
repair_folder_rank_cls = "repair/rank_cls/"
csv_uni_text_test_rank_cls_probs_all = folder_res_text+ repair_folder_rank_cls + 'test/probs_test_all_fold.csv'
csv_uni_text_test_rank_cls_probs_folds = folder_res_text + repair_folder_rank_cls+ 'test/probs_test_fold_'
csv_uni_text_test_rank_cls_scores = folder_res_text + repair_folder_rank_cls+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_rank_cls_res = folder_res_text + repair_folder_rank_cls+ 'test/res_bma_test.csv'
################################REPAIR RANK########################################################
repair_folder_rank = "repair/rank/"
csv_uni_text_test_rank_probs_all = folder_res_text+ repair_folder_rank + 'test/probs_test_all_fold.csv'
csv_uni_text_test_rank_probs_folds = folder_res_text + repair_folder_rank+ 'test/probs_test_fold_'
csv_uni_text_test_rank_scores = folder_res_text + repair_folder_rank+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_rank_res = folder_res_text + repair_folder_rank+ 'test/res_bma_test.csv'
################################REPAIR UNIFORM########################################################
repair_folder_uniform = "repair/uniform/"
csv_uni_text_test_uniform_probs_all = folder_res_text+ repair_folder_uniform + 'test/probs_test_all_fold.csv'
csv_uni_text_test_uniform_probs_folds = folder_res_text + repair_folder_uniform+ 'test/probs_test_fold_'
csv_uni_text_test_uniform_scores = folder_res_text + repair_folder_uniform+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_uniform_res = folder_res_text + repair_folder_uniform+ 'test/res_bma_test.csv'
################################REPAIR TRESHOLD########################################################
repair_folder_treshold = "repair/tresholding/"
csv_uni_text_test_treshold_probs_all = folder_res_text+ repair_folder_treshold + 'test/probs_test_all_fold.csv'
csv_uni_text_test_treshold_probs_folds = folder_res_text + repair_folder_treshold+ 'test/probs_test_fold_'
csv_uni_text_test_treshold_scores = folder_res_text + repair_folder_treshold+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_treshold_res = folder_res_text + repair_folder_treshold+ 'test/res_bma_test.csv'
################################REPAIR sample########################################################
repair_folder_sample = "repair/sample/"
csv_uni_text_test_sample_probs_all = folder_res_text+ repair_folder_sample + 'test/probs_test_all_fold.csv'
csv_uni_text_test_sample_probs_folds = folder_res_text + repair_folder_sample+ 'test/probs_test_fold_'
csv_uni_text_test_sample_scores = folder_res_text + repair_folder_sample+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_sample_res = folder_res_text + repair_folder_sample+ 'test/res_bma_test.csv'
################################REPAIR sample########################################################
repair_folder_censored_uniform = "repair/censored_uniform/"
csv_uni_text_test_censored_uniform_probs_all = folder_res_text+ repair_folder_censored_uniform + 'test/probs_test_all_fold.csv'
csv_uni_text_test_censored_uniform_probs_folds = folder_res_text + repair_folder_censored_uniform+ 'test/probs_test_fold_'
csv_uni_text_test_censored_uniform_scores = folder_res_text + repair_folder_censored_uniform+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_censored_uniform_res = folder_res_text + repair_folder_censored_uniform+ 'test/res_bma_test.csv'
################################REPAIR masked uniform########################################################
repair_folder_masked_uniform = "repair/masked_uniform/"
csv_uni_text_test_masked_uniform_probs_all = folder_res_text+ repair_folder_masked_uniform + 'test/probs_test_all_fold.csv'
csv_uni_text_test_masked_uniform_probs_folds = folder_res_text + repair_folder_masked_uniform+ 'test/probs_test_fold_'
csv_uni_text_test_masked_uniform_scores = folder_res_text + repair_folder_masked_uniform+ 'test/text_score_test_10fold.csv'
csv_uni_text_test_masked_uniform_res = folder_res_text + repair_folder_masked_uniform+ 'test/res_bma_test.csv'



#################################SYN#########################################
#csv_uni_text_syn_scores = folder_res_text + 'sintest/score_sin_test_text10fold.csv'
# csv with models predictions (probability) on test data (10Fold approach)
csv_uni_text_syn_probs_all = folder_res_text + 'sintest/probs_sin_test_all_fold.csv'
csv_uni_text_syn_probs = folder_res_text + 'sintest/probs_sin_test_all_fold.csv'
csv_uni_text_syn_scores = folder_res_text + 'sintest/score_sin_test_text10fold.csv'
csv_uni_text_new_syn_probs = folder_res_text + 'new_sintest/probs_sin_test_fold_'

###########################TEXT MASKED####################################################
csv_uni_text_syn_masked_probs_all= folder_res_text + 'masked_sintest/probs_sin_test_all_fold.csv'
csv_uni_text_syn_masked_probs_folds = folder_res_text + 'masked_sintest/probs_sin_test_fold_'
csv_uni_text_syn_masked_scores = folder_res_text + 'masked_sintest/score_sin_test_text10fold.csv'
csv_uni_text_syn_masked_res = folder_res_text + 'masked_sintest/text_res_bma_test2202.csv'

csv_uni_text_new_syn_masked_probs = folder_res_text + 'masked_new_sintest/probs_sin_test_fold_'
################################# TEXT MASKED COUNT##########################################
csv_uni_text_syn_masked_count_probs_all = folder_res_text + 'masked_count_sintest/probs_sin_test_all_fold.csv'
csv_uni_text_syn_masked_count_probs_folds = folder_res_text + 'masked_count_sintest/probs_sin_test_fold_'
csv_uni_text_syn_masked_count_scores = folder_res_text + 'masked_count_sintest/score_sin_test_text10fold.csv'
csv_uni_text_syn_masked_count_res = folder_res_text + 'masked_count_sintest/text_res_bma_test2202.csv'

csv_uni_text_new_syn_masked_count_probs = folder_res_text + 'masked_count_new_sintest/probs_sin_test_fold_'
#################################### TEXT CENSORED######################################################
csv_uni_text_syn_censored_probs_all = folder_res_text + 'censored_sintest/probs_sin_test_all_fold.csv'
csv_uni_text_syn_censored_probs_folds = folder_res_text + 'censored_sintest/probs_sin_test_fold_'
csv_uni_text_syn_censored_scores = folder_res_text + 'censored_sintest/score_sin_test_text10fold.csv'
csv_uni_text_syn_censored_res = folder_res_text + 'censored_sintest/score_sin_test_text10fold.csv'

######################################REPAIR RANK PER CLS######################################################
csv_uni_text_syn_rank_cls_probs_all = folder_res_text+ repair_folder_rank_cls + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_rank_cls_probs_folds = folder_res_text + repair_folder_rank_cls+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_rank_cls_scores = folder_res_text + repair_folder_rank_cls+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_rank_cls_res = folder_res_text + repair_folder_rank_cls+ 'sintest/res_bma_sintest.csv'
#/home/jimmy/Documenti/PhD/Project_BMA/data/results2strategy/text/repair/rank_cls/sintest/
######################################REPAIR RANK######################################################
csv_uni_text_syn_rank_probs_all = folder_res_text+ repair_folder_rank + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_rank_probs_folds = folder_res_text + repair_folder_rank+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_rank_scores = folder_res_text + repair_folder_rank+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_rank_res = folder_res_text + repair_folder_rank+ 'sintest/res_bma_sintest.csv'
######################################REPAIR UNIFORM ######################################################
csv_uni_text_syn_uniform_probs_all = folder_res_text+ repair_folder_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_uniform_probs_folds = folder_res_text + repair_folder_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_uniform_scores = folder_res_text + repair_folder_uniform+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_uniform_res = folder_res_text + repair_folder_uniform+ 'sintest/res_bma_sintest.csv'
######################################REPAIR TRESHOLD ######################################################
csv_uni_text_syn_treshold_probs_all = folder_res_text+ repair_folder_treshold + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_treshold_probs_folds = folder_res_text + repair_folder_treshold+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_treshold_scores = folder_res_text + repair_folder_treshold+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_treshold_res = folder_res_text + repair_folder_treshold+ 'sintest/res_bma_sintest.csv'
######################################REPAIR SAMPLE ######################################################
csv_uni_text_syn_sample_probs_all = folder_res_text+ repair_folder_sample + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_sample_probs_folds = folder_res_text + repair_folder_sample+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_sample_scores = folder_res_text + repair_folder_sample+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_sample_res = folder_res_text + repair_folder_sample+ 'sintest/res_bma_sintest.csv'
######################################REPAIR censored uniform ######################################################
csv_uni_text_syn_censored_uniform_probs_all = folder_res_text+ repair_folder_censored_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_censored_uniform_probs_folds = folder_res_text + repair_folder_censored_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_censored_uniform_scores = folder_res_text + repair_folder_censored_uniform+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_censored_uniform_res = folder_res_text + repair_folder_censored_uniform+ 'sintest/res_bma_sintest.csv'
######################################REPAIR masked uniform ######################################################
csv_uni_text_syn_masked_uniform_probs_all = folder_res_text+ repair_folder_masked_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_text_syn_masked_uniform_probs_folds = folder_res_text + repair_folder_masked_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_text_syn_masked_uniform_scores = folder_res_text + repair_folder_masked_uniform+ 'sintest/text_score_sintest_10fold.csv'
csv_uni_text_syn_masked_uniform_res = folder_res_text + repair_folder_masked_uniform+ 'sintest/res_bma_sintest.csv'


csv_uni_text_new_syn_censored_probs = folder_res_text + 'censored_new_sintest/probs_sin_test_fold_'
############################################################################################
# csv with averaged model performance on test data
csv_uni_text_syn_res = folder_res_text + 'sintest/text_res_bma_sin_test2202.csv'
csv_uni_text_new_syn_res = folder_res_text + 'new_sintest/text_res_bma_sin_test2202.csv'
## bma senza correzione



#############################OUTPUT CORREZIONI INFERENZA################################
out_bma_biased = "results/text_biased_bma/"
##output dei bma con correzioni applicate
out_corr_bma_neg = "results/text_bma_neg/"
out_corr_bma_pos = "results/text_bma_pos/"
out_corr_dyn_base = "results/text_dyn_corr/"
out_corr_dyn_bma = "results/text_dyn_corr_only_bma/"
out_corr_bma_neu = "results/text_bma_neu/"
out_corr_terms_base = "results/text_terms_corr_base/"
out_corr_terms_bma = "results/text_terms_corr_only_bma/"

###################################################################################################################
########################JIMMY RULES##### tags paths###########################################################################################
csv_uni_tags_test_probs = folder_res_tags + 'test/probs_test_fold_'
# csv with models F1 scores computed on 10 fold on whole training data
csv_uni_tags_test_scores = folder_res_tags + 'test/score_test_tags10fold_803.csv'
csv_uni_tags_test_res = folder_res_tags + 'test/tags_res_bma_test2202.csv'





csv_uni_tags_syn_scores = folder_res_tags + 'sintest/score_sin_test_tags10fold.csv'
# csv with models predictions (probability) on test data (10Fold approach)
csv_uni_tags_syn_probs = folder_res_tags + 'new_sintest/probs_sin_test_fold_'
# csv with averaged model performance on test data
csv_uni_tags_syn_res = folder_res_tags + 'new_sintest/tags_res_bma_sin_test2202.csv'

###########################TAG MASKED######################################
csv_uni_tags_test_masked_probs_all = folder_res_tags + 'masked_test/probs_test_all_fold.csv'
csv_uni_tags_test_masked_probs_folds = folder_res_tags + 'masked_test/probs_test_fold_'
csv_uni_tags_test_masked_scores = folder_res_tags + 'masked_test/score_test_tags10fold.csv'
csv_uni_tags_test_masked_res = folder_res_tags + 'masked_test/tags_res_bma_test2202.csv'

###########################TAG MASKED COUNT######################################
csv_uni_tags_test_masked_count_probs_all = folder_res_tags + 'masked_count_test/probs_test_all_fold.csv'
csv_uni_tags_test_masked_count_probs_folds= folder_res_tags + 'masked_count_test/probs_test_fold_'
csv_uni_tags_test_masked_count_scores = folder_res_tags + 'masked_count_test/score_test_tags10fold.csv'
csv_uni_tags_test_masked_count_res = folder_res_tags + 'masked_count_test/tags_res_bma_test2202.csv'
###########################TAG CENSORED######################################
csv_uni_tags_test_censored_probs_all = folder_res_tags + 'censored_test/probs_test_all_fold.csv'
csv_uni_tags_test_censored_probs_folds = folder_res_tags + 'censored_test/probs_test_fold_'
csv_uni_tags_test_censored_scores = folder_res_tags + 'censored_test/score_test_tags10fold.csv'
csv_uni_tags_test_censored_res = folder_res_tags + 'censored_test/tags_res_bma_test2202.csv'
################################REPAIR RANK CLS####################################################
csv_uni_tags_test_rank_cls_probs_all = folder_res_tags+ repair_folder_rank_cls + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_rank_cls_probs_folds = folder_res_tags + repair_folder_rank_cls+ 'test/probs_test_fold_'
csv_uni_tags_test_rank_cls_scores = folder_res_tags + repair_folder_rank_cls+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_rank_cls_res = folder_res_tags + repair_folder_rank_cls+ 'test/res_tags_bma_test.csv'
################################REPAIR RANK####################################################
csv_uni_tags_test_rank_probs_all = folder_res_tags+ repair_folder_rank + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_rank_probs_folds = folder_res_tags + repair_folder_rank+ 'test/probs_test_fold_'
csv_uni_tags_test_rank_scores = folder_res_tags + repair_folder_rank+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_rank_res = folder_res_tags + repair_folder_rank+ 'test/res_tags_bma_test.csv'
################################REPAIR UNIFORM####################################################
csv_uni_tags_test_uniform_probs_all = folder_res_tags+ repair_folder_uniform + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_uniform_probs_folds = folder_res_tags + repair_folder_uniform+ 'test/probs_test_fold_'
csv_uni_tags_test_uniform_scores = folder_res_tags + repair_folder_uniform+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_uniform_res = folder_res_tags + repair_folder_uniform+ 'test/res_tags_bma_test.csv'
################################REPAIR TRESHOLD####################################################
csv_uni_tags_test_treshold_probs_all = folder_res_tags+ repair_folder_treshold + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_treshold_probs_folds = folder_res_tags + repair_folder_treshold+ 'test/probs_test_fold_'
csv_uni_tags_test_treshold_scores = folder_res_tags + repair_folder_treshold+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_treshold_res = folder_res_tags + repair_folder_treshold+ 'test/res_tags_bma_test.csv'
################################REPAIR SAMPLE####################################################
csv_uni_tags_test_sample_probs_all = folder_res_tags+ repair_folder_sample + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_sample_probs_folds = folder_res_tags + repair_folder_sample+ 'test/probs_test_fold_'
csv_uni_tags_test_sample_scores = folder_res_tags + repair_folder_sample+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_sample_res = folder_res_tags + repair_folder_sample+ 'test/res_tags_bma_test.csv'
################################REPAIR Censored uniform####################################################
csv_uni_tags_test_censored_uniform_probs_all = folder_res_tags+ repair_folder_censored_uniform + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_censored_uniform_probs_folds = folder_res_tags + repair_folder_censored_uniform+ 'test/probs_test_fold_'
csv_uni_tags_test_censored_uniform_scores = folder_res_tags + repair_folder_censored_uniform+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_censored_uniform_res = folder_res_tags + repair_folder_censored_uniform + 'test/res_tags_bma_test.csv'
################################REPAIR masked uniform####################################################
csv_uni_tags_test_masked_uniform_probs_all = folder_res_tags+ repair_folder_masked_uniform + 'test/probs_test_all_fold.csv'
csv_uni_tags_test_masked_uniform_probs_folds = folder_res_tags + repair_folder_masked_uniform+ 'test/probs_test_fold_'
csv_uni_tags_test_masked_uniform_scores = folder_res_tags + repair_folder_masked_uniform+ 'test/tags_score_test_10fold.csv'
csv_uni_tags_test_masked_uniform_res = folder_res_tags + repair_folder_masked_uniform + 'test/res_tags_bma_test.csv'



csv_uni_tags_syn_res = folder_res_tags + 'new_sintest/tags_res_bma_sin_test2202.csv'

###########################TAG MASKED######################################
#csv_uni_tags_test_masked_probs = folder_res_tags + 'masked_new_sintest/probs_sin_test_fold_'
csv_uni_tags_syn_masked_probs_all = folder_res_tags + 'masked_sintest/probs_sin_test_all_fold.csv'
csv_uni_tags_syn_masked_probs_folds = folder_res_tags + 'masked_sintest/probs_sin_test_fold_'
csv_uni_tags_syn_masked_scores = folder_res_tags + 'masked_sintest/score_sin_test_tags10fold.csv'
csv_uni_tags_syn_masked_res = folder_res_tags + 'masked_sintest/tags_res_bma_sin_test2202.csv'


#csv_uni_tags_new_syn_masked_probs = folder_res_tags + 'masked_new_sintest/probs_sin_test_fold_'
###########################TAG MASKED COUNT######################################
csv_uni_tags_syn_masked_count_probs_all = folder_res_tags + 'masked_count_sintest/probs_sin_test_all_fold.csv'
csv_uni_tags_syn_masked_count_probs_folds = folder_res_tags + 'masked_count_sintest/probs_sin_test_fold_'
csv_uni_tags_syn_masked_count_scores = folder_res_tags + 'masked_count_sintest/score_sin_test_tags10fold.csv'
csv_uni_tags_syn_masked_count_res = folder_res_tags + 'masked_count_sintest/tags_res_bma_test2202.csv'
#csv_uni_tags_new_syn_masked_count_probs = folder_res_tags + 'masked_count_new_sintest/probs_sin_test_fold_'
###########################TAG CENSORED######################################
csv_uni_tags_syn_censored_probs_all = folder_res_tags + 'censored_sintest/probs_sin_test_all_fold.csv'
csv_uni_tags_syn_censored_probs_folds = folder_res_tags + 'censored_sintest/probs_sin_test_fold_'
csv_uni_tags_syn_censored_scores = folder_res_tags + 'censored_sintest/score_sin_test_tags10fold.csv'
csv_uni_tags_syn_censored_res = folder_res_tags + 'censored_sintest/tags_res_bma_sin_test2202.csv'
#csv_uni_tags_new_syn_censored_probs = folder_res_tags + 'censored_new_sintest/probs_sin_test_fold_'
################################REPAIR RANK CLS####################################################
csv_uni_tags_syn_rank_cls_probs_all = folder_res_tags+ repair_folder_rank_cls + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_rank_cls_probs_folds = folder_res_tags + repair_folder_rank_cls+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_rank_cls_scores = folder_res_tags + repair_folder_rank_cls+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_rank_cls_res = folder_res_tags + repair_folder_rank_cls+ 'sintest/res_bma_sintest.csv'
################################REPAIR RANK####################################################
csv_uni_tags_syn_rank_probs_all = folder_res_tags+ repair_folder_rank + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_rank_probs_folds = folder_res_tags + repair_folder_rank+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_rank_scores = folder_res_tags + repair_folder_rank+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_rank_res = folder_res_tags + repair_folder_rank+ 'sintest/res_bma_sintest.csv'
################################REPAIR UNIFORM####################################################
csv_uni_tags_syn_uniform_probs_all = folder_res_tags+ repair_folder_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_uniform_probs_folds = folder_res_tags + repair_folder_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_uniform_scores = folder_res_tags + repair_folder_uniform+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_uniform_res = folder_res_tags + repair_folder_uniform+ 'sintest/res_bma_sintest.csv'
################################REPAIR TRESHOLD####################################################
csv_uni_tags_syn_treshold_probs_all = folder_res_tags+ repair_folder_treshold + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_treshold_probs_folds = folder_res_tags + repair_folder_treshold+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_treshold_scores = folder_res_tags + repair_folder_treshold+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_treshold_res = folder_res_tags + repair_folder_treshold+ 'sintest/res_bma_sintest.csv'
################################REPAIR SAMPLE####################################################
csv_uni_tags_syn_sample_probs_all = folder_res_tags+ repair_folder_sample + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_sample_probs_folds = folder_res_tags + repair_folder_sample+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_sample_scores = folder_res_tags + repair_folder_sample+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_sample_res = folder_res_tags + repair_folder_sample+ 'sintest/res_bma_sintest.csv'
################################REPAIR censored uniform####################################################
csv_uni_tags_syn_sample_probs_all = folder_res_tags+ repair_folder_sample + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_sample_probs_folds = folder_res_tags + repair_folder_sample+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_sample_scores = folder_res_tags + repair_folder_sample+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_sample_res = folder_res_tags + repair_folder_sample+ 'sintest/res_bma_sintest.csv'
################################REPAIR censored uniform####################################################
csv_uni_tags_syn_censored_uniform_probs_all = folder_res_tags+ repair_folder_censored_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_censored_uniform_probs_folds = folder_res_tags + repair_folder_censored_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_censored_uniform_scores = folder_res_tags + repair_folder_censored_uniform+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_censored_uniform_res = folder_res_tags + repair_folder_censored_uniform+ 'sintest/res_bma_sintest.csv'
################################REPAIR masked uniform####################################################
csv_uni_tags_syn_masked_uniform_probs_all = folder_res_tags+ repair_folder_masked_uniform + 'sintest/probs_sintest_all_fold.csv'
csv_uni_tags_syn_masked_uniform_probs_folds = folder_res_tags + repair_folder_masked_uniform+ 'sintest/probs_sin_test_fold_'
csv_uni_tags_syn_masked_uniform_scores = folder_res_tags + repair_folder_masked_uniform+ 'sintest/tags_score_sintest_10fold.csv'
csv_uni_tags_syn_masked_uniform_res = folder_res_tags + repair_folder_masked_uniform+ 'sintest/res_bma_sintest.csv'


## bma senza correzione
out_bma_biased_tags = "results/tags_biased_bma/"
##output dei bma con correzioni applicate
################CORREZONI INFERENZA###################À
out_corr_bma_neg_tags = "results/tags_bma_neg/"
out_corr_bma_pos_tags = "results/tags_bma_pos/"
out_corr_dyn_base_tags = "results/tags_dyn_corr/"
out_corr_dyn_bma_tags = "results/tags_dyn_corr_only_bma/"
out_corr_bma_neu_tags = "results/tags_bma_neu/"
out_corr_terms_base_tags = "results/tags_object_corr_base/"
out_corr_terms_bma_tags = "results/tags_object_corr_only_bma/"


#################MULTIMODALE####################################

out_bma_biased_multi = "results/multi/multi_biased_bma/"
out_corr_bma_neg_multi = "results/multi/multi_bma_neg/"
out_corr_bma_pos_multi = "results/multi/multi_bma_pos/"
out_corr_dyn_base_multi = "results/multi/multi_dyn_corr/"
out_corr_dyn_bma_multi = "results/multi/multi_dyn_corr_only_bma/"
out_corr_bma_neu_multi = "results/multi/multi_bma_neu/"
out_corr_terms_base_multi = "results/multi/multi_tag_term_corr_base/"
out_corr_terms_bma_multi = "results/multi/multi_tag_term_corr_only_bma/"
out_corr_masked = "results/multi/multi_bma_masked/"
out_corr_masked_count = "results/multi/multi_bma_masked_count/"
out_corr_censored = "results/multi/multi_bma_censored/"
out_corr_repair_rank_cls = "results/multi/repair/multi_bma_rank_cls/"
out_corr_repair_rank = "results/multi/repair/multi_bma_rank/"
out_corr_repair_uniform = "results/multi/repair/multi_bma_uniform/"
out_corr_repair_treshold = "results/multi/repair/multi_bma_treshold/"
out_corr_repair_sample = "results/multi/repair/multi_bma_sample/"
out_corr_repair_censored_uniform = "results/multi/repair/multi_bma_censored_uniform/"
out_corr_repair_masked_uniform = "results/multi/repair/multi_bma_masked_uniform/"


#####################MULTI CORREGGO SOLO TESTO################################
out_bma_biased_multi_text = "results/multi/multi_biased_bma_only_text/"
out_corr_bma_neg_multi_text = "results/multi/multi_bma_neg_only_text/"
out_corr_bma_pos_multi_text = "results/multi/multi_bma_pos_only_text/"
out_corr_dyn_base_multi_text = "results/multi/multi_dyn_corr_only_text/"
out_corr_dyn_bma_multi_text = "results/multi/multi_dyn_corr_bma_only_text/"
out_corr_bma_neu_multi_text = "results/multi/multi_bma_neu_only_text/"
out_corr_terms_base_multi_text = "results/multi/multi_tag_term_corr_base_only_text/"
out_corr_terms_bma_multi_text = "results/multi/multi_tag_term_corr_only_bma_only_text/"
out_corr_masked_text = "results/multi/multi_bma_masked_only_text/"
out_corr_censored_text = "results/multi/multi_bma_censored_only_text/"
out_corr_repair_rank_cls_text = "results/multi/repair/multi_bma_rank_cls_only_text/"
out_corr_repair_rank_text = "results/multi/repair/multi_bma_rank_only_text/"
out_corr_repair_uniform_text = "results/multi/repair/multi_bma_uniform_only_text/"
out_corr_repair_treshold_text = "results/multi/repair/multi_bma_treshold_only_text/"
out_corr_repair_sample_text = "results/multi/repair/multi_bma_sample_only_text/"
out_corr_repair_censored_uniform_text = "results/multi/repair/multi_bma_censored_uniform_only_text/"
out_corr_repair_masked_uniform_text = "results/multi/repair/multi_bma_masked_uniform_only_text/"


#####################MULTI CORREGGO SOLO TAGS################################
out_bma_biased_multi_tags = "results/multi/multi_biased_bma_only_tags/"
out_corr_bma_neg_multi_tags = "results/multi/multi_bma_neg_only_tags/"
out_corr_bma_pos_multi_tags = "results/multi/multi_bma_pos_only_tags/"
out_corr_dyn_base_multi_tags = "results/multi/multi_dyn_corr_only_tags/"
out_corr_dyn_bma_multi_tags = "results/multi/multi_dyn_corr_bma_only_tags/"
out_corr_bma_neu_multi_tags = "results/multi/multi_bma_neu_only_tags/"
out_corr_terms_base_multi_tags = "results/multi/multi_tag_term_corr_base_only_tags/"
out_corr_terms_bma_multi_tags = "results/multi/multi_tag_term_corr_only_bma_only_tags/"
out_corr_masked_tags = "results/multi/multi_bma_masked_only_tags/"
out_corr_masked_count_tags = "results/multi/multi_bma_masked_count_only_tags/"
out_corr_censored_tags = "results/multi/multi_bma_censored_only_tags/"
out_corr_repair_rank_cls_tags = "results/multi/repair/multi_bma_rank_cls_only_tags/"
out_corr_repair_rank_tags = "results/multi/repair/multi_bma_rank_only_tags/"
out_corr_repair_uniform_tags = "results/multi/repair/multi_bma_uniform_only_tags/"
out_corr_repair_treshold_tags = "results/multi/repair/multi_bma_treshold_only_tags/"
out_corr_repair_sample_tags = "results/multi/repair/multi_bma_sample_only_tags/"
out_corr_repair_censored_uniform_tags = "results/multi/repair/multi_bma_censored_uniform_only_tags/"
out_corr_repair_masked_uniform_tags = "results/multi/repair/multi_bma_masked_uniform_only_tags/"

out_corr_terms_tags_bma_multi = "results/multi/multi_tag_term_corr_bma/"

#######################################################################################
#################################MASKED################################################
#######################################################################################
#csv_uni_tags_masked_test_probs = folder_res_tags + 'test/probs_test_fold_'
## csv with models F1 scores computed on 10 fold on whole training data
#csv_uni_tags_masked_test_scores = folder_res_tags + 'masked_test/score_test_tags_10fold.csv'
#csv_uni_tags_masked_test_res = folder_res_tags + 'masked_test/tags_res_bma_test.csv'
#
#
#csv_uni_tags_masked_syn_scores = folder_res_tags + 'masked_sintest/score_test_tags_10fold.csv'
## csv with models predictions (probability) on test data (10Fold approach)
#csv_uni_tags_masked_syn_probs_all = folder_res_tags + 'masked_test/probs_sin_test_fold_all.csv'
#csv_uni_tags_masked_syn_probs = folder_res_tags + 'masked_test/probs_sin_test_fold_'
## csv with averaged model performance on test data
#csv_uni_tags_masked_syn_res = folder_res_tags + 'new_sintest/tags_res_bma_sin_test2202.csv'
# csv with models predictions (probability) on test data (10Fold approach)
#csv_uni_text_test_probs = folder_res_test + 'unimodal_text_probs.csv'

# csv with averaged model performance on test data
#csv_uni_text_test_res = folder_res_test + 'unimodal_text_res.csv'

""" Unimodal text BMA _ Syntetic data """
# csv with models F1 scores computed on 10 fold on whole training data
#csv_uni_text_syn_scores = folder_res_syn + 'unimodal_text_scores.csv'

# csv with models predictions (probability) on test data (10Fold approach)
#csv_uni_text_syn_probs = folder_res_syn + 'unimodal_text_probs.csv'

# csv with averaged model performance on test data
#csv_uni_text_syn_res = folder_res_syn + 'unimodal_text_res.csv'

""" _______________________ Unimodal caps BMA  _______________________"""

""" Unimodal caps BMA _Training data """
# csv with models F1 scores coputed on training data (9Fold_8+1), used by BMA 
csv_uni_caps_train_scores = folder_res_train + 'unimodal_caps_scores.csv'

# csv with models predictions (probability) on training data (10Fold approach)
csv_uni_caps_train_probs = folder_res_train + 'unimodal_caps_probs.csv'

# csv with averaged model performance on training data
csv_uni_caps_train_res = folder_res_train + 'unimodal_caps_res.csv'

""" Unimodal caps BMA _ Test data """
# csv with models F1 scores computed on 10 fold on whole training data
csv_uni_caps_test_scores = folder_res_test + 'unimodal_caps_scores.csv'

# csv with models predictions (probability) on test data (10Fold approach)
csv_uni_caps_test_probs = folder_res_test + 'unimodal_caps_probs.csv'

# csv with averaged model performance on test data
csv_uni_caps_test_res = folder_res_test + 'unimodal_caps_res.csv'

""" Unimodal caps BMA _ Syntetic data """
# csv with models F1 scores computed on 10 fold on whole training data
csv_uni_caps_syn_scores = folder_res_syn + 'unimodal_caps_scores.csv'

# csv with models predictions (probability) on test data (10Fold approach)
csv_uni_caps_syn_probs = folder_res_syn + 'unimodal_caps_probs.csv'

# csv with averaged model performance on test data
csv_uni_caps_syn_res = folder_res_syn + 'unimodal_caps_res.csv'

""" _______________________ Unimodal tags BMA  _______________________"""

""" Unimodal tags BMA _Training data """
# csv with models F1 scores coputed on training data (9Fold_8+1), used by BMA 
#csv_uni_tags_train_scores = folder_res_train + 'unimodal_tags_scores.csv'
#
## csv with models predictions (probability) on training data (10Fold approach)
#csv_uni_tags_train_probs = folder_res_train + 'unimodal_tags_probs.csv'
#
## csv with averaged model performance on training data
#csv_uni_tags_train_res = folder_res_train + 'unimodal_tags_res.csv'
#
#""" Unimodal tags BMA _ Test data """
## csv with models F1 scores computed on 10 fold on whole training data
#csv_uni_tags_test_scores = folder_res_test + 'unimodal_tags_scores.csv'
#
## csv with models predictions (probability) on test data (10Fold approach)
#csv_uni_tags_test_probs = folder_res_test + 'unimodal_tags_probs.csv'
#
## csv with averaged model performance on test data
#csv_uni_tags_test_res = folder_res_test + 'unimodal_tags_res.csv'

""" Unimodal tags BMA _ Syntetic data """
# csv with models F1 scores computed on 10 fold on whole training data
#csv_uni_tags_syn_scores = folder_res_syn + 'unimodal_tags_scores.csv'
#
## csv with models predictions (probability) on test data (10Fold approach)
#csv_uni_tags_syn_probs = folder_res_syn + 'unimodal_tags_probs.csv'
#
## csv with averaged model performance on test data
#csv_uni_tags_syn_res = folder_res_syn + 'unimodal_tags_res.csv'