# Bias-Mitigation-in-Misogynous-Meme-Recognition


## Features

-   extract features from [SGG benchmark](https://github.com/microsoft/scene_graph_benchmark)
-   get caption from meme with [OSCAR](https://github.com/microsoft/Oscar)
-   BMA unimodal
-   BMA multimodal
-   Bias correction

## Installation of SGG becnhmark

```sh
# create a new environment from yaml
cd scene_graph_benchmark
conda env create -f sgg_benchmark.yml
conda activate sg_benchmark

#se quando esegui (step successivi) non funziona prova a rieseguire questo comando
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

```

## Run sgg

Run SGG for extract features  
file tools/mini\_tsv/tsv\_demo.py

-   prima di runnare, bisogna cambiare la destinazione degli output tra train/test/sintest e la cartella di input (cartella dei meme) (righe 17-22)  
    Assicurarsi che la scrittura in tsv prenda i file giusti (righe 66-74)

```sh
python tools/mini_tsv/tsv_demo.py

```

A questo punto nella cartella scelta dove ci sono gli output (ad esempio train.lineidx train.tsv ecc…) creare un file yaml con questo contenuto:  
img : “test.tsv”  
label : “test.label.tsv”  
hw : “test.hw.tsv”  
linelist : “test.linelist.tsv”

```sh

scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml cambiare variabile TEST di questo file con la cartella che punta allo yaml dei risultati

controlla che ci siano questi due file nelle cartelle, se non ci sono riscaricali, se li ho eliminati è per problema di spazio
# visualize VinVL object detection
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
# the associated labelmap at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
!IMPORTANTE! cambiare i path in extract_features.py
prima di runnare creare le cartelle di potput e nel codice cambiare le destinazioni ad esse.
python tools/demo/extract_features.py

```

Con il file extract features nella cartella /output ora sono presenti i file lineidx e tsv che prendono il nome dato nel file (esempio prediction\_train.tsv predictions\_train.lineidx)

```sh
tool/demo/prepare_vinvl_input.ipynb

```

Questo notebook guida alla creazione di features.tsv e label.tsv che servono ad oscar per generare caption e una yaml che punta a questi file

se non servono le caption passare a

```sh
tools/demo/merge_caps_obj.ipynb

```

questo notebook semplicemente genera da label tsv un altro tsv con gli header che dovrai spostare in  
/data/feature_extraction/sgg/  
Ora si può creare il dataset con una lista di oggetti in

```sh
data/datasets_creation.ipynb

```

il risultato può essere utilizzato per runnare

```sh
data/gvision_ocr.ipynb

```

questo notebook usa google OCR per estrarre testo dalle immagini

## Install OSCAR

```sh
# create a new environment from yaml
cd Oscar
conda env create -f oscar.yml
conda activate oscar

# anche qui l'ambiente dovrebbe già essere settato se non funziona prova a eseguire la build
python setup.py build develop

# non so se sono già inclusi quindo attivi l'ambiente, seno devi runnare anche questo comando
# install requirements
pip install -r requirements.txt
unset INSTALL_DIR

```

## Use OSCAR

Copia cartella di output contenente features.tsv e label.tsv, yaml, da SGG a OSCAR (io l’ho chiamata output\_train o output\_test o output_sintest).  
Scaricare modelli pretrainati  
Finetuned model checkpoint (w/ CIDEr optimization): coco\_captioning\_base_scst.zip. scaricare checkpoint scst [Nella sezione image captioning on COCO](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md)

```sh
python run_captioning.py --do_test --test_yaml oscar/output_train/vinvl_test_yaml.yaml --per_gpu_eval_batch_size 1 --num_beams 5 --max_gen_length 70 --eval_model_dir oscar/coco_captioning_base_scst_test

```

Questo crea nella cartella indicata come eval\_model\_dir un file tsv chiamato pred.coco.captions.parametriscelti.tsv  
bisogna spostare questo file nella cartella di sgg becnhmark nella cartella tools/demo.  
Nella stessa cartella c’è un notebook che serve a unire il tsv delle vcaptions con quello degli oggetti (label.tsv prodotto da prepare\_vinvl\_input.ipynb)

```sh
merge_caps_obj.ipynb

```

Il risultato è poi salvato in data/feature_extraction/

## BMA

### Workflow

#### Datasets

-   **Captions/analisi_captions.ipynb** confronta m2 azure e oscar
-   **data/dataset_creation.ipynb** creazione dei tre dataset testo caps e tags
-   **data/datasets/** cartella con tutti i dataset divisi per componente e i dataset di partenza test, sin e train

-   **BMA UNIMODALI** contiene k fold per train test e sintest per ogni componente inoltre è presente anche la correzione UBMA in ognuno dei tre file
	- unimodal_tags_bma.ipynb contiene modelli tags e per train test e sintest
	- unimodal_text_bma.ipynb contiene modelli text train test e sintest
	- unimodal_bma_tags_REPAIR.ipynb stessi modelli stessa procedura ma con i 5 dataset di 		repair
	- unimodal_bma_text_REPAIR.ipynb stessi modelli stessa procedura text con i dataset di repair
	
-  **UBMA_MULTI.py** contiene il codice per runnare i risultati del BMA_multimodale diventato text+tags (senza caption) prende 3 parametri:
 *-d dataset "test" o "syn"*
  -*c correzione*: [neu, neg, pos, dyn_base, dyn_bma, terms_bma, masked, censored] e i 5 tipi di repair [treshold, rank, rank_cls, uniform, sample]
 se *c = "none"* sto eseguendo il bma multimodale senza correzioni
 -m modalità di correzione "text" correggo solo le probs del testo, "tags" correggo solo le probs dei tags "multi" correggo tutte due le probs, al fine di valutare il bias tutte le correzioni e non vanno eseguite sia sul test che sul syn
```sh
conda activate bma_text
cd 2_strategy_test/

python UBMA_MULTI.py -d "test" -c "masked" -m "tags"
python UBMA_MULTI.py -d "syn" -c "masked" -m "tags"
python UBMA_MULTI.py -d "test" -c "masked" -m "text"
python UBMA_MULTI.py -d "syn" -c "masked" -m "text"
python UBMA_MULTI.py -d "test" -c "masked" -m "multi"
python UBMA_MULTI.py -d "syn" -c "masked" -m "multi"
python UBMA_MULTI.py -d "test" -c "censored" -m "tags"
python UBMA_MULTI.py -d "syn" -c "censored" -m "tags"
python UBMA_MULTI.py -d "syn" -c "censored" -m "text"
python UBMA_MULTI.py -d "test" -c "censored" -m "text"
python UBMA_MULTI.py -d "test" -c "censored" -m "multi"
python UBMA_MULTI.py -d "syn" -c "censored" -m "multi"
```
per i bma unimodali e le loro correzioni 
Ubma_tags.py Ubma_text.py
hanno solo correzioni in inferenza  [neu, pos, neg, dyn_base, dyn_bma, terms_bma]
poichè quelle in training sono già effettuate trainando i bma_unimodali nei predenti notebook corrispondenti questi file quindi hanno solo 2 parametri -d dataset e -c correzione
python Ubma_tags.py -d "test" -c "neu"
### bias
bias/identity_elements/
qui ci sono i notebook per calcolare identity terms e identity tags
#### cartella data/result2strategy
in questa cartella ci sono i risultati dei bma unimodali divisi nelle cartelle 
text/ e tags/ un csv per fold nelle cartelle test e new sintest poi per ogni correzione in training sono riportati i risultati nella cartella ad esempio masked/test masked/new_sintest
#### cartella 2strategy_test/results
qui ci sono tutti i risultati delle correzioni di UBMA_MULTI e Ubma_tags e Ubma_text
ogni cartella è chiamata con la componente + correzione es: text_bma_neg
per i bma unimodali non ci sono le cartelle della correzione in training perchè i risultati sono già in result2strategy, per il bma multi modale tutte le correzioni comprese quelle in training sono in results/multi/ perchè il bma multimodale masked e censored l'ho eseguito da UBMA_MULTI.py che salva qui i risultati
#### calcolo bias
bias_metric.py bias per bma unimodale testo
bias_tags_metric.py bias per bma unimodale tags
bias_multi_metric.py bias per bma multi
in futuro non serve tenere 3 file basta differenziare i path e inserire un parametro in più
bias_metric e bias_tags_metric ha 4 argomenti:
	- *modello -m* [bma, svm, knn, nby, dtr, mlp]
	- *sottogruppo -g* [both, pos, neg] si riferisce a quale sottogruppo prendere tra identity terms e tags se solo quelli positivi o solo quelli negativi tutti e due
	- *correzione -c*  [neu, neg, pos, dyn_base, dyn_bma, terms_bma, masked, censored, none] e i 5 tipi di repair [treshold, rank, rank_cls, uniform, sample]
	- *metrica -b* se metrica unimodale:  -b "text" o -b "tags" tiene conto solo degli identity elements della componente, "multi" calcola la metrica multimodale
```sh
python bias_metric.py -m "bma" -g "both" -c "dyn_base" -b "text"
python bias_metric.py -m "bma" -g "both" -c "dyn_base" -b "multi"
```
```sh
python bias_tags_metric.py -m "bma" -g "both" -c "masked" -b "tags"
python bias_tags_metric.py -m "bma" -g "both" -c "masked" -b "multi"
```
per bias_multi_metric.py  c'è un parametro in più 
*-u = modalità di correzione* ovvero text, tags o multi per considerare la correzione testuale nel bma multimodale, oppure la correzione visuale oppure la correzione multimodale
```sh
python bias_multi_metric.py -m "bma" -g "both" -c "none" -b "multi" -u "multi"
python bias_multi_metric.py -m "bma" -g "both" -c "masked" -b "multi" -u "multi"
python bias_multi_metric.py -m "bma" -g "both" -c "masked" -b "multi" -u "text"
python bias_multi_metric.py -m "bma" -g "both" -c "masked" -b "multi" -u "tags"
python bias_multi_metric.py -m "bma" -g "both" -c "censored" -b "multi" -u "text"
python bias_multi_metric.py -m "bma" -g "both" -c "censored" -b "multi" -u "tags"
python bias_multi_metric.py -m "bma" -g "both" -c "censored" -b "multi" -u "multi"
```


