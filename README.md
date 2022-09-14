# CLoSE: Contrastive Learning of Subframe Embeddings for Political Bias Classification of News Media

This is the official repository for the COLING 2022 long paper *CLoSE: Contrastive Learning of Subframe Embeddings for Political Bias Classification of News Media*. 
We provide codes and datasets for training and evaluating  CLoSE, contrastive learning of subframe embeddings model that jointly generates embeddings and predicts political bias of framed texts.

## Installation
We recommend **Python 3.7** or higher. 

Install the required libraries with `pip`:
```
pip install -r requirements.txt
```

## Getting Started
First, go to the `src` directory.
To train CLoSE:
```
python run.py --do_train
```

To evaluate CLoSE:
```
python run.py --do_eval --model_path $SAVED_MODEL_DIR
```

You can train and evaluate at the same time by running:
```
python run.py --do_train --do_eval
```

The full list of arguments of `run.py` with examples is here:
```
python run.py --data_dir ../data \
              --data_type abortion \
              --alpha 0.5 \
              --lr 2e-5 \
              --epoch 5 \
              --gpu_id 0 \
              --model_name bert-base-cased \
              --model_path ckpt.pt \
              --do_train
              --do_eval
```


## Datasets
The Framing Triplet Dataset can be found under the `data` directory.
There are three folders, one for each topic (*abortion, gun,* and *immigration*).
Data are saved in `csv`format and have four columns:
- Col 1: anchor sentence,   type = string
- Col 2: positive sentence, type = string
- Col 3: negative sentence, type = string
- Col 4: binary label,      type = int 
         0 for the right-biased media and 1 for the left-biased media


## Citation:
Please cite us if you find this useful: