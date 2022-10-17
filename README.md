# CLoSE: Contrastive Learning of Subframe Embeddings for Political Bias Classification of News Media

This is the official repository for the COLING 2022 long paper [CLoSE: Contrastive Learning of Subframe Embeddings for Political Bias Classification of News Media](https://aclanthology.org/2022.coling-1.245/). 
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
- Col 1: anchor sentence
- Col 2: positive sentence
- Col 3: negative sentence
- Col 4: binary label (0 for the right-biased media and 1 for the left-biased media)


## Citation
Please cite us if you find this useful:
```
@inproceedings{kim-johnson-2022-close,
    title = "{CL}o{SE}: Contrastive Learning of Subframe Embeddings for Political Bias Classification of News Media",
    author = "Kim, Michelle YoungJin  and
      Johnson, Kristen Marie",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.245",
    pages = "2780--2793",
    abstract = "Framing is a political strategy in which journalists and politicians emphasize certain aspects of a societal issue in order to influence and sway public opinion. Frameworks for detecting framing in news articles or social media posts are critical in understanding the spread of biased information in our society. In this paper, we propose CLoSE, a multi-task BERT-based model which uses contrastive learning to embed indicators of frames from news articles in order to predict political bias. We evaluate the performance of our proposed model on subframes and political bias classification tasks. We also demonstrate the model{'}s classification accuracy on zero-shot and few-shot learning tasks, providing a promising avenue for framing detection in unlabeled data.",
}
```