import torch
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer


class TripletDataset(torch.utils.data.Dataset):
    """
    This dataset reads the Framing Triplet Dataset.
    The triplet dataset has four columns:
     - Col 1: anchor sentence,   type = string
     - Col 2: positive sentence, type = string
     - Col 3: negative sentence, type = string
     - Col 4: binary label,      type = int 
              0 for the right-biased media and 1 for the left-biased media

    Arguments for the Dataset initialization are:
     - df: the triplet dataset,  type = Pandas Dataframe
     - model_name: encoder name, type = string
                   'roberta-base', 'bert-base-uncased', or 'bert-base-cased'
    """

    def __init__(self, df, model_name):
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)

        labels = {'0': 0, '1': 1}
        self.labels = [labels[label] for label in df['label']]
        self.s1 = [tokenizer(text, 
                             padding='max_length', max_length = 128, truncation=True,
                             return_tensors="pt") for text in df['s1']]
        self.s2 = [tokenizer(text, 
                             padding='max_length', max_length = 128, truncation=True,
                             return_tensors="pt") for text in df['s2']]
        self.s3 = [tokenizer(text, 
                             padding='max_length', max_length = 128, truncation=True,
                             return_tensors="pt") for text in df['s3']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return [self.s1[idx], self.s2[idx], self.s3[idx]]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y