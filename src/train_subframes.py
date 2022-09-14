import os
import torch
import csv
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from enum import Enum
import argparse
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data', required=False)
parser.add_argument('--data_type', type=str, default='abortion', required=False)
parser.add_argument('--lr', type=float, default=2e-5, required=False)
parser.add_argument('--epoch', type=int, default=1, required=False)
parser.add_argument('--gpu_id', type=int, default=0, required=False)
parser.add_argument('--model_name', type=str, default='bert-base-cased', required=False)
parser.add_argument('--model_path', type=str, default='checkpoint/subframes_by_group', required=False)
parser.add_argument('--num_group', type=int, default=22, required=False)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_eval', action='store_true')

args = parser.parse_args()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, group_labels, model_name):
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)


        #self.labels = [labels[label] for label in df['label']]
        self.labels = group_labels
        self.s1 = [tokenizer(text, 
                             padding='max_length', max_length = 128, truncation=True,
                             return_tensors="pt") for text in df['s1']]

    def classes(self):
        return self.s1

    def __len__(self):
        return len(self.s1)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.s1[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, model_name, dropout=0.5, num_group=20,  gpu_id = args.gpu_id):

        super(BertClassifier, self).__init__()

        if 'roberta' in model_name:
            self.bert = RobertaModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_group)
        self.relu = nn.ReLU()


    def forward(self, input_id, mask, device, batch_size):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        #final_layer = self.linear(pooled_output)

        return final_layer




def train(model, 
          model_name,
          train_data, 
          val_data, 
          train_group_data, 
          val_group_data, 
          learning_rate, 
          epochs, 
          gpu_id, 
          batch_size: int=4, 
          model_path: str=''):

    train, val = Dataset(train_data, train_group_data, model_name), Dataset(val_data, val_group_data, model_name)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, drop_last=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu_id if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    best_loss = 100

    if use_cuda:

            model = model.to(device)
            criterion = criterion.to(device)

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                final_layer = model(input_id, mask, device, batch_size)

                batch_loss = criterion(final_layer, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (final_layer.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)  
                    input_id = val_input['input_ids'].squeeze(1).to(device)  

                    final_layer = model(input_id, mask, device, batch_size)

                    batch_loss = criterion(final_layer, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (final_layer.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            # Save model
            total_loss_val = total_loss_val / len(val_data)
            #if total_loss_val < best_loss: 
            #    best_loss = total_loss_val
            #    print("save all model to {}".format(model_path))
            #    output = open(model_path, mode="wb")
            #    torch.save(model.state_dict(), output)

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Group Accuracy: {total_acc_train / len(train_data): .3f}')
                  
            print(f' | Val Loss: {total_loss_val: .3f} | Val Stance Accuracy: {total_acc_val / len(val_data): .3f} ')
                  

def evaluate(model, 
             model_name,
             test_data, 
             test_group_data, 
             gpu_id, 
             batch_size: int=16, 
             model_path: str=''):

    test = Dataset(test_data, test_group_data, model_name)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, drop_last=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu_id if use_cuda else "cpu")
    #model.load_state_dict(torch.load(model_path))

    if use_cuda:

        model = model.to(device)

    total_acc_test = 0

    embeddings = []
    pred_labels = []
    true_labels = []

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            true_labels += test_label.tolist()
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            final_layer = model(input_id, mask, device, batch_size)

            pred_labels += final_layer.argmax(dim=1).detach().cpu().tolist()
            acc = (final_layer.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    

    f1_group_micro =f1_score(true_labels, pred_labels, average='micro')
    f1_group_macro =f1_score(true_labels, pred_labels, average='macro')
    f1_group_weighted =f1_score(true_labels, pred_labels, average='weighted')

    print(f'Test Group Accuracy: {total_acc_test / len(test_data): .3f} | Test Group f1: (1) micro: {f1_group_micro: .3f}, (2) macro: {f1_group_macro: .3f}, (3) weighted: {f1_group_weighted: .3f}')

def read_csv(data_dir):
    dataset = []
    with open(data_dir, mode='r') as out_file:
        reader = csv.reader(out_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            dataset.append(row)
    return dataset

def list_to_df(dataset):
    return pd.DataFrame(dataset, columns=['s1','s2','s3','label'])

def read_group_label(data_dir):
    with open(data_dir, 'r') as in_file:
        group_data = in_file.readlines()
        group_data = [int(line) for line in group_data]
    return group_data


def main():

    print('=============================================================================') 
    print(f'Data: {args.data_type}, Model: {args.model_name}, Learning rate: {args.lr}, Epochs: {args.epoch}')
    print('=============================================================================') 

    np.random.seed(112)

    model = BertClassifier(args.model_name, num_group=args.num_group)
              
    file_dir = os.path.join(args.data_dir, args.data_type+'_data')
    save_prefix = os.path.join(args.model_path, args.data_type)
    #save_path = '{}_{}_lr_{}_epoch_{}_group.pt'.format(save_prefix, args.model_name, args.lr, args.epoch)
    save_path = '{}/immigration_bert-base-cased_lr_2e-05_epoch_1_group.pt'.format(args.model_path)

    #print('Load pre-trained-model: ', save_path)
    #model.load_state_dict(torch.load(save_path))

    if args.do_train:
        train_data_file = 'sent_triplets_wLabels_onSubframes_train.csv'
        train_data = read_csv(os.path.join(file_dir,train_data_file))
        train_data = train_data[:500]
        train_data = list_to_df(train_data)

        valid_data = train_data
        #valid_data_file = 'sent_triplets_wLabels_onSubframes_valid.csv'
        #valid_data = read_csv(os.path.join(file_dir,valid_data_file))
        #valid_data = list_to_df(valid_data)

        train_group_file = train_data_file[:-4] + '_group.csv'
        print(train_group_file)
        train_group_data = read_group_label(os.path.join(file_dir, train_group_file))
        print(train_group_data[:4])
        #val_group_file = valid_data_file[:-4] + '_group.csv'
        #val_group_data = read_group_label(os.path.join(file_dir, val_group_file))
        val_group_data = train_group_data

        train(model, args.model_name, train_data, valid_data, train_group_data, val_group_data, 
              learning_rate=args.lr, 
              epochs=args.epoch, 
              gpu_id=args.gpu_id,
              model_path=save_path)

    if args.do_eval:
        test_data_file = 'sent_triplets_wLabels_onSubframes_test.csv'
        test_data = read_csv(os.path.join(file_dir,test_data_file))
        test_data = list_to_df(test_data)

        test_group_file = test_data_file[:-4] + '_group.csv'
        test_group_data = read_group_label(os.path.join(file_dir, test_group_file))

        _ = evaluate(model, args.model_name, test_data, test_group_data, gpu_id=args.gpu_id, model_path=save_path)

if __name__=='__main__':
    main()
