import csv
import pandas as pd
import numpy as np
import argparse
from .src.CLoSE import CLoSE
from .src.train import train
from .src.evaluate import evaluate

def read_csv(data_dir):
    """
    Read a csv file saved in the 'data_dir' directory.
     - data_dir: file directory, type = string
                e.g., '../data/sent_triplets_wLabels_train.csv'
    """
    dataset = []
    with open(data_dir, mode='r') as out_file:
        reader = csv.reader(out_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            dataset.append(row)
    return dataset

def list_to_df(dataset):
    """
    Convert the 'dataset' list to a pandas dataframe.
     - dataset: a list of data, type = list[string, string, string, int]
                e.g., ['Roe v. Wade', 'Overturn Roe', 'March for Life', 1]
    """
    return pd.DataFrame(dataset, columns=['s1','s2','s3','label'])

def main():

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_dir', type=str, default='../data', required=False,
                            help='data file directory')
        parser.add_argument('--data_type', type=str, default='abortion', required=False,
                            help='data topics: abortion, gun, or immigration')

        parser.add_argument('--alpha', type=float, default=0.5, required=False)
        parser.add_argument('--lr', type=float, default=2e-5, required=False)
        parser.add_argument('--epoch', type=int, default=5, required=False)
        parser.add_argument('--gpu_id', type=int, default=0, required=False)

        parser.add_argument('--model_name', type=str, default='bert-base-cased', required=False,
                            help='encoder type: bert-base-cased, bert-base-uncased, or roberta-base')
        parser.add_argument('--model_path', type=str, default=None, required=False,
                            help='file directory of the saved model')

        parser.add_argument('--do_train', action='store_true')
        parser.add_argument('--do_eval', action='store_true')

        args = parser.parse_args()

        np.random.seed(112)

        file_dir = args.data_dir + '/' + args.data_type + '/sent_triplets_wLabels'

        model = CLoSE(args.model_name)

        if args.do_train:
            train_file = file_dir + '_train.csv'
            valid_file = file_dir + '_valid.csv'

            train_data = list_to_df(read_csv(train_file))
            valid_data = list_to_df(read_csv(valid_file))

            save_path = '{}_alpha_{}_lr_{}_epoch_{}.pt'.format(args.model_name, args.alpha, args.lr, args.epoch)

            train(model, args.model_name, train_data, valid_data, 
                  learning_rate=args.lr, 
                  epochs=args.epoch, 
                  gpu_id=args.gpu_id,
                  alpha=args.alpha,
                  model_path=save_path)

        if args.do_eval:
            test_file = file_dir + '_test.csv'
            test_data = list_to_df(read_csv(test_file))

            _ = evaluate(model, args.model_name, test_data, 
                        gpu_id=args.gpu_id, 
                        model_path=args.model_path)

    except Exception as e:
        print(e)

if __name__=='__main__':
    main()