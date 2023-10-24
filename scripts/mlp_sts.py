import os
import argparse
import csv
import hashlib
import torch
import json
import itertools
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from scipy.stats import pearsonr
from modules.nnets import BasicFFNet

load_dotenv()
os.chdir(os.path.expanduser(os.getenv('PROJECT_WORKING_DIRECTORY')))
def evals(true, predicted):
    rmse = np.sqrt(np.mean((true - predicted)**2))
    pearson = pearsonr(true, predicted)[0]
    # print("Pearson:", pearson)
    # print("RMSE:", rmse)
    return pearson, rmse

def train_nnet_stsb(hidden_layers, dropout, sentence_pair_method,
                    lmodel, n_epoch=20, lr=0.001, batch_size=32):
    train = pd.read_csv('data/stsbenchmark/train.csv')
    test = pd.read_csv('data/stsbenchmark/test.csv')
    dev = pd.read_csv('data/stsbenchmark/dev.csv')

    data = {'train': train, 'test': test, 'dev': dev}
    # normalise scores to 0-1
    for df in data.values():
        df['score'] = df['score'] / 5.

    # can change model
    embeddings = {}
    transformer = SentenceTransformer(lmodel)
    for split, df in data.items():
        result = {}
        result['s1'] = transformer.encode(df['sentence1'])
        result['s2'] = transformer.encode(df['sentence2'])
        result['prod'] = result['s1'] * result['s2']
        result['l1'] = abs(result['s1'] - result['s2'])
        result['concat'] = np.concatenate([result['s1'], result['s2']], axis=1)
        result['concatl1'] = np.concatenate([result['s1'], result['s2'], result['l1']], axis = 1)
        embeddings[split] = result

    X_train = torch.tensor(embeddings['train'][sentence_pair_method]).to('cuda')
    X_dev = torch.tensor(embeddings['dev'][sentence_pair_method]).to('cuda')
    Y_train = torch.tensor(data['train']['score']).to('cuda').float()
    Y_dev = data['dev']['score']
    input_dim = X_train.shape[1]
    model = BasicFFNet(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
    model.to('cuda')
    optimiser = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in (range(n_epoch)):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_pred = model(X_batch)
            Y_batch = Y_train[i:i + batch_size]
            Y_batch = Y_batch.unsqueeze(1)
            loss = loss_fn(Y_pred, Y_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    # evaluate nn
    with torch.no_grad():
        Y_pred = model(X_dev).cpu().numpy().flatten()
        Y_fitted = model(X_train).cpu().numpy().flatten()
        val_pearson, val_rmse = evals(Y_dev, Y_pred)
        train_pearson, train_rmse = evals(Y_train.cpu().numpy(), Y_fitted)
        res = {
            'train_pearson': train_pearson, 'train_rmse': train_rmse,
            'val_pearson': val_pearson, 'val_rmse': val_rmse
        }

    f_name = hashlib.sha256(f"{lmodel}{hidden_layers}{dropout}{sentence_pair_method}".encode()).hexdigest()
    results_path = 'models/stsb-ffnet'

    model_f_name = f_name + ".pt"

    param_list = {
        'hidden_layers' : hidden_layers,
        'dropout' : dropout,
        'sentence_pair_method' : sentence_pair_method,
        'lmodel':lmodel,
        'n_epoch':n_epoch,
        'batch_size':batch_size
    }

    # save model and results
    with open(os.path.join(results_path, 'results.csv'), mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f_name] + list(res.values()))

    with open(os.path.join(results_path, f_name + ".json"), 'w') as f:
        json.dump(param_list,f)

    torch.save(model.state_dict(), os.path.join(results_path, model_f_name))

def main():
    param_grid = {
        'hidden_layers':[[64,64], [128,64]],
        'dropout':[[0,0], [0.25,0.25]],
        'sentence_pair_method': ['concatl1'],
        'lmodel':["sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1"],
        'n_epoch' : [50],
        'lr' : [0.0005],
        'batch_size': [256,128],
    }
    param_names = list(param_grid.keys())
    combinations = itertools.product(*list(param_grid.values()))
    for p in combinations:
        train_nnet_stsb(*p)

    return

if __name__ == '__main__':
    main()


