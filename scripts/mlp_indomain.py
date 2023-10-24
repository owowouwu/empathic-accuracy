import os
import pandas as pd
import numpy as np
import torch
import settings
import itertools
import hashlib
import json
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.model_selection import KFold
from modules.nnets import BasicFFNet


os.chdir(settings.PROJECT_WORKING_DIRECTORY)
def encode_sentences(transformer,s1, s2, how = 'concat'):
    u = transformer.encode(s1)
    v = transformer.encode(s2)
    if how == 'concat':
        return np.concatenate([u,v], axis = 1)
    elif how == 'l1':
        return np.abs(u - v)
    elif how == 'prod':
        return u*v
    elif how == 'concatl1':
        return np.concatenate([u,v, np.abs(u-v)], axis = 1)

def make_sentence_pairs(u,v, how = 'concat'):
    if how == 'concat':
        return np.concatenate([u,v], axis = 1)
    elif how == 'l1':
        return np.abs(u - v)
    elif how == 'prod':
        return u*v
    elif how == 'concatl1':
        return np.concatenate([u,v, np.abs(u-v)], axis = 1)

def predict_scores(designer_embeddings, user_embeddings):
    return [1 - distance.cosine(designer_embeddings[i], user_embeddings[i])
                   for i in range(user_embeddings.shape[0])]

def evals(true, predicted):
    rmse = np.sqrt(np.mean((true - predicted)**2))
    pearson = pearsonr(true, predicted)[0]
    # print("Pearson:", pearson)
    # print("RMSE:", rmse)
    return pearson, rmse

def get_cv_results(hidden_layers, dropout, batch_size, n_epoch, learning_rate, concat_method):
    X = make_sentence_pairs(u_encodings, d_encodings, how=concat_method)
    Y = interview_data['avg_EA_scaled'].values
    input_dim = X.shape[1]

    loss_fn = nn.MSELoss()
    fold_train_results = []
    results = []
    for k, (train, val) in enumerate(splits.split(interview_data)):
        # reinitialise new isntance of network
        model = BasicFFNet(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
        model.to('cuda')
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
        # prepare data from folds
        X_train = torch.tensor(X[train]).to('cuda')
        X_val = torch.tensor(X[val]).to('cuda')
        Y_train = torch.tensor(Y[train]).to('cuda').float()
        Y_val = torch.tensor(Y[val]).to('cuda').float()
        # train nn
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

            with torch.no_grad():
                Y_pred = model(X_val).cpu().numpy().flatten()
                Y_fitted = model(X_train).cpu().numpy().flatten()
                val_pearson, val_rmse = evals(Y_val.cpu().numpy(), Y_pred)
                train_pearson, train_rmse = evals(Y_train.cpu().numpy(), Y_fitted)
                res = {
                    'epoch': epoch, 'fold': k,
                    'train_pearson': train_pearson, 'train_rmse': train_rmse,
                    'val_pearson': val_pearson, 'val_rmse': val_rmse
                }
                fold_train_results.append(res)

        # evaluate nn
        with torch.no_grad():
            Y_pred = model(X_val).cpu().numpy().flatten()
            Y_fitted = model(X_train).cpu().numpy().flatten()
            val_pearson, val_rmse = evals(Y_val.cpu().numpy(), Y_pred)
            train_pearson, train_rmse = evals(Y_train.cpu().numpy(), Y_fitted)
            res = {
                'train_pearson': train_pearson, 'train_rmse': train_rmse,
                'val_pearson': val_pearson, 'val_rmse': val_rmse
            }
            results.append(res)

    return res, fold_train_results

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)

def main():
    all_results = []
    param_grid = {
        'hidden_layers':[256, 512],
        'dropout':[0, 0.25, 0.5],
        'batch_size': [1],
        'n_epoch' : [100],
        'learning_rate' : [0.0005],
        'concat_method' : ['concatl1']
    }
    param_names = list(param_grid.keys())
    combinations = itertools.product(*list(param_grid.values()))
    for p in combinations:
        params = {k:v for (k,v) in zip(param_names, p)}
        f_name = hashlib.sha256(str(params).encode()).hexdigest()
        results, fold_train_results = get_cv_results(*p)
        all_results.append({
            'training_log': f_name,
            'param_grid': params,
            'results': results
        })
        pd.DataFrame(fold_train_results).to_csv(os.path.join('results/cvnnet/fold_logs', f"{f_name}.csv"))


    param_grid2 = {
        'hidden_layers' : [[256,256],[512,512],[256,128]],
        'dropout':[[0,0], [0.25,0.25],[0.5,0.5]],
        'batch_size': [1],
        'n_epoch': [100],
        'learning_rate': [0.0005],
        'concat_method': ['concatl1']
    }
    combinations = itertools.product(*list(param_grid2.values()))
    for p in combinations:
        params = {k: v for (k, v) in zip(param_names, combinations)}
        f_name = hashlib.sha256(str(params).encode()).hexdigest()
        results, fold_train_results = get_cv_results(*p)
        all_results.append({
            'training_log': f_name,
            'param_grid': params,
            'results': results
        })
        pd.DataFrame(fold_train_results).to_csv(os.path.join('results/cvnnet/fold_logs', f"{f_name}.csv"))

    param_grid3 = {
        'hidden_layers' : [[128,64,32], [256,128,64]],
        'dropout':[[0,0,0], [0.25,0.25,0.25],[0.5,0.5,0.5]],
        'batch_size': [1],
        'n_epoch': [100],
        'learning_rate': [0.0005],
        'concat_method': ['concatl1']
    }
    combinations = itertools.product(*list(param_grid3.values()))
    for p in combinations:
        params = {k: v for (k, v) in zip(param_names, combinations)}
        f_name = hashlib.sha256(str(params).encode()).hexdigest()
        results, fold_train_results = get_cv_results(*p)
        all_results.append({
            'training_log': f_name,
            'param_grid': params,
            'results': results
        })
        pd.DataFrame(fold_train_results).to_csv(os.path.join('results/cvnnet/fold_logs', f"{f_name}.csv"))

    with open(os.path.join('results/cvnnet', 'all_results2.json'), 'w') as f:
        json.dump(all_results, f, cls=NumpyFloatValuesEncoder)

if __name__ == '__main__':
    interview_data = pd.read_excel('data/full_data.xlsx', sheet_name=1)
    interview_data['avg_EA_scaled'] = interview_data['avg_EA'] / 2.

    u = interview_data['user_text']
    d = interview_data['designer_text']
    u = u.str.lower().str.replace(':', '')
    d = d.str.lower().str.replace(':', '')
    u = u.str.replace('i was', '')
    d = d.str.replace('i was', '').str.replace('s/he was', '').str.replace('she was', '').str.replace('he was', '')

    encoder = SentenceTransformer('sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')
    encoder.to('cuda')

    interview_data['user_text'] = u
    interview_data['designer_text'] = d

    u_encodings = encoder.encode(u)
    d_encodings = encoder.encode(d)

    splits = KFold(n_splits=10, shuffle=True, random_state=123)

    main()
