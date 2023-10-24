import os
import pandas as pd
import numpy as np
import torch
import settings
import itertools
import hashlib
import json
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
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

def get_cv_results(concat_method):
    X = make_sentence_pairs(u_encodings, d_encodings, how=concat_method)
    Y = interview_data['avg_EA_scaled'].values
    results = []
    losses = []
    for k, (train, val) in enumerate(splits.split(interview_data)):
        with pyro.get_param_store().scope():
            # reinitialise new isntance of network
            # prepare data from folds
            X_train = torch.tensor(X[train]).to('cuda')
            X_val = torch.tensor(X[val]).to('cuda')
            y_train = torch.tensor(Y[train]).to('cuda').float()
            y_val = Y[val]

            kernel = gp.kernels.Polynomial(input_dim = X_train.shape[1])
            gpr = gp.models.GPRegression(X_train, y_train, kernel = kernel)
            gpr.to('cuda')
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
            loss = gp.util.train(gpr, num_steps = 2000, optimizer = optimizer)
            losses.append(loss)

            y_pred = gpr(X_val)
            y_pred = y_pred[0].cpu().detach().numpy()
            pearson, rmse = evals(y_val, y_pred)
            results.append({'pearson':pearson, 'rmse':rmse})


    return results, losses


def main():
    results, losses = get_cv_results('concatl1')
    pd.DataFrame(results).to_csv('results/cv-indomain-gp-poly.csv', index=False)
    np.save('results/cv-indomain-gp-poly-losses.npy', losses)



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
