import os
import argparse
import csv
import hashlib
import pandas as pd
import numpy as np
import torch
import itertools
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from scipy.stats import pearsonr
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import matplotlib.pyplot as plt
load_dotenv()
os.chdir(os.path.expanduser(os.getenv('PROJECT_WORKING_DIRECTORY')))

def train_sparsegp(kernel_name, sparse_points, sentence_pair_method, lmodel):

    train = pd.read_csv('data/stsbenchmark/train.csv')
    test = pd.read_csv('data/stsbenchmark/test.csv')
    dev = pd.read_csv('data/stsbenchmark/dev.csv')

    data = {'train':train, 'test':test, 'dev':dev}
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
    Y_train = torch.tensor(data['train']['score']).to('cuda')
    Y_dev = data['dev']['score']

    # sparse points for sparse regression
    Xu = (X_train.clone())[np.random.choice(X_train.shape[0], sparse_points)]

    if kernel_name == 'rbf':
        kernel = gp.kernels.RBF(input_dim=X_train.shape[1])
    elif kernel_name == 'ratquad':
        kernel = gp.kernels.RationalQuadratic(input_dim=X_train.shape[1])

    likelihood = gp.likelihoods.Gaussian()
    vsgp = gp.models.VariationalSparseGP(X_train, Y_train, Xu=Xu, kernel = kernel,
                                        likelihood = likelihood, whiten=True)
    vsgp.to('cuda')
    # take only the mean
    losses = gp.util.train(vsgp, num_steps = 2000)
    Y_pred = vsgp(X_dev)[0].cpu().detach().numpy()
    f_name = hashlib.sha256(f"{lmodel}{kernel_name}{sparse_points}{sentence_pair_method}".encode()).hexdigest()
    results_path = 'models/stsb-sparsegp2'
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence")
    plot_f_name = f_name + ".png"
    plt.savefig(os.path.join(results_path, plot_f_name))
    plt.clf()
    pearson = pearsonr(Y_dev, Y_pred)[0]
    model_f_name = f_name + ".pt"
    
    # save model and results
    with open(os.path.join(results_path, 'results.csv'), mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([str(sparse_points),kernel_name,lmodel,sentence_pair_method,str(pearson),model_f_name])

    torch.save(vsgp.state_dict(), os.path.join(results_path, model_f_name))

def main():
    param_grid = {
        'kernel_name':['rbf','ratquad'],
        'sparse_points' : [50],
        'sentence_pair_method': ['concatl1', 'concat', 'l1'],
        'lmodel': ["sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1"],
    }
    param_names = list(param_grid.keys())
    combinations = itertools.product(*list(param_grid.values()))
    for p in combinations:
        print(p)
        train_sparsegp(*p)

    return



if __name__ == '__main__':
    main()


