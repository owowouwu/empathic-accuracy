#!/usr/bin/zsh
"""
Run pretrained sentence transformers from https://www.sbert.net/docs/pretrained_models.html and
compute cosine similarities between embeddings of user and designer text to predict scores
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import spatial, stats
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
os.chdir(os.path.expanduser(os.getenv('PROJECT_WORKING_DIRECTORY')))

def main():
    model_name = sys.argv[1]
    if '/' in model_name:
        model_name = model_name.split('/')[1]
    data = pd.read_excel('./data/full_data.xlsx', sheet_name='full_data')
    data = data.dropna(axis=1, how='all')
    user_texts = data['user_text']
    designer_texts = data['designer_text']
    # scores are between 0 and 2, normalise to 1
    scores = data['avg_EA'] / 2.

    model = SentenceTransformer(model_name)

    designer_embeddings = model.encode(designer_texts)
    user_embeddings = model.encode(user_texts)
    if not os.path.exists(os.path.join("./results/embeddings", model_name)):
        os.mkdir(os.path.join("./results/embeddings", model_name))
        np.save(f'./results/embeddings/{model_name}/designer.npy' ,designer_embeddings)
        np.save(f'./results/embeddings/{model_name}/user.npy' ,user_embeddings)


    predicted_scores = [1 - spatial.distance.cosine(designer_embeddings[i], user_embeddings[i])
                   for i in range(len(user_texts))]

    pearson = stats.pearsonr(scores, predicted_scores)[0]

    # write pearson
    pearson_scores = pd.read_csv('results/pearsons.csv', index_col=0)
    if model_name in pearson_scores.index:
        print(f'{model_name} has already been tested')
    else:
        pearson_scores.loc[model_name, :] = pearson
        pearson_scores.to_csv('results/pearsons.csv')

    # write cosine similarities
    df = pd.read_csv('results/cosine_sim.csv')
    df[model_name] = predicted_scores
    df.to_csv('results/cosine_sim.csv', index = False)


if __name__ == '__main__':
    main()



