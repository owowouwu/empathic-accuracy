import torch
import os
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
import settings
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from scipy.stats import pearsonr
from scipy.spatial import distance
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold


os.chdir(settings.PROJECT_WORKING_DIRECTORY)

def make_dataloader(df, batch_size):
    train_examples = []
    for _, row in df.iterrows():
        train_examples.append(InputExample(texts=[row['user_text'], row['designer_text']], label=row['avg_EA_scaled']))

    return DataLoader(train_examples, shuffle=True, batch_size=batch_size)

def predict_scores(designer_embeddings, user_embeddings):
    return [1 - distance.cosine(designer_embeddings[i], user_embeddings[i])
                   for i in range(user_embeddings.shape[0])]

def evals(true, predicted):
    rmse = np.sqrt(np.mean((true - predicted)**2))
    pearson = pearsonr(true, predicted)[0]
    print("Pearson:", pearson)
    print("RMSE:", rmse)
    return pearson, rmse

def eval_model(model,s1,s2, true_scores):
    s1 = model.encode(s1)
    s2 = model.encode(s2)
    scores = predict_scores(s1, s2)
    pearson,rmse = evals(true_scores, scores)
    return pearson, rmse
def main():
    interview_data = pd.read_excel('data/full_data.xlsx', sheet_name = 1)
    interview_data['avg_EA_scaled'] = interview_data['avg_EA'] / 2.
    u = interview_data['user_text']
    d = interview_data['designer_text']
    u = u.str.lower().str.replace(':', '')
    d = d.str.lower().str.replace(':', '')
    u = u.str.replace('i was', '')
    d = d.str.replace('i was', '').str.replace('s/he was', '').str.replace('she was', '').str.replace('he was', '')

    interview_data['user_text'] = u
    interview_data['designer_text'] = d
    k = 10
    splits = KFold(n_splits=k, shuffle=True, random_state=123)
    model_id = 'sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1'
    epochs = 50
    batch_size = 1

    results = []
    for k, (train, val) in enumerate(splits.split(interview_data)):
        model = SentenceTransformer(model_id)
        model.to('cuda')
        train_loss = losses.CosineSimilarityLoss(model=model)
        train_df = interview_data.iloc[train]
        val_df = interview_data.iloc[val]
        train_dataloader = make_dataloader(train_df, batch_size=batch_size)
        baseline_pearson, baseline_rmse = eval_model(model, val_df['user_text'].values, val_df['designer_text'].values,
                                                     val_df['avg_EA_scaled'].values
                                                     )
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs)
        train_pearson, train_rmse = eval_model(model, train_df['user_text'].values, train_df['designer_text'].values,
                                               train_df['avg_EA_scaled'].values
                                               )
        val_pearson, val_rmse = eval_model(model, val_df['user_text'].values, val_df['designer_text'].values,
                                           val_df['avg_EA_scaled'].values
                                           )

        results.append({
            'k': k,
            'train_pearson': train_pearson, 'train_rmse': train_rmse,
            'val_pearson': val_pearson, 'val_rmse': val_rmse,
            'baseline_pearson':baseline_pearson, 'baseline_rmse':baseline_rmse
        })

    pd.DataFrame(results).to_csv('results/cv-experiment2.csv')

if __name__ == '__main__':
    main()