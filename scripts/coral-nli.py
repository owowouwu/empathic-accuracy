import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader, random_split
import os
import logging
import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modules.coralnet as coralnet
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.dataset import proba_to_label
from datasets import load_dataset

os.chdir(settings.PROJECT_WORKING_DIRECTORY)
BASE_MODEL = 'sentence-transformers/all-distilroberta-v1'
DEVICE = 'cuda'

def make_sentence_pairs(u,v, how = 'concat'):
    if how == 'concat':
        return np.concatenate([u,v], axis = 1)
    elif how == 'l1':
        return np.abs(u - v)
    elif how == 'prod':
        return u*v
    elif how == 'concatl1':
        return np.concatenate([u,v, np.abs(u-v)], axis = 1)

def main():
    # hyperparameters
    batch_size = 512
    num_epochs = 50
    learning_rate = 0.001
    hidden_layers = [256, 32]
    dropout= [0.5,0.5]
    concat_method = 'concatl1'

    transformer = SentenceTransformer(BASE_MODEL)
    transformer.to(DEVICE)
    dataset = load_dataset("multi_nli")
    num_labels = 3
    encodings = {}
    for split, ds in dataset.items():
        encodings[split] = {}
        encodings[split]['s1'] = transformer.encode(ds['premise'].values)
        encodings[split]['s2'] = transformer.encode(ds['hypothesis'].values)

    # DATA

    X_train = make_sentence_pairs(encodings['train']['s1'], encodings['train']['s2'], concat_method)
    X_valid = make_sentence_pairs(encodings['validation_matched']['s1'], encodings['train']['s2'], concat_method)


    y_train = dataset['train']['label']
    y_valid = dataset['validation_matched']['label']
    train_dataset = coralnet.MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)


    levels = levels_from_labelbatch(range(num_labels), num_classes=num_labels)


    X_valid_t = torch.tensor(X_valid).to(DEVICE)
    y_valid_t = torch.tensor(y_valid).to(DEVICE)
    valid_levels = levels_from_labelbatch(y_valid, num_classes=num_labels)
    valid_levels = valid_levels.to(DEVICE)
    torch.manual_seed(1234)
    model = coralnet.CoralNet(input_dim=X_valid.shape[1], num_classes=num_labels, hidden_layers=hidden_layers, dropout=dropout)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    evals = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (features, class_labels) in enumerate(train_loader):

            ##### Convert class labels for CORAL
            levels = levels_from_labelbatch(class_labels,
                                            num_classes=num_labels)
            ###--------------------------------------------------------------------###

            features = features.to(DEVICE)
            levels = levels.to(DEVICE)
            logits, probas = model(features)

            #### CORAL loss
            loss = coral_loss(logits, levels)
            ###--------------------------------------------------------------------###

            # accuracy
            predicted_labels = proba_to_label(probas).float()
            class_labels = class_labels.to(DEVICE)
            acc = (predicted_labels == class_labels).sum() / class_labels.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            if not batch_idx % 200:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f | Acc: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss, acc))

        evals['train_loss'].append(loss.item())
        evals['train_acc'].append(acc.item())

        model = model.eval()
        with torch.no_grad():
            logit, probas = model(X_valid_t)
            valid_loss = coral_loss(logit, valid_levels)
            predicted_labels = proba_to_label(probas).float()
            acc = (predicted_labels == y_valid_t).sum() / y_valid_t.shape[0]
            print('| Valid Loss : %.4f | Valid Acc : %.4f' % (valid_loss, acc))
            evals['val_loss'].append(valid_loss.item())
            evals['val_acc'].append(acc.item())


    torch.save(model, 'models/nli-coral.pt')

if __name__ == '__main__':
    main()