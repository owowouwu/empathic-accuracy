import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
import numpy as np
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from torch.utils.data import DataLoader, Dataset

class CoralNet(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_layers=None, dropout=None):
        super().__init__()

        # default values
        if not hidden_layers:
            hidden_layers = [32]
        if not dropout:
            dropout = [0.25]

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_dim = input_dim

        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, size))
            input_dim = size

        for rate in dropout:
            self.dropout_layers.append(nn.Dropout(rate))

        self.output_layer = CoralLayer(size_in=hidden_layers[-1], num_classes=num_classes)

    def forward(self, x):
        for (hidden, dropout) in zip(self.hidden_layers, self.dropout_layers):
            x = hidden(x)
            x = F.gelu(x)
            x = dropout(x)

        logits = self.output_layer(x)
        probas = torch.sigmoid(logits)
        return logits, probas

class MyDataset(Dataset):

    def __init__(self, feature_array, label_array, dtype=np.float32):
        self.features = feature_array.astype(np.float32)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.labels.shape[0]