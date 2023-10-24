import torch.nn as nn
import torch.nn.functional as F

class BasicFFNet(nn.Module):
    def __init__(self, input_dim, hidden_layers = None, dropout = None):
        super().__init__()

        # default values
        if not hidden_layers: hidden_layers = [32]
        if not dropout: dropout = [0]
        if isinstance(hidden_layers, int): hidden_layers = [hidden_layers]
        if isinstance(dropout, float): dropout = [dropout]

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_dim = input_dim

        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, size))
            input_dim = size

        self.output = nn.Linear(input_dim, 1)

        for rate in dropout:
            self.dropout_layers.append(nn.Dropout(rate))

        self.act_output = nn.Tanh()

    def forward(self, x):
        for (hidden, dropout) in zip(self.hidden_layers, self.dropout_layers):
            x = hidden(x)
            x = F.relu(x)
            x = dropout(x)

        x = self.output(x)
        # rescale tanh to [0,1]
        x = (self.act_output(x) + 1) / 2
        return x