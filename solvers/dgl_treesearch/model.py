import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=torch.sigmoid))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class HindsightLoss(nn.Module):
    def __init__(self):
        super(HindsightLoss, self).__init__()
        self.ce_func = nn.BCELoss(reduction="none")

    def forward(self, output, labels):
        probmaps = output.shape[1]
        _labels = torch.unsqueeze(labels, 0)
        _labels = _labels.float().repeat(probmaps, 1)
        output = output.permute(1, 0)

        loss = torch.min(torch.mean(self.ce_func(output, _labels), axis=1))
        return loss
