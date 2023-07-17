import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)
    
class GAT(nn.Module):
    def __init__(self, nfeat, npred, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, npred, dropout=dropout, alpha=alpha, concat=False)
        self.loss_function = nn.MSELoss(reduction='none')

    def forward(self, features, labels, adj):
        x = F.dropout(features, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        pred = F.elu(self.out_att(x, adj))
        loss = self.loss_function(pred, labels).mean()
        batch_loss = self.loss_function(pred, labels).mean(dim=0)
        return Result(loss=loss, pred=pred, batch_loss=batch_loss)
