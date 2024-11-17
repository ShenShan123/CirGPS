import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
import torch

@register_head('san_graph_for_reg')
class SANGraphHead4Reg(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.gnn.agg]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l+1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.bn = cfg.gnn.batchnorm
        self.dropout = nn.Dropout(cfg.gnn.dropout)
        self.activation = register.act_dict[cfg.gnn.act]()
        self.BN_layers = nn.ModuleList([
            nn.BatchNorm1d(dim_in // 2 ** (l+1))
            for l in range(L)]) 
        self.regression_task = False
        # self.cir_statis_proj = False
        self.cir_statis_proj = cfg.gnn.cir_statis_proj
        if "regression" in cfg.dataset.task_type:
            self.regression_task = True
            # self.cir_statis_proj = cfg.gnn.cir_statis_proj

        if self.cir_statis_proj:
            # add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, dim_in, bias=True)
            self.dev_attr_layers = nn.Linear(17, dim_in, bias=True)
            self.pin_attr_layers = nn.Embedding(17, dim_in)

    def _apply_index(self, batch):
        if self.regression_task:
            return batch.graph_feature, batch.y_reg
        else:
            return batch.graph_feature, batch.y.long()

    def forward(self, batch):
        # compute the node attribute pooling
        if self.cir_statis_proj:
            net_node_mask = batch.node_type == 0
            dev_node_mask = batch.node_type == 1
            pin_node_mask = batch.node_type == 2
            
            node_attr_emb = torch.zeros(
                batch.x.size(), device=batch.x.device)
            
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            # graph_attr_emb = self.pooling_fun(node_attr_emb, batch.batch)
            batch.x += node_attr_emb
        
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
            if self.bn:
                graph_emb = self.BN_layers[l](graph_emb)
            graph_emb = self.dropout(graph_emb)
        
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
