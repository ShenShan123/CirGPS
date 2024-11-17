import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
import logging
# DRNL node labeling encoder, added by shan

@register_node_encoder('DRNL')
class DrnlNodeEncoder(torch.nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim
        pecfg = cfg.posenc_DRNL
        num_types = pecfg.max_drnl
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=dim_pe)

        if dim_emb - dim_pe < 1:
            raise ValueError(f"DRNL size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")
        # usually we have already got TypeDictNodeEncoder, 
        # so expand_x is set to False in composed_encoders.py
        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0
        print("Drnl dim_in", dim_in, "dim_emb", dim_emb, "dim_pe", dim_pe)

    def forward(self, batch):
        if not hasattr(batch, 'z') or batch.num_nodes != batch.z.shape[0]:
            raise AttributeError("Double raduis node labeling has not been calculated!"
                                 f"node num={batch.num_nodes}, x.shape={batch.x.shape}, in DrnlNodeEncoder.forward()")
        z_emb = self.encoder(batch.z.squeeze())

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, z_emb), 1)
        return batch
    

# Shortest distance graph encoder, same as DrnlNodeEncoder, added by shan

@register_node_encoder('SPD')
class ShortestPathDistEncoder(torch.nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim
        pecfg = cfg.posenc_SPD
        self.max_dist = pecfg.max_dist
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        # 2 shortest path to src and dst nodes
        self.dist_encoder = torch.nn.Embedding(self.max_dist + 10, int(dim_pe/2))
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        # self.edge_type_encoder = torch.nn.Embedding(
        #     cfg.dataset.edge_encoder_num_types+1, int(dim_pe/2))

        if dim_emb - dim_pe < 1:
            raise ValueError(f"DSPD size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")
        # usually we have already got TypeDictNodeEncoder, 
        # so expand_x is set to False in composed_encoders.py
        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0
        logging.info(f"SPD encoder dim_in={dim_in}, dim_emb={dim_emb}, dim_pe={dim_pe}")

    def forward(self, batch):
        if not hasattr(batch, 'spd'):
            raise AttributeError(
                "Shortest distance has not been calculated!"+
                f"node num={batch.num_nodes}, x.shape={batch.x.shape}, "+
                "in ShortestPathDistEncoder.forward()"
            )
        
        spd_emb = self.dist_encoder(batch.spd)
        N = spd_emb.size(0)
        # assert batch.shortest_path_etypes.size(1) == 2
        # assert batch.shortest_path_etypes.size(0) == N
        # shape of src_sp_etype_emb [N, 2, max_dist, dim_pe/2]
        # etype_emb = self.edge_type_encoder(batch.shortest_path_etypes.int())
        # shape of src_sp_etype_emb [N, max_dist, dim_pe/2]
        # src_etype_emb = etype_emb[:, 0, :, :]
        # dst_etype_emb = etype_emb[:, 1, :, :]
        # sum all etype_emb along with the shortest path and normalized them by path distance
        # dim [N, dim_pe/2]
        # src_etype_emb = src_etype_emb.sum(1) 
        # src_etype_emb = src_etype_emb / spd_emb[:, 0]
        # dst_etype_emb = dst_etype_emb.sum(1)
        # dst_etype_emb = dst_etype_emb / spd_emb[:, 1]
        # dim [N, dim_pe]
        # sp_emb = torch.cat((src_etype_emb, dst_etype_emb), dim=1) 

        if spd_emb.ndim == 2 and spd_emb.size(1) == 2:
            spd_emb = torch.cat((spd_emb[:, 0], spd_emb[:, 1]), dim=1)
        elif spd_emb.ndim == 3 and spd_emb.size(1) == 2:
            spd_emb = torch.cat((spd_emb[:, 0, :], spd_emb[:, 1, :]), dim=1)
        else:
            raise ValueError(f"Dimension number of SPD embedding is {spd_emb.ndim}, size", spd_emb.size())
        
        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, spd_emb), 1)
        return batch
    
@register_node_encoder('CirStatis')
class CircuitStatisticEncoder(torch.nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim
        pecfg = cfg.posenc_CirStatis
        dim_pe = pecfg.dim_pe
        self.dim_pe = dim_pe
        # add node_attr transform layer for net/device/pin nodes, by shan
        self.net_attr_layers = nn.Linear(17, dim_pe, bias=True)
        self.dev_attr_layers = nn.Linear(17, dim_pe, bias=True)
        self.pin_attr_layers = nn.Embedding(17, dim_pe)
        if dim_emb - dim_pe < 1:
            raise ValueError(f"CirStatis size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")
        # usually we have already got TypeDictNodeEncoder, 
        # so expand_x is set to False in composed_encoders.py
        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0
        logging.info(f"CirStatis encoder dim_in={dim_in}, dim_emb={dim_emb}, "+
                     f"dim_pe={dim_pe}, expand_x={self.expand_x}")

    def forward(self, batch):
        if not hasattr(batch, 'node_type'):
            raise AttributeError(
                "node_type has not been incorporated in batch!"+
                f"node num={batch.num_nodes}, x.shape={batch.x.shape}, "+
                "in CircuitStatisticEncoder.forward()"
            )
        if not hasattr(batch, 'node_attr'):
            raise AttributeError(
                "node_attr has not been incorporated in batch!"+
                f"node num={batch.num_nodes}, x.shape={batch.x.shape}, "+
                "in CircuitStatisticEncoder.forward()"
            )
        # compute the node attribute pooling
        # if 'regression' in cfg.dataset.task_type:
        net_node_mask = batch.node_type == 0
        dev_node_mask = batch.node_type == 1
        pin_node_mask = batch.node_type == 2
        node_attr_emb = torch.zeros(
            (batch.node_attr.size(0), self.dim_pe), device=batch.node_type.device)
        node_attr_emb[net_node_mask] = \
            self.net_attr_layers(batch.node_attr[net_node_mask])
        node_attr_emb[dev_node_mask] = \
            self.dev_attr_layers(batch.node_attr[dev_node_mask])
        node_attr_emb[pin_node_mask] = \
            self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
        # assert graph_emb.size(0) == graph_attr_emb.size(0)
        # assert graph_emb.size(1) == graph_attr_emb.size(1)
        # print("\ngraph_emb shape", graph_emb.shape)
        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, node_attr_emb), 1)
        return batch