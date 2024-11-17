import torch
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph #, drnl_node_labeling
import os
from torch_geometric.data import Data, InMemoryDataset, download_url
# from torch_geometric.utils import from_dgl
from torch_geometric.utils import to_undirected, structured_negative_sampling, negative_sampling
import copy
import dgl
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.separate import separate
import logging

""" This version is based on the dataset from dgl/exmaple/pytorch/obg/ngnn_seal """

def get_pos_neg_edges(
        g, sample_type='structured', 
        force_undirected=False, 
        neg_ratio=1.0,
    ):
    r""" we got 3 types target edges so far, cc_p2n, cc_p2p, cc_n2n.
    So we need to generate negative edges for each target edge type.

    Args:
        g (pyg graph): the orignal homogenous graph.
        type (string): 'global' sampling or 'structured' sampling.
        force_undirected (bool): whether negative edges are undirected
    Return:
        pos_edge_index (LongTensor 2xN), neg_edge_index (LongTensor 2xN),
        neg_edge_type (LongTensor N)
    """
    if neg_ratio > 1.0 or sample_type == 'global':
        neg_edge_index = negative_sampling(
            g.tar_edge_index, g.num_nodes,
            force_undirected=force_undirected,
            num_neg_samples=int(g.tar_edge_index.size(1)*neg_ratio),
        )
        
        neg_edge_type = torch.zeros(neg_edge_index.size(1), dtype=torch.long)
        for i in range(neg_edge_index.size(1)):
            node_pair = neg_edge_index[:, i]
            ntypes = set(g.node_type[node_pair].tolist())
            # for neg edge types are related to target edge types
            # weight for neg Cc_p2n
            if ntypes == {0, 2}: 
                neg_edge_type[i] = 2
            # weight for Cc_p2p
            elif ntypes == {2}:
                neg_edge_type[i] = 3
            # weight for Cc_n2n
            elif ntypes == {0}:
                neg_edge_type[i] = 4
        
        legal_mask = neg_edge_type > 0
        logging.info(
            f"Using global negtive sampling, #pos={g.tar_edge_index.size(1)}, " + 
            f"#neg={neg_edge_index[:, legal_mask].size(1)}")
        return g.tar_edge_index, neg_edge_index[:,legal_mask], neg_edge_type[legal_mask]
    
    neg_edge_index = []
    pos_edge_index = []
    neg_edge_type  = []

    for i in range(g.num_tar_etypes):
        edge_mask = (g.tar_edge_type - g.tar_edge_type[0]) == i
        pos_edge_index.append(g.tar_edge_index[:,edge_mask])
        pos_edge_src, pos_edge_dst, neg_edge_dst = structured_negative_sampling(
            pos_edge_index[-1], g.num_nodes, contains_neg_self_loops=False,
        )
        indices = torch.randperm(pos_edge_src.size(0))[
            :int(pos_edge_src.size(0) * neg_ratio), 
        ]
        neg_edge_index.append(
            torch.stack((pos_edge_src[indices], neg_edge_dst[indices]), dim=0)
        )
        neg_edge_type.append(
            torch.ones(indices.size(0), dtype=torch.long) * (i + g.tar_edge_type[0])
        )
        logging.info(
            f"Using global negtive sampling for target etype {i}, " + 
            f"pos={pos_edge_index[-1].size(1)}, #neg={neg_edge_index[-1].size(1)}")
    
    return torch.cat(pos_edge_index, 1), torch.cat(neg_edge_index, 1), torch.cat(neg_edge_type)

def add_tar_edges_to_g(g, neg_edge_index, neg_edge_type):
        added_edges_index = torch.cat((g.tar_edge_index, neg_edge_index), 1)
        added_edge_type = torch.cat((g.tar_edge_type, neg_edge_type))
        added_edges_index, added_edge_type = to_undirected(
            added_edges_index, added_edge_type, g.num_nodes, reduce='mean'
        )
        logging.info(f"#added edges={g.tar_edge_index.size(1)+neg_edge_index.size(1)} "+
                     f"#undirected added edges={added_edges_index.size(1)}")
        aug_g = Data()
        aug_g.edge_index = torch.cat((g.edge_index, added_edges_index), dim=1)
        aug_g.edge_type = torch.cat((g.edge_type, added_edge_type)).long()
        # aug_g.neg_edge_mask = aug_g.edge_type >= g._num_etypes
        aug_g.node_type = g.node_type.long()
        aug_g.num_pos_etype = g._num_etypes
        aug_g.num_ntypes = g._num_ntypes
        aug_g.tar_edge_type_offset = g.tar_edge_type.min()
        # aug_g.tar_edge_dist = g.tar_edge_dist
        # del g.tar_edge_index
        # del g.tar_edge_type
        print("aug_g", aug_g)
        return aug_g

def k_hop_subgraph_wo_target_edges(
    node_idx: Tensor,
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: int = None,
    flow: str = 'source_to_target',
    directed: bool = False,
    sample_prob: float = 1.0,
) -> tuple:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (str, optional): The flow direction of :math:`k`-hop aggregation
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    # if isinstance(node_idx, (int, list, tuple)):
    #     node_idx = torch.tensor([node_idx], device=row.device).flatten()
    # else:
    node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_subset = col[edge_mask].unique()
        indices = torch.randperm(new_subset.size(0))[
            :int(new_subset.size(0)*sample_prob)]
        # print("new_subset len", new_subset.size(0), "sampled len", indices.size(0))
        subsets.append(new_subset[indices])
    # assert 0
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    # remove the target edges from the subgraph, added by shan
    src_dst_edge_mask  = (row == node_idx[0]) & (col == node_idx[1]) 
    src_dst_edge_mask |= (row == node_idx[1]) & (col == node_idx[0]) 
    edge_mask &= ~src_dst_edge_mask

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def shortest_path_to_src_dst(
    data,
    edge_index: Tensor,
    num_nodes: int = None,
    max_spd: int = 128,
    flow: str = 'source_to_target',
) -> tuple:
    r""" This is a modified version of torch_geometric.utils.k_hop_subgraph,
    this modified version calculate the shortest distrance between the src and dst nodes,
    and return as the last int value.
    Args:
        data: seal sampling graph
        edge_index: full graph in dataset
        num_nodes: #nodes of full graph
        max_spd: maximum shortest path distance
        flow: defualt from k_hop_subgraph

    Return: 
        data: torch_geometry.data.Data
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    # find the src and dst nodes in the sampled graph
    src, dst = data.ori_node_index[data.mapping].view(-1, 1)
    # print("src", src, "dst", dst)
    src_subsets = [src]
    dst_subsets = [dst]
    dist2src = [torch.tensor([0])]
    dist2dst = [torch.tensor([0])]
    dist_counter = 0
    # max_spd = 128
    subset = data.ori_node_index
    spd = torch.ones(subset.size(0), 2, dtype=torch.long) * (max_spd-1)
    spd_mask = torch.zeros(spd.size(), dtype=torch.bool)
    # set for src and dst node
    spd[data.mapping[0], 0] = 0
    spd[data.mapping[1], 1] = 0

    for i in range(max_spd - 1):
        dist_counter += 1
        # calculate the shortest path between src, dst
        node_mask.fill_(False)
        node_mask[src_subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_subset = col[edge_mask].unique()
        src_subsets.append(new_subset)
        # dist2src.append(torch.ones(new_subset.size())*src_dist_counter)
        for node in new_subset:
            mask = node == subset
            if mask.any():
                spd_mask[mask, 0] = True
                if spd[mask, 0] > dist_counter:
                    spd[mask, 0] = dist_counter
        # if dst.item() in new_subset.tolist():
        #     dst_found = True

        # calculate the shortest path between src, dst
        node_mask.fill_(False)
        node_mask[dst_subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_subset = col[edge_mask].unique()
        dst_subsets.append(new_subset)
        # dist2dst.append(torch.ones(new_subset.size())*dst_dist_counter)
        for node in new_subset:
            mask = node == subset
            if mask.any():
                spd_mask[mask, 1] = True
                if spd[mask, 1] > dist_counter:
                    spd[mask, 1] = dist_counter
        # if src.item() in new_subset.tolist():
        #     src_found = True
        
        # if src_found and dst_found:
        #     break
        if spd_mask.view(-1).all():
            break
    data.spd = spd
    # print("data.spd", spd)
    return data

def unique_w_index(A):
    r""" Equals to torch.unique with the indices of returned unique tensor """
    unique, idx, counts = torch.unique(A, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return unique, first_indices

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        num_hops=1,
        add_target_edges=False,
        ratio_per_hop=1.0,
        neg_edge_ratio=1.0,
        transform=None, 
        pre_transform=None
    ) -> None:
        self.name = name
        self.folder = root
        self.num_hops = num_hops
        # self.percent = percent
        self.add_target_edges = add_target_edges
        self.ratio_per_hop = ratio_per_hop
        self.neg_edge_ratio = neg_edge_ratio
        self.split_path = os.path.join(self.folder, f'shuffle_split_dict.pt')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("dataset.data", self.data)
        print("dataset.slices", self.slices)
        
    def sram_graph_load(self):
        raw_path = os.path.join(self.folder, "raw/"+self.name+".pt")
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path)
        hg = hg[0]
        # print("hg", hg)
        # print("slice", slice)
        power_net_ids = torch.empty(1, 0)
        
        if self.name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif self.name == "ultra_8T":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif self.name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif self.name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif self.name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        elif self.name == "8T_digitized_timing_top_fast":
            power_net_ids = torch.tensor([0, 1])
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False
        hg = hg.subgraph(subset_dict)
        hg = hg.edge_type_subgraph([
            ('device', 'device-pin', 'pin'),
            ('pin', 'pin-net', 'net'),
            ('pin', 'cc_p2n', 'net'),
            ('pin', 'cc_p2p', 'pin'),
            ('net', 'cc_n2n', 'net'),
        ])
        print(hg)
        ### transform hetero g into homo g
        g = hg.to_homogeneous()
        print("to_homogeneous g:", g)
        # assert 0
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        # {'net': 0, 'device': 1, 'pin': 2}
        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n

        g._e2type = {}
        tar_edge_dist = []
        num_tar_edges = 0

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if 'cc' in edge[1]:
                tar_edge_y.append(store['y'])
                tar_edge_dist.append(store['edge_index'].shape[1])
                num_tar_edges += store['edge_index'].shape[1]
            else:
                # edge_index's shape [2, num_edges]
                edge_offset += store['edge_index'].shape[1]
        
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        g.num_tar_edges = num_tar_edges
        g.num_tar_etypes = len(tar_edge_dist)
        g.tar_edge_dist = torch.tensor(tar_edge_dist)
        # print(shg)
        print("g._n2type", g._n2type)
        print("g._e2type", g._e2type)
        g.tar_edge_offset = edge_offset
        g.tar_edge_index = g.edge_index[:, edge_offset:]
        g.tar_edge_type = g.edge_type[edge_offset:]
        g.tar_edge_y = torch.cat(tar_edge_y)
        # tar_edge_weight = tar_edge_dist.max() / tar_edge_dist
        # g.tar_edge_weight = tar_edge_weight / tar_edge_weight.sum()
        # g.tar_edge_weight = tar_edge_weight
        # print("g.tar_edge_weight=", g.tar_edge_weight)
        # remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]
        g.edge_index, g.edge_type = to_undirected(
            g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
        )
        del g.y
        return g
    
    def process(self):
        # we can have multiple graphs
        graph = self.sram_graph_load()
        print("pyg", graph)
        print("neg_edge_ratio", self.neg_edge_ratio)
        # add neg edges for each graph
        pos_edge_index, neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)
        if self.add_target_edges:
            # self.data is actually the augmented graph
            self.data = add_tar_edges_to_g(graph, neg_edge_index, neg_edge_type)
        else:
            self.data = graph
        # generate links for train / val / test
        links = torch.cat([pos_edge_index, neg_edge_index], 1)  # [2, Np + Nn]
        link_types = torch.cat([graph.tar_edge_type, neg_edge_type])  # [Np + Nn]
        link_types -= graph.tar_edge_type.min() # link type in {0,1,2}
        print("links shape", links.shape)
        print("link_types shape", link_types.shape)
        labels = torch.tensor(
            [1]*pos_edge_index.shape[1] + [0]*neg_edge_index.shape[1]
        )
        # prepare the weights for both pos & neg edges
        weights = graph.tar_edge_dist.max() / graph.tar_edge_dist # shape [3]
        weights = torch.gather(
            weights.expand(links.size(1), -1), 
            dim=1, 
            index=link_types.view(-1, 1)
        )
        # we sample subgraphs for each full graph, and append to data_list
        logging.info("Do seal_sampling ...")
        data_list = [
            self.seal_sampling(
                i, links[:, i].view(-1), labels[i], weights[i], 
            )
            for i in tqdm(range(links.size(1)), ncols=80)
        ]
            
        if self.pre_transform is not None:
            logging.info("Do spd calculation ...")
            data_list = [
                self.pre_transform(data, graph.edge_index, graph.num_nodes) 
                for data in tqdm(data_list)
            ]

        self.create_shuffle_split(len(data_list), 0.1, 0.1, link_types)
        data, slices = self.collate(data_list)

        logging.info('Saving processed data_list to %s ...' % self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

    def seal_sampling(
            self, idx: int, node_pair: Tensor, 
            label: torch.Tensor, weight: float=1.0
        ) -> Data:

        g = self.data
        subset, subg_edge_index, mapped_node_pair, subg_edge_mask = \
            k_hop_subgraph_wo_target_edges(
                node_pair, self.num_hops, g.edge_index, 
                relabel_nodes=True, num_nodes=g.num_nodes, 
                directed=g.is_directed(),
                sample_prob=self.ratio_per_hop,
            )
        data = Data()
        data.edge_index = subg_edge_index
        # data.ori_node_index = subset
        data.mapped_node_pair = mapped_node_pair
        # data.spd = shortest_distance
        # data.edge_mask = subg_edge_mask.nonzero().view(-1)
        data.edge_attr = g.edge_type[subg_edge_mask].view(-1)
        data.y = label
        data.x = g.node_type[subset].view(-1, 1)
        
        # adj = csr_matrix(to_dense_adj(data.edge_index).squeeze())
        # data.z = drnl_node_labeling(adj, mapped_node_pair[0], mapped_node_pair[1]).view(-1, 1)

        # dgl_z = dgl.double_radius_node_labeling(subg, 0, 1) 
        # print(f"data #n={data.num_nodes}, #e={data.num_edges}, src={src}, dst={dst}")
        
        data.weight = weight
        # data = shortest_path_to_src_dst(data, g.edge_index, g.num_nodes, max_spd=64)
        # print(f"src:{nodes[0]}, dst:{nodes[1]}, dist:{dist}, etype:{self.graph.tar_edge_type[pos_idx]} label:{label}")
        return data 

    def create_shuffle_split(self, N, val_ratio, test_ratio, edge_type):
        """ Create a random shuffle split and saves it to disk.
        Args:
            N: Total size of the dataset to split.
        """

        if N == 2* len(edge_type):
            stratify = torch.cat((edge_type, edge_type))
        else:
            stratify = edge_type

        train_ind, val_ind = train_test_split(
            np.arange(N), test_size=val_ratio+test_ratio, 
            random_state=123, stratify=stratify,
        )

        val_ind, test_ind = train_test_split(
            val_ind, test_size=test_ratio/(val_ratio+test_ratio), 
            random_state=123, stratify=stratify[val_ind],
        )
        # assert self._check_splits(N, [train_ind, val_ind, test_ind],
        #                           [train_ratio, val_ratio, test_ratio])

        shuffle_split = {'train': train_ind, 'val': val_ind, 'test': test_ind}
        self.split_path = os.path.join(self.folder, f'shuffle_split_dict.pt')
        print('Saving shuffle_split to %s ...' % self.split_path)
        torch.save(shuffle_split, self.split_path)
    
    def get_idx_split(self, split_name):
        """ Get dataset splits.

        Args:
            split_name: Split type: 'shuffle', 'num-atoms'

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = os.path.join(
            self.folder,
            f"{split_name.replace('-', '_')}_split_dict.pt"
        )
        split_dict = torch.load(self.split_path)
        return split_dict

    @property
    def raw_file_names(self):
        return self.name+".pt"

    @property
    def processed_file_names(self):
        name = f"hop_{self.num_hops}"
        if self.add_target_edges:
            if self.ratio_per_hop < 1.0:
                name += f"_aug_p{self.ratio_per_hop:.1f}"
            else:
                name += "_aug"
        if self.neg_edge_ratio == 1.0:
            name += "_processed.pt"
        else:
            name += f"_nr{self.neg_edge_ratio:.1f}_processed.pt"
        return name
    
    @property
    def num_classes(self) -> int:
        return 2
    
    def save_pe_codes(self, pe_types):
        r""" this method will only be revoked after  
        pre_transform_in_memory to get position encodings.
        """
        data0 = self.get(0)
        if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
            if not hasattr(data0, 'EigVals') or not hasattr(data0, 'EigVecs'):
                raise AttributeError("data in dataset doesn't have attribute 'EigVals' or 'EigVecs, using {pe_types} encoder/encoders")
        if 'SignNet' in pe_types:
            if not hasattr(data0, 'eigvals_sn') or not hasattr(data0, 'eigvecs_sn'):
                raise AttributeError("data in dataset doesn't have attribute 'eigvals_sn' or 'eigvecs_sn, using {pe_types} encoder/encoders")
        if 'RWSE' in pe_types:
            if not hasattr(data0, 'pestat_RWSE'):
                raise AttributeError("data in dataset doesn't have attribute 'pestat_RWSE', using {pe_types} encoder/encoders")
        if 'DRNL' in pe_types:
            if not hasattr(data0, 'z'):
                raise AttributeError("data in dataset doesn't have attribute 'z', using {pe_types} encoder/encoders")
        if 'SPD' in pe_types:
            if not hasattr(data0, 'spd'):
                raise AttributeError("data in dataset doesn't have attribute 'spd', using {pe_types} encoder/encoders")
        if 'GraphormerBias' in pe_types:
            if not hasattr(data0, 'in_degrees') or not hasattr(data0, 'out_degrees'):
                raise AttributeError("data in dataset doesn't have attribute 'in_degrees' or 'out_degrees', using {pe_types} encoder/encoders")
            
        for pe_type in pe_types:
            # graphormer encoding is too large to save
            if pe_type == "GraphormerBias":
                continue 
            path = self.processed_paths[0].replace('.pt',f'_{pe_type}.pt')
            logging.info(f"Saving data_list with {pe_type} encoding to {path} ...")
            torch.save((self.data, self.slices), path)

    def has_processed_pe(self, pe_types):
        data_list = self.len() * [None]
        if hasattr(self, '_data_list') or self.cached_data_list is not None:
            data_list = self.cached_data_list
        else:
            for i in len(self):
                data_list[i] = self[i]
        
        pe_types_need_to_calculate = []

        for p, pe_type in enumerate(pe_types):
            path = self.processed_paths[0].replace('.pt',f'_{pe_type}.pt')

            if os.path.exists(path):
                loaded_data, loaded_slices = torch.load(path)
                if (loaded_data.num_edges != self.data.num_edges) or \
                    (loaded_data.num_nodes != self.data.num_nodes):
                    pe_types_need_to_calculate.append(pe_type)
                    logging.info(
                        f"self.data and loaded data are mismatch," + 
                        f"pe {pe_type} needs to be re-computed" +
                        f"#edges: {loaded_data.num_edges} and {self.data.num_edges}, " +
                        f"#nodes: {loaded_data.num_nodes} and {self.data.num_nodes}, " +
                        f"load path {path}")
                    continue
                
                logging.info(f"updating the dataset with {pe_type} encoding, load path {path}")
                
                for i in range(len(self)):
                    data = separate(
                        cls=loaded_data.__class__,
                        batch=loaded_data,
                        idx=i,
                        slice_dict=loaded_slices,
                        decrement=False,
                    )

                    if (self[i].num_nodes != data.num_nodes) or \
                        (self[i].num_edges != data.num_edges):
                        pe_types_need_to_calculate.append(pe_type)
                        raise AttributeError(
                            f"data_list{i} and data in disk file are mismatched" +
                            f"with #nodes {self[i].num_nodes} and {data.num_nodes}, " +
                            f"#edges {self[i].num_edges} and {data.num_edges}"
                        )

                    if pe_type == 'DRNL':
                        data_list[i].z = data.z
                    elif pe_type == 'LapPE' or pe_type == 'EquivStableLapPE':
                        data_list[i].EigVals = data.EigVals
                        data_list[i].EigVecs = data.EigVecs
                    elif pe_type == 'SignNet':
                        data_list[i].eigvals_sn = data.eigvals_sn
                        data_list[i].eigvecs_sn = data.eigvecs_sn
                    elif pe_type == 'RWSE':
                        data_list[i].pestat_RWSE = data.pestat_RWSE
                    elif pe_type == 'GraphormerBias':
                        data_list[i].in_degrees = data.in_degrees
                        data_list[i].out_degrees = data.out_degrees
                    elif pe_type == 'SPD':
                        data_list[i].spd = data.spd
                        if hasattr(data, 'shortest_path_etypes'):
                            data_list[i].shortest_path_etypes = data.shortest_path_etypes
                    else:
                        raise NotImplementedError(f"PE encoder {pe_type} hasn't been supported in this version")

            else:
                pe_types_need_to_calculate.append(pe_type)
        # update self.data & self.slices with the loaded data & slices
        if len(pe_types) > len(pe_types_need_to_calculate):
            logging.info("We have updated self data.")
            self.data, self.slices = self.collate(data_list)
            print(self.data)
        return pe_types_need_to_calculate
    
if __name__ == '__main__':
    dataset = SealSramDataset(name='8T_digitized_timing_top_fast', root='/data1/shenshan/dgl/GraphGPS/datasets')
    print("dataset.data", dataset.data)
    print("dataset.graph", dataset.graph)
    print("dataset.links", dataset.links)
    print("dataset.labels", dataset.labels)
    print("dataset.split_idxs", dataset.split_idxs)
    print("dataset._num_ntypes", dataset._num_ntypes)
    print("dataset.num_hops", dataset.num_hops)