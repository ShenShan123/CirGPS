import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
# from torch_geometric.utils import k_hop_subgraph #, drnl_node_labeling
import os
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from graphgps.loader.dataset.seal_sampling import (
    seal_sampling, mp_seal_sampling, get_pos_neg_edges, 
    add_tar_edges_to_g, get_balanced_edges, collated_data_separate,
    k_hop_subgraph_wo_target_edges, ArgsIterator)
import copy 

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        num_hops=1,
        add_target_edges=False,
        ratio_per_hop=1.0,
        neg_edge_ratio=1.0,
        to_undirected=True,
        shuffle_split=True,
        num_sampler=1,
        sample_rates=[1.0],
        task_type='classification',
        transform=None, 
        pre_transform=None
    ) -> None:
        self.name = 'sram'
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        self.folder = os.path.join(root, self.name)
        self.num_hops = num_hops
        self.add_target_edges = add_target_edges
        self.ratio_per_hop = ratio_per_hop
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        self.shuffle_split = shuffle_split
        self.num_sampler = num_sampler
        self.data_lengths = {}
        self.data_offsets = {}
        self.task_type = task_type
        self.max_net_node_feat = torch.ones((1, 17))
        self.max_dev_node_feat = torch.ones((1, 17))
        super().__init__(self.folder, transform, pre_transform)
        # combine multiple subgraphs into 1 data
        data_list = []

        for i, name in enumerate(self.names):
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
            self.data_offsets[name] = len(data_list)

            # we temporarily change the data_len for fast training
            # should be remove when we have time and computing resources
            # if name == "sandwich":
            #     data_len = int(loaded_data.y.size(0)* 0.3)
            #     self.data_lengths[name] = data_len
            # elif name == "ssram":
            #     data_len = int(loaded_data.y.size(0)* 0.5)
            #     self.data_lengths[name] = data_len
            # else:
            self.data_lengths[name] = loaded_data.y.size(0)

            data_list += collated_data_separate(loaded_data, loaded_slices)#[:data_len]
            logging.info(f"load processed {name}, "+
                         f"len(data_list)={self.data_lengths[name]}, "+
                         f"data_offset={self.data_offsets[name]} ")
        
        logging.info(f"total length of data_list {len(data_list)}, collating ...")
        # self.data, self.slices = self.collate(data_list)
        # normalize node feat for net and device nodes
        # self.norm_nfeat([0, 1])
        # do not collate here as PE encoding is going to be computed later
        self._data_list = data_list

    # we temporarily normlize node_attr in dataset.__init__
    def norm_nfeat(self, ntypes):
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0
            # if ntype == 1:
            #     dev_node_attr = self.data.node_attr[node_mask]
            #     max_id = torch.argmax(dev_node_attr[:, 3])
            #     print("dev_node_attr", self.data.node_attr[node_mask][max_id])
            #     print("min max", dev_node_attr.min(), dev_node_attr.max())
            logging.info(f"normalizing tar_node_attr {ntype}: {max_node_feat} ...")
            self._data.node_attr[node_mask] /= max_node_feat
        
        # y_reg stands for Cc values, ranging from [0, 1)
        if "node" not in self.task_type:
            self._data.y_reg = torch.log10(self._data.y_reg * 1e21) / 6
            # mask out y_reg of neg edges
            self._data.y_reg[self._data.y_reg < 0] = 0.0
        # y_reg stands for Cg values for net and pin nodes, ranging from [0, 1]
        else: # we make 1e-21 < Cg < 1e-15
            y = self._data.y
            y_reg = self._data.y_reg
            y_reg = torch.log10(y_reg * 1e21) / 6
            # mask out y_reg of neg edges
            y_reg[y_reg < 0] = 0.0
            y_reg[y_reg > 1] = 1.0
            hist, bin_edges = torch.histogram(y_reg, bins=6)
            y[y_reg <= bin_edges[2]] = 0
            y[(y_reg > bin_edges[2]) & (y_reg <= bin_edges[3])] = 1
            y[(y_reg > bin_edges[3]) & (y_reg <= bin_edges[4])] = 2
            y[y_reg > bin_edges[4]] = 3
            self._data.y_reg = y_reg
            self._data.y = y
            print("DEBUG: unique(self._data.y)", torch.unique(y, return_counts=True))

    def sram_graph_load(self, name, raw_path):
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path)
        if isinstance(hg, list):
            hg = hg[0]
        # print("hg", hg)
        power_net_ids = torch.tensor([0, 1])
        
        if name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif name == "ultra8t":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        elif name == "digtime":
            power_net_ids = torch.tensor([0, 1])
        elif name == "timing_ctrl":
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
        # virtual y for device nodes
        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30
        print(hg)
        ### transform hetero g into homo g
        g = hg.to_homogeneous()
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        node_feat = []
        max_feat_dim = 17
        net_x_union = torch.tensor(
            [1e3]*4 + [1e-6]*2 + # MOS
            [1e3, 1e-6, 1e-6] + # rupolym 
            [1e3, 1e3, 1e-6] + # cfmom
            [1e3, 1e-12, 1e-6] + # dio
            [1, 1], # port and lvl
        ).view(1, -1)
        dev_x_union = torch.tensor(
            [1e3] + [1e-6]*4 + # MOS
            [1e3, 1e-6, 1e-6] + # rupolym 
            [1e3, 1e3, 1e-6] + # cfmom
            [1e3, 1e-12, 1e-6] + # dio
            [1e3, 1], # num ports and type_code
        ).view(1, -1)
        # {'net': 0, 'device': 1, 'pin': 2}
        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n
            feat = hg[ntype].x
            if ntype == 'net':
                feat /= net_x_union
            elif ntype == 'device':
                feat /= dev_x_union
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
            
        g.node_attr = torch.cat(node_feat, dim=0)
        g.tar_node_y = torch.cat(tar_node_y, dim=0)

        assert g.num_nodes == g.tar_node_y.size(0)
        del g.y
        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if 'cc' in edge[1]:
                tar_edge_y.append(store['y'])
                # print("tar_edge_type", e, "y", store['y'][0])
            else:
                # edge_index's shape [2, num_edges]
                edge_offset += store['edge_index'].shape[1]
        
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")
        g.tar_edge_offset = edge_offset
        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)

        # testing
        # for i in range(tar_edge_type.min(), tar_edge_type.max()+1):
        #     mask = tar_edge_type == i
        #     print("tar_edge_type", tar_edge_type[mask][0], "tar_edge_y", tar_edge_y[mask][0])
        # assert 0

        # restrict the target value range 
        legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
        tar_edge_src_y = g.tar_node_y[tar_edge_index[0, :]].squeeze()
        tar_edge_dst_y = g.tar_node_y[tar_edge_index[1, :]].squeeze()
        legel_node_mask = (tar_edge_src_y < 1e-13) & (tar_edge_src_y > 1e-23)
        legel_node_mask &= (tar_edge_dst_y < 1e-13) & (tar_edge_dst_y > 1e-23)
        
        g.tar_edge_y = tar_edge_y[legel_edge_mask]# & legel_node_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]# & legel_node_mask]
        g.tar_edge_type = tar_edge_type[legel_edge_mask]# & legel_node_mask]
        logging.info(f"we filter out the edges with Cc > 1e-15 and Cc < 1e-21 " + 
                     f"{legel_edge_mask.size(0)-legel_edge_mask.sum()}")
        logging.info(f"we filter out the edges with src/dst Cg > 1e-13 and Cg < 1e-23 " +
                     f"{legel_node_mask.size(0)-legel_node_mask.sum()}")
        # assert 0
        # calculate target edge distributions
        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
        
        # remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]

        # convert to undirected edges
        if self.to_undirected:
            g.edge_index, g.edge_type = to_undirected(
                g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
            )
        return g


    """
    def seal_sampling(
        self,
        arg_tuple,
    ) -> Data:
        g, node_pairs, labels, targets = arg_tuple
        edge_index = g.edge_index 
        edge_type = g.edge_type
        node_type = g.node_type
        node_attr = g.node_attr
        num_nodes = node_type.size(0)
        num_pairs = node_pairs.size(1)
        directed = g.is_directed()
        num_hops = self.num_hops
        ratio_per_hop = self.ratio_per_hop
        # found bug if we use tqdm
        # for i in tqdm(range(num_pairs), ncols=80):
        for i in range(num_pairs):
            subset, subg_edge_index, mapped_node_pair, subg_edge_mask = \
            k_hop_subgraph_wo_target_edges(
                node_pairs[:, i], num_hops, edge_index, 
                relabel_nodes=True, num_nodes=num_nodes, 
                directed=directed,
                sample_prob=ratio_per_hop,
            )
            # print("return from seal_sampling")
            data = Data()
            data.edge_index = subg_edge_index
            data.mapped_node_pair = mapped_node_pair
            # data.edge_mask = subg_edge_mask.nonzero().view(-1)
            data.edge_attr = edge_type[subg_edge_mask].view(-1)
            data.y = labels[i].to(torch.int8)
            data.y_reg = targets[i]
            data.x = node_type[subset].view(-1, 1)
            # data.weight = weight
        
            if node_attr is not None:
                # same size(0) as data.x
                assert node_type.size(0) == node_attr.size(0)
                data.node_attr = node_attr[subset]
                data.node_type = node_type[subset].view(-1).to(torch.int8)
                # data.ori_node_index = subset
            return data 
    
    def mp_seal_sampling(self, g, links, labels, targets):
        start = time.perf_counter() 
        # pool = mp.get_context('spawn').Pool(self.num_sampler)
        result_list = []
        argsIter = ArgsIterator(g, links, labels, targets)
        with mp.get_context('spawn').Pool(self.num_sampler) as pool:
        # for i in range(self.num_sampler):
        #     links_for_worker = links[:, i+indices]
        #     labels_for_worker = labels[i+indices]
        #     targets_for_worker = targets[i+indices]
            # result_list.append(pool.apply_async(func=seal_sampling, args=args))
            data_list = list(tqdm(
            # # data_list = \
                pool.imap_unordered(self.seal_sampling, argsIter, chunksize=10), 
                total=labels.size(0), ncols=80))
        return data_list
    """


    def single_g_process(self, idx: int):
        logging.info(f"processing dataset {self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        # we can have multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        logging.info(f"loaded graph {graph}")
        # add neg edges for each graph
        neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)
        if self.add_target_edges:
            # self.data is actually the augmented graph
            aug_graph = add_tar_edges_to_g(graph, neg_edge_index, neg_edge_type)
        else:
            aug_graph = graph
        
        # sample a portion of pos/neg edges
        (
            pos_edge_index, pos_edge_type, pos_edge_y,
            neg_edge_index, neg_edge_type
        ) = get_balanced_edges(
            graph, neg_edge_index, neg_edge_type, 
            self.neg_edge_ratio, self.sample_rates[idx]
        )
        # task node_regression make little modification just links change to nodes,
        # labels are invalide and targets change to g.tar_ndoe_y
        if 'node' in self.task_type:
            links = pos_edge_index.view(-1).unique()
            # Cg for src and dst nodes in pos_edge
            targets = graph.tar_node_y[links].view(-1)
            nonezero_mask = targets > 0.0
            # links = links[nonezero_mask]
            # targets = targets[nonezero_mask]
            labels = labels = torch.tensor(
                [1]*links.size(0)
            )
            logging.info(
                f"DEBUG: task_type:{self.task_type}, "+
                f"target len {targets.size(0)}")
            print("DEBUG: links",links)
            print("DEBUG: node_type",graph.node_type[links])
            print("DEBUG: targets",targets)
            print("DEBUG: target range ", targets.min().item(), targets.max().item())
        # linke prediction task
        else:
            links = torch.cat([pos_edge_index, neg_edge_index], 1)  # [2, Np + Nn]
            # link_types = torch.cat([pos_edge_type, neg_edge_type])  # [Np + Nn]
            # link_types -= link_types.min() # link type in {0,1,2}
            labels = torch.tensor(
                [1]*pos_edge_index.size(1) + [0]*neg_edge_index.size(1)
            )
            targets = torch.cat(
                # Cc of neg edges is set to the minimum value
                (pos_edge_y, torch.ones(neg_edge_type.size())*1e-30),
            )
        # we sample subgraphs for each full graph with multiprocessing
        logging.info(f"Do seal_sampling. pid={os.getpid()} ...")
        data_length = mp_seal_sampling(
            aug_graph, links, labels, targets, 
            self.num_sampler, self.num_hops, self.ratio_per_hop,
            self.processed_paths[idx], 
        ) 
        # load the saved chunks from each process, and concatenate them into one data_list
        data_list = []
        for j in range(self.num_sampler):
            collated_data, collated_slices = torch.load(
                self.processed_paths[idx]+f".stash{j}"
            )
            data_list += collated_data_separate(collated_data, collated_slices)
            os.remove(self.processed_paths[idx]+f".stash{j}")
            del collated_data, collated_slices
        data, slices = self.collate(data_list)
        del data_list

        logging.info('Re-Saving processed entire data_list to %s ...' % self.processed_paths[idx])
        torch.save((data, slices), self.processed_paths[idx])
        del data, slices
        
        return data_length
        # if self.pre_transform is not None:
        #     logging.info("Do pre_transform calculation ...")
        #     data_list = [
        #         self.pre_transform(data, graph.edge_index, graph.num_nodes) 
        #         for data in tqdm(data_list)
        #     ]
        
        # return data_list

    def process(self):
        data_lens_for_split = []
        p = Path(self.processed_dir)
        # we can have multiple graphs
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
                        
            data_lens_for_split.append(
                self.single_g_process(i)
            )
            
        # TODO: need to be changed with different # of test datasets
        self.create_shuffle_split(data_lens_for_split, 0.1, 0.1)

    def create_shuffle_split(self, Ns, val_ratio, test_ratio, num_test=1):
        """ Create a random shuffle split and saves it to disk.

        Args:
            N (list): Total size of the dataset to split.
            val_ratio (float): ratio of the train dataset to be used for validation.
            test_ratio (float): ratio of the train dataset to be used for testing.
            No use when we have multiple SRAM cases.
            num_test (int): the number of dataset that are used as test data
        """
        test_size = 0
        # 2 SRAM cases
        if len(Ns) > 1:
            num_train = len(Ns) - num_test
            assert num_train, f"number of train sets is zero!"
            train_size = np.array(Ns[0:num_train]).sum()
            test_size = np.array(Ns[num_train:]).sum()
            train_ind, val_ind = train_test_split(
                np.arange(train_size), 
                test_size=val_ratio, 
                random_state=123, 
                shuffle=self.shuffle_split,
                #stratify=stratify,
            )
            test_ind = np.arange(train_size, train_size+test_size)
        # 1 SRAM case
        else:
            train_size = Ns[0]
            train_ind, val_ind = train_test_split(
                np.arange(train_size), test_size=val_ratio+test_ratio, 
                random_state=123, 
                shuffle=self.shuffle_split,
                #stratify=stratify,
            )
            val_ind, test_ind = train_test_split(
                val_ind, test_size=test_ratio/(val_ratio+test_ratio), 
                random_state=245, 
                shuffle=self.shuffle_split,
                #stratify=stratify[val_ind],
            )

        shuffle_split = {'train': train_ind, 'val': val_ind, 'test': test_ind}
        logging.info(f"Saving shuffle_split to {self.split_path()}")
        torch.save(shuffle_split, self.split_path())
        return shuffle_split
    
    def get_idx_split(self, split_name):
        """ Get dataset splits.

        Args:
            split_name: Split type: 'shuffle', 'num-atoms'

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        if os.path.exists(self.split_path()):
            split_dict = torch.load(self.split_path())
            return split_dict
        else:
            # temporary implementation
            assert hasattr(self, 'data_lengths')
            return self.create_shuffle_split(list(self.data_lengths.values()), 0.1, 0.1)

    @property
    def raw_file_names(self):
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_type == 'regression' or \
            self.task_type == 'classification':
            return os.path.join(self.root, 'processed')
        elif self.task_type == 'node_regression' or self.task_type == 'node_classification':
            return os.path.join(self.root, 'processed_for_nodes')
        else:
            raise ValueError(f"No defination of task {self.task_type}!")

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            name += f"_hop{self.num_hops}"
            if self.sample_rates[i] < 1.0:
                name += f"_s{self.sample_rates[i]}"
            if self.add_target_edges:
                name += "_aug"
                if self.ratio_per_hop < 1.0:
                    name += f"_p{self.ratio_per_hop:.1f}"
            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names
    
    # @property
    def split_file_name(self):
        file_name = ""
        for data_name, len in self.data_lengths.items():
            file_name += f"{data_name}_L{len}_"
        file_name += f"hop_{self.num_hops}"
        if self.add_target_edges:
            file_name += "_aug"
            if self.ratio_per_hop < 1.0:
                file_name += f"_p{self.ratio_per_hop:.1f}"
        if self.neg_edge_ratio == 1.0:
            file_name += "_split_dict.pt"
        else:
            file_name += f"_nr{self.neg_edge_ratio:.1f}_split_dict.pt"
        return file_name
    
    def split_path(self):
        return os.path.join(self.folder, self.split_file_name())
    
    #TODO: temporary implementation
    @property
    def num_classes(self) -> int:
        if self.task_type == "classification":
            return 2
        elif self.task_type == "node_classification":
            return 4
        else:
            return 'n/a'
    
    def save_pe_codes(self, data_name, start, end, pe_types):
        r""" this method will only be revoked after  
        pre_transform_in_memory to get position encodings.
        """
        # saving encoding for different SRAM cases
        for pe_type in pe_types:
            # graphormer/ER encodings are too large to save
            if "GraphormerBias" == pe_type or \
                "ER" in pe_type:
                continue 

            assert data_name in self.names, \
                f"cannot find {data_name} in dataset {self.names}"
            path = self.processed_paths[
                self.names.index(data_name)
            ].replace('.pt', f'_{pe_type}.pt')

            assert self._data_list is not None, \
                "during saving PE, self._data_list is None!!!"

            # we need to delet other attributes to save memory
            data, slices = self.collate(self._data_list[start:end])
            for key in self._data_list[start].keys:
                if key == 'y' or key == 'mapped_node_pair': continue
                elif pe_type == 'SPD' and key == 'spd': continue
                elif pe_type == 'RWSE' and key == 'pestat_RWSE': continue
                elif pe_type == 'DRNL' and key == 'z': continue
                elif pe_type == 'exp' and key == 'expander_edges': continue
                elif pe_type == 'ERN' and key == 'er_emb': continue
                elif pe_type == 'ERE' and key == 'er_edge': continue
                elif 'Lap' in pe_type and 'Eig' in key: continue
                elif pe_type ==  'SignNet' and 'eig' in key: continue
                elif 'HK' in pe_type and key == 'pestat_HKdiagSE': continue
                elif pe_type == 'ElstaticSE' and key == 'pestat_ElstaticSE': 
                    continue
                else:
                    delattr(data, key)
                    del slices[key]

            logging.info(
                f"Data: {data_name}, pe_type: {pe_type} "+ 
                f"data.keys:{data.keys}, "+ f"saving_path: {path}")
            
            torch.save((data, slices), path)
            del data, slices
            logging.info(
                f"Done Saving! data_len: {self.data_lengths[data_name]}, " + 
                f"data_offsets: {self.data_offsets[data_name]}")
                
        # if self._data_list is not None:
        #     self.data, self.slices = self.collate(self._data_list)
        #     self._data_list = None

    def has_processed_pe(self, pe_types):
        # for debugging
        pe_types_need_to_calculate = []
        assert hasattr(self, 'data_lengths')
        assert hasattr(self, 'data_offsets')
        updated = False
        for i, (data_name, data_len) in enumerate(self.data_lengths.items()):
            pe_types_recom = []
            start = self.data_offsets[data_name]
            end = start + self.data_lengths[data_name]
            self_data0 = self._data_list[start]

            for pe_type in pe_types:
                if pe_type == "CirStatis":
                    assert hasattr(self_data0, 'node_attr'), \
                        f"data must have 'node_attr' when 'CirStatis' encoder is used."
                    assert hasattr(self_data0, 'node_type'), \
                        f"data must have 'node_type' when 'CirStatis' encoder is used."
                    continue
                path = self.processed_paths[i].replace('.pt',f'_{pe_type}.pt')
                if not os.path.exists(path):
                    pe_types_recom.append(pe_type)
                    continue

                loaded_pe_data, loaded_pe_slices = torch.load(path)
                loaded_pe_data0 = collated_data_separate(loaded_pe_data, loaded_pe_slices, idx=0)
                assert self._data_list is not None, "self._data_list is None!!!"
                match_cond = False

                # this is for node sampling scenario
                if self.task_type == "node_regression" or \
                    self.task_type == "node_classification":
                    match_cond = (loaded_pe_data0.mapped_node_pair[0] == \
                                  self_data0.mapped_node_pair[0])
                # this is for link sampling scenario
                else:
                    match_cond = (self_data0.y == loaded_pe_data0.y) and \
                        (loaded_pe_data0.mapped_node_pair[0] == 
                         self_data0.mapped_node_pair[0]) and \
                        (loaded_pe_data0.mapped_node_pair[1] == 
                         self_data0.mapped_node_pair[1])
                
                if match_cond:
                    logging.info(
                        f"updating {data_name} with {pe_type} encoding, " + 
                        f"start index {start}, data_list length {data_len} "+
                        f"load path {path}")
                    self.update_pe(
                        start, data_len, pe_type, loaded_pe_data, loaded_pe_slices)
                    updated = True
                else:
                    pe_types_recom.append(pe_type)
                    logging.info(
                        f"self.data and loaded data are mismatch," + 
                        f"pe {pe_type} needs to be re-computed " +
                        f"{data_name} len {data_len} vs. " + 
                        f"loaded_data len {loaded_pe_data.y.size(0)}, " + 
                        f"node_pair: {loaded_pe_data0.mapped_node_pair} vs. " + 
                        f"{self_data0.mapped_node_pair}, " +
                        f"#nodes: {loaded_pe_data0.num_nodes} vs. {self_data0.num_nodes}, " +
                        f"load path {path}")
            
            pe_types_need_to_calculate.append((data_name, start, end, pe_types_recom))

        # update self.data & self.slices with the loaded data & slices
        if updated:
            pass
        return pe_types_need_to_calculate

    def update_pe(self, start, data_len, pe_type, loaded_data, loaded_slices):
        assert self._data_list is not None, "during updating pe, self._data_list is None!!!"

        for i, data in enumerate(
            tqdm(collated_data_separate(loaded_data, loaded_slices))
        ):
            if i >= data_len: break

            if pe_type == 'DRNL':
                self._data_list[start+i].z = data.z
            elif pe_type == 'LapPE' or pe_type == 'EquivStableLapPE':
                self._data_list[start+i].EigVals = data.EigVals
                self._data_list[start+i].EigVecs = data.EigVecs
            elif pe_type == 'SignNet':
                self._data_list[start+i].eigvals_sn = data.eigvals_sn
                self._data_list[start+i].eigvecs_sn = data.eigvecs_sn
            elif pe_type == 'RWSE':
                self._data_list[start+i].pestat_RWSE = data.pestat_RWSE
            elif pe_type == 'GraphormerBias':
                self._data_list[start+i].in_degrees = data.in_degrees
                self._data_list[start+i].out_degrees = data.out_degrees
            elif pe_type == 'SPD':
                self._data_list[start+i].spd = data.spd
                if hasattr(data, 'shortest_path_etypes'):
                    self._data_list[start+i].shortest_path_etypes = \
                        data.shortest_path_etypes
            # These are for Exphormer
            elif pe_type == 'exp':
                self._data_list[start+i].expander_edges = data.expander_edges
            elif pe_type == 'ERN':
                self._data_list[start+i].er_emb = data.er_emb
            elif pe_type == 'ERE':
                self._data_list[start+i].er_edge = data.er_edge
            elif pe_type == 'dist':
                self._data_list[start+i].dist = data.dist
                self._data_list[start+i].prev_node = data.prev_node
                self._data_list[start+i].prev_edge_id = data.prev_edge_id
                self._data_list[start+i].in_degree = data.in_degree
                self._data_list[start+i].out_degree = data.out_degree
            elif pe_type == 'ElstaticSE':
                self._data_list[start+i].pestat_ElstaticSE = data.pestat_ElstaticSE
            elif 'HK' in pe_type:
                self._data_list[start+i].pestat_HKdiagSE = data.pestat_HKdiagSE
            else:
                raise NotImplementedError(
                    f"PE encoder {pe_type} hasn't been supported in this version")

if __name__ == '__main__':
    dataset = SealSramDataset(
        name='8T_digitized_timing_top_fast', 
        root='/data1/shenshan/dgl/GraphGPS/datasets')
    print("dataset.data", dataset.data)
    print("dataset.graph", dataset.graph)
    print("dataset.links", dataset.links)
    print("dataset.labels", dataset.labels)
    print("dataset.split_idxs", dataset.split_idxs)
    print("dataset._num_ntypes", dataset._num_ntypes)
    print("dataset.num_hops", dataset.num_hops)