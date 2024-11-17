import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from graphgps.transform.posenc_stats import compute_posenc_stats
import math

def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

# added by shan
def pre_encoding_in_memory(dataset, data_pe_tuple, cfg):
    r"""  This function is a substitute for pre_transform_in_memory()
    """
    if len(data_pe_tuple) == 0:
        return
    logging.info(f"Getting data_list...")

    # data_list = [dataset[i] for i in tqdm(range(len(dataset)))]
    # dataset.data = None
    # dataset.slices = None
    # dataset._indices = None

    if dataset._data_list is None or dataset._data_list[-1] is None:
        logging.info("get dataset._data_list for preparing PE encoding")
        dataset._data_list = [dataset[i] for i in tqdm(range(len(dataset)))]
        dataset._data = None
        dataset.slices = None

    # added from Exphormer
    # if 'ERN' in pe_types or 'ERE' in pe_types:
    MaxK = max(
    [
        min(
        math.ceil(data.num_nodes//2), 
        math.ceil(8 * math.log(data.num_edges) / (cfg.posenc_ERN.accuracy**2))
        ) 
        for data in dataset._data_list
    ]
    )
    cfg.posenc_ERN.er_dim = MaxK
    logging.info(f"Choosing ER pos enc dim = {MaxK}")
        
    # if 'dist' in pe_types:
    Max_N = max([data.num_nodes for data in dataset._data_list])
    cfg.prep.max_n = Max_N
    logging.info(f"Choosing dist pos enc max_n = {Max_N}")

    for (data_name, start, end, pe_types) in data_pe_tuple:
        if len(pe_types) == 0:
            continue

        logging.info(f"{data_name}: Computing  {pe_types} ...")
        for i in tqdm(range(start, end),ncols=80):
            compute_posenc_stats(
                dataset._data_list[i], pe_types, 
                dataset._data_list[i].is_undirected(), cfg)
        dataset.save_pe_codes(data_name, start, end, pe_types)

    # data_list = list(filter(None, data_list))
    # if dataset._data_list is not None:
    #     dataset.data, dataset.slices = dataset.collate(dataset._data_list)
    #     dataset._data_list = None

def generate_splits(data, g_split):
    n_nodes = len(data.x)
    train_mask = torch.zeros(n_nodes, dtype=bool)
    valid_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    idx = torch.randperm(n_nodes)
    val_num = test_num = int(n_nodes * (1 - g_split) / 2)
    train_mask[idx[val_num + test_num:]] = True
    valid_mask[idx[:val_num]] = True
    test_mask[idx[val_num:val_num + test_num]] = True
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    return data

def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
