from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from graphgps.encoder.graphormer_encoder import graphormer_pre_processing
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_dense_adj, to_networkx
import networkx
from graphgps.transform.expander_edges import generate_random_expander
from graphgps.transform.dist_transforms import (add_dist_features,
                                       effective_resistance_embedding, 
                                       effective_resistances_from_embedding)
from torch_geometric.data import Data

def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'Graphormer': Computes spatial types and optionally edges along shortest paths.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    # 'DRNL', 'SPD' are added by shan
    # 'exp', 'ERN', 'ERE' are added by Exphormer
    for t in pe_types: 
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE',
                     'HKfullPE', 'ElstaticSE', 'GraphormerBias', 'DRNL', 'SPD', 
                     'exp', 'ERN', 'ERE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    
    if 'exp' in pe_types:
        for j in range(cfg.prep.exp_count):
            generate_random_expander(
                data=data,
                degree=cfg.prep.exp_deg,
                algorithm=cfg.prep.exp_algorithm,
                rng=None,
                max_num_iters=cfg.prep.exp_max_num_iters,
                exp_index=j
            )
    
    if 'ERN' in pe_types or 'ERE' in pe_types:
        assert cfg.posenc_ERN.er_dim > 0, f"Invalid ER_dim={cfg.posenc_ERN.er_dim}"
        effective_resistance_embedding(
            data=data,
            MaxK = cfg.posenc_ERN.er_dim,
            accuracy = cfg.posenc_ERN.accuracy,
            which_method = 0
        )
        effective_resistances_from_embedding(
            data=data, normalize_per_node=False
        )
    if 'dist' in pe_types:
        add_dist_features(
            data,
            max_n = cfg.prep.max_n,
            is_undirected = is_undirected,
            cutoff = cfg.prep.dist_cutoff
        )
    
    # drnl is calculated here,  added by shan
    if 'DRNL' in pe_types:
        adj = csr_matrix(to_dense_adj(data.edge_index).squeeze())
        data.z = drnl_node_labeling(
            adj, data.mapped_node_pair[0], data.mapped_node_pair[1]
        ).view(-1, 1)

    # spd encoding is added by shan
    if 'SPD' in pe_types:
        data.spd = get_dspd(
            N, data.edge_index, data.mapped_node_pair.tolist(), cfg)

    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    if 'GraphormerBias' in pe_types:
        data = graphormer_pre_processing(
            data,
            cfg.posenc_GraphormerBias.num_spatial_types
        )

    return data


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(
        adj_wo_dst, directed=False, unweighted=True, indices=src
    )
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = (
        torch.div(dist, 2, rounding_mode="floor"),
        dist % 2,
    )

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    # shortest path may include inf values
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)

def get_dspd(N, edge_index, nodes, cfg):
    max_dist = cfg.posenc_SPD.max_dist
    graph: networkx.DiGraph = to_networkx(Data(edge_index=edge_index, num_nodes=N))

    # assert data.mapped_node_pair.size(0) == 2
    
    # Calculate the shortest path of mapped src & dst nodes as the start node.
    # This function return a dict {node_id: list[node_id0, node_id1, ...]},
    # which is indexed by node id that has a shortest path to
    # the mapped src & dst nodes node.
    shortest_paths = [
        networkx.shortest_path(graph, source=node)
        for node in nodes
    ]
    # spatial_types is the tensor of shortest path distance,
    # each row is for 1 node in subgraph, 
    # dim 0 is SPD to src node dim 1 is SPD to dst node
    spatial_types = torch.empty((N, 2), dtype=torch.long).fill_(cfg.posenc_SPD.max_dist)

    # if hasattr(data, "edge_attr") and data.edge_attr is not None:
    #     # dim 0 must be N for dataset.collate()
    #     shortest_path_etypes = torch.zeros((N, 2, max_dist), dtype=torch.int8)
    #     edge_attr = torch.zeros(N, N, dtype=torch.int8)
    #     # the edge_attr starts from 0 but we use 0 represent non-existing edges
    #     edge_attr[data.edge_index[0], data.edge_index[1]] = data.edge_attr + 1

    # j is the mapped nodes (src or dst node)
    for j, shortest_path in enumerate(shortest_paths):
        # i is the end node of shortest path
        for i, path in shortest_path.items():
            if len(path) > max_dist:
                path = path[:max_dist]
            assert len(path) >= 1
            # len(path) is #nodes in the path, so len(path) -1 is the path length
            spatial_types[i, j] = len(path) - 1 

            # if len(path) > 1 and hasattr(data, "edge_attr") and data.edge_attr is not None:
            #     # find the edge types on the path
            #     path_attr = [
            #         edge_attr[path[k], path[k + 1]] 
            #         for k in range(len(path) - 1)  # len(path) * (num_edge_types)
            #     ]
            #     shortest_path_etypes[i, j, :len(path)-1] = torch.tensor(
            #         path_attr, dtype=torch.int8)
    
    # Once we got 1 node center, same distance value at dim 0 and 1 of spatial_types
    if len(nodes) == 1:
        spatial_types[:,1] = spatial_types[:,0]

    # data.spd = spatial_types
    return spatial_types
    # if hasattr(data, "edge_attr"):
    #     data.shortest_path_etypes = shortest_path_etypes#.reshape(N*2, max_dist)

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs
