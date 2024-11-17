import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from graphgps.encoder.ast_encoder import ASTNodeEncoder
from graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder, \
    HKdiagSENodeEncoder, ElstaticSENodeEncoder
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.ppa_encoder import PPANodeEncoder
from graphgps.encoder.signnet_pos_encoder import SignNetNodeEncoder
from graphgps.encoder.voc_superpixels_encoder import VOCNodeEncoder
from graphgps.encoder.type_dict_encoder import TypeDictNodeEncoder
from graphgps.encoder.drnl_pos_encoder import DrnlNodeEncoder, ShortestPathDistEncoder, CircuitStatisticEncoder # added by shan
from graphgps.encoder.linear_node_encoder import LinearNodeEncoder
from graphgps.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from graphgps.encoder.graphormer_encoder import GraphormerEncoder
from graphgps.encoder.ER_node_encoder import ERNodeEncoder
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()
            
            if cfg.posenc_EquivStableLapPE.enable: # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
                self.encoder1 = self.enc1_cls(dim_emb)
                self.encoder2 = self.enc2_cls(dim_emb)
            else:
                # PE dims can only be gathered once the cfg is loaded.
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
                # 1st encoder is dataset encoder 2nd encoder is position encoder, 
                # total dim is dim_emb, which is cfg.gt.dim_hidden/cfg/gnn/dim_inner
                self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
                self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            return batch
    
    class Concat4NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None
        enc4_cls = None
        enc4_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            enc4_dim_pe = getattr(cfg, f"posenc_{self.enc4_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe - enc4_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe - enc4_dim_pe, expand_x=False)
            self.encoder3 = self.enc3_cls(dim_emb - enc4_dim_pe, expand_x=False)
            self.encoder4 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            batch = self.encoder4(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    elif len(encoder_classes) == 4:
        Concat4NodeEncoder.enc1_cls = encoder_classes[0]
        Concat4NodeEncoder.enc2_cls = encoder_classes[1]
        Concat4NodeEncoder.enc3_cls = encoder_classes[2]
        Concat4NodeEncoder.enc4_cls = encoder_classes[3]
        Concat4NodeEncoder.enc2_name = pe_enc_names[0]
        Concat4NodeEncoder.enc3_name = pe_enc_names[1]
        Concat4NodeEncoder.enc4_name = pe_enc_names[2]
        return Concat4NodeEncoder
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")


# Dataset-specific node encoders.
ds_encs = {'Atom': AtomEncoder,
           'ASTNode': ASTNodeEncoder,
           'PPANode': PPANodeEncoder,
           'TypeDictNode': TypeDictNodeEncoder,
           'VOCNode': VOCNodeEncoder,
           'LinearNode': LinearNodeEncoder,
           'CirStatis': CircuitStatisticEncoder,}

# Positional Encoding node encoders.
pe_encs = {'LapPE': LapPENodeEncoder,
           'RWSE': RWSENodeEncoder,
           'HKdiagSE': HKdiagSENodeEncoder,
           'ElstaticSE': ElstaticSENodeEncoder,
           'SignNet': SignNetNodeEncoder,
           'EquivStableLapPE': EquivStableLapPENodeEncoder,
           'GraphormerBias': GraphormerEncoder,
           # added by shan
           'DRNL': DrnlNodeEncoder, 
           'SPD': ShortestPathDistEncoder,
           # added by Exphormer
           'ERN': ERNodeEncoder,
           'ERE': EREdgeEncoder,
        }

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

# Combine both DRNL and SPD graph encodings, by shan
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+DRNL+SPD",
        concat_node_encoders([ds_enc_cls, DrnlNodeEncoder, ShortestPathDistEncoder],
                             ['DRNL', 'SPD'])
    )
# Combine SPD with RWSE graph encodings, by shan
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SPD+RWSE",
        concat_node_encoders([ds_enc_cls, ShortestPathDistEncoder, RWSENodeEncoder],
                             ['SPD', 'RWSE'])
    )

# This is only for node regression for Cg predicting, by shan
register_node_encoder(
    f"TypeDictNode+CirStatis+SPD",
    concat_node_encoders([TypeDictNodeEncoder, CircuitStatisticEncoder, ShortestPathDistEncoder],
                         ['CirStatis','SPD'])
)
register_node_encoder(
    f"TypeDictNode+CirStatis",
    concat_node_encoders([TypeDictNodeEncoder, CircuitStatisticEncoder],
                         ['CirStatis'])
)
# This is only for node regression for Cg predicting, by shan
register_node_encoder(
    f"TypeDictNode+CirStatis+SPD+RWSE",
    concat_node_encoders([TypeDictNodeEncoder, CircuitStatisticEncoder, ShortestPathDistEncoder, RWSENodeEncoder],
                         ['CirStatis', 'SPD', 'RWSE'])
)
# For Exphormer
register_node_encoder(
    f"TypeDictNode+ERN+ERE",
    concat_node_encoders([TypeDictNodeEncoder, ERNodeEncoder, EREdgeEncoder],
                         ['ERN', 'ERE'])
)

# Combine both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )

# Combine both SignNet and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SignNet+RWSE",
        concat_node_encoders([ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder],
                             ['SignNet', 'RWSE'])
    )

# Combine GraphormerBias with LapPE or RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+LapPE",
        concat_node_encoders([ds_enc_cls, GraphormerEncoder, LapPENodeEncoder],
                             ['GraphormerBias', 'LapPE'])
    )
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+RWSE",
        concat_node_encoders([ds_enc_cls, GraphormerEncoder, RWSENodeEncoder],
                             ['GraphormerBias', 'RWSE'])
    )

# # Combine GraphormerBias with DSPD positional encodings.
# for ds_enc_name, ds_enc_cls in ds_encs.items():
#     register_node_encoder(
#         f"{ds_enc_name}+GraphormerBias+SPD",
#         concat_node_encoders([ds_enc_cls, GraphormerEncoder, ShortestPathDistEncoder],
#                              ['GraphormerBias', 'SPD'])
#     )