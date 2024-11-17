import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

# adding external weights, by shan

@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true, weight=None):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy' and weight is None:
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)
    # added by shan
    elif weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            pred, true.float(), weight=weight
        )
        return loss, torch.sigmoid(pred)
