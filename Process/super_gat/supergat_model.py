import torch
from torch import nn
from supergat import SuperGAT
from torch_geometric.data import DataLoader
from typing import Tuple, List
import torch.nn.functional as F
from utils import get_args

def getattr_d(dataset_or_loader, name):
    if isinstance(dataset_or_loader, DataLoader):
        return getattr(dataset_or_loader.dataset, name)
    else:
        return getattr(dataset_or_loader, name)

def _get_gat_cls(attention_name: str):
    if attention_name in ["GAT", "GATPPI", "LargeGAT"]:
        return SuperGAT
    else:
        raise ValueError("{} is not proper name".format(attention_name))



class TdSuperGATNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        gat_cls = _get_gat_cls(self.args.model_name)

        # num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        # num_classes = getattr_d(dataset_or_loader, "num_classes")
        args = get_args()
        self.conv1 = gat_cls(
            input_dim, args.hidden_dim,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        self.conv2 = gat_cls(
            args.num_hidden_features * args.heads, out_dim,
            heads=(args.out_heads or args.heads), dropout=args.dropout, concat=False,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        print(next(self.modules()))

    def forward_for_all_layers(self, x, edge_index, batch=None, **kwargs):
        x1 = F.dropout(x, p=self.args.dropout, training=self.training)
        x1 = self.conv1(x1, edge_index, batch=batch, **kwargs)
        x2 = F.elu(x1)
        x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        x2 = self.conv2(x2, edge_index, batch=batch, **kwargs)
        return x1, x2

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index, batch=batch, **kwargs)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index, batch=batch, **kwargs)

        if self.training and self.args.verbose >= 3:
            _inspect_attention_tensor(x, edge_index, self.conv2.cache)

        return x

    def set_layer_attrs(self, name, value):
        setattr(self.conv1, name, value)
        setattr(self.conv2, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        return [
            self.conv1.get_attention_dist(edge_index, num_nodes),
            self.conv2.get_attention_dist(edge_index, num_nodes),
        ]

def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res["att_with_negatives"] is not None \
            and (num_pos_samples == 13264 or
                 num_pos_samples == 12431 or
                 num_pos_samples == 0):

        att_with_negatives = att_res["att_with_negatives"].mean(dim=-1)
        att_with_negatives_cloned = att_with_negatives.clone()
        att_with_negatives_cloned = torch.sigmoid(att_with_negatives_cloned)

        pos_samples = att_with_negatives_cloned[:num_pos_samples]
        neg_samples = att_with_negatives_cloned[num_pos_samples:]


        pos_m, pos_s = float(pos_samples.mean()), float(pos_samples.std())
        print("TPos: {} +- {} ({})".format(pos_m, pos_s, pos_samples.size()), "blue")
        neg_m, neg_s = float(neg_samples.mean()), float(neg_samples.std())
        print("TNeg: {} +- {} ({})".format(neg_m, neg_s, neg_samples.size()), "blue")
        return True
    else:
        return False

class BuSuperGATNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        gat_cls = _get_gat_cls(self.args.model_name)

        # num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        # num_classes = getattr_d(dataset_or_loader, "num_classes")
        args = get_args()
        self.conv1 = gat_cls(
            input_dim, args.hidden_dim,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        self.conv2 = gat_cls(
            args.num_hidden_features * args.heads, out_dim,
            heads=(args.out_heads or args.heads), dropout=args.dropout, concat=False,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        print(next(self.modules()))

    def forward_for_all_layers(self, x, edge_index, batch=None, **kwargs):
        x1 = F.dropout(x, p=self.args.dropout, training=self.training)
        x1 = self.conv1(x1, edge_index, batch=batch, **kwargs)
        x2 = F.elu(x1)
        x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        x2 = self.conv2(x2, edge_index, batch=batch, **kwargs)
        return x1, x2

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index, batch=batch, **kwargs)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index, batch=batch, **kwargs)

        if self.training and self.args.verbose >= 3:
            _inspect_attention_tensor(x, edge_index, self.conv2.cache)

        return x

    def set_layer_attrs(self, name, value):
        setattr(self.conv1, name, value)
        setattr(self.conv2, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        return [
            self.conv1.get_attention_dist(edge_index, num_nodes),
            self.conv2.get_attention_dist(edge_index, num_nodes),
        ]

def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res["att_with_negatives"] is not None \
            and (num_pos_samples == 13264 or
                 num_pos_samples == 12431 or
                 num_pos_samples == 0):

        att_with_negatives = att_res["att_with_negatives"].mean(dim=-1)
        att_with_negatives_cloned = att_with_negatives.clone()
        att_with_negatives_cloned = torch.sigmoid(att_with_negatives_cloned)

        pos_samples = att_with_negatives_cloned[:num_pos_samples]
        neg_samples = att_with_negatives_cloned[num_pos_samples:]


        pos_m, pos_s = float(pos_samples.mean()), float(pos_samples.std())
        print("TPos: {} +- {} ({})".format(pos_m, pos_s, pos_samples.size()), "blue")
        neg_m, neg_s = float(neg_samples.mean()), float(neg_samples.std())
        print("TNeg: {} +- {} ({})".format(neg_m, neg_s, neg_samples.size()), "blue")
        return True
    else:
        return False
