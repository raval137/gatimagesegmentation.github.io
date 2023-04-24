# from typing import Callable, List, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from torch_sparse import spspmm

# from torch_geometric.nn import GATConv, TopKPooling
# from torch_geometric.typing import OptTensor, PairTensor
# from torch_geometric.utils import (
#     add_self_loops,
#     remove_self_loops,
#     sort_edge_index,
# )
# from torch_geometric.utils.repeat import repeat
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import SAGEConv


class GAT(nn.Module):
    # Define a constructor with in_feats, layer_sizes, n_classes, heads, residuals, activation, feat_drops, attn_drops, negative_slopes as inputs
    def __init__(self, in_feats, layer_sizes, n_classes, heads, residuals, activation=F.relu, feat_drops=0.5, attn_drops=0.5, negative_slopes=0.1):
        # Call the constructor of nn.Module
        super().__init__()
        # Create a list of layers
        self.layers = nn.ModuleList()
        # Save the activation function
        self.activation = activation
        # Add the input projection layer to the list of layers
        self.layers.append(GATConv(
            in_feats, layer_sizes[0], heads[0],
            feat_drops, attn_drops, negative_slopes, False, self.activation))
        # Loop over the hidden layers
        for i in range(1, len(layer_sizes)):
            # Add a GATConv layer to the list of layers
            self.layers.append(GATConv(
                layer_sizes[i-1] * heads[i-1], layer_sizes[i], heads[i],
                feat_drops, attn_drops, negative_slopes, residuals[i], self.activation))
        # Add the output projection layer to the list of layers
        self.layers.append(GATConv(
            layer_sizes[-1] * heads[-1], n_classes, 1,
            feat_drops, attn_drops, negative_slopes, False, None))

    # Define the forward pass method of the GAT class with inputs G and inputs
    def forward(self, G, inputs):
            # Save the inputs
            i = inputs
            # Loop over all the layers except the output projection layer
            for l in range(len(self.layers)-1):
                # Apply the lth layer to the input and flatten the output
                i = self.layers[l](G, i).flatten(1)
            # Apply the output projection layer to the flattened output and compute the mean
            logits = self.layers[-1](G, i).mean(1)
            # Return the output
            return logits



def init_graph_net(hp):

    net = GAT(in_feats=hp.in_feats, layer_sizes=hp.layer_sizes, n_classes=hp.out_classes,
              heads=hp.gat_heads, residuals=hp.gat_residuals)
    return net
