import torch
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels=[256, 64]):
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: GATConv(in_channels, hidden_channels[0], heads=2, concat=True) 
                for edge_type in metadata[1]
        }, aggr='sum')
        self.conv2 = HeteroConv({
            edge_type: GATConv(2*hidden_channels[0], hidden_channels[1], heads=2, concat=True) 
                for edge_type in metadata[1]
        }, aggr='sum')
        self.lin = Linear(hidden_channels[1] * 2, 1)  # Output layer for pairwise classification 

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        node_embeddings = x_dict['node']
        
        i, j = torch.meshgrid(torch.arange(node_embeddings.size(0)), torch.arange(node_embeddings.size(0)), indexing='ij')
        node_pairs = node_embeddings[i.flatten()] + node_embeddings[j.flatten()]
        return self.lin(node_pairs)