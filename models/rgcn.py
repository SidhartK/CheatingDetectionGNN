import torch
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, Linear


class RelationalGCN(torch.nn.Module):
    def __init__(self, num_relations, in_channels, hidden_channels=[256, 64]):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels[0], num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv4 = RGCNConv(hidden_channels[0], hidden_channels[1], num_relations=num_relations)
        self.lin = Linear(hidden_channels[1], 1)  # Output layer for pairwise classification

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        node_embeddings = self.conv4(x, edge_index, edge_type)

        i, j = torch.meshgrid(torch.arange(node_embeddings.size(0)), torch.arange(node_embeddings.size(0)), indexing='ij')
        node_pairs = node_embeddings[i.flatten()] + node_embeddings[j.flatten()]
        return self.lin(node_pairs)
