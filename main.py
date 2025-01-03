import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, Linear
import time
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter

heterogat = True
rgcn = True
num_epochs = 250
print_every = 10

# Initialize TensorBoard writers
writer_hetero = SummaryWriter(log_dir='runs/heterogat')
writer_rgcn = SummaryWriter(log_dir='runs/rgcn')

# 1. Heterogeneous GNN with GAT
class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels=[256, 64]):
    """
    Initializes the HeteroGAT model with two heterogeneous graph attention convolutional layers
    and a linear layer for pairwise classification.

    Args:
        metadata (tuple): Contains metadata about types of nodes and edges in the heterogeneous graph.
        in_channels (int): Number of input features for each node.
        hidden_channels (list): List containing the number of hidden units for each layer.
    """
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
        """
        Forward pass of the HeteroGAT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.

        Returns:
            torch.Tensor: Output of the model for pairwise classification.

        Summary:
            Performs two heterogeneous graph attention convolutional layers followed by a linear layer for pairwise classification.
        """
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        node_embeddings = x_dict['node']
        
        i, j = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        node_pairs = node_embeddings[i.flatten()] + node_embeddings[j.flatten()]
        return self.lin(node_pairs)

# 2. Relational GCN
class RelationalGCN(torch.nn.Module):
    def __init__(self, num_relations, in_channels, hidden_channels=[256, 64]):
        """
        __init__ method of RelationalGCN class.

        Args:
            num_relations (int): Number of edge types in the graph.
            in_channels (int): Number of input features for each node.
            hidden_channels (list): List containing the number of hidden units for each layer.

        Summary:
            Initializes the RelationalGCN model with the given number of relations, in_channels, and hidden_channels.
        """

        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels[0], num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv4 = RGCNConv(hidden_channels[0], hidden_channels[1], num_relations=num_relations)
        self.lin = Linear(hidden_channels[1], 1)  # Output layer for pairwise classification

    def forward(self, x, edge_index, edge_type):
        """
        Forward pass of the RelationalGCN model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_type (torch.Tensor): Edge types.

        Returns:
            torch.Tensor: Output of the model for pairwise classification.

        Summary:
            Performs four relational graph convolutional layers followed by a linear layer for pairwise classification.
        """
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        node_embeddings = self.conv4(x, edge_index, edge_type)

        i, j = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        node_pairs = node_embeddings[i.flatten()] + node_embeddings[j.flatten()]
        return self.lin(node_pairs)

def evaluate_model(model, data, labels, loss_func, metadata, dict_format=False):
    """
    Evaluates the given model on the given data and labels using the given loss function.

    Args:
        model (nn.Module): The model to evaluate.
        data (HeteroData): The data to evaluate on.
        labels (torch.Tensor): The labels to evaluate against.
        loss_func (nn.Module): The loss function to use.
        metadata (HeteroData.metadata()): The metadata of the data.
        dict_format (bool): Whether the data is in dictionary format or not.

    Returns:
        tuple: (loss, accuracy, precision, recall, f1, roc_auc)

    Summary:
        Evaluates the given model on the given data and labels using the given loss function and returns evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        # Make predictions on the data
        if dict_format:
            predictions = model(
                x_dict={'node': data['node'].x},
                edge_index_dict={
                    edge_type: data[edge_type].edge_index
                        for edge_type in metadata[1]
                },
            )
        else:
            predictions = model(
                x=data['node'].x,
                edge_index=torch.cat(
                    [data[edge_type].edge_index for edge_type in metadata[1]], dim=1
                ),
                edge_type=torch.cat(
                    [
                        torch.full((data[edge_type].edge_index.size(1),), i, dtype=torch.long)
                            for i, edge_type in enumerate(metadata[1])
                    ]
                ),
            )
        # Calculate loss
        loss = loss_func(predictions, labels)
        # Calculate accuracy, precision, recall, f1, and roc_auc
        probs = torch.sigmoid(predictions).flatten()
        preds_binary = (probs > 0.5).long()
        true_labels = labels[:, 2].long()
        correct = (preds_binary == true_labels).sum().item()
        total = true_labels.size(0)
        accuracy = correct / total
        true_positive = ((preds_binary == 1) & (true_labels == 1)).sum().item()
        false_positive = ((preds_binary == 1) & (true_labels == 0)).sum().item()
        false_negative = ((preds_binary == 0) & (true_labels == 1)).sum().item()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = true_labels[sorted_indices]
        sorted_probs = probs[sorted_indices]
        tpr = torch.cumsum(sorted_labels, dim=0) / sorted_labels.sum()
        fpr = torch.cumsum(1 - sorted_labels, dim=0) / (1 - sorted_labels).sum()
        roc_auc = torch.trapz(tpr, fpr).item() if tpr.numel() > 1 else 0.0
        return loss.item(), accuracy, precision, recall, f1, roc_auc
    
if __name__ == '__main__':
    with open("data/data.pkl", "rb") as f:
        data, labels = pickle.load(f)

    # Extract useful data from features
    num_nodes = data['node'].x.size(0)
    in_channels = data['node'].x.size(1)
    metadata = data.metadata()
    # Initialize the HeteroGAT and RelationalGCN model with the metadata and input channels
    hetero_model = HeteroGAT(metadata, in_channels=in_channels)
    # Initialize the RelationalGCN model with the number of relations and input channels
    relational_model = RelationalGCN(num_relations=len(metadata[1]), in_channels=in_channels)

    # Define the heuristic and team loss criteria with the positive weight
    heuristic_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(25.0))
    team_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
    # Define a custom train loss function
    def train_loss_func(pred, target, reg_param=0.1):
        heuristic_rec_loss = heuristic_criterion(pred.flatten(), target[:, 2])
        team_rec_loss = team_criterion(pred.flatten(), target[:, 1])
        return (heuristic_rec_loss * reg_param) + team_rec_loss

    # Define custom test loss function
    test_criterion = torch.nn.BCEWithLogitsLoss()
    def test_loss_func(pred, target, reg_param=0.1):
        label_rec_loss = test_criterion(pred.flatten(), target[:, 2])
        return label_rec_loss

    # Initialize optimizers for both models
    optimizer_hetero = torch.optim.Adam(hetero_model.parameters(), lr=0.01)
    optimizer_relational = torch.optim.Adam(relational_model.parameters(), lr=0.01)

    # Train the HeteroGAT model if specified
    if heterogat:
        print("-" * 50)
        start = time.time()
        for epoch in range(num_epochs):
            verbose = (epoch + 1) % print_every == 0 
            hetero_model.train()
            optimizer_hetero.zero_grad()
            predictions = hetero_model(
                x_dict={'node': data['node'].x},
                edge_index_dict={
                    edge_type: data[edge_type].edge_index
                        for edge_type in metadata[1]
                },
            )
            loss = train_loss_func(predictions, labels)
            loss.backward()
            optimizer_hetero.step()
            if verbose:
                print(f"HeteroGAT Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                test_loss, accuracy, precision, recall, f1, roc_auc = evaluate_model(
                    hetero_model, data, labels, test_loss_func, metadata, dict_format=True
                )
                print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
                writer_hetero.add_scalar('Loss/train', loss.item(), epoch + 1)
                writer_hetero.add_scalar('Loss/test', test_loss, epoch + 1)
                writer_hetero.add_scalar('Metrics/Accuracy', accuracy, epoch + 1)
                writer_hetero.add_scalar('Metrics/Precision', precision, epoch + 1)
                writer_hetero.add_scalar('Metrics/Recall', recall, epoch + 1)
                writer_hetero.add_scalar('Metrics/F1-Score', f1, epoch + 1)
                writer_hetero.add_scalar('Metrics/ROC-AUC', roc_auc, epoch + 1)
        print(f"Time taken: {time.time() - start:.2f}s")
        print("-" * 50)

    # Train the RelationalGCN model if specified
    if rgcn:
        print("-" * 50)
        start = time.time()
        for epoch in range(num_epochs):
            verbose = (epoch + 1) % print_every == 0
            relational_model.train()
            optimizer_relational.zero_grad()
            predictions = relational_model(
                x=data['node'].x,
                edge_index=torch.cat(
                    [data[edge_type].edge_index for edge_type in metadata[1]], dim=1
                ),
                edge_type=torch.cat(
                    [
                        torch.full((data[edge_type].edge_index.size(1),), i, dtype=torch.long)
                            for i, edge_type in enumerate(metadata[1])
                    ]
                ),
            )
            loss = train_loss_func(predictions, labels)
            loss.backward()
            optimizer_relational.step()
            if verbose:
                print(f"RelationalGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                test_loss, accuracy, precision, recall, f1, roc_auc = evaluate_model(
                    relational_model, data, labels, test_loss_func, metadata
                )
                print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
                writer_rgcn.add_scalar('Loss/train', loss.item(), epoch + 1)
                writer_rgcn.add_scalar('Loss/test', test_loss, epoch + 1)
                writer_rgcn.add_scalar('Metrics/Accuracy', accuracy, epoch + 1)
                writer_rgcn.add_scalar('Metrics/Precision', precision, epoch + 1)
                writer_rgcn.add_scalar('Metrics/Recall', recall, epoch + 1)
                writer_rgcn.add_scalar('Metrics/F1-Score', f1, epoch + 1)
                writer_rgcn.add_scalar('Metrics/ROC-AUC', roc_auc, epoch + 1)
        print(f"Time taken: {time.time() - start:.2f}s")
        print("-" * 50)

    # Close the TensorBoard writers
    writer_hetero.close()
    writer_rgcn.close()
