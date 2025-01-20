import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset for protein expression prediction using pre-loaded embeddings."""

    def __init__(
        self,
        embeddings_dict: dict,
        adj_matrix: pd.DataFrame,
        expression_data: pd.DataFrame,
        sample_ids: list[str],
        device: str = "cuda",
    ):
        self.device = device
        self.embeddings = embeddings_dict
        self.adj_matrix = adj_matrix
        self.expression_data = expression_data
        self.sample_ids = sample_ids

        # Create node ordering for consistent tensor creation
        self.node_order = sorted(adj_matrix.index)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_order)}

        # Get embedding dimension from first embedding
        first_embedding = next(iter(embeddings_dict.values()))
        self.embedding_dim = first_embedding.shape[0]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item from dataset.

        Returns
        -------
            Tuple of (node_features, adj_matrix, expression_values)
        """
        sample_id = self.sample_ids[idx]

        # Create node feature matrix using pre-loaded embeddings
        node_features = torch.stack([self.embeddings[node] for node in self.node_order]).to(self.device)

        # Create adjacency matrix tensor
        adj_matrix = torch.FloatTensor(self.adj_matrix.loc[self.node_order, self.node_order].values).to(self.device)

        # Get expression values
        expression_values = torch.FloatTensor(
            [self.expression_data.loc[node, sample_id] for node in self.node_order]
        ).to(self.device)

        return node_features, adj_matrix, expression_values


class ExpressionDataset(Dataset):
    """Dataset for protein expression prediction using GAT."""

    def __init__(
        self,
        embeddings: pd.DataFrame,
        adjacency_matrix: pd.DataFrame,
        expression_data: pd.DataFrame,
        sample_ids: list[str],
        device: str = "cuda",
    ):
        """
        Initialize the dataset.

        Args:
            embeddings: DataFrame containing protein embeddings
            adjacency_matrix: DataFrame containing adjacency matrix
            expression_data: DataFrame containing expression data
            sample_ids: List of sample IDs to use
            device: Device to load tensors to
        """
        self.device = device
        self.embeddings = embeddings
        self.sample_ids = sample_ids

        # Create networkx graph from adjacency matrix
        # Transpose because we want edges from source to target
        self.graph = nx.from_pandas_adjacency(adjacency_matrix.T, create_using=nx.DiGraph)

        # Store expression data
        self.expression_data = expression_data

        # Create node ordering for consistent tensor creation
        self.node_order = sorted(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_order)}

        # Get embedding dimension
        self.embedding_dim = len(embeddings.columns)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item at index.

        Returns
        -------
            Tuple of:
            - node_features: Tensor of shape [num_nodes, embedding_dim]
            - adj_matrix: Tensor of shape [num_nodes, num_nodes]
            - expression_values: Tensor of shape [num_nodes]
        """
        sample_id = self.sample_ids[idx]

        # Get embeddings for all nodes
        node_features = torch.FloatTensor(self.embeddings.loc[self.node_order].values).to(self.device)

        # Create adjacency matrix tensor
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=self.node_order)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Get expression values for this sample
        expression_values = torch.FloatTensor(
            [self.expression_data.loc[node, sample_id] for node in self.node_order]
        ).to(self.device)

        return node_features, adj_matrix, expression_values
