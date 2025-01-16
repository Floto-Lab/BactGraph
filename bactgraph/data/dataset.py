from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset


class ExpressionDataset(Dataset):
    """Dataset for protein expression prediction using GAT."""

    def __init__(
        self,
        embeddings_dir: str,
        adjacency_matrix: pd.DataFrame,
        expression_data: pd.DataFrame,
        sample_ids: list[str],
        device: str = "cuda",
    ):
        """
        Initialize the dataset.

        Args:
            embeddings_dir: Directory containing ESM-2 embeddings
            adjacency_matrix: DataFrame containing adjacency matrix
            expression_data: DataFrame containing expression data
            sample_ids: List of sample IDs to use
            device: Device to load tensors to
        """
        self.device = device
        self.embeddings_dir = Path(embeddings_dir)
        self.sample_ids = sample_ids

        # Create networkx graph from adjacency matrix
        # Transpose because we want edges from source to target
        self.graph = nx.from_pandas_adjacency(adjacency_matrix.T, create_using=nx.DiGraph)

        # Store expression data
        self.expression_data = expression_data

        # Create node ordering for consistent tensor creation
        self.node_order = sorted(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_order)}

        # Load a single embedding to get embedding dimension
        sample_embedding = self._load_sample_embedding(sample_ids[0])
        self.embedding_dim = next(iter(sample_embedding.values())).shape[0]

    def _load_sample_embedding(self, sample_id: str) -> dict[str, torch.Tensor]:
        """Load embeddings for a single sample from file"""
        embedding_path = self.embeddings_dir / f"{sample_id}.npy"
        return torch.load(embedding_path)

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

        # Load embeddings for this sample
        embeddings = self._load_sample_embedding(sample_id)

        # Create node feature matrix
        node_features = torch.stack([torch.FloatTensor(embeddings[node]) for node in self.node_order]).to(self.device)

        # Create adjacency matrix tensor
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=self.node_order)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Get expression values for this sample
        expression_values = torch.FloatTensor(
            [self.expression_data.loc[node, sample_id] for node in self.node_order]
        ).to(self.device)

        return node_features, adj_matrix, expression_values
