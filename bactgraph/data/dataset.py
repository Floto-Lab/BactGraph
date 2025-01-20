import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset class for gene embeddings and expression data."""

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

        # Create separate orderings for regulators and targets
        self.regulator_order = sorted(adj_matrix.index)
        self.target_order = sorted(adj_matrix.columns)
        self.regulator_to_idx = {node: idx for idx, node in enumerate(self.regulator_order)}
        self.target_to_idx = {node: idx for idx, node in enumerate(self.target_order)}

        # Get embedding dimension from first embedding
        first_embedding = next(iter(embeddings_dict.values()))
        self.embedding_dim = first_embedding.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item from dataset.

        Returns
        -------
            Tuple of (node_features, adj_matrix, expression_values)
        """
        sample_id = self.sample_ids[idx]

        # Create node feature matrices for both regulators and targets
        regulator_features = torch.stack([self.embeddings[node] for node in self.regulator_order]).to(self.device)
        target_features = torch.stack([self.embeddings[node] for node in self.target_order]).to(self.device)

        # Create asymmetric adjacency matrix tensor
        adj_matrix = torch.FloatTensor(self.adj_matrix.loc[self.regulator_order, self.target_order].values).to(
            self.device
        )

        # Get expression values for target genes
        target_expression = torch.FloatTensor(
            [self.expression_data.loc[node, sample_id] for node in self.target_order]
        ).to(self.device)

        return (regulator_features, target_features), adj_matrix, target_expression
