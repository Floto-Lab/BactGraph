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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        sample_id = self.sample_ids[idx]

        # node feature matrices for regulators and targets
        regulator_features = torch.stack([self.embeddings[node] for node in self.regulator_order]).to(
            self.device
        )  # [num_regulators, embedding_dim]

        target_features = torch.stack([self.embeddings[node] for node in self.target_order]).to(
            self.device
        )  # [num_targets, embedding_dim]

        # adjacency matrix tensor [num_regulators, num_targets]
        adj_matrix = torch.FloatTensor(self.adj_matrix.loc[self.regulator_order, self.target_order].values).to(
            self.device
        )

        # expression values for target genes [num_targets]
        target_expression = torch.FloatTensor(
            [self.expression_data.loc[node, sample_id] for node in self.target_order]
        ).to(self.device)

        # Add batch dimension of 1 to all tensors
        regulator_features = regulator_features.unsqueeze(0)  # [1, num_regulators, embedding_dim]
        target_features = target_features.unsqueeze(0)  # [1, num_targets, embedding_dim]
        adj_matrix = adj_matrix.unsqueeze(0)  # [1, num_regulators, num_targets]
        target_expression = target_expression.unsqueeze(0)  # [1, num_targets]

        if len(regulator_features.shape) > 3:
            regulator_features = regulator_features.reshape(1, regulator_features.size(1), -1)
        if len(target_features.shape) > 3:
            target_features = target_features.reshape(1, target_features.size(1), -1)

        return (regulator_features, target_features), adj_matrix, target_expression
