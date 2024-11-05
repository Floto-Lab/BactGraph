import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset


class ProteinExpressionDataset(Dataset):
    """Dataset for protein expression prediction using GAT."""

    def __init__(
        self, embeddings_path: str, adjacency_matrix_path: str, expression_data_path: str, device: str = "cuda"
    ):
        """
        Dataset for protein expression prediction using GAT.

        Args:
            embeddings_path: Path to ESM-2 embeddings
            adjacency_matrix_path: Path to adjacency matrix
            expression_data_path: Path to expression data
            device: Device to load tensors to
        """
        self.device = device

        self.embeddings = self._load_embeddings(embeddings_path)
        self.graph = self._create_graph(adjacency_matrix_path)
        self.expressions = self._load_expressions(expression_data_path)

        self.sample_ids = list(self.embeddings.keys())

        # Create node ordering for consistent tensor creation
        self.node_order = sorted(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_order)}

    def _load_embeddings(self, path: str) -> dict[str, dict[str, torch.Tensor]]:
        # TODO: Implement loading of ESM-2 embeddings
        raise NotImplementedError

    def _create_graph(self, path: str) -> nx.DiGraph:
        # Read adjacency matrix and create networkx graph
        adj_df = pd.read_csv(path, sep="\t", index_col=0)
        # NB: transpose so that the rows are sources and columns are targets
        G = nx.from_pandas_adjacency(adj_df.T, create_using=nx.DiGraph)
        return G

    def _load_expressions(self, path: str) -> dict[str, dict[str, float]]:
        # TODO: Implement loading of expression data
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item from dataset

        Returns
        -------
            node_features: Tensor of shape [num_nodes, embedding_dim]
            adj_matrix: Tensor of shape [num_nodes, num_nodes]
            expression_values: Tensor of shape [num_nodes]
        """
        sample_id = self.sample_ids[idx]

        # Create node feature matrix
        node_features = torch.stack([self.embeddings[sample_id][node] for node in self.node_order]).to(self.device)

        # Create adjacency matrix tensor
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=self.node_order)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Create expression values tensor
        expression_values = torch.FloatTensor([self.expressions[sample_id][node] for node in self.node_order]).to(
            self.device
        )

        return node_features, adj_matrix, expression_values
