import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Graph Attention Layer"""

    def __init__(self, in_features: int, out_features: int, n_heads: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        # Linear transformation for input features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        nn.init.xavier_uniform_(self.W.data)

        # Attention parameters (one for each direction in the graph)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes] where adj[i,j] = 1 means
                there's an edge from node i to node j

        Returns
        -------
            Output features [batch_size, num_nodes, out_features * n_heads]
        """
        batch_size, N = x.size(0), x.size(1)

        # Linear transformation
        h = torch.matmul(x, self.W)
        h = h.view(batch_size, N, self.n_heads, self.out_features)

        # Prepare source and target representations
        # For each target node j, we only want to attend to its source nodes i
        # where adj[i,j] = 1
        source_nodes = h.repeat(1, 1, N, 1, 1)  # Shape: [batch, N, N, heads, out_features]
        target_nodes = h.repeat(1, N, 1, 1, 1)  # Shape: [batch, N, N, heads, out_features]

        # Compute attention coefficients
        a_input = torch.cat([source_nodes, target_nodes], dim=-1)
        attention = self.leakyrelu(torch.matmul(a_input, self.a))
        attention = attention.squeeze(-1)  # Shape: [batch, N, N, heads]

        # Mask attention scores for directed edges
        # adj.T because we want attention to flow from source to target
        # (if adj[i,j] = 1, node i should attend to node j)
        attention = attention.masked_fill((1 - adj).bool().unsqueeze(-1).repeat(1, 1, 1, self.n_heads), float("-inf"))

        # Apply softmax to source nodes
        # (each target node j attends to its source nodes i where adj[i,j] = 1)
        attention = F.softmax(attention, dim=1)  # Normalize over source dimension
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Compute output features
        # For each target node j, aggregate features from its source nodes i
        h_out = torch.matmul(attention.transpose(1, 2), h)  # Shape: [batch, N, heads, out_features]

        return h_out.view(batch_size, N, -1)


class GAT(nn.Module):
    """Graph Attention Network"""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, n_heads: list[int], dropout: float = 0.6
    ):
        super().__init__()

        # Input layer
        self.input_layer = GATLayer(input_dim, hidden_dims[0], n_heads[0], dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [
                GATLayer(hidden_dims[i] * n_heads[i], hidden_dims[i + 1], n_heads[i + 1], dropout)
                for i in range(len(hidden_dims) - 1)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1] * n_heads[-1], output_dim)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass

        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
                where adj[i,j] = 1 means there's an edge from node i to node j

        Returns
        -------
            predictions: Output predictions [batch_size, num_nodes, output_dim]
            attention_weights: List of attention weights from each layer
        """
        attention_weights = []

        # Input layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.input_layer(x, adj)
        x = F.elu(x)
        attention_weights.append(x)  # Store attention weights

        # Hidden layers
        for layer in self.hidden_layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, adj)
            x = F.elu(x)
            attention_weights.append(x)

        # Output layer
        x = self.output_layer(x)

        return x, attention_weights
