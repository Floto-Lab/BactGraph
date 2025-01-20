import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingGATLayer(nn.Module):
    """Graph Attention Layer"""

    def __init__(self, in_features: int, out_features: int, n_heads: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        # Transform node features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        nn.init.xavier_uniform_(self.W.data)

        # Attention mechanism (single vector per head)
        self.att = nn.Parameter(torch.zeros(size=(1, n_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.att.data)

        # Learnable weight for self-connection
        self.self_weight = nn.Parameter(torch.ones(1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit message passing.

        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
                where adj[i,j] = 1 means there's an edge from i to j

        Returns
        -------
            Node features [batch_size, num_nodes, out_features * n_heads]
        """
        batch_size, N = x.size(0), x.size(1)

        # Transform node features for all heads
        # [batch_size, num_nodes, n_heads, out_features]
        h = torch.matmul(x, self.W).view(batch_size, N, self.n_heads, self.out_features)

        # Compute attention scores
        # First, prepare pairs of nodes for attention
        h_i = h.unsqueeze(-2).expand(-1, -1, N, -1, -1)  # Source nodes
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # Target nodes

        # Concatenate source and target features
        pair_features = torch.cat((h_i, h_j), dim=-1)  # [batch, num_nodes, num_nodes, n_heads, 2*out_features]

        # Compute attention scores using attention weights
        # [batch, num_nodes, num_nodes, n_heads]
        attention = self.leakyrelu(torch.einsum("bnmhf,hf->bnmh", pair_features, self.att.squeeze(0)))

        # Mask attention scores using adjacency matrix
        # We mask before softmax to ensure attention flows only along edges
        mask = adj.unsqueeze(-1) == 0  # Add head dimension
        attention = attention.masked_fill(mask, float("-inf"))

        # Apply softmax to normalize attention scores
        attention = F.softmax(attention, dim=2)  # Normalize over source nodes
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention scores to compute messages
        # [batch, num_nodes, n_heads, out_features]
        messages = torch.einsum("bnmh,bnhf->bnhf", attention, h)

        # Add weighted self-connection
        self_message = h * self.self_weight
        messages = messages + self_message

        # Reshape to final output dimension
        return messages.reshape(batch_size, N, -1)


class GAT(nn.Module):
    """Graph Attention Network"""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, n_heads: list[int], dropout: float = 0.6
    ):
        super().__init__()

        # Input layer
        self.input_layer = MessagePassingGATLayer(input_dim, hidden_dims[0], n_heads[0], dropout)

        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                MessagePassingGATLayer(hidden_dims[i] * n_heads[i], hidden_dims[i + 1], n_heads[i + 1], dropout)
            )
        self.hidden_layers = nn.ModuleList(layers)

        # Output layer (combines all heads into single prediction)
        self.output_layer = nn.Linear(hidden_dims[-1] * n_heads[-1], output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
                where adj[i,j] = 1 means there's an edge from i to j

        Returns
        -------
            predictions: Output predictions [batch_size, num_nodes, output_dim]
            attention_weights: List of attention weights from each layer
        """
        attention_weights = []

        # Input layer
        x = self.dropout(x)
        x = self.input_layer(x, adj)
        x = F.elu(x)
        attention_weights.append(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer(x, adj)
            x = F.elu(x)
            attention_weights.append(x)

        # Output layer
        predictions = self.output_layer(x)

        return predictions, attention_weights
