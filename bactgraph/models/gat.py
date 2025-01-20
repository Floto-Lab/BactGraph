import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingGATLayer(nn.Module):
    """Graph Attention Layer for directed graph"""

    def __init__(
        self,
        regulator_dim: int,
        target_dim: int,
        out_features: int,
        n_heads: int,
        dropout: float = 0.6,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.regulator_dim = regulator_dim
        self.target_dim = target_dim
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        # Transform node features for regulators and targets
        self.W_reg = nn.Parameter(torch.zeros(size=(regulator_dim, out_features * n_heads)))
        self.W_tgt = nn.Parameter(torch.zeros(size=(target_dim, out_features * n_heads)))
        nn.init.xavier_uniform_(self.W_reg.data)
        nn.init.xavier_uniform_(self.W_tgt.data)

        # Attention mechanism (single vector per head)
        self.att = nn.Parameter(torch.zeros(size=(1, n_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.att.data)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit message passing.

        Args:
            x: Tuple of (regulator_features, target_features)
                regulator_features: [batch_size, num_regulators, regulator_dim]
                target_features: [batch_size, num_targets, target_dim]
            adj: Adjacency matrix [batch_size, num_regulators, num_targets]
                where adj[i,j] = 1 means there's an edge from regulator i to target j

        Returns
        -------
            Updated target features [batch_size, num_targets, out_features * n_heads]
        """
        regulator_features, target_features = x
        batch_size = regulator_features.size(0)
        num_regulators = regulator_features.size(1)
        num_targets = target_features.size(1)

        # Transform features
        h_reg = torch.matmul(regulator_features, self.W_reg).view(batch_size, num_regulators, self.n_heads, -1)
        h_tgt = torch.matmul(target_features, self.W_tgt).view(batch_size, num_targets, self.n_heads, -1)

        # Compute attention scores between regulators and targets
        h_reg_expanded = h_reg.unsqueeze(2)  # [batch, num_regulators, 1, heads, features]
        h_tgt_expanded = h_tgt.unsqueeze(1)  # [batch, 1, num_targets, heads, features]

        # Concatenate along feature dimension
        pair_features = torch.cat(
            (h_reg_expanded.expand(-1, -1, num_targets, -1, -1), h_tgt_expanded.expand(-1, num_regulators, -1, -1, -1)),
            dim=-1,
        )

        # Compute attention scores
        attention = self.leakyrelu(torch.einsum("bnmhf,hf->bnmh", pair_features, self.att.squeeze(0)))

        # Mask attention scores using adjacency matrix
        mask = adj.unsqueeze(-1) == 0
        attention = attention.masked_fill(mask, float("-inf"))

        # Apply softmax to normalize attention scores
        attention = F.softmax(attention, dim=1)  # Normalize over regulators
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to compute messages
        messages = torch.einsum("bnmh,bnhf->nmhf", attention, h_reg)

        # Reshape to final output dimension
        return messages.reshape(batch_size, num_targets, -1)


class GAT(nn.Module):
    """Graph Attention Network for directed graph"""

    def __init__(
        self,
        regulator_dim: int,
        target_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        n_heads: list[int],
        dropout: float = 0.6,
    ):
        super().__init__()

        self.input_layer = MessagePassingGATLayer(
            regulator_dim=regulator_dim,
            target_dim=target_dim,
            out_features=hidden_dims[0],
            n_heads=n_heads[0],
            dropout=dropout,
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                MessagePassingGATLayer(
                    regulator_dim=hidden_dims[i] * n_heads[i],
                    target_dim=hidden_dims[i] * n_heads[i],
                    out_features=hidden_dims[i + 1],
                    n_heads=n_heads[i + 1],
                    dropout=dropout,
                )
            )
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(hidden_dims[-1] * n_heads[-1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor], adj: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the network."""
        attention_weights = []

        # Input layer
        x = (self.dropout(x[0]), self.dropout(x[1]))
        x = self.input_layer(x, adj)
        x = F.elu(x)
        attention_weights.append(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer((x, x), adj)  # After first layer, use same features for both inputs
            x = F.elu(x)
            attention_weights.append(x)

        # Output layer
        predictions = self.output_layer(x)

        return predictions, attention_weights
