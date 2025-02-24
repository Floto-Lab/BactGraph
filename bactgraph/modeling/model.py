import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GCNConv

from bactgraph.modeling.utils import (
    batch_into_single_graph,
    compute_binary_metrics,
    compute_regression_metrics,
    unbatch_single_graph,
)


class GNNModel(nn.Module):
    """Graph Attention Network (GAT) model."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: int, num_heads: int):
        """Initialize the GAT model.

        Args:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of hidden units for each GAT layer.
            output_dim (int): Dimensionality of the output.
            num_layers (int): Total number of GAT layers.
            dropout (float): Dropout probability.
            num_heads (int): Number of attention heads for the GAT layers.
        """
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Build the GAT layers
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GCNConv(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    # heads=1,
                    # concat=False,  # don't concat the heads for the output
                    add_self_loops=False,
                )
            )
            return

        # 1) First GAT layer: input_dim -> hidden_dim
        self.convs.append(
            GCNConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                # heads=num_heads,
                add_self_loops=False,
                # concat=True,  # if True => output_dim = hidden_dim * num_heads
            )
        )

        # 2) Middle GAT layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    # heads=num_heads,
                    add_self_loops=False,
                    # concat=True,  # if True => output_dim = hidden_dim * num_heads
                )
            )

        # 3) Final GAT layer: hidden_dim -> output_dim
        #    Typically for regression, we use a single head (heads=1) and concat=False
        self.convs.append(
            GCNConv(
                in_channels=hidden_dim,
                out_channels=output_dim,
                # heads=num_heads,
                add_self_loops=False,
                # concat=True,  # if True => output_dim = hidden_dim * num_heads
            )
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
            x (Tensor): Node features of shape [num_nodes, input_dim].
            edge_index (LongTensor): Graph connectivity of shape [2, num_edges].

        Returns
        -------
            Tensor: Output of shape [num_nodes, output_dim].
        """
        # Pass through all but the last GAT layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer (typically no nonlinear activation if it's a regression/logits)
        x = self.convs[-1](x, edge_index)
        return x


class BactGraphModel(pl.LightningModule):
    """PyTorch Lightning BactGraph model."""

    def __init__(self, config: dict, phenotype_prediction: bool = False):
        """Initialize the model

        Config dictionary can include:
            config = {
                "input_dim": 480,
                "hidden_dim": 480,
                "output_dim": 1,
                "num_layers": 3,
                "dropout": 0.2,
                "num_heads": 4,
                "lr": 1e-3,
            }
        """
        super().__init__()
        self.config = config
        self.phenotype_prediction = phenotype_prediction

        # Build the underlying GNN model (nn.Module)
        self.gnn_module = GNNModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_heads=config["num_heads"],
        )

        if self.phenotype_prediction:
            self.linear = nn.Linear(config["output_dim"], 1)
            self.droput = nn.Dropout(0.2)
        else:
            self.gene_bias = torch.nn.Parameter(torch.zeros(config["n_genes"]), requires_grad=True)  # .unsqueeze(1)

        # Learning rate (default to 1e-3 if not specified)
        self.lr = config.get("lr", 1e-3)
        self.save_hyperparameters(logger=False)

    def forward(self, x_batch: torch.Tensor, edge_index_batch: torch.Tensor, gene_indices: torch.Tensor):
        """Expects a PyG data object with data.x (node features) and data.edge_index (graph connectivity)."""
        x, edge_index, batch_vector = batch_into_single_graph(x_batch, edge_index_batch.type(torch.long))
        batch_size = x_batch.shape[0]
        logits = self.gnn_module(x, edge_index).squeeze()
        # last_hidden_state = self.gat_module(x, edge_index)
        # last_hidden_state = group_by_label(self.dropout(last_hidden_state), gene_indices.view(-1))
        # logits = torch.einsum(
        #     "bnm,bm->bn", last_hidden_state, self.gene_matrix.to(last_hidden_state.device)
        # ) + self.bias.to(last_hidden_state.device)
        # logits = []
        # for idx, gene_lhs in enumerate(last_hidden_state):
        #     logits.append(self.gene_layers[idx](gene_lhs))
        # logits = torch.stack(logits, dim=1).squeeze()
        # logits = last_hidden_state.squeeze() + self.bias.to(last_hidden_state.device)
        if self.phenotype_prediction:
            # re-batch the tensors from one graph to strain graphs to have a probability for a strain
            logits = unbatch_single_graph(logits, batch_vector)
            # take the mean of all nodes in the graph
            logits = logits.mean(dim=1)
            # predict the phenotype
            logits = self.linear(self.droput(logits)).squeeze()
            return logits
        return F.softplus(logits + self.gene_bias.repeat(batch_size))

    def training_step(self, batch, batch_idx):
        """Training step."""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        # y = group_by_label(y.view(-1).unsqueeze(-1), gene_indices.view(-1))
        # preds = preds.view(-1)
        # y = y.view(-1)

        if self.phenotype_prediction:
            loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))
        else:
            preds = preds[y.view(-1) != -100.0]
            y = y[y != -100.0]
            loss = F.mse_loss(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        if self.phenotype_prediction:
            loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))
            res = compute_binary_metrics(preds, y, split="val")
        else:
            preds_flat = preds[y.view(-1) != -100.0]
            y_flat = y[y != -100.0]
            loss = F.mse_loss(preds_flat, y_flat)
            res = compute_regression_metrics(preds, y, gene_indices, split="val")

        res["loss"] = loss
        self.log_dict(res, prog_bar=True, batch_size=self.config["batch_size"])
        return res

    def test_step(self, batch, batch_idx) -> dict:
        """Test step."""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        if self.phenotype_prediction:
            loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))
            res = compute_binary_metrics(preds, y, split="test")
        else:
            preds_flat = preds[y.view(-1) != -100.0]
            y_flat = y[y != -100.0]
            loss = F.mse_loss(preds_flat, y_flat)
            res = compute_regression_metrics(preds, y, gene_indices, split="test")

        res["loss"] = loss
        self.log_dict(res, prog_bar=True, batch_size=self.config["batch_size"])
        return res

    def configure_optimizers(self):
        """Configure the optimizer and add a Cosine Annealing LR scheduler."""
        optimizer = AdamW(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.config["weight_decay"],
        )
        return optimizer
