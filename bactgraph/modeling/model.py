import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GATv2Conv
from torchmetrics.functional import pearson_corrcoef, r2_score

from bactgraph.modeling.utils import batch_into_single_graph, group_by_label


class GATModel(nn.Module):
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

        # 1) First GAT layer: input_dim -> hidden_dim
        self.convs.append(
            GATv2Conv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,  # if True => output_dim = hidden_dim * num_heads
            )
        )

        # 2) Middle GAT layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * num_heads,  # since we concat above
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                )
            )

        # 3) Final GAT layer: hidden_dim -> output_dim
        #    Typically for regression, we use a single head (heads=1) and concat=False
        self.convs.append(
            GATv2Conv(
                in_channels=hidden_dim * num_heads,
                out_channels=output_dim,
                heads=1,
                concat=False,  # don't concat the heads for the output
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

    def __init__(self, config: dict):
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

        # Build the underlying GAT model (nn.Module)
        self.gat_module = GATModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_heads=config["num_heads"],
        )

        self.bias = torch.nn.Parameter(torch.zeros(config["n_genes"]), requires_grad=True).unsqueeze(1)
        self.dropout = nn.Dropout(config["dropout"])
        self.gene_matrix = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(config["n_genes"], config["output_dim"])), requires_grad=True
        )

        # self.gene_layers = nn.ModuleList([nn.Linear(config["output_dim"], 1) for _ in range(config["n_genes"])])

        # Learning rate (default to 1e-3 if not specified)
        self.lr = config.get("lr", 1e-3)
        self.save_hyperparameters(logger=False)

    def forward(self, x_batch: torch.Tensor, edge_index_batch: torch.Tensor, gene_indices: torch.Tensor):
        """Expects a PyG data object with data.x (node features) and data.edge_index (graph connectivity)."""
        x, edge_index, _ = batch_into_single_graph(x_batch, edge_index_batch.type(torch.long))
        # batch_size = x_batch.shape[0]
        # logits = self.gat_module(x, edge_index).squeeze() + self.bias.repeat(batch_size)
        last_hidden_state = self.gat_module(x, edge_index)
        last_hidden_state = group_by_label(self.dropout(last_hidden_state), gene_indices.view(-1))
        logits = torch.einsum(
            "bnm,bm->bn", last_hidden_state, self.gene_matrix.to(last_hidden_state.device)
        ) + self.bias.to(last_hidden_state.device)
        # logits = []
        # for idx, gene_lhs in enumerate(last_hidden_state):
        #     logits.append(self.gene_layers[idx](gene_lhs))
        # logits = torch.stack(logits, dim=1).squeeze()
        # logits = last_hidden_state.squeeze() + self.bias.to(last_hidden_state.device)
        return F.softplus(logits)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        y = group_by_label(y.view(-1).unsqueeze(-1), gene_indices.view(-1))
        preds = preds.view(-1)
        y = y.view(-1)
        preds = preds[y.view(-1) != -100.0]
        y = y[y != -100.0]
        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        y = group_by_label(y.view(-1).unsqueeze(-1), gene_indices.view(-1))
        preds = preds.view(-1)
        y = y.view(-1)
        preds = preds[y.view(-1) != -100.0]
        y = y[y != -100.0]
        loss = F.mse_loss(preds, y)
        pearson = pearson_corrcoef(preds, y)
        r2 = r2_score(preds, y)

        res = {"val_loss": loss, "val_pearson": pearson, "val_r2": r2}
        self.log_dict(res, prog_bar=True, batch_size=self.config["batch_size"])

        return res

    def test_step(self, batch, batch_idx) -> dict:
        """Test step."""
        x_batch, edge_index_batch, y, gene_indices = batch
        preds = self.forward(x_batch, edge_index_batch.type(torch.long), gene_indices)

        y = group_by_label(y.view(-1).unsqueeze(-1), gene_indices.view(-1))
        preds = preds.view(-1)
        y = y.view(-1)
        preds = preds[y.view(-1) != -100.0]
        y = y[y != -100.0]
        loss = F.mse_loss(preds, y)
        pearson = pearson_corrcoef(preds, y)
        r2 = r2_score(preds, y)

        res = {"test_loss": loss, "test_pearson": pearson, "test_r2": r2}
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

        # # If user doesn't specify, default T_max to 10
        # T_max = self.config.get("t_max", 10)
        # scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0)
        #
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",  # step every epoch
        #         "frequency": 1,
        #     },
        # }
