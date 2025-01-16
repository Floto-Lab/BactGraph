import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from bactgraph.data.dataset import ExpressionDataset
from bactgraph.data.preprocessing import create_train_val_split, load_and_validate_data
from bactgraph.models.gat import GAT
from torch.utils.data import DataLoader


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train GAT model for protein expression prediction")
    parser.add_argument("--adj-matrix", type=str, required=True, help="Path to adjacency matrix (tab-delimited)")
    parser.add_argument("--expression-data", type=str, required=True, help="Path to expression data (tab-delimited)")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Directory containing ESM-2 embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of samples to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str,
) -> None:
    """Train GAT model"""
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0

        for _batch_idx, (features, adj, labels) in enumerate(train_loader):
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions, _ = model(features, adj)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for features, adj, labels in val_loader:
                features = features.to(device)
                adj = adj.to(device)
                labels = labels.to(device)

                predictions, _ = model(features, adj)
                val_loss += criterion(predictions, labels).item()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")

        model.train()


def main():
    """Run training"""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    adj_df, expr_df, _ = load_and_validate_data(args.adj_matrix, args.expression_data, args.embeddings_dir)

    # Create train/val split
    train_samples, val_samples = create_train_val_split(expr_df.columns.tolist(), args.train_split, args.seed)

    # Create datasets
    train_dataset = ExpressionDataset(
        embeddings_dir=args.embeddings_dir,
        adjacency_matrix=adj_df,
        expression_data=expr_df,
        sample_ids=train_samples,
        device=device,
    )

    val_dataset = ExpressionDataset(
        embeddings_dir=args.embeddings_dir,
        adjacency_matrix=adj_df,
        expression_data=expr_df,
        sample_ids=val_samples,
        device=device,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Initialize model
    model = GAT(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )

    criterion = nn.MSELoss()

    # Train model
    train(model, train_loader, val_loader, optimizer, criterion, config["training"]["num_epochs"], device)


if __name__ == "__main__":
    main()
