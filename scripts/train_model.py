import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from bactgraph.data.dataset import EmbeddingDataset
from bactgraph.data.preprocessing import create_dataset_splits, load_and_validate_data
from bactgraph.models.gat import GAT
from torch.utils.data import DataLoader


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train GAT model for protein expression prediction")
    parser.add_argument("--adj-matrix", type=str, required=True, help="Path to adjacency matrix (tab-delimited)")
    parser.add_argument("--expression-data", type=str, required=True, help="Path to expression data (tab-delimited)")
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to embeddings parquet file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of samples to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoints")

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
    output_dir: Path,
) -> None:
    """Train GAT model

    Args:
        model: GAT model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Model optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train for
        device: Device to train on
        output_dir: Directory to save model checkpoints
    """
    best_val_loss = float("inf")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        for _batch_idx, (features, adj, labels) in enumerate(train_loader):
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions, _ = model(features, adj)
            loss = criterion(predictions.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for features, adj, labels in val_loader:
                features = features.to(device)
                adj = adj.to(device)
                labels = labels.to(device)

                predictions, _ = model(features, adj)
                val_loss += criterion(predictions.squeeze(), labels).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                output_dir / "best_model.pt",
            )

        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            output_dir / "latest_model.pt",
        )


def main():
    """Train GAT model for protein expression prediction"""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj_df, expr_df, embeddings_dict, _ = load_and_validate_data(
        args.adj_matrix, args.expression_data, args.embeddings_path
    )

    train_samples, val_samples, test_samples = create_dataset_splits(
        expr_df.columns.tolist(), train_split=0.7, val_split=0.15, seed=args.seed
    )

    train_dataset = EmbeddingDataset(
        embeddings_dict=embeddings_dict,
        adj_matrix=adj_df,
        expression_data=expr_df,
        sample_ids=train_samples,
        device=device,
    )

    val_dataset = EmbeddingDataset(
        embeddings_dict=embeddings_dict,
        adj_matrix=adj_df,
        expression_data=expr_df,
        sample_ids=val_samples,
        device=device,
    )

    test_dataset = EmbeddingDataset(
        embeddings_dict=embeddings_dict,
        adj_matrix=adj_df,
        expression_data=expr_df,
        sample_ids=test_samples,
        device=device,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Save test set indices for later evaluation
    torch.save(
        {
            "test_samples": test_samples,
        },
        f"{args.output_dir}/test_split.pt",
    )

    print("Initializing model...")
    model = GAT(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )

    criterion = nn.MSELoss()

    # Train model
    print("Starting training...")
    train(
        model, train_loader, val_loader, optimizer, criterion, config["training"]["num_epochs"], device, args.output_dir
    )

    print("Evaluating best model on test set...")
    best_model = torch.load(Path(args.output_dir) / "best_model.pt")
    model.load_state_dict(best_model["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)

    results = {
        "test_loss": test_loss,
        "train_loss": best_model["train_loss"],
        "val_loss": best_model["val_loss"],
        "epochs_trained": best_model["epoch"],
    }
    torch.save(results, Path(args.output_dir) / "evaluation_results.pt")


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """
    Evaluate model on test set.

    Args:
        model: Trained GAT model
        test_loader: DataLoader for test set
        criterion: Loss function
        device: Device to run evaluation on

    Returns
    -------
        Test loss
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for features, adj, labels in test_loader:
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

            predictions, _ = model(features, adj)
            test_loss += criterion(predictions.squeeze(), labels).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss


if __name__ == "__main__":
    main()
