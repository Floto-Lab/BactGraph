import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from bactgraph.data.dataset import EmbeddingDataset
from bactgraph.data.preprocessing import create_dataset_splits, load_and_validate_data
from bactgraph.models.gat import GAT
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stopping handler to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute various regression metrics"""
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    return {
        "mse": np.mean((predictions - labels) ** 2),
        "mae": np.mean(np.abs(predictions - labels)),
        "r2": r2_score(labels, predictions),
        "correlation": np.corrcoef(predictions.ravel(), labels.ravel())[0, 1],
    }


def analyze_attention_weights(attention_weights: list[torch.Tensor], adj_matrix: torch.Tensor) -> dict[str, float]:
    """Analyze attention weight patterns"""
    metrics = {}

    # For each attention layer
    for layer_idx, layer_weights in enumerate(attention_weights):
        # Average attention to connected vs unconnected nodes
        connected_attention = layer_weights[adj_matrix == 1].mean().item()
        unconnected_attention = layer_weights[adj_matrix == 0].mean().item()

        metrics[f"layer_{layer_idx}_connected_attention"] = connected_attention
        metrics[f"layer_{layer_idx}_unconnected_attention"] = unconnected_attention
        metrics[f"layer_{layer_idx}_attention_ratio"] = connected_attention / (unconnected_attention + 1e-10)

    return metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = []

    for features, adj, labels in train_loader:
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions, attention_weights = model(features, adj)
        loss = criterion(predictions.squeeze(), labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        batch_metrics = compute_metrics(predictions.squeeze(), labels)
        all_metrics.append(batch_metrics)
        total_loss += loss.item()

        # Analyze attention patterns
        attention_metrics = analyze_attention_weights(attention_weights, adj)
        all_metrics[-1].update(attention_metrics)

    # Average metrics across batches
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    return total_loss / len(train_loader), avg_metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float]]:
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for features, adj, labels in val_loader:
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

            predictions, attention_weights = model(features, adj)
            loss = criterion(predictions.squeeze(), labels)

            batch_metrics = compute_metrics(predictions.squeeze(), labels)
            attention_metrics = analyze_attention_weights(attention_weights, adj)

            batch_metrics.update(attention_metrics)
            all_metrics.append(batch_metrics)
            total_loss += loss.item()

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    return total_loss / len(val_loader), avg_metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str,
    output_dir: Path,
    patience: int = 10,
) -> dict:
    """Train the model with early stopping and detailed monitoring"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float("inf")
    training_history = []

    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Log metrics
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        training_history.append(epoch_stats)

        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train R2: {train_metrics['r2']:.4f}, Val R2: {val_metrics['r2']:.4f}")

        # Save attention analysis
        print("\nAttention Analysis:")
        for k, v in val_metrics.items():
            if "attention" in k:
                print(f"{k}: {v:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            output_dir / "latest_model.pt",
        )

        # Save training history
        torch.save(training_history, output_dir / "training_history.pt")

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    return training_history


def main():
    """Train GAT model with improved monitoring"""
    parser = argparse.ArgumentParser(description="Train GAT model for protein expression prediction")
    parser.add_argument("--adj-matrix", type=str, required=True, help="Path to adjacency matrix")
    parser.add_argument("--expression-data", type=str, required=True, help="Path to expression data")
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training data fraction")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation data fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    adj_df, expr_df, embeddings_dict, genes_with_embeddings = load_and_validate_data(
        args.adj_matrix, args.expression_data, args.embeddings_path
    )

    # Create data splits
    train_samples, val_samples, test_samples = create_dataset_splits(
        expr_df.columns.tolist(),
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Create datasets
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # Save test samples for later evaluation
    torch.save({"test_samples": test_samples}, Path(args.output_dir) / "test_split.pt")

    # Initialize model
    print("\nInitializing model...")
    model = GAT(
        regulator_dim=config["model"]["input_dim"],
        target_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Initialize optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    # Train model
    print("\nStarting training...")
    training_history = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        config["training"]["num_epochs"],
        device,
        args.output_dir,
        args.patience,
    )

    # Final evaluation on test set
    print("\nEvaluating best model on test set...")
    best_model = torch.load(Path(args.output_dir) / "best_model.pt")
    model.load_state_dict(best_model["model_state_dict"])
    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    results = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "model_info": {
            "train_loss": best_model["train_loss"],
            "val_loss": best_model["val_loss"],
            "epochs_trained": best_model["epoch"],
        },
    }

    torch.save(results, Path(args.output_dir) / "evaluation_results.pt")

    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        if "attention" not in metric:  # Only print main metrics
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
