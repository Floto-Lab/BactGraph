import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.data.dataset import ProteinExpressionDataset
from src.models.gat import GAT
from torch.utils.data import DataLoader


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
    """Run training loop"""
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ProteinExpressionDataset(
        embeddings_path=config["data"]["embeddings_path"],
        adjacency_matrix_path=config["data"]["adjacency_matrix_path"],
        expression_data_path=config["data"]["expression_data_path"],
        device=device,
    )

    val_dataset = ProteinExpressionDataset(
        embeddings_path=config["data"]["val_embeddings_path"],
        adjacency_matrix_path=config["data"]["adjacency_matrix_path"],
        expression_data_path=config["data"]["val_expression_path"],
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
