from pathlib import Path

import pandas as pd
import torch


def load_and_validate_data(
    adj_matrix_path: str, expression_data_path: str, embeddings_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """
    Load and validate input data files.

    Args:
        adj_matrix_path: Path to adjacency matrix file
        expression_data_path: Path to expression data file
        embeddings_dir: Directory containing ESM-2 embeddings

    Returns
    -------
        Tuple containing:
        - Processed adjacency matrix
        - Processed expression data
        - Set of genes present in network
    """
    # Load adjacency matrix
    adj_df = pd.read_csv(adj_matrix_path, sep="\t", index_col=0)
    network_genes = set(adj_df.index) | set(adj_df.columns)
    print(f"Number of genes in network: {len(network_genes)}")

    # Load expression data
    expr_df = pd.read_csv(expression_data_path, sep="\t", index_col=0)

    # Filter expression data to network genes
    expr_df = expr_df.loc[expr_df.index.isin(network_genes)]
    print(f"Number of genes with expression data in network: {len(expr_df.index)}")

    # Check for missing genes
    missing_genes = network_genes - set(expr_df.index)
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes in network missing from expression data:")
        print(sorted(missing_genes))

    # Subset adjacency matrix to genes with expression data
    adj_df = adj_df.loc[expr_df.index, expr_df.index]

    # Verify embeddings exist
    embeddings_dir = Path(embeddings_dir)
    sample_id = expr_df.columns[0]
    if not (embeddings_dir / f"{sample_id}.npy").exists():
        raise FileNotFoundError(f"Could not find embedding file for sample {sample_id}")

    print("\nFinal dataset dimensions:")
    print(f"Adjacency matrix: {adj_df.shape}")
    print(f"Expression matrix: {expr_df.shape}")

    return adj_df, expr_df, network_genes


def create_train_val_split(sample_ids: list, train_split: float, seed: int = 42) -> tuple[list, list]:
    """
    Create train/validation split of sample IDs.

    Args:
        sample_ids: List of all sample IDs
        train_split: Fraction of samples to use for training
        seed: Random seed for reproducibility

    Returns
    -------
        Tuple of (train_samples, val_samples)
    """
    n_train = int(len(sample_ids) * train_split)

    torch.manual_seed(seed)
    shuffled_indices = torch.randperm(len(sample_ids))

    train_samples = [sample_ids[i] for i in shuffled_indices[:n_train]]
    val_samples = [sample_ids[i] for i in shuffled_indices[n_train:]]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    return train_samples, val_samples
