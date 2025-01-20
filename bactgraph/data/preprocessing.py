import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch


def get_embedding_for_proteins(
    file_path: str, protein_ids: set[str], sample_ids: set[str], batch_size: int = 1000
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Get embeddings for a subset of proteins from a parquet file."""
    dataset = ds.dataset(file_path, format="parquet")
    available_columns = dataset.schema.names
    valid_proteins = list(protein_ids & set(available_columns))

    if not valid_proteins:
        raise ValueError("None of the requested proteins found in the dataset")
    print(f"Loading embeddings for {len(valid_proteins)} proteins...")
    sample_filter = pc.field("sample").isin(list(sample_ids))
    columns_to_read = valid_proteins + ["sample"]
    embedding_dict = {protein: [] for protein in valid_proteins}
    samples_with_embeddings = set()

    for batch in dataset.to_batches(batch_size=batch_size, columns=columns_to_read, filter=sample_filter):
        df_chunk = batch.to_pandas()
        samples_with_embeddings.update(df_chunk.index)

        for protein in valid_proteins:
            embeddings = df_chunk[protein].apply(lambda x: np.array(x, dtype=np.float32))
            embedding_dict[protein].extend(embeddings.tolist())

    return (
        {
            protein: torch.tensor(np.array(embeddings), dtype=torch.float32)
            for protein, embeddings in embedding_dict.items()
        },
        list(samples_with_embeddings),
    )


def load_and_validate_data(
    adj_matrix_path: str, expression_data_path: str, embeddings_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor], set[str]]:
    """Load and validate input data files."""
    adj_df = pd.read_csv(adj_matrix_path, sep="\t", index_col=0)
    network_genes = set(adj_df.index) | set(adj_df.columns)
    print(f"Number of genes in network: {len(network_genes)}")

    expr_df = pd.read_csv(expression_data_path, sep="\t", index_col=0)
    sample_ids = set(expr_df.columns)

    # TODO: if only using embeddings for hotspot genes (and not their regulons
    # then only pass set(adj_df.index) instead of network_genes)
    embeddings_dict, samples_with_embeddings = get_embedding_for_proteins(
        embeddings_path, network_genes, sample_ids, batch_size=1000
    )
    genes_with_embeddings = set(embeddings_dict.keys())

    print(f"Number of network genes with embeddings: {len(genes_with_embeddings)}")
    print(f"Number of samples with embeddings: {len(samples_with_embeddings)}")

    common_genes = list(genes_with_embeddings & set(expr_df.index))
    expr_df = expr_df.loc[common_genes, samples_with_embeddings]

    # Drop missing genes from adjacency matrix
    missing_embeddings = network_genes - genes_with_embeddings
    if missing_embeddings:
        print(f"\nWarning: {len(missing_embeddings)} genes in network missing from embeddings:")
        print(sorted(missing_embeddings))

    missing_expr = network_genes - set(expr_df.index)
    if missing_expr:
        print(f"\nWarning: {len(missing_expr)} genes in network missing from expression:")
        print(sorted(missing_expr))

    missing_rows = missing_expr & set(adj_df.index)
    missing_cols = missing_expr & set(adj_df.columns)

    adj_df = adj_df.drop(index=missing_rows, columns=missing_cols)

    print("\nFinal dataset dimensions:")
    print(f"Adjacency matrix: {adj_df.shape}")
    print(f"Expression matrix: {expr_df.shape}")
    print(f"Number of genes with embeddings: {len(embeddings_dict)}")

    return adj_df, expr_df, embeddings_dict, genes_with_embeddings


def create_dataset_splits(
    sample_ids: list[str], train_split: float = 0.7, val_split: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Create train/validation/test split of sample IDs.

    Args:
        sample_ids: List of all sample IDs
        train_split: Fraction of samples to use for training
        val_split: Fraction of samples to use for validation
        seed: Random seed for reproducibility

    Returns
    -------
        Tuple of (train_samples, val_samples, test_samples)
    """
    if train_split + val_split >= 1.0:
        raise ValueError("Train and validation splits must sum to less than 1.0 to leave room for test set")

    n_samples = len(sample_ids)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    torch.manual_seed(seed)
    shuffled_indices = torch.randperm(n_samples)

    train_samples = [sample_ids[i] for i in shuffled_indices[:n_train]]
    val_samples = [sample_ids[i] for i in shuffled_indices[n_train : n_train + n_val]]
    test_samples = [sample_ids[i] for i in shuffled_indices[n_train + n_val :]]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples
