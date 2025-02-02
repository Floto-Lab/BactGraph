import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from bactgraph.modeling.dataset import BactGraphDataset

BACTMAP_PROTEINS_FILE_NAME = "bactmap_proteins_prot_embeds.parquet"
NORMALISED_EXPRESSION_FILE_NAME = "norm_dat_pao1.tsv"
PERTURB_NETWORK_FILE_NAME = "llcb_perturb_hits_adj_matrix.tsv"


def preprocess_data_for_training(
    input_dir: str,
    transform_norm_expression_fn: Callable = np.log10,
    train_size: float = 0.7,
    test_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Preprocess the data for training the BactGraph model."""
    # read the data
    protein_embeddings = pd.read_parquet(os.path.join(input_dir, BACTMAP_PROTEINS_FILE_NAME))
    expression_df = pd.read_csv(os.path.join(input_dir, NORMALISED_EXPRESSION_FILE_NAME), sep="\t").set_index(
        "feature_id"
    )
    perturb_network = pd.read_csv(os.path.join(input_dir, PERTURB_NETWORK_FILE_NAME), sep="\t").set_index("gene_id")

    # keep only genes which are in all files
    prot_emb_genes = set(protein_embeddings.columns.tolist())
    expression_genes = set(expression_df.index.tolist())
    perturb_network_genes = set(perturb_network.index.tolist() + perturb_network.columns.tolist())

    genes_of_interest = list(prot_emb_genes.intersection(expression_genes).intersection(perturb_network_genes))
    genes_of_interest = genes_of_interest[: len(genes_of_interest) // 2]
    print(f"Total nr of genes available: {len(genes_of_interest)}")

    # subset the genes of interest
    protein_embeddings = protein_embeddings[genes_of_interest]
    expression_df = expression_df[expression_df.index.isin(genes_of_interest)]
    perturb_network = perturb_network[[g for g in genes_of_interest if g in perturb_network.columns]]
    perturb_network = perturb_network[perturb_network.index.isin(genes_of_interest)]

    # subset to the strains with expression data
    strains_w_expression = expression_df.columns.tolist()
    strains_w_prot_emb = protein_embeddings.index.tolist()
    strains_of_interest = list(set(strains_w_expression).intersection(strains_w_prot_emb))
    expression_df = expression_df[strains_of_interest]
    protein_embeddings = protein_embeddings.loc[strains_of_interest]

    # split the data
    random.seed(random_seed)
    random.shuffle(strains_of_interest)
    train_size = int(len(strains_of_interest) * train_size)
    test_size = int(len(strains_of_interest) * test_size)
    train_strains = strains_of_interest[:train_size]
    test_strains = strains_of_interest[train_size : train_size + test_size]
    val_strains = strains_of_interest[train_size + test_size :]

    gene2idx = {gene: idx for idx, gene in enumerate(protein_embeddings.columns)}

    # create datasets
    train_dataset = BactGraphDataset(
        protein_embeddings=protein_embeddings.loc[train_strains],
        expression_df=expression_df[train_strains],
        gene2idx=gene2idx,
        perturb_network=perturb_network,
        transform_norm_expression_fn=transform_norm_expression_fn,
        random_seed=random_seed,
    )
    val_dataset = BactGraphDataset(
        protein_embeddings=protein_embeddings.loc[val_strains],
        expression_df=expression_df[val_strains],
        gene2idx=gene2idx,
        perturb_network=perturb_network,
        transform_norm_expression_fn=transform_norm_expression_fn,
        random_seed=random_seed,
    )
    test_dataset = BactGraphDataset(
        protein_embeddings=protein_embeddings.loc[test_strains],
        expression_df=expression_df[test_strains],
        gene2idx=gene2idx,
        perturb_network=perturb_network,
        transform_norm_expression_fn=transform_norm_expression_fn,
        random_seed=random_seed,
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dict(  # noqa
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        n_train_size=len(train_strains),
        gene2idx=gene2idx,
    )
