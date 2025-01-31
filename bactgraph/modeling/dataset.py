import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BACTMAP_PROTEINS_FILE_NAME = "bactmap_proteins_prot_embeds.parquet"
NORMALISED_EXPRESSION_FILE_NAME = "norm_dat_pao1.tsv"
PERTURB_NETWORK_FILE_NAME = "bactmap_proteins_prot_embeds.parquet"


def perturb_mtx_to_triples(df: pd.DataFrame, gene2idx: dict[str, int]) -> torch.Tensor:
    """Conver perturbation dataframe to triples with non-zero values for training."""
    # 1. "Stack" the DataFrame so that rows become part of a MultiIndex
    nonzero_stacked = df.stack()  # This will convert the DataFrame into a Series

    # 2. Filter out zero values
    nonzero_stacked = nonzero_stacked[nonzero_stacked != 0]

    # 3. Convert to a list of (index_name, column_name, value) tuples
    triples = list(
        zip(
            nonzero_stacked.index.get_level_values(0),  # index name
            nonzero_stacked.index.get_level_values(1),  # column name
            nonzero_stacked.values,
            strict=False,  # value
        )
    )

    triples = torch.tensor(
        [
            [gene2idx[gene1] for gene1, _, _ in triples],
            [gene2idx[gene2] for _, gene2, _ in triples],
            [val for _, _, val in triples],
        ],
        dtype=torch.float32,
    )
    return triples


class BactGraphDataset(Dataset):
    """Dataset of gene networks in bacteria for BactGraph project."""

    def __init__(
        self,
        input_dir: str,
        transform_norm_expression_fn: Callable = np.log10,
    ):
        # read the data
        self.protein_embeddings = pd.read_parquet(os.path.join(input_dir, BACTMAP_PROTEINS_FILE_NAME))
        self.expression_df = pd.read_csv(os.path.join(input_dir, NORMALISED_EXPRESSION_FILE_NAME), sep="\t").set_index(
            "feature_id"
        )
        perturb_network = pd.read_csv(os.path.join(input_dir, PERTURB_NETWORK_FILE_NAME), sep="\t").set_index("gene_id")

        # keep only genes which are in all files
        prot_emb_genes = set(self.protein_embeddings.columns.tolist())
        expression_genes = set(self.expression_df.index.tolist())
        perturb_network_genes = set(perturb_network.index.tolist() + perturb_network.columns.tolist())

        genes_of_interest = list(prot_emb_genes.intersection(expression_genes).intersection(perturb_network_genes))
        print(f"Total nr of genes available: {len(genes_of_interest)}")

        # subset the genes of interest
        self.protein_embeddings = self.protein_embeddings[genes_of_interest]
        self.expression_df = self.expression_df[self.expression_df.index.isin(genes_of_interest)]
        perturb_network = perturb_network[genes_of_interest]
        perturb_network = perturb_network[perturb_network.index.isin(genes_of_interest)]

        # subset to the strains with expression data
        strains_w_expression = self.expression_df.columns.tolist()
        strains_w_prot_emb = self.protein_embeddings.index.tolist()
        strains_of_interest = list(set(strains_w_expression).intersection(strains_w_prot_emb))
        self.expression_df = self.expression_df[strains_of_interest]
        self.protein_embeddings = self.protein_embeddings.loc[strains_of_interest]

        # get triples
        self.gene2idx = {gene: idx for idx, gene in enumerate(self.protein_embeddings.columns)}
        self.triples = perturb_mtx_to_triples(perturb_network, self.gene2idx)

        # normalise the expression data
        # revert previous log2 transformation (the data was provided like this)
        self.expression_df = self.expression_df.apply(np.exp2)
        # transform the data with the provided function
        self.expression_df = self.expression_df.apply(transform_norm_expression_fn)

        self.strains = self.expression_df.columns.tolist()

    def __len__(self):
        return len(self.expression_df.columns)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the expression data for the idx-th strain
        strain = self.strains[idx]
        # get protein embeddings
        prot_emb = torch.tensor(self.protein_embeddings.loc[strain].values, dtype=torch.float32)
        expr_values = torch.tensor(
            [self.expression_df.loc[gene, strain] for gene in self.protein_embeddings.columns], dtype=torch.float32
        )
        return prot_emb, expr_values, self.triples
