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
        protein_embeddings: pd.DataFrame,
        expression_df: pd.DataFrame,
        gene2idx: dict[str, int],
        perturb_network: pd.DataFrame,
        transform_norm_expression_fn: Callable = np.log10,
        random_seed: int = 42,
    ):
        self.protein_embeddings = protein_embeddings
        self.expression_df = expression_df
        self.gene2idx = gene2idx

        # get triples
        self.triples = perturb_mtx_to_triples(perturb_network, self.gene2idx)[:2, :]
        # reverse the direction
        # self.triples = self.triples[:2, :].flip(0)
        # randomize the network experiment
        # print("Randomizing the network experiment by randomly sampling edges.")
        # torch.manual_seed(random_seed)
        # self.triples = torch.randint(0, len(self.gene2idx), self.triples.shape)
        # fully connected network
        # self.triples = torch.stack(
        #     [torch.arange(len(self.gene2idx)), torch.arange(len(self.gene2idx)), torch.ones(len(self.gene2idx))],
        #     dim=0,
        # )

        # normalise the expression data
        # revert previous log2 transformation (the data was provided like this)
        self.expression_df = self.expression_df.apply(np.exp2)
        # transform the data with the provided function
        self.expression_df = self.expression_df.apply(transform_norm_expression_fn).fillna(-100.0)

        self.strains = self.expression_df.columns.tolist()

    def __len__(self):
        return len(self.expression_df.columns)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the expression data for the idx-th strain
        strain = self.strains[idx]
        # get protein embeddings
        prot_emb = torch.tensor(np.stack(self.protein_embeddings.loc[strain].values), dtype=torch.float32)
        expr_values = torch.tensor(
            [self.expression_df.loc[gene, strain] for gene in self.protein_embeddings.columns], dtype=torch.float32
        )
        gene_idx = torch.arange(len(self.protein_embeddings.columns), dtype=torch.long)
        return prot_emb, self.triples, expr_values, gene_idx
