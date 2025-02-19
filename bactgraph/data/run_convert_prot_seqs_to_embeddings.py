from itertools import islice

import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe before embedding the protein sequences."""
    df_stacked = df.stack().reset_index()
    df_stacked = df_stacked.rename(columns={"level_0": "strain", "level_1": "gene", 0: "sequence"})
    df_stacked = df_stacked.dropna(subset=["sequence"])
    # groupby sequence to prevent duplicates
    df_stacked_groupped = df_stacked.groupby("sequence").agg(list).reset_index()
    return df_stacked_groupped


def postprocess_df(df: pd.DataFrame, prot_seqs_df: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the dataframe after embedding the protein sequences."""
    for _, row in prot_seqs_df.iterrows():
        for strain, gene in zip(row.strain, row.gene, strict=False):
            df.loc[strain, gene] = row.protein_embedding
    return df


def batcher(iterable, batch_size):
    """Batch the iterable."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def main(
    input_file_path: str,
    output_file_path: str,
    esm_model_name: str = "esm2_t12_35M_UR50D",
    batch_size: int = 256,
    max_aa_seq_len: int = 1280,
):
    """Convert amino acid sequences to protein embeddings."""
    # read the data
    df = pd.read_parquet(input_file_path)
    prot_seqs_df = preprocess_df(df)

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_model_name}", use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        f"facebook/{esm_model_name}",
    ).esm
    model.eval()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        model = model.cuda()
        device = "cuda"

    prot_seqs_df["seq_len"] = prot_seqs_df["sequence"].apply(len)
    # sort by sequence len, this makes inference more efficient as we can batch sequences of similar length
    # reducing the amount of padding required
    prot_seqs_df = prot_seqs_df.sort_values("seq_len", ascending=False)
    batches = batcher(prot_seqs_df["sequence"].tolist(), batch_size)

    with torch.no_grad():
        prot_representations = []
        for batch in tqdm(batches, mininterval=300):
            batch = tokenizer(batch, padding="longest", truncation=True, return_tensors="pt", max_length=max_aa_seq_len)
            input_ids = batch["input_ids"].to(device)
            att_mask = batch["attention_mask"].to(device)

            token_representations = model(
                input_ids=input_ids,
                attention_mask=att_mask,
            ).last_hidden_state[:, 1:, :]

            att_mask = att_mask[:, 1:].float()

            # get the average of non-padding tokens
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            avg_prot_representations = torch.einsum("ijk,ij->ik", token_representations, att_mask) / att_mask.sum(
                1
            ).unsqueeze(1)
            prot_representations += list(avg_prot_representations.cpu().numpy())
    prot_seqs_df = prot_seqs_df.drop(columns=["Protein Sequence"])
    prot_seqs_df["protein_embedding"] = prot_representations
    # delete to save memory and it's not necessary anymore
    del prot_representations

    print("Postprocessing the dataframe...")
    output_df = postprocess_df(df, prot_seqs_df)

    output_df.to_parquet(output_file_path)


class ArgParser(Tap):
    """Arguments for converting amino acid sequences to protein embeddings."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_file_path: str  # path to the input parquet file
    output_file_path: str  # path to the output parquet file
    esm_model_name: str = "esm2_t12_35M_UR50D"  # name of the ESM model
    batch_size: int = 256  # batch size for inference
    max_aa_seq_len: int = 1280  # maximum amino acid sequence length


if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(
        input_file_path=args.input_file_path,
        output_file_path=args.output_file_path,
        esm_model_name=args.esm_model_name,
        batch_size=args.batch_size,
        max_aa_seq_len=args.max_aa_seq_len,
    )
