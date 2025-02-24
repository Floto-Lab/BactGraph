import torch
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, pearson_corrcoef, r2_score


def batch_into_single_graph(x_batch: torch.Tensor, edge_index_batch: torch.Tensor):
    """
    Merges a batch of graphs into a single 'big' graph.

    Parameters
    ----------
    x_batch : torch.Tensor
        Shape: [B, N, d]
        - B: Number of graphs in the batch
        - N: Number of nodes per graph
        - d: Dimension of each node's feature vector

    edge_index_batch : torch.Tensor
        Shape: [B, 2, E]
        - B: Number of graphs in the batch
        - 2: Each edge is (src, dst)
        - E: Number of edges per graph

    Returns
    -------
    merged_x : torch.Tensor
        Shape: [B*N, d]
        Node features for all graphs, concatenated into one big tensor.

    merged_edge_index : torch.Tensor
        Shape: [2, B*E]
        Merged adjacency. Node indices are offset so each subgraph
        occupies a distinct index range in [0 .. B*N-1].

    batch_vector : torch.Tensor
        Shape: [B*N]
        A 1D tensor of subgraph indices, specifying which mini-graph
        each node belongs to (0 <= value < B).
    """
    # -------------------------------
    # 1) Flatten node features
    # -------------------------------
    # x_batch: [B, N, d] => [B*N, d]
    B, N, d = x_batch.shape
    merged_x = x_batch.view(B * N, d)

    # -------------------------------
    # 2) Offset and merge edges
    # -------------------------------
    # edge_index_batch: [B, 2, E]
    # We'll offset each edge_index by i*N for graph i.
    _, _, E = edge_index_batch.shape

    edge_index_list = []
    for i in range(B):
        offset = i * N
        # edge_index_batch[i] has shape [2, E], offset it by offset
        # This adds 'offset' to both src and dst node indices
        ei = edge_index_batch[i] + offset
        edge_index_list.append(ei)

    merged_edge_index = torch.cat(edge_index_list, dim=1)  # => shape [2, B*E]

    # -------------------------------
    # 3) Create batch vector
    # -------------------------------
    # For graph i, we have N nodes => batch index i for those nodes.
    # shape: [B*N]
    batch_vector = []
    for i in range(B):
        # A tensor of length N, filled with 'i'
        b_i = torch.full((N,), i, dtype=torch.long)
        batch_vector.append(b_i)
    batch_vector = torch.cat(batch_vector, dim=0)  # => shape [B*N]

    return merged_x, merged_edge_index, batch_vector


def unbatch_single_graph(
    merged_logits: torch.Tensor,  # [B*N, d_out]
    batch_vector: torch.Tensor,  # [B*N], each entry in [0..B-1]
) -> torch.Tensor:
    """Unbatch a single graph

    Given a merged tensor of node (or features) and a batch_vector
    that indicates which subgraph each row belongs to, reconstruct a
    [B, N, d_out] tensor.

    Parameters
    ----------
    merged_logits : torch.Tensor
        Shape: [B*N, d_out]
        Node-level logits/features after merging.

    batch_vector : torch.Tensor
        Shape: [B*N]
        For each node index in range [0..B*N-1],
        `batch_vector[i]` is the subgraph ID (0 <= ID < B).

    Returns
    -------
    logits_unbatched : torch.Tensor
        Shape: [B, N, d_out]
        The node logits reshaped into B subgraphs, each with N nodes.
    """
    # 1) Determine how many distinct subgraphs (B) we have
    B = batch_vector.max().item() + 1  # subgraphs labeled 0..(B-1)

    # 2) We know total_nodes = B*N (from shape of merged_logits)
    total_nodes, d_out = merged_logits.shape

    # 3) Infer N by dividing the total nodes among B subgraphs
    if total_nodes % B != 0:
        raise ValueError(f"Cannot evenly split {total_nodes} nodes into {B} subgraphs.")
    N = total_nodes // B

    # 4) Allocate output [B, N, d_out]
    logits_unbatched = torch.zeros(B, N, d_out, dtype=merged_logits.dtype, device=merged_logits.device)

    # 5) Fill each subgraph slice according to batch_vector
    node_counts = [0] * B  # how many nodes we've placed in each subgraph
    for global_node_idx in range(total_nodes):
        subgraph_idx = batch_vector[global_node_idx].item()  # which subgraph this node belongs to
        local_node_idx = node_counts[subgraph_idx]
        if local_node_idx >= N:
            raise ValueError(
                f"Subgraph {subgraph_idx} has more than {N} nodes " f"based on the batch_vector assignment."
            )

        # Place the row in the correct [subgraph, node, :] slot
        logits_unbatched[subgraph_idx, local_node_idx] = merged_logits[global_node_idx]
        node_counts[subgraph_idx] += 1

    return logits_unbatched


def group_by_label(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Group rows of X by the integer labels in Y.

    X: [N, M]
    Y: [N]    (contains integer labels)

    Returns: grouped: [Z, W, M],
        where Z = number of unique labels,
              W = number of rows per label,
              M = original feature dim.
    """
    unique_labels = torch.unique(Y)  # e.g. [0,1,2,3]
    groups = []
    for label in unique_labels:
        # Mask out rows of X that correspond to this label
        group = X[Y == label]  # shape [W, M]
        groups.append(group)
    # Stack groups along a new dimension 0 => [Z, W, M]
    grouped = torch.stack(groups, dim=0)
    return grouped


def compute_regression_metrics(
    preds: torch.Tensor,
    y: torch.Tensor,
    gene_indices: torch.Tensor,
    split: str = "val",
) -> dict[str, torch.Tensor]:
    """Compute regression metrics"""
    preds_flat = preds[y.view(-1) != -100.0]
    y_flat = y[y != -100.0]

    y = group_by_label(y.view(-1).unsqueeze(-1), gene_indices.view(-1)).squeeze(-1)
    preds = group_by_label(preds.view(-1).unsqueeze(-1), gene_indices.view(-1)).squeeze(-1)

    pearson_arr = []
    r2_arr = []
    for idx in range(y.shape[0]):
        y_gene = y[idx, :]
        preds_gene = preds[idx, :]
        preds_gene = preds_gene[y_gene != -100.0]
        y_gene = y_gene[y_gene != -100.0]

        if len(y_gene) < 10:
            continue
        pearson_gene = pearson_corrcoef(preds_gene, y_gene)
        r2_gene = r2_score(preds_gene, y_gene)

        if torch.isnan(pearson_gene):
            continue
        if torch.isnan(r2_gene):
            continue
        pearson_arr.append(pearson_gene)
        r2_arr.append(r2_gene)

    pearson_gene = torch.tensor(pearson_arr).mean()
    r2_gene = torch.tensor(r2_arr).mean()
    pearson = pearson_corrcoef(preds_flat, y_flat)
    r2 = r2_score(preds_flat, y_flat)

    res = {
        f"{split}_pearson": pearson,
        f"{split}_r2": r2,
        f"{split}_gene_pearson": pearson_gene,
        f"{split}_gene_r2": r2_gene,
    }
    return res


def compute_binary_metrics(preds: torch.Tensor, y: torch.Tensor, split: str = "val") -> dict[str, torch.Tensor]:
    """Compute binary metrics"""
    return {
        f"{split}_f1": f1_score(preds, y, task="binary"),
        f"{split}_accuracy": accuracy(preds, y, task="binary"),
        f"{split}_auroc": auroc(preds, y, task="binary"),
        f"{split}_auprc": average_precision(preds, y, task="binary"),
    }
