import torch


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
