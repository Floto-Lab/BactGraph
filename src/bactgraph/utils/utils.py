import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch


@dataclass
class CausalNetwork:
    """
    Class for loading and processing causal network

    Args:
        edge_list (pd.DataFrame): edge list of the network
        feature_ids (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes
    ----------
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, edge_list, feature_ids, node_map):
        """Initialize CausalNetwork class"""
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(
            self.edge_list,
            source="source_gene_name",
            target="target_gene_name",
            edge_attr=["signed_weight"],
            create_using=nx.DiGraph(),
        )
        self.feature_ids = feature_ids
        for n in self.feature_ids:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        edge_attr = nx.get_edge_attributes(self.G, "signed_weight")
        signed_weights = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(signed_weights)


def read_edge_list(edge_list_path: str, threshold: float = 0, data_path: str = "./data") -> pd.DataFrame:
    """
    Read edge list from file

    Args:
        edge_list_path (str): path to edge list file
        threshold (float): threshold for edge weights
    """
    fname = os.path.join(data_path, f"causal_network_{threshold}.csv")

    if os.path.exists(fname):
        return pd.read_csv(fname, sep="\t")

    edge_list = pd.read_csv(edge_list_path, sep="\t")
    edge_list = edge_list[abs(edge_list["weight"]) > threshold]
    edge_list.to_csv(fname, index=False, sep="\t")

    return edge_list


def read_mic_data(mic_path: str, data_path: str = "./data") -> pd.DataFrame:
    """
    Read MIC data from file

    Args:
        mic_path (str): path to MIC data file
        threshold (float): threshold for edge weights
    """
    fname = os.path.join(data_path, "mic_data.csv")

    if os.path.exists(fname):
        return pd.read_csv(fname, sep="\t")

    mic_data = pd.read_csv(mic_path, sep="\t")

    return mic_data


def read_metadata(mic_path: str, data_path: str = "./data") -> pd.DataFrame:
    """
    Read MIC data from file

    Args:
        mic_path (str): path to MIC data file
        threshold (float): threshold for edge weights
    """
    fname = os.path.join(data_path, "mic_data.csv")

    if os.path.exists(fname):
        return pd.read_csv(fname, sep="\t")

    mic_data = pd.read_csv(mic_path, sep="\t")

    return mic_data
