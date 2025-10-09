import torch
import torch.nn as nn
from torch_geometric.nn import (
                                    to_hetero,
                                    global_mean_pool,
                                    TopKPooling,
                                )
from torch_geometric.nn.models import GraphSAGE

from src.model.GraphBased.AiidkitTEAVGraphEmbedder import AiidkitTEAVGraphEmbedder

# Because torch.fx._symbolic_trace.List was removed in PyTorch ≥ 2.3, and PyG 2.6.1 (the latest stable as of now) hasn’t yet
# adapted fully to the FX API changes introduced in PyTorch 2.7.x
# That is why we need to execute the following lines BEFORE using to_hetero from Pytorch Geometric
import torch.fx._symbolic_trace as st
from typing import List
if not hasattr(st, "List"):
    st.List = List

class HeteroGraphSage(nn.Module):
    def __init__(
                    self,
                    in_channels,
                    hidden_channels,
                    out_channels,
                    dropout,
                    num_layers,
                    possible_values_all_patients,
                    categorical_ent_attr_pairs,
                    metadata,
                    graph_pool_strategy="mean",
                    graph_pool_fusion="stack",
                    act='ReLU',
                    aggr='mean'
                ):
        """
        Graph-based module using GraphSAGE.
        """
        super().__init__()
        self.graph_pool_strategy = graph_pool_strategy
        self.graph_pool_fusion = graph_pool_fusion
        
        # Graph embedder for child features
        self.graph_embedder = AiidkitTEAVGraphEmbedder(
                                                        possible_values_all_patients=possible_values_all_patients,
                                                        categorical_ent_attr_pairs=categorical_ent_attr_pairs,
                                                        emb_dim_ent_attr=8,
                                                        emb_dim_ent_attr_vals=8
                                                     )
        
        # Input projection layers (necessary for heterogeneous graphs where different types of nodes have different dimensions)
        self.in_proj_layer = nn.ModuleDict({
                                                node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                                for node_type in in_channels
                                            })


        # GNN
        self.gnn = GraphSAGE(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, act=act)
        self.gnn = to_hetero(self.gnn, metadata, aggr=aggr)

        # Graph-level Pooling
        if (self.graph_pool_strategy.lower() == 'topk'):
            self.topk_per_node = nn.ModuleDict({
                                                    node_type: TopKPooling(hidden_channels, ratio=0.5)
                                                    for node_type in in_channels
                                                })
        
        # Output projection layers
        if (self.graph_pool_fusion.lower() == "stack"):
            self.graph_lin = nn.Linear(hidden_channels, out_channels)
        elif (self.graph_pool_fusion.lower() == "concatenation"):
            n_nodes = len(in_channels)
            self.graph_lin = nn.Linear(n_nodes*hidden_channels, out_channels)


    #def forward(self, x, ids, edge_index, edge_attr, timestamps):
    def forward(self, data):
        """Forward pass."""
        x_dict = {}
        # Central node
        x_dict['central'] = self.in_proj_layer['central'](data['central'].x)
        
        # Child nodes
        child_cont_features, child_categ_features = self.graph_embedder(data)
        x_dict['child_cont'] = self.in_proj_layer['child_cont'](child_cont_features)
        x_dict['child_categ'] = self.in_proj_layer['child_categ'](child_categ_features)
        
        # Message passing
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)

        # Graph-level pooling
        # Each node type has a batch vector
        graph_embeds = []
        for node_type, x in x_dict.items():
            batch = data[node_type].batch  # tells which graph each node belongs to
            if (self.graph_pool_strategy.lower() == "mean"):
                pooled = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
            elif (self.graph_pool_strategy.lower() == "topk"):
                raise NotImplementedError()
            else:
                raise ValueError(f"Graph pool strategy {self.graph_pool_strategy} is not valid")
            graph_embeds.append(pooled)

        # Combine node-type pooled embeddings
        if (self.graph_pool_fusion.lower() == "stack"):
            graph_emb = torch.stack(graph_embeds, dim=0).mean(dim=0)  # average across types
        elif (self.graph_pool_fusion.lower() == "concatenation"):
            graph_emb = torch.cat(graph_embeds, dim=-1) 
        else:
            raise ValueError(f"Graph pool fusion strategy {self.graph_pool_fusion} is not valid")
        
        return self.graph_lin(graph_emb)  # graph-level prediction

