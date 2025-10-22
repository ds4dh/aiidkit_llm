import torch
import torch.nn as nn
from torch_geometric.nn import (
                                    HGTConv,
                                    global_mean_pool,
                                    TopKPooling,
                                )
from edl_pytorch import Dirichlet

from src.model.GraphBased.AiidkitTEAVGraphEmbedder import AiidkitTEAVGraphEmbedder


class HeteroGNN(nn.Module):
    """
        Simple HeteroGNN where the base was generated with ChatGPT
    """
    def __init__(
                    self,
                    in_channels,
                    hidden_channels,
                    out_channels,
                    possible_values_all_patients,
                    categorical_ent_attr_pairs,
                    metadata,
                    graph_pool_strategy="mean",
                    graph_pool_fusion="stack",
                    evidential=False
                ):
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
        
        # One HGTConv layer
        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=2)
        self.lin_dict = nn.ModuleDict({
                                            node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                            for node_type in in_channels
                                        })

        # Graph-level Pooling
        if (self.graph_pool_strategy.lower() == 'topk'):
            self.topk_per_node = nn.ModuleDict({
                                                    node_type: TopKPooling(hidden_channels, ratio=0.5)
                                                    for node_type in in_channels
                                                })

        # Output layer
        if (evidential):
            if (self.graph_pool_fusion.lower() == "stack"):
                self.graph_lin = Dirichlet(hidden_channels, out_channels)
            elif (self.graph_pool_fusion.lower() == "concatenation"):
                n_nodes = len(in_channels)
                self.graph_lin = Dirichlet(n_nodes*hidden_channels, out_channels)
        else:
            if (self.graph_pool_fusion.lower() == "stack"):
                self.graph_lin = nn.Linear(hidden_channels, out_channels)
            elif (self.graph_pool_fusion.lower() == "concatenation"):
                n_nodes = len(in_channels)
                self.graph_lin = nn.Linear(n_nodes*hidden_channels, out_channels)

    def compute_graph_embedding(self, data):
        x_dict = {}
        # Central node
        x_dict['central'] = self.lin_dict['central'](data['central'].x)
        
        # Child nodes
        child_cont_features, child_categ_features = self.graph_embedder(data)
        x_dict['child_cont'] = self.lin_dict['child_cont'](child_cont_features)
        x_dict['child_categ'] = self.lin_dict['child_categ'](child_categ_features)

        # Message passing
        x_dict = self.conv1(x_dict, data.edge_index_dict)

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
        
        return graph_emb

    def forward(self, data):
        # Get graph embedding
        graph_emb = self.compute_graph_embedding(data)

        # Compute classification output
        out = self.graph_lin(graph_emb)  # graph-level prediction
        
        return out