import torch

def positional_encoding(pos, d):
    """
        Generates sinusoidal positional encoding for a given position/time
        
        Parameters:
        -----------
        pos: int or torch.tensor of shape [num_nodes]
            Time (int) used to generate the positional embedding        
        d: intr
            Embedding dimension

        Returns:
        --------
        pe: torch.tensor
            Embedding tensor
    """
    if (type(pos) == torch.Tensor):
        pe = torch.zeros(len(pos), d)
    else:
        pos = torch.tensor(pos)
        pe = torch.zeros(1, d)
    for k in range(d):
        if k % 2 == 0: # Even k = 2*i
            div_term = 10000 ** (k / d)
            pe[:, k] = torch.sin(pos / div_term)
        else: # Odd k = 2*i+1
            div_term = 10000 ** ((k-1) / d)
            pe[:, k] = torch.cos(pos / div_term)
    return pe



class AiidkitTEAVGraphEmbedder(torch.nn.Module):
    def __init__(
                    self,
                    possible_values_all_patients,
                    categorical_ent_attr_pairs,
                    emb_dim_ent_attr=8,
                    emb_dim_ent_attr_vals=8
                ):
        """
            Model to get the embeddings of a TEAV AIIDKIT data representation, that
            can then be used with GNN models.
        """
        super().__init__()
        # Create IDs for all the entity-attribute pairs
        possible_ent_attr_pairs = list(possible_values_all_patients)
        ids_ent_attr_pairs = {possible_ent_attr_pairs[i]: i for i in range(len(possible_ent_attr_pairs))}
        self.inv_ids_ent_attr_pairs = {ids_ent_attr_pairs[ent_attr_pair]: ent_attr_pair for ent_attr_pair in ids_ent_attr_pairs}
        
        # Create embedding layer
        self.ent_attr_pair_emb_layer = torch.nn.Embedding(len(self.inv_ids_ent_attr_pairs), emb_dim_ent_attr)  # one embedding per entity-attr pair
        
        # Creating embedding layers for categorical values for entity-attribute pairs
        categorical_vals_vocabs = {}
        categorical_vals_emb_layers = {}
        for ent, attr in categorical_ent_attr_pairs:
            # Remove Unknown if exists
            categorical_ent_attr_pairs[(ent, attr)].discard('Unknown')
            categorical_ent_attr_pairs[(ent, attr)].discard('unknown')
            # Add Unknown token <UNK>
            categorical_ent_attr_pairs[(ent, attr)].add('<UNK>')
            categorical_vals_vocabs[(ent, attr)] = {value: index for index, value in enumerate(categorical_ent_attr_pairs[(ent, attr)])}
            tmp_ent = ent.replace(".", "") # Because torch.nn.ModuleDict does not accept . in the keys
            tmp_attr = attr.replace(".", "") # Because torch.nn.ModuleDict does not accept . in the keys
            categorical_vals_emb_layers[f"{tmp_ent}\\SEP\\{tmp_attr}"] = torch.nn.Embedding(len(categorical_vals_vocabs[(ent, attr)]), emb_dim_ent_attr_vals)
        self.categorical_vals_emb_layers = torch.nn.ModuleDict(categorical_vals_emb_layers)

    def forward(self, pat_data_graph):
        
        # Child node features
        # Continuous
        ent_attr_pair_cont_emb = self.ent_attr_pair_emb_layer(pat_data_graph['child_cont'].ent_attr_ids)
        ent_attr_pair_cont_emb = torch.cat([ent_attr_pair_cont_emb, pat_data_graph['child_cont'].vals.unsqueeze(-1)], dim=-1)
        pos_enc_cont_emb = positional_encoding(pos=pat_data_graph['child_cont'].days_since_tpx, d=ent_attr_pair_cont_emb.shape[1]).to(ent_attr_pair_cont_emb.device)
        child_cont_features = ent_attr_pair_cont_emb + pos_enc_cont_emb
        
        # Categorical
        ent_attr_pair_categ_emb = self.ent_attr_pair_emb_layer(pat_data_graph['child_categ'].ent_attr_ids)
        n = pat_data_graph['child_categ'].vocab_ids.shape[0]
        children_ent_attr_categorical_emb = []
        for j in range(n):
            vocab_id = pat_data_graph['child_categ'].vocab_ids[j]
            tmp_ent, tmp_attr = self.inv_ids_ent_attr_pairs[int(pat_data_graph['child_categ'].ent_attr_ids[j])]
            tmp_ent = tmp_ent.replace(".", "") # Because torch.nn.ModuleDict does not accept . in the keys
            tmp_attr = tmp_attr.replace(".", "") # Because torch.nn.ModuleDict does not accept . in the keys
            emb_layer_name = f"{tmp_ent}\\SEP\\{tmp_attr}"
            children_ent_attr_categorical_emb.append(self.categorical_vals_emb_layers[emb_layer_name](vocab_id))
        children_ent_attr_categorical_emb = torch.stack(children_ent_attr_categorical_emb)
        ent_attr_pair_categ_emb = ent_attr_pair_categ_emb + children_ent_attr_categorical_emb
        pos_enc_categ_emb = positional_encoding(pos=pat_data_graph['child_categ'].days_since_tpx, d=ent_attr_pair_categ_emb.shape[1]).to(ent_attr_pair_categ_emb.device)
        child_categ_features = ent_attr_pair_categ_emb + pos_enc_categ_emb

        return child_cont_features, child_categ_features



