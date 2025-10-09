import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig
from transformers.utils import ModelOutput
from transformers.data.data_collator import DataCollatorMixin
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from dataclasses import dataclass, field
from typing import Optional

from src.model.model_utils import TimeEmbedding, PositionalEncoding, FocalLoss
import src.constants as constants
csts = constants.ConstantsNamespace()


@dataclass
class PatientTokenEmbeddingOutput(ModelOutput):
    """
    Custom output class with fields for a potiential supervised task
    This ensures a consistent output structure:
    - loss (total),
    - mlm_loss, mlm_logits,
    - supervised_task_loss, supervised_task_logits
    - supervised_task_hidden_states, supervised_task_pooled_embeddings
    """
    loss: Optional[torch.FloatTensor] = None  # total loss
    mlm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    cutoff_days: Optional[torch.FloatTensor] = None
    supervised_task_loss: Optional[torch.FloatTensor] = None
    supervised_task_logits: Optional[torch.FloatTensor] = None
    supervised_task_hidden_states: Optional[tuple[torch.Tensor]] = None
    supervised_task_pooled_embeddings: Optional[torch.Tensor] = None


class PatientTokenEmbeddingModel(nn.Module):
    def __init__(
        self,
        vocabs: dict[str, dict[str, int]],
        original_model_id: str,
        original_model_task: str,
        original_model_params: dict[str, int]=None,
        send_hidden_states_to_cpu: bool=True,
        use_supervised_task: bool=False,
        use_uncertainty_weighting: bool=False,    
        supervised_task_type: str="binary",
        supervised_task_weight: float=0.0,
        use_positional_encoding_for_input_layer: bool=True,
        use_pretrained_embeddings_for_input_layer: bool=True,
        pretrained_model_name_for_input_layer: str="NeuML/pubmedbert-base-embeddings",
        *args, **kwargs,
    ):
        super().__init__()
        assert original_model_task in ["masked", "causal"],\
            "original_model_task must be masked or causal"
        self.is_causal = original_model_task == "causal"
        self.send_hidden_states_to_cpu = send_hidden_states_to_cpu

        # Load parameters of the given model and modify them as required
        config = AutoConfig.from_pretrained(original_model_id)
        if original_model_params is not None:
            for value, key in original_model_params.items():
                setattr(config, value, key)

        # Initialize model from an original pre-trained LLM and return hidden states
        if original_model_task == "masked":
            self.llm = AutoModelForMaskedLM.from_config(config)
        elif original_model_task == "causal":
            self.llm = AutoModelForCausalLM.from_config(config)
        self.hidden_size = self.llm.config.hidden_size
        self.num_tokens_max = self.llm.config.max_position_embeddings

        # Create embedding layer and replace the LLM one to prevent incompatibility
        self.input_embedding_layer = PatientEmbeddingLayer(
            embedding_dim=self.hidden_size,
            vocabs=vocabs,
            use_positional_encoding=use_positional_encoding_for_input_layer,
            use_pretrained_embeddings=use_pretrained_embeddings_for_input_layer,
            pretrained_model_name=pretrained_model_name_for_input_layer,
        )

        # Modify the LLM head (classifier) to match the number of value tokens
        num_value_tokens = len(vocabs["value_binned"])
        self.llm.config.vocab_size = num_value_tokens
        new_decoder_layer = nn.Linear(
            in_features=self.hidden_size, 
            out_features=num_value_tokens,
            bias=True,
        )
        self.llm.set_output_embeddings(new_decoder_layer)

        # Supervised task setup, if required
        self.use_supervised_task = use_supervised_task
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.supervised_task_weight = supervised_task_weight
        if use_supervised_task:
            self._init_supervised_modules_and_loss(supervised_task_type)

    def _init_supervised_modules_and_loss(
        self,
        supervised_task_type: str,
    ) -> None:
        """
        Initialize modules and loss function for a potential supervised task
        """
        # Module to pool token embeddings into a single vector
        self.pooler = Pooling(
            word_embedding_dimension=self.hidden_size,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True,
            pooling_mode_mean_tokens=False,
        )

        # Classifier for the supervised task, taking pooled embeddings as input
        self.supervised_task_num_classes = 2 if supervised_task_type == "binary" else 4
        # self.supervised_task_decoder = nn.Sequential(
        #         nn.Linear(self.pooler.pooling_output_dimension, self.hidden_size),
        #         nn.LayerNorm(self.hidden_size),
        #         nn.GELU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(self.hidden_size, self.supervised_task_num_classes),
        #     )
        self.supervised_task_decoders = nn.ModuleDict({
            str(cutoff_day): nn.Sequential(
                nn.Linear(self.pooler.pooling_output_dimension, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.supervised_task_num_classes),
            )
            for cutoff_day in csts.CUTOFF_DAYS
        })

        # Uncertainty weighting parameters, if required
        if self.use_uncertainty_weighting:
            self.log_var_mlm = nn.Parameter(torch.tensor(0.0))
            self.log_var_sup = nn.Parameter(torch.tensor(0.0))

        # Loss to learn the supervised task
        self.supervised_task_loss_fn = FocalLoss(gamma=2.0)

    def forward(
        self,
        unmasked_input_dict: dict[str: list[str|int]],
        masked_input_dict: dict[str: list[str|int]],
        attention_mask: Optional[torch.Tensor]=None,
        mlm_labels: Optional[torch.LongTensor]=None,
        supervised_task_labels: Optional[torch.LongTensor]=None,
        output_attentions: Optional[bool]=None,
        **kwargs,
    ) -> PatientTokenEmbeddingOutput:
        """
        Forward function of the patient embedding model
        - during training: use masked inputs for both the MLM and supervised task
        - during inference: use unmasked inputs for the supervised task
        """
        # Initialize conditional outputs to default values
        # This prevents inconsistencies when collecting predictions in an evaluation loop
        device = attention_mask.device if attention_mask is not None else "cpu"
        hidden_states, pooled_embeddings = torch.empty(0), torch.empty(0)
        cutoff_days = torch.empty(0)
        sup_logits, mlm_logits = torch.empty(0), torch.empty(0)
        sup_loss = torch.tensor(0.0, device=device)

        # Perform forward pass through the embedding layer and the LLM
        output_mlm_hidden_states = self.training and self.use_supervised_task
        mlm_input_embeddings = self.input_embedding_layer(**masked_input_dict)
        mlm_outputs = self.llm(
            inputs_embeds=mlm_input_embeddings,
            attention_mask=attention_mask,
            labels=mlm_labels,
            output_hidden_states=output_mlm_hidden_states,
            output_attentions=output_attentions,
        )

        # Extract necessary outputs
        mlm_loss = mlm_outputs.loss
        mlm_logits = mlm_outputs.logits
        if output_mlm_hidden_states:
            hidden_states = self._lighten_hidden_states(mlm_outputs.hidden_states, attention_mask)

        # Process supervised task outputs
        if self.use_supervised_task:

            # If in evaluation mode, use unmasked embeddings as input
            if not self.training:
                llm_input_embeddings = self.input_embedding_layer(**unmasked_input_dict)
                llm_outputs = self.llm(
                    inputs_embeds=llm_input_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
                hidden_states = self._lighten_hidden_states(
                    hidden_states=llm_outputs.hidden_states,
                    attention_mask=attention_mask,
                )

            # Pool the embeddings to get a single vector representation for the sequence
            pooled_embeddings = self.pooler({
                "token_embeddings": hidden_states[-1],
                "attention_mask": attention_mask,
            })["sentence_embedding"]

            # # /!\ DEBUG /!\ #
            # linear_readout_mode = True
            # if linear_readout_mode:
            #     pooled_embeddings = pooled_embeddings.detach()
            # # /!\ DEBUG /!\ #

            # Compute supervised logits and loss if labels are provided
            # sup_logits = self.supervised_task_decoder(pooled_embeddings)
            sup_logits = torch.empty(
                pooled_embeddings.size(0),
                self.supervised_task_num_classes,
                device=pooled_embeddings.device,
            )
            cutoffs = [str(c) for c in kwargs["cutoff"]]
            for cutoff_day in set(cutoffs):
                indices = [i for i, c in enumerate(cutoffs) if c == cutoff_day]
                cutoff_embeddings = pooled_embeddings[indices]
                cutoff_decoder = self.supervised_task_decoders[cutoff_day]
                cutoff_logits = cutoff_decoder(cutoff_embeddings)
                sup_logits[indices] = cutoff_logits.to(sup_logits.dtype)
            
            cutoffs = [int(c) if c != "full" else -1 for c in cutoffs]
            cutoff_days = torch.tensor(cutoffs, device=device)

            sup_loss = self.supervised_task_loss_fn(sup_logits, supervised_task_labels)

        # Compute final loss (case where there is no supervised task)
        if not self.use_supervised_task:
            total_loss = mlm_loss

        # Use uncertainty weighting to mix losses, see in Kandall et al. (2018)
        elif self.use_uncertainty_weighting:

            # The first part of the loss is the precision-weighted task losses
            weighted_mlm_loss = torch.exp(-self.log_var_mlm) * mlm_loss
            weighted_sup_loss = torch.exp(-self.log_var_sup) * sup_loss
            regularization = 0.5 * (self.log_var_mlm + self.log_var_sup)
            total_loss = weighted_mlm_loss + weighted_sup_loss + regularization

        # Combine losses with a "fixed" task weight (might be modified by callback scheduler)
        else:
            weighted_mlm_loss = (1 - self.supervised_task_weight) * mlm_loss  # mlm_loss
            weighted_sup_loss = self.supervised_task_weight * sup_loss
            total_loss = weighted_mlm_loss + weighted_sup_loss

        # Optionally, send hidden states to CPU to save GPU memory
        if self.send_hidden_states_to_cpu:
            hidden_states = tuple(h.detach().cpu() for h in hidden_states)

        # Return all relevant outputs in a structured object
        return PatientTokenEmbeddingOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            mlm_logits=mlm_logits,
            cutoff_days=cutoff_days,
            supervised_task_loss=sup_loss,
            supervised_task_logits=sup_logits,
            supervised_task_hidden_states=hidden_states,
            supervised_task_pooled_embeddings=pooled_embeddings,
        )

    def _lighten_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
    ):
        """ Avoid keeping all hidden states in memory (only the last one)
        """
        # Extract last hidden states
        last_hidden_state = hidden_states[-1]

        # Exceptions for some models when using flash-attn-2
        if last_hidden_state.ndim != 3:
            batch_size, seq_len = attention_mask.shape
            hidden_size = last_hidden_state.shape[-1]
            unflattened = torch.zeros(
                batch_size, seq_len, hidden_size,
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype
            )
            unflattened[attention_mask.bool()] = last_hidden_state
            last_hidden_state = unflattened  # chelou non
        
        return (last_hidden_state,)


class PatientEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocabs: dict[str, dict[str, int]],
        vocab_mapping: dict[str, str]={
            "entity": "entity_id",
            "attribute": "attribute_id",
            "value_binned": "value_id",
        },
        time_key: str="days_since_tpx",
        use_positional_encoding: bool=True,
        use_pretrained_embeddings: bool=True,
        pretrained_model_name: str="NeuML/pubmedbert-base-embeddings",
    ):
        """
        Layer to generate input embedddings given patient data sequences
        Args:
            embedding_dim (int): dimension of the generated embeddings
            vocabs (dict[str, dict[str, int]]): dictionary mapping feature names
                to their respective vocabulary dictionaries, from string to int id
            vocab_mapping (dict[str, str]): dictionary mapping the keys in `vocabs`
                to the corresponding keys in the input `kwargs` during the forward pass
            time_key (str): key in forward `kwargs` that corresponds to the time feature
            use_positional_encoding (bool): Whether to apply positional encoding to
                the combined embeddings
            use_pretrained_embeddings (bool): whether to initialize embedding layers
                with embeddings from a pretrained SentenceTransformer model.
            pretrained_model_name (str): model used for `use_pretrained_embeddings`
            freeze_pretrained (bool): freeze model for `use_pretrained_embeddings`
        """
        super().__init__()

        # Create embedding layers for each feature vocabulary
        if vocabs.keys() != vocab_mapping.keys():
            raise KeyError(f"Required vocab keys: {list(vocab_mapping.keys())}")
        self.vocabs = vocabs
        self.vocab_mapping = vocab_mapping
        self.embedding_dim = embedding_dim

        # Initialize pretrained embedding model, if required
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_model_name = pretrained_model_name
        self._init_embedding_strategy()

        # Special embedding layer for time feature
        self.time_key = time_key
        self.time_embedding = TimeEmbedding(embedding_dim)

        # Positional encoding to encode relative position between events
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(embedding_dim)
        else:
            self.positional_encoding = None

    def _init_embedding_strategy(self) -> nn.ModuleDict:
        """
        Initialize ModuleDict for token embeddings and create embedding layers
        from pretrained embeddings or from scratch for each feature vocabulary
        """
        # Create embedding layers from scratch for each feature vocabulary
        if not self.use_pretrained_embeddings:
            token_embedding_dict = nn.ModuleDict()
            
            for vocab_key, vocab in self.vocabs.items():
                input_key = self.vocab_mapping[vocab_key]
                token_embedding_dict[input_key] = nn.Embedding(
                    num_embeddings=len(vocab), 
                    embedding_dim=self.embedding_dim
                )

            self.required_eav_keys = list(self.vocab_mapping.values())
            self.token_embedding_dict = token_embedding_dict
            self.event_sentence_embedding_model_list = None

        # Create embedding layers from pretrained embeddings for each feature vocabulary
        else:
            sentence_model = SentenceTransformer(self.pretrained_model_name, device="cpu")
            sentence_model.eval()
            for param in sentence_model.parameters():
                param.requires_grad = False
            
            # Check if the model's embedding dimension matches the desired one
            model_embedding_dim = sentence_model.get_sentence_embedding_dimension()
            if model_embedding_dim != self.embedding_dim:
                raise ValueError(
                    f"The `embedding_dim` ({self.embedding_dim}) does not match the "
                    f"pretrained model's dimension ({model_embedding_dim}). "
                    "Please adjust `embedding_dim` or choose a different model."
                )
            
            # Note: the sentence embedding model is registered as a list to avoid
            # automatic submodule registration which would send it to GPU
            self.required_eav_keys = list(self.vocab_mapping.keys())
            self.event_sentence_embedding_model_list = [sentence_model]
            self.event_embedding_cache = {}  # avoid re-computing already seen ones
            self.token_embedding_dict = None

    def _get_embeddings_from_scratch_from_eav_ids(
        self,
        kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute embeddings by summing individual feature embeddings
        """
        return sum(
            self.token_embedding_dict[key](sequence)
            for key, sequence in kwargs.items()
            if key in self.required_eav_keys
        )

    def _construct_eav_event_sentence(
        self,
        entity: str,
        attribute: str,
        value: str,
    ) -> str:
        """
        Helper function to construct a sentence from an EAV triplets
        """
        # Special case which do not follow the structure of the EAV event sentence
        if entity == "[BOS]": return "Start of patient medical event sequence."
        if entity == "[EOS]": return "End of patient medical event sequence."

        return f"Event for the patient's {entity}: the value of {attribute} was {value}"

    def _get_pretrained_embeddings_with_sentence_model(
        self,
        kwargs: dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Calculates embeddings using a pretrained sentence embedding model
        """
        # Extract EAV data as flattened arrays after getting their shape
        batch_size, seq_len = kwargs["entity"].shape
        entities = kwargs["entity"].reshape(-1)
        attributes = kwargs["attribute"].reshape(-1)
        values = kwargs["value_binned"].reshape(-1)

        # Store all triplets as a list to preserve order in the final embeddings
        all_triplets = list(zip(entities, attributes, values))

        # Identify which triplets are new and which are cached
        new_triplets_to_encode = []
        unique_new_triplets = set()
        for triplet in all_triplets:

            # Skip padding tokens and triplets that were already seen
            if triplet[0] == "[PAD]" or triplet in self.event_embedding_cache:
                continue
            if triplet not in unique_new_triplets:
                unique_new_triplets.add(triplet)
                new_triplets_to_encode.append(triplet)

        # If there are new, unseen triplets, encode them
        event_embedding_model = self.event_sentence_embedding_model_list[0]
        if new_triplets_to_encode:

            # Build sentences from unseen event triplets
            sentences_to_encode = [
                self._construct_eav_event_sentence(e, a, v)
                for e, a, v in new_triplets_to_encode
            ]

            # Encode newly constructed sentences
            with torch.no_grad():
                new_embeddings: torch.Tensor = event_embedding_model.encode(
                    sentences=sentences_to_encode,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )

            # Populate the cache with the newly computed embeddings
            for triplet, embedding in zip(new_triplets_to_encode, new_embeddings):
                self.event_embedding_cache[triplet] = embedding

        # Pre-fill a tensor for the whole batch for performance
        embedding_dim = event_embedding_model.get_sentence_embedding_dimension()
        flat_embeddings = torch.zeros(len(all_triplets), embedding_dim, device="cpu")

        # Populate the final embedding tensor using the cache
        # Note: order is preserved thanks to all_triplets
        for i, triplet in enumerate(all_triplets):
            if triplet[0] != "[PAD]":  # leave [PAD] tokens as vectors of zeros
                flat_embeddings[i] = self.event_embedding_cache[triplet]

        # Reshape the flattened tensor to the final required embedding shape
        eav_embeddings = flat_embeddings.view(batch_size, seq_len, embedding_dim)

        return eav_embeddings

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for the patient embedding layer
        """
        # Check input arguments
        required_keys = self.required_eav_keys + [self.time_key]
        if not all(key in kwargs for key in required_keys):
            raise KeyError(f"Missing one or more required inputs. Expected: {required_keys}")

        # Get the event triplet embeddings
        if self.use_pretrained_embeddings:
            eav_embeddings = self._get_pretrained_embeddings_with_sentence_model(kwargs)
        else:
            eav_embeddings = self._get_embeddings_from_scratch_from_eav_ids(kwargs)
        eav_embeddings = eav_embeddings.to(kwargs[self.time_key].device)

        # Apply time embeddings using the time input
        time_embeddings = self.time_embedding(x=eav_embeddings, times=kwargs[self.time_key])
        final_embeddings = eav_embeddings + time_embeddings

        # Apply relative positional encoding, if required
        if self.positional_encoding is not None:
            final_embeddings = self.positional_encoding(final_embeddings)

        return final_embeddings


@dataclass
class PatientDataCollatorForLanguageModelling(DataCollatorMixin):
    """
    Data collator used for the PatientEmbedding-based language model
    Modified from transformers.data.data_collator.py
    """
    mlm: bool=True
    pad_id: int=0
    mask_id: int=1
    bos_id: int=2
    eos_id: int=3
    unk_id: int=4
    input_keys: list[str] = field(default_factory=lambda: ["entity_id", "attribute_id"])
    mlm_masked_key: str="value_id"
    mlm_label_key: str="value_id"
    time_key: str="days_since_tpx"
    use_supervised_task: bool=False
    supervised_task_key: str="infection_label_binary"
    use_pretrained_embeddings: bool=True
    num_tokens_max: int=512
    num_mlm_labels: Optional[int]=None
    mlm_probability: float=0.15
    return_tensors: str="pt"

    def __post_init__(self):
        """
        Prepare data collator for embedding text sentences directly, with the time
        """
        if self.use_pretrained_embeddings:

            # Replace input keys by their corresponding text
            self.pad_id = "[PAD]"  # no ids, but texts
            self.mask_id = "[MASK]"  # no ids, but texts
            self.bos_id = "[BOS]"  # no ids, but texts
            self.eos_id = "[EOS]"  # no ids, but texts
            self.unk_id = "[UNK]"  # no ids, but texts
            self.input_keys = ["entity", "attribute"]
            self.mlm_masked_key = "value_binned"

            # But time key and MLM labels stay the same (int ids)  
            self.mlm_label_key = "value_id"
            self.time_key = "days_since_tpx"

        added_keys = [self.mlm_masked_key, self.mlm_label_key, self.time_key]
        self.features_to_process = list(set(self.input_keys + added_keys))

    def _separate_non_processed_features(
        self,
        samples: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Separate non-feature labels (e.g., for classification) from the samples
        """
        # Identify which keys are the non-features
        sample_keys = samples[0].keys()
        non_processed_keys = [k for k in sample_keys if k not in self.features_to_process]
        if not non_processed_keys or not samples: return samples, {}  # useful?

        # Extract the labels by popping them out of each sample
        external_labels = {
            key: [s.pop(key) for s in samples] for key in non_processed_keys
        }

        return samples, external_labels

    def _truncate_sequences(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        """
        Truncate samples longer than the maximum allowed sequence length
        """
        effective_max_len = self.num_tokens_max - 2  # for bos and eos tokens
        for i, sample in enumerate(samples):
            seq_len = next(iter(sample.values())).shape[0]
            if seq_len > effective_max_len:
                start_idx = random.randint(0, seq_len - effective_max_len)
                end_idx = start_idx + effective_max_len
                samples[i] = {key: val[start_idx:end_idx] for key, val in sample.items()}

        return samples

    def _add_bos_eos_ids(
        self,
        sequence: np.ndarray,
        data_key: str,
    ) -> np.ndarray:
        """
        Add bos and eos token ids or first and last time to a sequence
        given the data it contains
        """
        if data_key == self.time_key:
            to_add = ([sequence[0]], [sequence[-1]])
        else:
            to_add = ([self.bos_id], [self.eos_id])

        return np.concatenate([to_add[0], sequence, to_add[-1]], axis=0)
        # return torch.cat([to_add[0], sequence, to_add[-1]], dim=0)
        
    def _add_special_tokens(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        """
        For now, only add BOS and EOS tokens to each sequence in all samples
        """
        for i, sample in enumerate(samples):
            samples[i] = {k: self._add_bos_eos_ids(v, k) for k, v in sample.items()}
        return samples

    def _pad_batch(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        """
        Pad all sequences in the batch to the same length
        """
        # Pad sequences into rectangular array
        padded_batch: dict[str, np.ndarray] = {}
        for key in self.features_to_process:
            sequences = [s[key] for s in samples]
            padding_value = 0.0 if key == self.time_key else self.pad_id
            padded_batch[key] = self._pad_sequences_numpy(
                sequences, batch_first=True, padding_value=padding_value,
            )

        # Compute attention mask, given the padding performed on the sequences
        attention_mask = padded_batch[self.features_to_process[0]] != self.pad_id
        attention_mask = attention_mask.astype(int)

        return padded_batch, attention_mask

    def _process_batch(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        """
        Pipeline to truncate, encapsulate, pad samples, and compute attention masks
        """
        samples = self._truncate_sequences(samples)
        samples = self._add_special_tokens(samples)
        padded_batch, attention_mask = self._pad_batch(samples)

        return padded_batch, attention_mask

    def torch_call(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]|np.ndarray]:
        """
        Collate patient embedding samples and create labels for LM training
        Used by the huggingface trainer
        """
        # Preprocess the input samples
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)

        # Prepare unmasked data and labels for the supervised task
        if self.use_supervised_task:
            unmasked_input_dict = {k: v.copy() for k, v in padded_batch.items()}
            if self.use_pretrained_embeddings: del unmasked_input_dict[self.mlm_label_key]
            supervised_task_labels = np.array(non_features[self.supervised_task_key])
        else:
            unmasked_input_dict = None
            supervised_task_labels = None

        # Prepare inputs and labels for the MLM task
        masked_input_dict = padded_batch
        if self.mlm:
            masked_values, mlm_labels = self._masked_modelling(masked_input_dict)
            masked_input_dict[self.mlm_masked_key] = masked_values
        else:  # causal LM (not sure if I keep this, but we never know)
            mlm_labels = self._causal_modelling(masked_input_dict)

        # Assemble the final input batch
        batch = {
            "masked_input_dict": self._check_types(masked_input_dict),       # dict[str, np.ndarray|torch.tensor]
            "unmasked_input_dict": self._check_types(unmasked_input_dict),   # dict[str, np.ndarray|torch.tensor]
            "attention_mask": torch.tensor(attention_mask),                  # torch.tensor
            "mlm_labels": torch.tensor(mlm_labels),                          # torch.tensor
            "supervised_task_labels": torch.tensor(supervised_task_labels),  # torch.tensor
        }
        batch.update(non_features)  # in case useful somewhere

        return batch

    def _check_types(
        self,
        input_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray|torch.Tensor]:
        """ Make sure the type of the batch elements is the correct one.
            The reason for all this is that strings can only be put to np.ndarray,
            while other elements are better inside tensors
        """
        for key, value in input_dict.items():
            if key == self.time_key or not self.use_pretrained_embeddings:
                input_dict[key] = torch.tensor(value)

        return input_dict

    @staticmethod
    def _pad_sequences_numpy(
        sequences: list[np.ndarray],
        batch_first: bool=False,
        padding_value: int|float|str=0,
    ):
        """
        Pads a list of variable-length sequences with a padding value.

        This function is a NumPy equivalent of torch.nn.utils.rnn.pad_sequence.

        Args:
            sequences (list of np.ndarray): sequences to pad
            batch_first (bool, optional): batch dimension as the first dimension
            padding_value (float, optional): value to use for padding

        Returns:
            np.ndarray: The padded sequences as a single NumPy array.
        """
        # Define the output shape
        max_len = max([s.shape[0] for s in sequences])
        trailing_dims = sequences[0].shape[1:]
        if batch_first:  # (batch_size, max_len, *trailing_dims)
            out_shape = (len(sequences), max_len) + trailing_dims
        else:  # (max_len, batch_size, *trailing_dims)
            out_shape = (max_len, len(sequences)) + trailing_dims

        # Create the padded array, copying values from the original sequences
        padded_sequences = np.full(out_shape, padding_value, dtype=sequences[0].dtype)
        for i, s in enumerate(sequences):
            length = s.shape[0]
            if batch_first:
                padded_sequences[i, :length, ...] = s
            else:
                padded_sequences[:length, i, ...] = s

        return padded_sequences

    def _masked_modelling(
        self,
        masked_input_dict: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs and labels for masked language modeling
        Modified from transformers.data.data_collator.py
        """
        # Take input values to mask and output ids to retrieve. Using copy()
        # ensures labels and inputs do not refer to the same object when the
        # masked key and the label key are the same
        mlm_inputs = masked_input_dict.get(self.mlm_masked_key).copy()
        mlm_labels = masked_input_dict.pop(self.mlm_label_key)

        # Sample masked locations in the batch, using mlm_probability
        probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
        no_mask_ids = [self.bos_id, self.eos_id, self.unk_id, self.pad_id]
        no_mask_pad = torch.tensor(np.isin(mlm_labels, no_mask_ids))
        probability_matrix.masked_fill_(no_mask_pad, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Build the label array
        mlm_labels[~masked_indices] = -100  # special code to only compute loss on masked tokens
        mlm_labels = mlm_labels.astype(int)  # special tokens are -100, then all the rest is int

        # 80% of the time, replace masked input tokens with mask token
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        mlm_inputs[indices_replaced] = self.mask_id

        # 10% of the time, replace masked input tokens with a random entry of the batch
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        flat_mlm_inputs = mlm_inputs.flatten()
        valid_choices = flat_mlm_inputs[~np.isin(flat_mlm_inputs, no_mask_ids)]
        random_tokens = valid_choices[np.random.randint(0, valid_choices.size, mlm_labels.shape)]
        mlm_inputs[indices_random] = random_tokens[indices_random]

        return mlm_inputs, mlm_labels

    def _causal_modelling(
        self,
        masked_input_dict: dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Prepare labels for causal language modeling by shifting tokens to the right
        Modified from transformers.data.data_collator.py
        """
        raise NotImplementedError
        # TODO: CHECK WHAT TO CHANGE TO MAKE IT WORK WITH TEXT AND NOT JUST IDS
        # # Create labels from inputs (no need of manual shift: done in trainer)
        # inputs = masked_input_dict[self.mlm_masked_key]
        # labels = inputs.clone()

        # # Ensure original padding tokens are also ignored in the labels
        # if self.pad_id is not None:
        #     labels[inputs == self.pad_id] = -100

        # return labels
