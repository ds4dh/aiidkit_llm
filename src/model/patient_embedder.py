import os
import types
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from transformers.data.data_collator import DataCollatorMixin
from sentence_transformers import SentenceTransformer
from src.model.model_utils import TimeEmbedding, PositionalEncoding


class PatientEmbeddingModelFactory:
    """
    Factory to handle the lifecycle of a patient embedding model:
    1. create_from_backbone: hybrid HF backbone with patient input embedding layer
    2. from_pretrained: restores pretrained patient embedding model for finetuning/inference
    """
    @classmethod
    def create_from_backbone(
        cls,
        model_id: str,
        task: str,
        embedding_layer_config: dict,
        config_args: dict={},
        model_args: dict={},
        load_backbone_weights: bool=False,
    ):
        """
        Creates a new patient model by grafting a custom embedding layer onto a standard backbone
        """
        # Parse potential complex-type arguments
        for key in ["dtype", "torch_dtype"]:
            if isinstance(model_args.get(key), str):
                model_args[key] = getattr(torch, model_args[key])

        # Load model configuration from huggingface's directory
        config = AutoConfig.from_pretrained(model_id, **config_args, **model_args)

        # Initialize the Backbone (Pre-trained or Random)
        model_cls = cls._get_model_class(task)
        if load_backbone_weights:
            print(f"Initializing {model_id} backbone with pre-trained weights.")
            model = model_cls.from_pretrained(
                model_id, config=config, ignore_mismatched_sizes=True, **model_args
            )
        else:
            print(f"Initializing {model_id} backbone with random weights.")
            model = model_cls.from_config(config, **model_args)

        # Initialize Custom Layer (Always random at creation)
        custom_embed = PatientEmbeddingLayer(
            embedding_dim=config.hidden_size, **embedding_layer_config
        )

        # Graft custom input patient embedding layer to the classic backbone
        return cls._patch_model(model, custom_embed, embedding_layer_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: str,
        task: str,
        embedding_layer_config: dict,
        model_args: dict = {},
        **kwargs,
    ):
        """
        Restores a patient model from a local directory where it was saved
        """
        print(f"From pretrained unused arguments: {kwargs}")
        # Load Config from the local directory
        config = AutoConfig.from_pretrained(pretrained_dir, **model_args)

        # Build the skeleton with random weights
        model_cls = cls._get_model_class(task)
        model = model_cls.from_config(config, **model_args)

        # Re-create the custom Layer
        custom_embed = PatientEmbeddingLayer(
            embedding_dim=config.hidden_size, **embedding_layer_config
        )

        # Patch model structure before loading weights
        model = cls._patch_model(model, custom_embed, embedding_layer_config)

        # Load all weights (backbone and custom embedding layer)
        safe_path = os.path.join(pretrained_dir, "model.safetensors")
        if os.path.exists(safe_path):
            print(f"Loading patient model weights from {safe_path}...")
            state_dict = load_file(safe_path)
            # strict=False allows switching from MLM head (saved) to classification head (new)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No model.safetensors found in {pretrained_dir}")

        return model

    @staticmethod
    def _get_model_class(task: str):
        if task == "masked": return AutoModelForMaskedLM
        if task == "classification": return AutoModelForSequenceClassification
        raise ValueError("Task must be 'masked' or 'classification'")

    @staticmethod
    def _patch_model(model, custom_embed, config_dict):
        """
        Attaches the custom layer and overrides the forward method.
        """
        # Attach layer as a submodule so it saves/loads with state_dict
        custom_embed.to(dtype=model.dtype, device=model.device)
        model.patient_embedder = custom_embed

        # Define custom forward
        def custom_forward(self, input_dict=None, inputs_embeds=None, **kwargs):
            inputs_embeds = self.patient_embedder(**input_dict)
            outputs = self.original_forward(inputs_embeds=inputs_embeds, **kwargs)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                outputs.hidden_states = (outputs.hidden_states[-1],)
            return outputs

        # Apply Monkey Patch
        model.original_forward = model.forward
        model.forward = types.MethodType(custom_forward, model)

        # Expose config attributes
        model.use_pretrained_embeddings = custom_embed.use_pretrained_embeddings
        model.pretrained_model_name = custom_embed.pretrained_model_name
        model.config.max_position_embeddings = config_dict.get("max_position_embeddings", 512)

        return model


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
        self.use_positional_encoding = use_positional_encoding
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
class PatientDataCollatorForMaskedLanguageModelling(DataCollatorMixin):
    """
    Data collator used for the PatientEmbedding-based language model
    Modified from transformers.data.data_collator.py
    """
    pad_token_id: int=0
    mask_token_id: int=1
    bos_token_id: int=2
    eos_token_id: int=3
    unk_token_id: int=4
    input_keys: list[str] = field(default_factory=lambda: ["entity_id", "attribute_id"])
    mlm_masked_key: str="value_id"
    mlm_label_key: str="value_id"
    time_key: str="days_since_tpx"
    use_pretrained_embeddings: bool=True
    max_position_embeddings: int=512
    mlm_probability: float=0.15
    truncation_probability: float=0.5
    return_tensors: str="pt"

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Filter the kwargs to only include keys that exist in the dataclass
        Valid keys will automatically include keys from the subclass if called on the subclass
        """
        # cls.__dataclass_fields__ contains fields from the class and its parents
        return cls(**{
            k: v for k, v in kwargs.items() 
            if k in cls.__dataclass_fields__
        })

    def __post_init__(self):
        """
        Prepare data collator for embedding text sentences directly, with the time
        """
        if self.use_pretrained_embeddings:

            # Replace input keys by their corresponding text
            self.pad_token_id = "[PAD]"  # no ids, but texts
            self.mask_token_id = "[MASK]"  # no ids, but texts
            self.bos_token_id = "[BOS]"  # no ids, but texts
            self.eos_token_id = "[EOS]"  # no ids, but texts
            self.unk_token_id = "[UNK]"  # no ids, but texts
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
        effective_max_len = self.max_position_embeddings - 2  # for bos and eos tokens
        for i, sample in enumerate(samples):
            seq_len = next(iter(sample.values())).shape[0]

            # Augment the data by taking partial sequences only
            if self.truncation_probability is not None and self.truncation_probability > 0:
                if random.random() < self.truncation_probability:
                    truncated_len = random.randint(1, seq_len - 1)
                    samples[i] = {key: val[:truncated_len] for key, val in sample.items()}

            # Ensure the sequence does not go past the maximum model length
            if seq_len > effective_max_len:
                start_idx = random.randint(0, seq_len - effective_max_len)
                end_idx = start_idx + effective_max_len
                samples[i] = {key: val[start_idx:end_idx] for key, val in sample.items()}

        return samples

    def _add_bos_eos_token_ids(
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
            to_add = ([self.bos_token_id], [self.eos_token_id])

        return np.concatenate([to_add[0], sequence, to_add[-1]], axis=0)
        # return torch.cat([to_add[0], sequence, to_add[-1]], dim=0)
        
    def _add_special_token_ids(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        """
        For now, only add BOS and EOS tokens to each sequence in all samples
        """
        for i, sample in enumerate(samples):
            samples[i] = {k: self._add_bos_eos_token_ids(v, k) for k, v in sample.items()}
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
            padding_value = 0.0 if key == self.time_key else self.pad_token_id
            padded_batch[key] = self._pad_sequences_numpy(
                sequences, batch_first=True, padding_value=padding_value,
            )

        # Compute attention mask, given the padding performed on the sequences
        attention_mask = padded_batch[self.features_to_process[0]] != self.pad_token_id
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
        samples = self._add_special_token_ids(samples)
        padded_batch, attention_mask = self._pad_batch(samples)

        return padded_batch, attention_mask

    def torch_call(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]|np.ndarray]:
        """
        Collate patient sequences and create labels for LM training
        Used by HuggingFace's trainer
        """
        # Preprocess the input samples
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)

        # Prepare inputs and labels for the MLM task
        masked_input_dict = padded_batch
        masked_values, labels = self._masked_modelling(masked_input_dict)
        masked_input_dict[self.mlm_masked_key] = masked_values

        # Assemble the final input batch
        batch = {
            "input_dict": self._check_types(masked_input_dict),  # dict[str, np.ndarray|torch.tensor]
            "attention_mask": torch.tensor(attention_mask),      # torch.tensor[float]
            "labels": torch.tensor(labels).long(),               # torch.tensor[int]
        }
        batch.update(non_features)  # in case useful somewhere

        return batch

    def _check_types(
        self,
        input_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray|torch.Tensor]:
        """ Make sure the type of the batch elements is the correct one.
            The reason for this is that strings can only be put to np.ndarray,
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
        no_mask_ids = [self.bos_token_id, self.eos_token_id, self.unk_token_id, self.pad_token_id]
        no_mask_pad = torch.tensor(np.isin(mlm_labels, no_mask_ids))
        probability_matrix.masked_fill_(no_mask_pad, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Build the label array
        mlm_labels[~masked_indices] = -100  # special code to only compute loss on masked tokens
        mlm_labels = mlm_labels.astype(int)  # special tokens are -100, then all the rest is int

        # 80% of the time, replace masked input tokens with mask token
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        mlm_inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, replace masked input tokens with a random entry of the batch
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        flat_mlm_inputs = mlm_inputs.flatten()
        valid_choices = flat_mlm_inputs[~np.isin(flat_mlm_inputs, no_mask_ids)]
        random_tokens = valid_choices[np.random.randint(0, valid_choices.size, mlm_labels.shape)]
        mlm_inputs[indices_random] = random_tokens[indices_random]

        return mlm_inputs, mlm_labels


@dataclass
class PatientDataCollatorForClassification(PatientDataCollatorForMaskedLanguageModelling):
    """
    Inherits padding/truncation logic from the MLM collator, 
    but disables masking and extracts a specific target label.
    """
    label_key: str|None = None

    def torch_call(self, samples: list[dict[str, np.ndarray]]) -> dict:
        """
        Collate patient sequences and create labels for supervised fine-tuning
        Used by HuggingFace's trainer
        """
        # Separate features and labels
        if self.label_key is None:
            raise ValueError("Label key missing in patient data collator for classification")
        labels = [s.pop(self.label_key) for s in samples]

        # Preprocess the input samples
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)

        # Assemble batch
        batch = {
            "input_dict": self._check_types(padded_batch),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels).long(),  # classification loss expects long tensors
        }
        batch.update(non_features)  # in case useful somewhere

        return batch