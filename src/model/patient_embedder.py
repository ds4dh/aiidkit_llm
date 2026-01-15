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
from src.model.model_utils import TimeEmbedding


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

        # Initialize the backbone (pre-trained or random)
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
        task: str,
        pretrained_dir: str,
        embedding_layer_config: dict,
        model_args: dict = {},
        reset_weights: bool = False,
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

        # Return a randomly initialized model if requested
        if reset_weights:
            print("Resetting model weights to random initialization.")
            return model

        # Load pretrained weights (backbone and custom embedding layer)
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

        # Apply monkey patch
        model.original_forward = model.forward
        model.forward = types.MethodType(custom_forward, model)

        # Expose config attributes
        model.use_direct_text_input = custom_embed.use_direct_text_input
        model.sentence_embedding_model = custom_embed.sentence_embedding_model
        model.config.max_position_embeddings = config_dict.get("max_position_embeddings", 512)

        return model


class PatientEmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        eav_keys: list[str]=["entity_id", "attribute_id", "value_id"],
        time_key: str="days_since_tpx",
        use_direct_text_input: bool=False,
        sentence_embedding_model: str|None=None,
    ):
        """
        Layer to generate input embedddings given patient data sequences
        Args:
            vocab_size (int): size of the vocabulary for token embeddings
            embedding_dim (int): dimension of the generated embeddings
            time_key (str): key in forward `kwargs` that corresponds to the time feature
            use_direct_text_input (bool): whether to initialize embedding layers
                with embeddings from a pretrained SentenceTransformer model.
            sentence_embedding_model (str): model used for `use_direct_text_input`
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eav_keys = eav_keys
        self.time_key = time_key
        self.use_direct_text_input = use_direct_text_input
        self.sentence_embedding_model = sentence_embedding_model

        # Time embedding layer (always used)
        self.time_embedding = TimeEmbedding(embedding_dim)

        # Train from scratch (using medical code vocabulary and embedding layer)
        if not use_direct_text_input:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Use a pretrained sentence transformer to embed textual event descriptions
        else:
            self.sentence_model = SentenceTransformer(sentence_embedding_model, device="cpu")
            self.sentence_model.eval()
            for param in self.sentence_model.parameters():
                param.requires_grad = False
            self.event_embedding_cache = {}

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

    def _get_sentence_embedding_model(
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
        Forward pass for the patient embedding layer.
        """
        # Compute base embeddings
        if self.use_direct_text_input:
            x = self._get_sentence_embedding_model(kwargs)
        else:
            x = sum(self.token_embedding(kwargs[key]) for key in self.eav_keys)

        # Add time embeddings to the base embeddings
        x = x.to(kwargs[self.time_key].device)
        x = x + self.time_embedding(x, kwargs[self.time_key])
            
        return x


@dataclass
class PatientDataCollatorForMaskedLanguageModelling(DataCollatorMixin):
    """
    Data collator used for the PatientEmbedding-based language model
    """
    # Special tokens
    pad_token_id: int | str = 0
    mask_token_id: int | str = 1
    bos_token_id: int | str = 2
    eos_token_id: int | str = 3
    unk_token_id: int | str = 4
    
    # Configuration
    feature_keys: list[str] = field(default_factory=lambda: [
        "days_since_tpx",
        "entity_id",
        "attribute_id",
        "value_id",
    ])
    time_key: str = "days_since_tpx"
    mlm_masking_rules: dict[str, float] = field(default_factory=lambda: {"value_id": 0.15})
    mlm_label_keys: list[str] = field(default_factory=lambda: ["value_id"])
    
    # Flags
    do_augmentation: bool = False  # updated for different dataset splits
    max_position_embeddings: int = 512
    return_tensors: str = "pt"
    
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
    
    def _separate_non_processed_features(
        self,
        samples: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Separate non-feature labels (e.g., for classification) from the samples
        """
        # Identify which keys are the non-features
        sample_keys = samples[0].keys()
        non_processed_keys = [k for k in sample_keys if k not in self.feature_keys]
        if not non_processed_keys or not samples: return samples, {}  # useful?

        # Extract the labels by popping them out of each sample
        external_labels = {
            key: [s.pop(key) for s in samples] for key in non_processed_keys
        }

        return samples, external_labels

    def _augment_shuffle_concurrent_events(
        self, 
        samples: list[dict[str, np.ndarray]]
    ) -> list[dict[str, np.ndarray]]:
        """
        Data augmentation: shuffle events that share the same timestamp
        """
        if not self.do_augmentation: return samples
        
        for i, sample in enumerate(samples):
            
            # Find unique timestamps and their occurrence counts
            times = sample[self.time_key]
            unique_times, counts = np.unique(times, return_counts=True)
            if len(unique_times) == len(times): continue

            # Iterate over timestamps that appear more than once
            indices = np.arange(len(times))
            for t in unique_times[counts > 1]:
                
                # Extract indices for this time, shuffle them in-place, and re-assign
                mask = times == t
                subset = indices[mask]
                np.random.shuffle(subset)
                indices[mask] = subset

            # Reorder all feature arrays in the sample using the shuffled indices
            samples[i] = {k: v[indices] for k, v in sample.items()}

        return samples

    def _augment_jitter_time(self, samples: list[dict], noise_level: float=0.2) -> list[dict]:
        """
        Data augmentation: add small Gaussian noise to time entries
        """
        if not self.do_augmentation: return samples

        for sample in samples:
            time = np.array(sample[self.time_key], dtype=np.float32)
            noise = np.random.normal(0, noise_level, size=time.shape)          
            sample[self.time_key] = time + noise.astype(np.float32)
            
        return samples

    def _augment_random_crop(
        self, 
        samples: list[dict[str, np.ndarray]]
    ) -> list[dict[str, np.ndarray]]:
        """
        Data augmentation: randomly crop the sequence to a shorter length.
        """
        if not self.do_augmentation: return samples
        
        for i, sample in enumerate(samples):
            seq_len = next(iter(sample.values())).shape[0]
            if seq_len > 1 and random.random() < 0.5:  # only 50% of the time
                truncated_len = random.randint(1, seq_len - 1)
                samples[i] = {key: val[:truncated_len] for key, val in sample.items()}

        return samples

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
        for key in self.feature_keys:
            sequences = [s[key] for s in samples]
            if key == self.time_key or key in self.mlm_label_keys:
                padding_value = 0
            else:
                padding_value = self.pad_token_id
            padded_batch[key] = self._pad_sequences_numpy(
                sequences, batch_first=True, padding_value=padding_value,
            )

        # Compute attention mask, given the padding performed on the sequences
        attention_mask = padded_batch[self.feature_keys[0]] != self.pad_token_id
        attention_mask = attention_mask.astype(int)

        return padded_batch, attention_mask

    def _process_batch(
        self,
        samples: list[dict[str, np.ndarray]],
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        """
        Pipeline to truncate, encapsulate, pad samples, and compute attention masks
        """
        if self.do_augmentation:
            samples = self._augment_shuffle_concurrent_events(samples)
            samples = self._augment_jitter_time(samples)
            # samples = self._augment_random_crop(samples)
            
        samples = self._truncate_sequences(samples)
        samples = self._add_special_token_ids(samples)
        padded_batch, attention_mask = self._pad_batch(samples)

        return padded_batch, attention_mask

    def torch_call(self, samples: list[dict[str, np.ndarray]]) -> dict:
        """
        Main entry point used by HuggingFace Trainer.
        """
        self.do_augmentation = (samples[0]["split"] == "train")
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)
        masked_input_dict, labels = self._masked_modelling(padded_batch)
        batch = {
            # "input_dict": self._check_types(masked_input_dict),
            "input_dict": {k: torch.tensor(v) for k, v in masked_input_dict.items()},
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels).long(),
        }
        batch.update(non_features)

        return batch

    # def _check_types(
    #     self,
    #     input_dict: dict[str, np.ndarray],
    # ) -> dict[str, np.ndarray|torch.Tensor]:
    #     """ Make sure the type of the batch elements is the correct one.
    #         The reason for this is that strings can only be put to np.ndarray,
    #         while other elements are better inside tensors
    #     """
    #     for key, value in input_dict.items():
    #         if key == self.time_key or not self.use_direct_text_input:
    #             input_dict[key] = torch.tensor(value)
    #     return input_dict

    def _masked_modelling(
        self,
        masked_input_dict: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Generalized MLM with mutually exclusive masking per token.
        At any given position, maximum one feature is masked.
        """
        # Initialization
        any_key = next(iter(self.mlm_masking_rules.keys()))
        batch_shape = masked_input_dict[any_key].shape
        final_labels = np.full(batch_shape, -100, dtype=int)  # will be filled
        global_mask_map = torch.zeros(batch_shape, dtype=torch.bool)  # who is masked
        features_to_process = list(self.mlm_masking_rules.keys())
        random.shuffle(features_to_process)  # shuffle which feature gets priority
        # ground_truth_dict = {k: v.copy() for k, v in masked_input_dict.items()}

        # Process each feature one by one
        for feature_key in features_to_process:
            if feature_key not in masked_input_dict:
                continue
                
            # Generate candidates for this feature key
            input_tokens = masked_input_dict[feature_key].copy()
            probability = self.mlm_masking_rules[feature_key]
            prob_matrix = torch.full(input_tokens.shape, probability)
            
            # Initial mask candidates for this feature (skipping PAD, CLS, SEP, etc.)
            no_mask_ids = [self.bos_token_id, self.eos_token_id, self.unk_token_id, self.pad_token_id]
            special_tokens_mask = torch.tensor(np.isin(input_tokens, no_mask_ids))
            prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
            candidate_mask = torch.bernoulli(prob_matrix).bool()

            # Enforce mutual exclusivity
            final_mask = candidate_mask & (~global_mask_map)
            global_mask_map = global_mask_map | final_mask
            if not final_mask.any():
                continue
            
            # Update labels if part of label keys (using standard MLM labeling)
            if feature_key in self.mlm_label_keys:
                target_values = masked_input_dict[feature_key]
                final_labels[final_mask] = target_values[final_mask]
                
            # Replace 80% of selected masked token locations using [MASK]
            indices_replaced = torch.bernoulli(torch.full(final_labels.shape, 0.8)).bool() & final_mask
            input_tokens[indices_replaced] = self.mask_token_id

            # Replace 10% of selected masked token locations with a random token from the batch (10% last untouched)
            indices_random = torch.bernoulli(torch.full(final_labels.shape, 0.5)).bool() & final_mask & ~indices_replaced
            flat_tokens = input_tokens.flatten()
            valid_choices = flat_tokens[~np.isin(flat_tokens, no_mask_ids)]
            if valid_choices.size > 0:
                random_tokens = valid_choices[np.random.randint(0, valid_choices.size, input_tokens.shape)]
                input_tokens[indices_random] = random_tokens[indices_random]

            # Update the batch dictionary
            masked_input_dict[feature_key] = input_tokens

        return masked_input_dict, final_labels


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
        self.do_augmentation = (samples[0]["split"] == "train")
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)

        # Assemble batch
        batch = {
            # "input_dict": self._check_types(padded_batch),
            "input_dict": {k: torch.tensor(v) for k, v in padded_batch.items()},
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels).long(),  # classification loss expects long tensors
        }
        batch.update(non_features)  # in case useful somewhere

        return batch