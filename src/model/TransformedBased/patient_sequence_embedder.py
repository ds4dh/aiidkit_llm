import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from typing import Iterator
from dataclasses import dataclass
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from safetensors.torch import save_file
from sentence_transformers.sampler import BatchSampler, SetEpochMixin
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from src.model.patient_token_embedder import (
    PatientTokenEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
)


class PatientTokenEmbeddingModule(nn.Module):
    """
    Simple wrapper for the core patient token embedding model, formatting its output
    for the pooling layer of the sentence tranformer model
    """
    def __init__(self, token_embedder: PatientTokenEmbeddingModel):
        super().__init__()
        self.token_embedder = token_embedder
        self.pad_id = 0

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """
        Dummy function to satisfy the SentenceTransformerModelCardCallback
        """
        return {}
    
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add required "token_embeddings" and "attention_mask" keys for the pooling layer
        """
        # Separate the attention mask from the actual features
        attention_mask = features.pop("attention_mask")

        # Pass the batch through your core model
        outputs = self.token_embedder(
            input_dict=features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Update the dictionary with the outputs the pooling layer expects
        features.update({
            "token_embeddings": outputs.hidden_states[-1],
            "attention_mask": attention_mask,
        })

        return features
    
    def save(self, output_path: str):
        model_path = os.path.join(output_path, "model.safetensors")
        save_file(self.token_embedder.state_dict(), model_path)


class PatientTrainer(SentenceTransformerTrainer):
    """
    Custom sentence transformer trainer that overrides the default loss computation
    """
    def __init__(self, supervised_mode: bool=False, *args, **kwargs):
        """
        Custom __init__ to handle a model without a tokenizer
        """
        self.supervised_mode = supervised_mode
        model = kwargs.get("model")
        if model is not None and not hasattr(model, "tokenizer"):
            model.tokenizer = None
        super().__init__(*args, **kwargs)
    
    def add_model_card_callback(self, default_args_dict) -> None:
        """
        Overridden to do nothing, to disable the original model card creation
        """
        pass

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool=False,
        **kwargs,
    ):
        """
        Compute loss given the mode in which the sentence transformer is used
        In supervised mode, external labels are used for contrastive learning
        Else, anchor / positive pairs are already provided by the dataloader
        """
        # Mode which uses labels for contrastive loss
        if self.supervised_mode:
            losses = []
            for label_name in self.label_names:
                labels = inputs.pop(label_name)
                labels = torch.tensor(labels, device=model.device)
                losses.append(self.loss([inputs], labels))

            loss = sum(losses)
            return (loss, {"loss": loss}) if return_outputs else loss

        # Training mode: inputs is [anchor, positive]
        if isinstance(inputs, list):
            loss = self.loss(inputs, labels=None)
            return (loss, {}) if return_outputs else loss

        # Evaluation mode: input is a single batch dictionary
        else:
            outputs = model(inputs)
            return (torch.tensor(0.0, device=model.device), outputs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Use a sampler that balances label distribution by batch in supervised_mode, 
        and otherwise falls back to the original implementation of SentenceTransformer
        """
        if self.supervised_mode:
            return DataLoader(
                self.train_dataset,
                batch_sampler=PKClassSampler(
                    dataset=self.train_dataset,
                    batch_size=self.args.train_batch_size,
                    valid_label_columns=self.label_names,
                ),
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super().get_train_dataloader()


@dataclass
class PatientDataCollatorForSequenceEmbedding(PatientDataCollatorForLanguageModelling):
    """
    Data collator for sequence embedding, preprocessing batches like the LM collator
    """
    # Overriding parent arguments that are not relevant to the sequence embedding task
    mlm: bool=False
    num_mlm_labels: int|None=None
    mlm_probability: float=0.0

    def __init__(self, valid_label_columns=None):
        super().__init__()
        self.supervised_mode = (valid_label_columns is not None)
        self.valid_label_columns = valid_label_columns

    def _prepare_batch(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """
        Prepare a batch by padding sequence features and adding labels
        and the attention mask separately
        """
        to_pad, non_processed_features = self._separate_non_processed_features(features)
        padded_batch, attention_mask = self._process_batch(to_pad)
        padded_batch["attention_mask"] = attention_mask
        padded_batch.update(non_processed_features)

        return padded_batch

    def __call__(
        self,
        features: list[dict[str, dict]],
    ) -> list[dict[str, torch.Tensor]]|dict[str, torch.Tensor]:
        """
        Batching function used by the sentence-transformer trainer pipeline
        """
        # Supervised mode: expects a flat list of features (including labels)
        if self.supervised_mode:
            return self._prepare_batch(features)

        # Continuing to unsupervised mode
        # Identify if the data collator is being used during training or evaluation
        feature_keys = features[0].keys()
        training_mode = ("anchor" in feature_keys and "positive" in feature_keys)

        # Training mode, labels, and anchor + positive samples containing sequence dicts
        if training_mode:
            anchor_dicts = [feat["anchor"] for feat in features]
            positive_dicts = [feat["positive"] for feat in features]
            prepared_anchors = self._prepare_batch(anchor_dicts)
            prepared_positives = self._prepare_batch(positive_dicts)

            return [prepared_anchors, prepared_positives]

        # Evaluation mode, directly sample dicts with sequence and labels
        else:
            return self._prepare_batch(features)


class PKClassSampler(SetEpochMixin, BatchSampler):
    """
    A robust batch sampler for metric learning.

    This sampler guarantees that each batch contains a diverse set of classes.
    It works by first selecting P distinct classes and then sampling K instances
    from each of those classes. This is a standard approach for training with
    losses like TripletLoss that require negative examples within a batch.

    The batch size is implicitly defined as `num_classes_per_batch * num_samples_per_class`.

    Args:
        dataset (Dataset): The dataset to sample from.
        num_classes_per_batch (int): (P) The number of distinct classes in each batch.
        num_samples_per_class (int): (K) The number of samples to draw for each class.
        oversampling_power (float): Controls the strength of oversampling for rare
            classes. 0 means uniform class sampling, > 0 gives more weight to
            rarer classes. Defaults to 0.5.
        num_batches (int): The number of batches to yield per epoch. If None, it's
            estimated based on the dataset size.
        valid_label_columns (list[str]): Potential names for the label column.
        generator (torch.Generator): A PyTorch generator for reproducibility.
        seed (int): A seed for the generator.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int=16,
        num_classes_per_batch: int=2,
        num_samples_per_class: int=8,
        oversampling_power: float=0.5,
        num_batches: int=None,
        valid_label_columns: list[str]=None,
        generator: torch.Generator=None,
        seed: int=0,
    ):
        dummy_sampler = SequentialSampler(dataset)
        if batch_size % (num_classes_per_batch * num_samples_per_class) != 0:
            raise ValueError(
                "The batch_size must be a multiple of"
                "(num_classes_per_batch * num_samples_per_class)."
            )
        super().__init__(sampler=dummy_sampler,batch_size=batch_size, drop_last=True)

        self.dataset = dataset
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class
        self.oversampling_power = oversampling_power
        self.generator = generator
        self.seed = seed
        self.epoch = 0

        # Group samples by class label
        labels = self._determine_labels_to_use(dataset, valid_label_columns or [])
        print("Label distribution: ", Counter(labels))
        self.groups: dict[str|int, list[int]] = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            self.groups[label].append(sample_idx)

        # Filter for classes that have at least K samples
        self.used_labels = sorted([
            label for label, indices in self.groups.items()
            if len(indices) >= self.num_samples_per_class
        ])
        if not self.used_labels:
            raise ValueError(
                f"No classes have at least {self.num_samples_per_class} samples."
            )
        if len(self.used_labels) < self.num_classes_per_batch:
            raise ValueError(
                f"Not enough classes ({len(self.used_labels)}) to form a batch "
                f"with {self.num_classes_per_batch} classes."
            )

        # Calculate class sampling probabilities based on inverse frequency
        class_sizes = np.array([len(self.groups[label]) for label in self.used_labels])
        weights = (1 / class_sizes) ** self.oversampling_power
        self.class_probabilities = torch.from_numpy(weights / np.sum(weights))

        # Determine the number of batches for one epoch
        if num_batches is None:
            self._num_batches = len(dataset) // self.batch_size
        else:
            self._num_batches = num_batches

    @staticmethod
    def _determine_labels_to_use(
        dataset: Dataset,
        valid_label_columns: list[str],
    ) -> list:
        for column_name in valid_label_columns:
            if column_name in dataset.column_names:
                return dataset[column_name]
        if "label" in dataset.features:
            return dataset["label"]
        raise ValueError(
            f"None of the valid_label_columns {valid_label_columns}"
             "are in the dataset."
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is None: self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + self.epoch)

        for _ in range(self._num_batches):
            current_batch = []
            
            # Select P classes for the batch without replacement
            class_indices = torch.multinomial(
                self.class_probabilities,
                num_samples=self.num_classes_per_batch,
                replacement=False,
                generator=self.generator,
            )

            # For each selected class, sample K instances
            for class_idx in class_indices:
                label = self.used_labels[class_idx]
                class_samples = self.groups[label]
                
                # Sample K indices with replacement from the chosen class
                # to prevent issues if K is large relative to the class size
                sample_indices_in_class = torch.randint(
                    0, len(class_samples), (self.num_samples_per_class,),
                    generator=self.generator
                )
                
                for idx in sample_indices_in_class:
                    current_batch.append(class_samples[idx])

            yield current_batch

    def __len__(self) -> int:
        return self._num_batches