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
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
)
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.data.data_collator import DataCollatorMixin
from src.model.model_utils import TimeEmbedding, NumericalValueEmbedding


class PatientEmbeddingModelFactory:
    """
    Factory to handle the lifecycle of a patient embedding model:
    1. create_from_backbone: hybrid HF backbone with patient input embedding layer
    2. from_pretrained: restores pretrained patient embedding model for finetuning/inference
    Supports both discriminative bidirectional encoders (ModernBERT) and generative
    causal decoder-only backbones (Qwen2, Llama, Mistral, etc.).
    """
    _CAUSAL_MODEL_TYPES = {"qwen", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma2", "gpt2", "smollm"}

    @classmethod
    def _is_causal_backbone(cls, model_id_or_config, architecture_type: str | None = None) -> bool:
        if architecture_type and str(architecture_type).lower() in {"causal", "generative", "decoder"}:
            return True
        if isinstance(model_id_or_config, str):
            low = model_id_or_config.lower()
            return any(tok in low for tok in cls._CAUSAL_MODEL_TYPES)
        if getattr(model_id_or_config, "is_causal_decoder", False):
            return True
        if str(getattr(model_id_or_config, "architecture_type", "")).lower() in {"causal", "generative", "decoder"}:
            return True
        return str(getattr(model_id_or_config, "model_type", "")).lower() in cls._CAUSAL_MODEL_TYPES

    @classmethod
    def _load_or_build_config(
        cls,
        model_id: str,
        config_args: dict,
        model_args: dict,
        architecture_type: str | None = None,
    ):
        cfg_kwargs = dict(config_args)
        is_causal = cls._is_causal_backbone(model_id, architecture_type=architecture_type)
        # Map unified YAML/Optuna `dropout_<part>` keys onto HuggingFace `<part>_dropout` keys
        for src_k, dst_k in (
            ("dropout_attention", "attention_dropout"),
            ("dropout_embedding", "embedding_dropout"),
            ("dropout_mlp", "mlp_dropout"),
            ("dropout_classifier", "classifier_dropout"),
        ):
            if src_k in cfg_kwargs and dst_k not in cfg_kwargs:
                cfg_kwargs[dst_k] = cfg_kwargs[src_k]
        # Input embeddings come from PatientEmbeddingLayer (not HF's default tok_embeddings),
        # so disable HF tie_word_embeddings globally to keep MLM/CausalLM output heads trainable.
        cfg_kwargs.setdefault("tie_word_embeddings", False)
        if is_causal:
            cfg_kwargs.setdefault("use_cache", False)

        try:
            config = AutoConfig.from_pretrained(model_id, **cfg_kwargs, **model_args)
        except Exception:
            if not is_causal:
                raise
            # Offline fallback for causal decoder configs (e.g. on compute nodes without Hub access)
            low = model_id.lower()
            causal_defaults = dict(
                vocab_size=cfg_kwargs.pop("vocab_size", 1200),
                hidden_size=cfg_kwargs.pop("hidden_size", 768),
                intermediate_size=cfg_kwargs.pop("intermediate_size", 1152),
                num_hidden_layers=cfg_kwargs.pop("num_hidden_layers", 22),
                num_attention_heads=cfg_kwargs.pop("num_attention_heads", 12),
                num_key_value_heads=cfg_kwargs.pop("num_key_value_heads", 4),
                max_position_embeddings=cfg_kwargs.pop("max_position_embeddings", 1024),
                pad_token_id=cfg_kwargs.pop("pad_token_id", 0),
                bos_token_id=cfg_kwargs.pop("bos_token_id", 2),
                eos_token_id=cfg_kwargs.pop("eos_token_id", 3),
                tie_word_embeddings=cfg_kwargs.pop("tie_word_embeddings", False),
                use_cache=cfg_kwargs.pop("use_cache", False),
            )
            causal_defaults.update(cfg_kwargs)
            if "llama" in low:
                config_cls = LlamaConfig
            elif "qwen3" in low:
                config_cls = Qwen3Config
            else:
                config_cls = Qwen2Config
            config = config_cls(**causal_defaults)

        if hasattr(config, "text_config") and config.text_config is not None:
            text_cfg = config.text_config
            for k, v in cfg_kwargs.items():
                setattr(text_cfg, k, v)
            for k in ("dtype", "torch_dtype", "attn_implementation", "_attn_implementation"):
                if k in model_args:
                    setattr(text_cfg, k, model_args[k])
            config = text_cfg

        if hasattr(config, "layer_types") and isinstance(config.layer_types, list) and len(config.layer_types) != config.num_hidden_layers:
            if "layer_types" in cfg_kwargs and isinstance(cfg_kwargs["layer_types"], list) and len(cfg_kwargs["layer_types"]) == config.num_hidden_layers:
                config.layer_types = list(cfg_kwargs["layer_types"])
            else:
                pat = config.layer_types or ["full_attention"]
                config.layer_types = [pat[i % len(pat)] for i in range(config.num_hidden_layers)]
        if hasattr(config, "max_window_layers") and isinstance(config.max_window_layers, int):
            config.max_window_layers = min(config.max_window_layers, config.num_hidden_layers)
        if (
            hasattr(config, "head_dim")
            and ("hidden_size" in cfg_kwargs or "num_attention_heads" in cfg_kwargs)
            and "head_dim" not in cfg_kwargs
        ):
            config.head_dim = config.hidden_size // config.num_attention_heads

        if is_causal:
            config.is_causal_decoder = True
            config.architecture_type = "causal"
            config.classifier_pooling = config_args.get("classifier_pooling", getattr(config, "classifier_pooling", "last"))
            config.dropout_classifier = config_args.get("dropout_classifier", getattr(config, "dropout_classifier", 0.1))
        if hasattr(config, "validate"):
            config.validate()
        return config

    @classmethod
    def create_from_backbone(
        cls,
        model_id: str,
        task: str,
        embedding_layer_config: dict,
        config_args: dict={},
        model_args: dict={},
        load_backbone_weights: bool=False,
        enforce_monotonicity: bool=False,
        architecture_type: str | None = None,
    ):
        """
        Creates a new patient model by grafting a custom embedding layer onto a standard backbone
        """
        # Copy mutable dict arguments so caller's YAML config dict remains JSON-serializable
        config_args = dict(config_args or {})
        model_args = dict(model_args or {})
        for key in ["dtype", "torch_dtype"]:
            if isinstance(model_args.get(key), str):
                model_args[key] = getattr(torch, model_args[key])

        # Load model configuration from huggingface's directory (or offline causal config)
        if "modernbert" in (model_id or "").lower() and isinstance(config_args.get("layer_types"), list) and len(config_args["layer_types"]) >= 2 and "sliding_attention" not in config_args["layer_types"]:
            config_args["layer_types"] = ["sliding_attention"] + list(config_args["layer_types"][1:])
        config = cls._load_or_build_config(model_id, config_args, model_args, architecture_type=architecture_type)
        config.enforce_monotonicity = enforce_monotonicity
        is_causal = cls._is_causal_backbone(config, architecture_type=architecture_type)

        # Initialize the backbone (pre-trained or random)
        model_cls = cls._get_model_class(task, is_causal=is_causal)
        if load_backbone_weights:
            print(f"Initializing {model_id} backbone with pre-trained weights.")
            model = model_cls.from_pretrained(
                model_id, config=config, ignore_mismatched_sizes=True, **model_args
            )
        else:
            print(f"Initializing {model_id} ({'causal decoder' if is_causal else 'bidirectional encoder'}) backbone with random weights.")
            from_cfg_kwargs = {k: v for k, v in model_args.items() if k in ("dtype", "torch_dtype", "attn_implementation")}
            model = model_cls.from_config(config, **from_cfg_kwargs)

        # Initialize custom layer (always random at creation)
        emb_cfg = dict(embedding_layer_config)
        if "dropout_prob" not in emb_cfg and ("dropout_embedding" in config_args or "embedding_dropout" in config_args):
            emb_cfg["dropout_prob"] = float(config_args.get("dropout_embedding", config_args.get("embedding_dropout", 0.1)) or 0.0)
        custom_embed = PatientEmbeddingLayer(
            embedding_dim=config.hidden_size, **emb_cfg,
        )

        # Graft custom input patient embedding layer to the backbone
        return cls._patch_model(model, custom_embed, config_args, task=task)

    @classmethod
    def from_pretrained(
        cls,
        task: str,
        pretrained_dir: str,
        embedding_layer_config: dict,
        model_args: dict = {},
        reset_weights: bool = False,
        enforce_monotonicity: bool | list[int] | None = None,
        **kwargs,
    ):
        """
        Restores a patient model from a local directory where it was saved
        """
        print(f"From pretrained unused arguments: {kwargs}")
        cfg_overrides = kwargs.get("config_args", {}) or {}
        # Load Config from the local directory (mutating in-memory only; never rewrite pretrained_dir/config.json)
        model_args = {k: v for k, v in model_args.items() if k not in ("model_id", "load_backbone_weights", "config_args", "architecture_type")}
        for key in ["dtype", "torch_dtype"]:
            if isinstance(model_args.get(key), str):
                model_args[key] = getattr(torch, model_args[key])
        config = AutoConfig.from_pretrained(pretrained_dir, **model_args)
        if getattr(config, "model_type", None) == "modernbert" and isinstance(getattr(config, "layer_types", None), list) and len(config.layer_types) >= 2 and "sliding_attention" not in config.layer_types:
            config.layer_types = ["sliding_attention"] + list(config.layer_types[1:])
        if enforce_monotonicity is not None:
            config.enforce_monotonicity = enforce_monotonicity
        if "classifier_pooling" in cfg_overrides:
            config.classifier_pooling = cfg_overrides["classifier_pooling"]
        for src_k, dst_k in (
            ("dropout_attention", "attention_dropout"),
            ("dropout_embedding", "embedding_dropout"),
            ("dropout_mlp", "mlp_dropout"),
            ("dropout_classifier", "classifier_dropout"),
        ):
            val = cfg_overrides.get(src_k, cfg_overrides.get(dst_k))
            if val is not None:
                setattr(config, src_k, val)
                if hasattr(config, dst_k):
                    setattr(config, dst_k, val)
        if hasattr(config, "layer_types") and isinstance(config.layer_types, list) and len(config.layer_types) != config.num_hidden_layers:
            pat = config.layer_types or ["full_attention"]
            config.layer_types = [pat[i % len(pat)] for i in range(config.num_hidden_layers)]
        if hasattr(config, "max_window_layers") and isinstance(config.max_window_layers, int):
            config.max_window_layers = min(config.max_window_layers, config.num_hidden_layers)

        is_causal = cls._is_causal_backbone(config, architecture_type=kwargs.get("architecture_type"))
        if is_causal:
            config.is_causal_decoder = True
        if hasattr(config, "validate"):
            config.validate()

        # Build the skeleton with random weights
        model_cls = cls._get_model_class(task, is_causal=is_causal)
        from_cfg_kwargs = {k: v for k, v in model_args.items() if k in ("dtype", "torch_dtype", "attn_implementation")}
        model = model_cls.from_config(config, **from_cfg_kwargs)

        # Re-create the custom Layer
        emb_cfg = dict(embedding_layer_config)
        emb_drop_override = cfg_overrides.get(
            "dropout_embedding",
            cfg_overrides.get("embedding_dropout", getattr(config, "dropout_embedding", None)),
        )
        if "dropout_prob" not in emb_cfg and emb_drop_override is not None:
            emb_cfg["dropout_prob"] = float(emb_drop_override)
        custom_embed = PatientEmbeddingLayer(
            embedding_dim=config.hidden_size, **emb_cfg,
        )

        # Patch model structure before loading weights
        model = cls._patch_model(
            model,
            custom_embed,
            cfg_overrides or embedding_layer_config,
            task=task,
            survival_head_config=kwargs.get("survival_head_config"),
        )

        # Return a randomly initialized model if requested
        if reset_weights:
            print("Resetting model weights to random initialization.")
            return model

        # Load pretrained weights (backbone and custom embedding layer)
        safe_path = os.path.join(pretrained_dir, "model.safetensors")
        if os.path.exists(safe_path):
            print(f"Loading patient model weights from {safe_path}...")
            state_dict = load_file(safe_path)
            # Backward compatibility: if loading a legacy checkpoint saved before
            # `numerical_embedding` was added, disable `use_numerical_embedding`
            # unless the user explicitly forced it.
            has_num_emb = any(k.startswith("patient_embedder.numerical_embedding.") for k in state_dict)
            if not has_num_emb and getattr(model.patient_embedder, "use_numerical_embedding", False):
                print("  [Note] Checkpoint predates NumericalValueEmbedding; falling back to discrete value_id embeddings.")
                model.patient_embedder.use_numerical_embedding = False
            # strict=False allows switching from MLM/CausalLM head (saved) to classification head (new)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No model.safetensors found in {pretrained_dir}")

        return model

    @staticmethod
    def _get_model_class(task: str, is_causal: bool = False):
        if task == "masked":
            return AutoModelForCausalLM if is_causal else AutoModelForMaskedLM
        if task == "classification":
            return AutoModelForSequenceClassification
        if task == "survival":
            return AutoModelForSequenceClassification  # 1 logit = log hazard ratio
        raise ValueError("Task must be 'masked', 'classification' or 'survival'")

    @classmethod
    def _patch_model(cls, model, custom_embed, config_dict, task: str | None = None, survival_head_config: dict | None = None):
        """
        Attaches the custom layer (and dedicated continuous MLM head / causal next-event query head
        when task == 'masked', or `SurvivEHRHead` when `survival_head_config` is supplied) and overrides forward.
        """
        # Attach layer as a submodule so it saves/loads with state_dict
        custom_embed.to(dtype=model.dtype, device=model.device)
        model.patient_embedder = custom_embed
        is_causal = cls._is_causal_backbone(model.config)
        hidden_size = model.config.hidden_size

        # Freeze the unused default input token embedding table on the HF backbone
        # (since inputs_embeds comes from model.patient_embedder) so DDP never sees unused gradients,
        # while strictly keeping any output LM projection head (e.g. decoder / lm_head) trainable!
        base_in_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        out_emb_params = {id(p) for p in out_emb.parameters()} if out_emb is not None else set()
        if base_in_emb is not None:
            for p in base_in_emb.parameters():
                if id(p) not in out_emb_params:
                    p.requires_grad = False

        if task == "survival" and survival_head_config:
            from src.model.survival_utils import SurvivEHRHead
            cls_drop = float(
                config_dict.get(
                    "dropout_classifier",
                    getattr(model.config, "dropout_classifier", getattr(model.config, "classifier_dropout", 0.1)),
                ) or 0.0
            )
            surv_kwargs = dict(survival_head_config)
            surv_kwargs.setdefault("dropout_prob", cls_drop)
            model.survival_head = SurvivEHRHead(hidden_size=hidden_size, **surv_kwargs).to(device=model.device)
            # Freeze unused default sequence classifier weights so optimizer/DDP stay clean
            for unused_attr in ("head", "classifier", "score"):
                sub_mod = getattr(model, unused_attr, None)
                if isinstance(sub_mod, nn.Module):
                    for p in sub_mod.parameters():
                        p.requires_grad = False

        if is_causal and task == "masked":
            # Causal Next-Event Query projector: fuses causal history state h_{t-1} with
            # partial next-event query q_t = PatientEmbeddingLayer(time_t, masked_eav_t)
            model.next_event_query_projector = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
            ).to(dtype=model.dtype, device=model.device)

        if is_causal and task in ("classification", "survival"):
            # Replace HF causal sequence classifier's bare `score` linear layer with the same
            # 2-stage projection head (`head` -> `drop` -> `classifier`) used in ModernBERT
            if hasattr(model, "score"):
                delattr(model, "score")
            num_labels = getattr(model.config, "num_labels", 1)
            cls_drop = float(
                config_dict.get(
                    "dropout_classifier",
                    getattr(model.config, "dropout_classifier", getattr(model.config, "classifier_dropout", 0.1)),
                ) or 0.0
            )
            model.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=1e-5, bias=False),
            ).to(dtype=model.dtype, device=model.device)
            model.drop = nn.Dropout(cls_drop)
            model.classifier = nn.Linear(hidden_size, num_labels).to(dtype=model.dtype, device=model.device)

        if task == "masked" and getattr(custom_embed, "use_numerical_embedding", True):
            model.numerical_mlm_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
                nn.Tanh(),
            ).to(dtype=model.dtype, device=model.device)
            # Learnable log-variance parameters for uncertainty-weighted loss aggregation
            model.mlm_log_vars = nn.Parameter(
                torch.zeros(2, dtype=torch.float32, device=model.device)
            )

        def _compute_dual_mlm_loss(
            model_self,
            disc_loss_raw: torch.Tensor | None,
            logits_tensor: torch.Tensor,
            pred_num_tensor: torch.Tensor | None,
            labels_num_tensor: torch.Tensor | None,
            has_num_head: bool,
        ) -> torch.Tensor:
            """Combine discrete CE MLM loss and continuous Smooth-L1 value loss via uncertainty weighting."""
            loss_disc = disc_loss_raw if (disc_loss_raw is not None and torch.isfinite(disc_loss_raw)) else None
            loss_cont = None
            if has_num_head and pred_num_tensor is not None and labels_num_tensor is not None:
                num_mask = torch.isfinite(labels_num_tensor)
                if bool(num_mask.any()):
                    # Variance of Uniform[-1, 1] is 1/3 -> multiplying by 3.0 normalizes unit scale
                    loss_cont = 3.0 * torch.nn.functional.smooth_l1_loss(
                        pred_num_tensor[num_mask].float(),
                        labels_num_tensor[num_mask].float(),
                        beta=0.15,
                    )
            if has_num_head and pred_num_tensor is not None:
                log_vars = torch.clamp(model_self.mlm_log_vars, min=-4.0, max=4.0)
                total = torch.tensor(0.0, device=logits_tensor.device, dtype=torch.float32)
                if loss_disc is not None and loss_cont is not None:
                    prec_d = torch.exp(-log_vars[0])
                    prec_c = torch.exp(-log_vars[1])
                    total = (
                        0.5 * prec_d * loss_disc.float() + 0.5 * torch.nn.functional.softplus(log_vars[0])
                        + 0.5 * prec_c * loss_cont.float() + 0.5 * torch.nn.functional.softplus(log_vars[1])
                    )
                elif loss_disc is not None:
                    total = loss_disc.float()
                elif loss_cont is not None:
                    total = loss_cont.float()
                return total + 0.0 * pred_num_tensor.float().sum() + 0.0 * logits_tensor.float().sum() + 0.0 * model_self.mlm_log_vars.sum()
            return (loss_disc.float() if loss_disc is not None else 0.0) + 0.0 * logits_tensor.float().sum()

        # Define custom forward
        def custom_forward(self, input_dict=None, inputs_embeds=None, labels=None, labels_num=None, landmark_fup=None, **kwargs):
            is_causal_model = getattr(self.config, "is_causal_decoder", False)
            has_num_mlm_head = hasattr(self, "numerical_mlm_head") and getattr(self.patient_embedder, "use_numerical_embedding", True)
            has_surv_ehr_head = hasattr(self, "survival_head")

            if has_surv_ehr_head and input_dict is not None:
                inputs_embeds = self.patient_embedder(**input_dict)
                need_hidden = bool(kwargs.pop("output_hidden_states", False))
                attn_mask = kwargs.get("attention_mask", None)
                if hasattr(self, "model"):
                    enc_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attn_mask, "return_dict": True}
                    if is_causal_model:
                        enc_kwargs["use_cache"] = False
                    backbone_out = self.model(**enc_kwargs)
                    last_hidden = backbone_out.last_hidden_state
                else:
                    fw_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
                    backbone_out = self.original_forward(inputs_embeds=inputs_embeds, output_hidden_states=True, **fw_kwargs)
                    last_hidden = backbone_out.hidden_states[-1]
                time_key = getattr(self.patient_embedder, "time_key", "days_since_tpx")
                tok_times = input_dict.get(time_key, None) if isinstance(input_dict, dict) else None
                logits = self.survival_head(
                    last_hidden=last_hidden,
                    attention_mask=attn_mask,
                    token_times=tok_times,
                    landmark_fup=landmark_fup,
                )
                return SequenceClassifierOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=(last_hidden,) if need_hidden else None,
                )

            # ------------------------------------------------------------------
            # PATH 1: Generative / Causal Decoder Pre-Training (task == 'masked')
            # Clean-History Causal State h_{t-1} + Partial Next-Event Query q_t
            # ------------------------------------------------------------------
            if is_causal_model and hasattr(self, "next_event_query_projector"):
                if self.training and hasattr(self.model, "gradient_checkpointing_enable") and not getattr(self.model, "is_gradient_checkpointing", False):
                    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if "_clean_entity_id" in input_dict:
                    clean_dict = dict(input_dict)
                    for k in list(self.patient_embedder.eav_keys) + ["value_num"]:
                        ck = f"_clean_{k}"
                        if ck in input_dict:
                            clean_dict[k] = input_dict[ck]
                    clean_embeds = self.patient_embedder(**clean_dict)
                else:
                    clean_embeds = self.patient_embedder(**input_dict)

                need_hidden = kwargs.pop("output_hidden_states", getattr(self.config, "output_hidden_states", True))
                attn_mask = kwargs.get("attention_mask", None)
                backbone_out = self.model(
                    inputs_embeds=clean_embeds,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = backbone_out.last_hidden_state
                query_embeds = self.patient_embedder(**input_dict) if "_clean_entity_id" in input_dict else clean_embeds

                # Shift causal hidden state by 1 step: at position t, prev_hidden[:, t] = last_hidden[:, t-1]
                # summarizes strictly past events (0 .. t-1) with zero leakage from event t or future events.
                prev_hidden = torch.zeros_like(last_hidden)
                prev_hidden[:, 1:, :] = last_hidden[:, :-1, :]

                proj_dtype = self.next_event_query_projector[0].weight.dtype
                fused_hidden = self.next_event_query_projector(
                    torch.cat([prev_hidden.to(dtype=proj_dtype), query_embeds.to(dtype=proj_dtype)], dim=-1)
                )
                logits = self.lm_head(fused_hidden.to(dtype=self.lm_head.weight.dtype))

                pred_num = None
                if has_num_mlm_head:
                    pred_num = self.numerical_mlm_head(
                        fused_hidden.to(dtype=self.numerical_mlm_head[0].weight.dtype)
                    ).squeeze(-1)

                total_loss = None
                if labels is not None:
                    disc_labels = torch.where(labels >= 0, labels, torch.full_like(labels, -100))
                    loss_d = (
                        torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)).float(),
                            disc_labels.view(-1),
                            ignore_index=-100,
                        )
                        if bool((disc_labels != -100).any())
                        else None
                    )
                    total_loss = _compute_dual_mlm_loss(self, loss_d, logits, pred_num, labels_num, has_num_mlm_head)

                hidden_out = None
                if need_hidden and (not self.training or labels is None):
                    hidden_out = (pred_num, last_hidden) if pred_num is not None else (last_hidden,)
                return MaskedLMOutput(loss=total_loss, logits=logits, hidden_states=hidden_out)

            # ------------------------------------------------------------------
            # PATH 2: Generative / Causal Decoder Classification & Survival
            # ------------------------------------------------------------------
            if is_causal_model and hasattr(self, "classifier") and hasattr(self, "head"):
                if self.training and hasattr(self.model, "gradient_checkpointing_enable") and not getattr(self.model, "is_gradient_checkpointing", False):
                    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                inputs_embeds = self.patient_embedder(**input_dict)
                need_hidden = bool(kwargs.pop("output_hidden_states", False))
                attn_mask = kwargs.get("attention_mask", None)
                backbone_out = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = backbone_out.last_hidden_state
                pooling_mode = getattr(self.config, "classifier_pooling", "last")
                if pooling_mode == "last" and attn_mask is not None:
                    seq_idx = attn_mask.sum(dim=1).clamp(min=1) - 1
                    pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_idx]
                elif attn_mask is not None:
                    mask = attn_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
                    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    pooled = last_hidden[:, -1, :]

                head_dtype = self.head[0].weight.dtype
                pooled_head = self.head(pooled.to(dtype=head_dtype))
                logits = self.classifier(self.drop(pooled_head).to(dtype=self.classifier.weight.dtype))
                outputs = SequenceClassifierOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=(last_hidden,) if need_hidden else None,
                )
            else:
                # --------------------------------------------------------------
                # PATH 3: Bidirectional Encoder (ModernBERT MLM & Classification)
                # --------------------------------------------------------------
                inputs_embeds = self.patient_embedder(**input_dict)
                if has_num_mlm_head and labels is not None:
                    # Separate discrete masked targets (labels >= 0) from continuous masked targets (labels < 0, labels != -100)
                    disc_labels = torch.where(labels >= 0, labels, torch.full_like(labels, -100))
                    need_hidden = kwargs.pop("output_hidden_states", getattr(self.config, "output_hidden_states", True))
                    outputs = self.original_forward(
                        inputs_embeds=inputs_embeds,
                        labels=disc_labels,
                        output_hidden_states=True,
                        **kwargs,
                    )
                    last_hidden = outputs.hidden_states[-1]
                    pred_num = self.numerical_mlm_head(last_hidden.to(dtype=self.numerical_mlm_head[0].weight.dtype)).squeeze(-1)

                    has_disc = bool((disc_labels != -100).any())
                    loss_d = outputs.loss if has_disc else None
                    outputs.loss = _compute_dual_mlm_loss(self, loss_d, outputs.logits, pred_num, labels_num, True)
                    if need_hidden:
                        outputs.hidden_states = (pred_num, last_hidden)
                    else:
                        outputs.hidden_states = None
                    return outputs

                if labels is not None:
                    if labels.dtype in (torch.int32, torch.int64):
                        labels = torch.where(labels >= 0, labels, torch.full_like(labels, -100))
                    kwargs["labels"] = labels
                outputs = self.original_forward(inputs_embeds=inputs_embeds, **kwargs)
            
            # Enforce monotonic logits multi-label classification (discrete survival analysis)
            is_monotonic = getattr(self.config, "enforce_monotonicity", False)
            is_multilabel = getattr(self.config, "problem_type", None) == "multi_label_classification"
            if is_monotonic and is_multilabel and hasattr(outputs, "logits"):
                raw_logits = outputs.logits
                if raw_logits.shape[-1] > 1:  # only if we have more than one horizon
                    # 'enforce_monotonicity' may be True (all columns form ONE
                    # horizon-ordered group, classic single-task run) or a list
                    # of group sizes for multi-task runs: each task's horizon
                    # block is monotonic on its own (columns are task-major
                    # ordered: [t1_h1, t1_h2, ..., t2_h1, t2_h2, ...]).
                    if isinstance(is_monotonic, (list, tuple)):
                        group_sizes = [int(g) for g in is_monotonic]
                        if sum(group_sizes) != raw_logits.shape[-1]:
                            raise ValueError(
                                f"enforce_monotonicity groups {group_sizes} do not cover "
                                f"{raw_logits.shape[-1]} output columns"
                            )
                    else:
                        group_sizes = [raw_logits.shape[-1]]

                    chunks, start = [], 0
                    for size in group_sizes:
                        group = raw_logits[:, start:start + size]
                        start += size
                        if size <= 1:
                            chunks.append(group)
                            continue
                        # Force the next horizon's logit to be strictly greater
                        # than the previous one WITHIN the group
                        monotonic_logits = [group[:, 0]]
                        for i in range(1, size):
                            monotonic_logits.append(
                                monotonic_logits[-1] + torch.nn.functional.softplus(group[:, i])
                            )
                        chunks.append(torch.stack(monotonic_logits, dim=1))

                    outputs.logits = torch.cat(chunks, dim=1)
            
            # Remove hidden states from output to save memory if not needed
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                outputs.hidden_states = (outputs.hidden_states[-1],)
                
            return outputs

        # Apply monkey patch
        model.original_forward = model.forward
        model.forward = types.MethodType(custom_forward, model)
        model.config.max_position_embeddings = config_dict.get("max_position_embeddings", getattr(model.config, "max_position_embeddings", 512))

        return model


class PatientEmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        eav_mappings: dict[str, str] = {
            "entity_id": "entity",
            "attribute_id": "attribute",
            "value_id": "value_binned",
        },
        time_mapping: dict[str, str] = {"days_since_tpx": "time"},
        use_numerical_embedding: bool = True,
        dropout_prob: float = 0.1,
        **_unused_kwargs,
    ):
        """
        Layer to generate input embeddings given patient t-EAV sequences.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_key = list(time_mapping.keys())[0]
        self.eav_keys = list(eav_mappings.keys())
        self.use_numerical_embedding = use_numerical_embedding

        self.time_embedding = TimeEmbedding(embedding_dim)
        self.delta_time_embedding = TimeEmbedding(embedding_dim)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.numerical_embedding = (
            NumericalValueEmbedding(embedding_dim) if self.use_numerical_embedding else None
        )
        self.eav_projector = nn.Sequential(
            nn.Linear(embedding_dim * len(self.eav_keys), embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_prob),
        )
        self.time_projector = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_prob),
        )

        # Explicit hooks for Captum attribution analysis (identity layers)
        self.value_embedding_hook = nn.Identity()
        self.aggregated_embedding_hook = nn.Identity()

    def add_time_embeddings(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add both absolute and delta time embeddings to the input tensor
        """
        delta_times = times.clone()
        delta_times[:, 1:] = times[:, 1:] - times[:, :-1]
        delta_times[:, 0] = 0  # default delta for the very first event
        t_abs = self.time_embedding(times).to(dtype=x.dtype)
        t_delta = self.delta_time_embedding(delta_times).to(dtype=x.dtype)
        
        return self.time_projector(torch.cat([x, t_abs, t_delta], dim=-1))

    def encode_values(
        self,
        value_ids: torch.Tensor,
        value_nums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Embed the value slot of each EAV triplet:
        - Categorical tokens and special tokens ([PAD]=0, [MASK]=1, [BOS]=2, [EOS]=3, [UNK]=4)
          use the discrete `token_embedding(value_ids)`.
        - Numerical entries (`torch.isfinite(value_nums)` and `value_ids >= 5`) use the
          continuous `numerical_embedding(value_nums)` module on the quantile-normalized
          scalar in `[-1, 1]`.
        """
        tok_embeds = self.token_embedding(value_ids)
        if not self.use_numerical_embedding or self.numerical_embedding is None or value_nums is None:
            return tok_embeds
        is_num = torch.isfinite(value_nums) & (value_ids >= 5)
        safe_nums = torch.where(is_num, value_nums, torch.zeros_like(value_nums))
        num_embeds = self.numerical_embedding(safe_nums).to(dtype=tok_embeds.dtype)
        return torch.where(is_num.unsqueeze(-1), num_embeds, tok_embeds)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for the patient embedding layer.
        """
        eav_embeds = []
        for key in self.eav_keys:
            if key == "value_id":
                v_emb = self.encode_values(kwargs[key], kwargs.get("value_num"))
                v_emb = self.value_embedding_hook(v_emb)  # for captum analysis
                eav_embeds.append(v_emb)
            else:
                eav_embeds.append(self.token_embedding(kwargs[key]))

        x = self.eav_projector(torch.cat(eav_embeds, dim=-1))
        x = self.aggregated_embedding_hook(x)  # for captum analysis
        x = x.to(kwargs[self.time_key].device)
        return self.add_time_embeddings(x, kwargs[self.time_key])


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
    time_mapping: dict[str, str] = field(default_factory=lambda: {"days_since_tpx": "time"})
    eav_mappings: dict[str, str] = field(default_factory=lambda: {
        "entity_id": "entity", "attribute_id": "attribute", "value_id": "value_binned"
    })
    value_num_key: str = "value_num"
    mlm_masking_rules: dict[str, float] = field(default_factory=lambda: {"value_id": 0.15})
    mlm_label_keys: list[str] = field(default_factory=lambda: ["value_id"])
    
    # Flags
    use_numerical_embedding: bool = True  # True: continuous value embedding + dual-head MLM; False: binned token embedding + CE MLM
    do_augmentation: bool = False  # updated for different dataset splits
    keep_recent_tail: bool = False  # True for supervised landmark prediction so recent events before fup are never cropped
    max_position_embeddings: int = 512
    return_tensors: str = "pt"
    
    def __post_init__(self):
        """Anything that must be initialized dynamically"""
        self.time_key = list(self.time_mapping.keys())[0]
        self.eav_keys = list(self.eav_mappings.keys())
        self.feature_keys = [self.time_key] + self.eav_keys
    
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
        samples = [dict(s) for s in samples]
        sample_keys = list(samples[0].keys())
        seq_keys = set(self.feature_keys) | {self.value_num_key}
        non_processed_keys = [k for k in sample_keys if k not in seq_keys]
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
        head_budget: int = 128,  # more or less how many static or past items exist
    ) -> list[dict[str, np.ndarray]]:
        """
        Truncate samples longer than the maximum allowed sequence length.
        - Training: Random continuous crop (data augmentation).
        - Eval/Test: Middle truncation (keep the first `head_budget` static events 
                     and the most recent tail events).
        """
        effective_max_len = self.max_position_embeddings - 2  # for bos and eos tokens
        
        for i, sample in enumerate(samples):
            seq_len = next(iter(sample.values())).shape[0]
            
            if seq_len > effective_max_len:
                if self.do_augmentation and not self.keep_recent_tail:  # random crop for MLM pre-training robustness
                    start_idx = random.randint(0, seq_len - effective_max_len)
                    end_idx = start_idx + effective_max_len
                    samples[i] = {key: val[start_idx:end_idx] for key, val in sample.items()}
                else:  # middle truncation: head (static baseline data) + tail (recent data anchored at fup)
                    head_keep = min(head_budget, max(1, effective_max_len // 2))
                    tail_keep = effective_max_len - head_keep
                    indices = np.concatenate([
                        np.arange(head_keep),
                        np.arange(seq_len - tail_keep, seq_len),
                    ])
                    samples[i] = {key: val[indices] for key, val in sample.items()}

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
            to_add = ([sequence[0] if len(sequence) > 0 else 0], [sequence[-1] if len(sequence) > 0 else 0])
        elif data_key == self.value_num_key:
            return np.concatenate(
                [np.array([np.nan], dtype=np.float32), np.asarray(sequence, dtype=np.float32), np.array([np.nan], dtype=np.float32)],
                axis=0,
            )
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
        if samples and self.value_num_key in samples[0]:
            num_sequences = [np.asarray(s[self.value_num_key], dtype=np.float32) for s in samples]
            padded_batch[self.value_num_key] = self._pad_sequences_numpy(
                num_sequences, batch_first=True, padding_value=np.nan,
            )

        # Compute attention mask, given the padding performed on the sequences
        seq_lengths = np.array([len(s[self.feature_keys[0]]) for s in samples], dtype=np.int64)
        max_len = int(seq_lengths.max()) if len(seq_lengths) > 0 else 0
        attention_mask = (np.arange(max_len)[None, :] < seq_lengths[:, None])
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
        self.do_augmentation = bool(samples and samples[0].get("split") == "train")
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)
        # Snapshot clean (unmasked) arrays before _masked_modelling mutates padded_batch in-place.
        # Used by causal next-event pretraining (`tfg`) so history h_{t-1} is computed on
        # clean events while the query at step t uses the partially-masked event.
        clean_snapshot = {
            f"_clean_{k}": v.copy()
            for k, v in padded_batch.items()
            if k in self.mlm_masking_rules or k == self.value_num_key
        }
        masked_input_dict, labels, labels_num = self._masked_modelling(padded_batch)
        input_dict_tensors = {k: torch.tensor(v) for k, v in masked_input_dict.items()}
        for ck, cv in clean_snapshot.items():
            input_dict_tensors[ck] = torch.tensor(cv)
        batch = {
            "input_dict": input_dict_tensors,
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels).long(),
            "labels_num": torch.tensor(labels_num, dtype=torch.float32),
        }
        batch.update(non_features)

        return batch

    def _masked_modelling(
        self,
        masked_input_dict: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Generalized MLM with mutually exclusive masking per token.
        At any given position, maximum one feature is masked.
        - Discrete tokens (entities, attributes, and categorical values) get positive token IDs in `final_labels`
          and `NaN` in `final_labels_num` (trained with CrossEntropyLoss on the discrete vocabulary head).
        - Continuous numerical values (`value_id` where `value_num` is finite) get their ground-truth scalar
          in `final_labels_num` (trained with SmoothL1Loss on `numerical_mlm_head`) and negative bin sentinel
          `-1 - bin_idx` in `final_labels` (excluded from discrete CrossEntropyLoss, used for unified MLM accuracy).
        """
        # Initialization
        any_key = next(iter(self.mlm_masking_rules.keys()))
        batch_shape = masked_input_dict[any_key].shape
        final_labels = np.full(batch_shape, -100, dtype=int)  # will be filled
        final_labels_num = np.full(batch_shape, np.nan, dtype=np.float32)
        global_mask_map = torch.zeros(batch_shape, dtype=torch.bool)  # who is masked
        features_to_process = list(self.mlm_masking_rules.keys())
        # On validation/test (`not self.do_augmentation`), use a local batch-deterministic RNG so
        # mid-training eval checkpoints are scored on identical masks without perturbing training RNG!
        if self.do_augmentation:
            random.shuffle(features_to_process)
            torch_gen = None
            np_rng = None
        else:
            batch_seed = int(np.asarray(masked_input_dict[any_key], dtype=np.int64).sum() * 2654435761) & 0x7FFFFFFF
            local_py_rng = random.Random(batch_seed)
            local_py_rng.shuffle(features_to_process)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(batch_seed)
            np_rng = np.random.default_rng(batch_seed)

        ground_truth_num = (
            masked_input_dict[self.value_num_key].copy()
            if self.value_num_key in masked_input_dict
            else None
        )

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
            candidate_mask = torch.bernoulli(prob_matrix, generator=torch_gen).bool()

            # Enforce mutual exclusivity
            final_mask = candidate_mask & (~global_mask_map)
            global_mask_map = global_mask_map | final_mask
            if not final_mask.any():
                continue
            final_mask_np = final_mask.numpy()
            
            # Update labels if part of label keys
            if feature_key in self.mlm_label_keys:
                target_values = masked_input_dict[feature_key]
                if self.use_numerical_embedding and feature_key == "value_id" and ground_truth_num is not None:
                    is_num_target = final_mask_np & np.isfinite(ground_truth_num)
                    is_cat_target = final_mask_np & (~np.isfinite(ground_truth_num))
                    # Categorical value targets go to discrete vocabulary head (labels >= 0)
                    final_labels[is_cat_target] = target_values[is_cat_target]
                    # Continuous value targets go to continuous regression head (labels_num in [-1, 1], labels = -1 - bin_idx)
                    if is_num_target.any():
                        gt_u = ground_truth_num[is_num_target]
                        final_labels_num[is_num_target] = gt_u
                        bin_idx = np.clip(np.floor((gt_u + 1.0) * 0.5 * 7.0).astype(int), 0, 6)
                        final_labels[is_num_target] = -1 - bin_idx
                else:
                    final_labels[final_mask_np] = target_values[final_mask_np]
                
            # Replace 80% of selected masked token locations using [MASK]
            indices_replaced = torch.bernoulli(torch.full(final_labels.shape, 0.8), generator=torch_gen).bool() & final_mask
            input_tokens[indices_replaced] = self.mask_token_id

            # Replace 10% of selected masked token locations with a random token/value (10% left untouched)
            indices_random = torch.bernoulli(torch.full(final_labels.shape, 0.5), generator=torch_gen).bool() & final_mask & ~indices_replaced
            flat_tokens = input_tokens.flatten()
            valid_choices = flat_tokens[~np.isin(flat_tokens, no_mask_ids)]
            if valid_choices.size > 0:
                rand_idx = (
                    np_rng.integers(0, valid_choices.size, size=input_tokens.shape)
                    if np_rng is not None
                    else np.random.randint(0, valid_choices.size, input_tokens.shape)
                )
                random_tokens = valid_choices[rand_idx]
                input_tokens[indices_random] = random_tokens[indices_random]

            # Update the batch dictionary
            masked_input_dict[feature_key] = input_tokens
            if feature_key == "value_id" and ground_truth_num is not None:
                masked_num = ground_truth_num.copy()
                replaced_np = indices_replaced.numpy()
                random_np = indices_random.numpy()
                # 80% [MASK] positions -> NaN so encode_values uses the discrete [MASK] token embedding
                masked_num[replaced_np] = np.nan
                # 10% random positions -> for continuous tokens, sample random Uniform(-1, 1); for categorical, keep NaN
                is_num_rand = random_np & np.isfinite(ground_truth_num)
                is_cat_rand = random_np & (~np.isfinite(ground_truth_num))
                if is_num_rand.any():
                    n_r = int(is_num_rand.sum())
                    masked_num[is_num_rand] = (
                        np_rng.uniform(-1.0, 1.0, size=n_r).astype(np.float32)
                        if np_rng is not None
                        else np.random.uniform(-1.0, 1.0, size=n_r).astype(np.float32)
                    )
                masked_num[is_cat_rand] = np.nan
                masked_input_dict[self.value_num_key] = masked_num

        return masked_input_dict, final_labels, final_labels_num
  

@dataclass
class PatientDataCollatorForClassification(PatientDataCollatorForMaskedLanguageModelling):
    """
    Inherits padding/truncation logic from the MLM collator, 
    but disables masking and extracts a specific target label.
    """
    label_keys: str | list[str] | None = None
    keep_recent_tail: bool = True

    def __post_init__(self):
        """Anything that must be initialized dynamically"""
        super().__post_init__()
        if self.label_keys is None:
            raise ValueError("Label keys missing in patient data collator for classification")
        if isinstance(self.label_keys, str):
            self.label_keys = [self.label_keys]

    def torch_call(self, samples: list[dict[str, np.ndarray]]) -> dict:
        """
        Collate patient sequences and create labels for supervised fine-tuning
        Used by HuggingFace's trainer
        """
        samples = [dict(s) for s in samples]
        batch_labels = []
        for s in samples:
            missing_keys = [k for k in self.label_keys if k not in s]
            if missing_keys:
                raise KeyError(
                    f"Missing required label key(s) {missing_keys} in sample (expected {self.label_keys})."
                )
            batch_labels.append([s.pop(k) for k in self.label_keys])

        # Preprocess the input samples
        self.do_augmentation = bool(samples and samples[0].get("split") == "train")
        samples, non_features = self._separate_non_processed_features(samples)
        padded_batch, attention_mask = self._process_batch(samples)

        # Assemble batch
        batch = {
            "input_dict": {k: torch.tensor(v) for k, v in padded_batch.items()},
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(batch_labels, dtype=torch.float),
        }
        batch.update(non_features)  # in case useful somewhere

        return batch


@dataclass
class PatientDataCollatorForSurvival(PatientDataCollatorForClassification):
    """
    Data collator for single-task or multi-task survival fine-tuning.

    Single-task: pass `time_label` and `status_label` -> labels tensor `[batch, 2]`.
    Multi-task (K outcomes): pass `survival_label_pairs=[(t_0, s_0), ..., (t_{K-1}, s_{K-1})]`
    -> labels tensor `[batch, 2*K]` (plus optional trailing Horvitz-Thompson `ipw_weight`
       column `[batch, 2*K + 1]` when present on subsampled training splits), and
       top-level float32 `landmark_fup` tensor `[batch]` when `fup` is present.
    """
    time_label: str | None = None
    status_label: str | None = None
    survival_label_pairs: list[tuple[str, str]] | None = None

    def __post_init__(self):
        if self.survival_label_pairs is not None:
            if not self.survival_label_pairs:
                raise ValueError("PatientDataCollatorForSurvival: 'survival_label_pairs' must not be empty.")
            self.label_keys = [col for t_col, s_col in self.survival_label_pairs for col in (t_col, s_col)]
        else:
            if self.time_label is None or self.status_label is None:
                raise ValueError(
                    "PatientDataCollatorForSurvival requires 'time_label' and 'status_label' (or 'survival_label_pairs')."
                )
            self.label_keys = [self.time_label, self.status_label]
        super().__post_init__()

    def torch_call(self, samples: list[dict[str, np.ndarray]]) -> dict:
        ipw_vals = None
        if samples and "ipw_weight" in samples[0]:
            samples = [dict(s) for s in samples]
            ipw_vals = [float(s.pop("ipw_weight")) for s in samples]
        batch = super().torch_call(samples)
        if ipw_vals is not None:
            w_col = torch.tensor(ipw_vals, dtype=torch.float32).unsqueeze(1)
            batch["labels"] = torch.cat([batch["labels"], w_col], dim=1)
        if "fup" in batch:
            batch["landmark_fup"] = torch.as_tensor(batch["fup"], dtype=torch.float32)
        return batch