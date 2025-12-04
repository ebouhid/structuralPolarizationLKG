import torch
import os
import json
from omegaconf import DictConfig
from transformer_lens import HookedTransformer
from sae_lens import SAE, SAEConfig
import logging
from typing import List, Set, Union, Dict

logger = logging.getLogger(__name__)


class LKGModelWrapper:
    def __init__(self, config: Union[DictConfig, Dict]):
        self.config = config
        self.device = config.model.device if isinstance(
            config, DictConfig) else config['model']['device']

        # Get model name
        model_name = config.model.name if isinstance(
            config, DictConfig) else config['model']['name']

        # 1. Load Base LLM (FP16)
        logger.info(
            f"Loading Base Model: {model_name} on {self.device} (FP16)")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=torch.float16
        )

        # Ensure we have a pad token for masking later
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        # 2. Load SAE (Manual Local)
        sae_release = config.sae.release if isinstance(
            config, DictConfig) else config['sae']['release']
        sae_id = config.sae.id if isinstance(
            config, DictConfig) else config['sae']['id']
        full_path = os.path.join(sae_release, sae_id)

        logger.info(f"Loading SAE from: {full_path}")

        with open(os.path.join(full_path, "cfg.json"), 'r') as f:
            cfg_dict = json.load(f)

        cfg_dict['device'] = self.device
        cfg_dict['dtype'] = "float16"

        self.sae = SAE.from_dict(cfg_dict)

        state_dict = torch.load(os.path.join(
            full_path, "sae_weights.pt"), map_location=self.device)
        state_dict = {k: v.to(dtype=torch.float16)
                      for k, v in state_dict.items()}

        self.sae.load_state_dict(state_dict)
        self.sae.to(self.device, dtype=torch.float16)

        assert self.sae.cfg.d_in == self.model.cfg.d_model, "Dimension Mismatch!"

        # Get activation threshold
        if isinstance(config, DictConfig):
            self.activation_threshold = config.pipeline.get(
                'activation_threshold', 1.0)
        else:
            self.activation_threshold = config['pipeline'].get(
                'activation_threshold', 1.0)

    def get_active_features(self, tokens: torch.Tensor) -> List[Set[int]]:
        """
        Runs forward pass, ignores padding tokens, returns sparse features.
        """
        # List to store sets of features for VALID tokens only
        active_indices_valid_only = []

        # Create a mask: 1 if real token, 0 if padding
        # tokens shape: [batch, seq_len]
        pad_id = self.model.tokenizer.pad_token_id
        valid_mask = (tokens != pad_id)

        def sae_hook(resid_pre, hook):
            # resid_pre: [batch, seq, d_model]
            resid_pre = resid_pre.to(dtype=torch.float16)

            # 1. Encode
            feature_acts = self.sae.encode(resid_pre)

            # 2. Threshold
            mask = feature_acts > self.activation_threshold

            # 3. Extract Coordinates [batch, seq, feature_idx]
            coords = torch.nonzero(mask)
            coords_cpu = coords.detach().cpu().numpy()

            # 4. Filter Padding
            # We need to map back to the flattened list structure.
            # But we only want to keep entries where valid_mask[b, s] is True.

            # Optimization: Instead of pre-allocating for everything and filtering later,
            # we iterate the sparse coordinates and check the mask.

            # First, create a temporary lookup for this batch
            # shape: [batch, seq] -> set of features
            batch_sets = [[set() for _ in range(tokens.shape[1])]
                          for _ in range(tokens.shape[0])]

            for b, s, f in coords_cpu:
                # CRITICAL CHECK: Is this a padding token?
                if valid_mask[b, s]:
                    batch_sets[b][s].add(int(f))

            # Flatten only valid tokens into the result list
            for b in range(tokens.shape[0]):
                for s in range(tokens.shape[1]):
                    if valid_mask[b, s]:
                        active_indices_valid_only.append(batch_sets[b][s])

            return resid_pre

        # Get target hook point
        if isinstance(self.config, DictConfig):
            target_point = self.config.sae.id
            target_layer = self.config.model.target_layer
        else:
            target_point = self.config['sae']['id']
            target_layer = self.config['model']['target_layer']

        with torch.no_grad():
            self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(target_point, sae_hook)],
                stop_at_layer=target_layer + 1
            )

        return active_indices_valid_only
