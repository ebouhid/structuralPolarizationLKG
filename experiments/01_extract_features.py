import hydra
from omegaconf import DictConfig
from src.model_utils import LKGModelWrapper
from src.data_loader import LKGDataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def pad_and_tensorize(batch_input_ids, pad_token_id, device):
    """
    Pads a list of lists to the longest sequence in the batch.
    """
    # 1. Find max length in this specific batch
    max_len = max(len(seq) for seq in batch_input_ids)

    # 2. Pad shorter sequences
    padded_batch = []
    for seq in batch_input_ids:
        # Pad Right (standard for analysis, though Llama usually generates left-pad)
        # We use standard right-padding here as we aren't generating new tokens
        num_pads = max_len - len(seq)
        padded_seq = seq + [pad_token_id] * num_pads
        padded_batch.append(padded_seq)

    return torch.tensor(padded_batch).to(device)


def run_extraction(dataset_name, dataset, model_wrapper, batch_size, output_dir):
    print(f"\n--- Extracting features for: {dataset_name} ---")
    all_active_features = []
    total_rows = len(dataset)

    # Get pad token from the model wrapper
    pad_token_id = model_wrapper.model.tokenizer.pad_token_id

    # Processing Loop
    for i in tqdm(range(0, total_rows, batch_size)):
        # Slice dataset (returns dictionary of lists)
        batch = dataset[i: i + batch_size]

        # --- THE FIX: Pad dynamically ---
        input_ids_list = batch['input_ids']
        tokens = pad_and_tensorize(
            input_ids_list, pad_token_id, model_wrapper.device)

        # Run Model + SAE (The wrapper now ignores the pad tokens we just added)
        batch_features = model_wrapper.get_active_features(tokens)

        all_active_features.extend(batch_features)

    # Save Result
    output_path = output_dir / f"{dataset_name}_features.pkl"
    print(f"Saving {len(all_active_features)} contexts to {output_path}...")

    with open(output_path, 'wb') as f:
        pickle.dump(all_active_features, f)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()

    output_dir = Path(orig_cwd) / "results/activation_caches"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Data Loader...")
    loader = LKGDataLoader(cfg)

    print("Initializing Model & SAE...")
    model_wrapper = LKGModelWrapper(cfg)

    print("Preparing Balanced Corpora...")
    neutral_ds, political_ds = loader.prepare_balanced_corpora()

    batch_size = cfg.pipeline.batch_size

    run_extraction("neutral", neutral_ds, model_wrapper,
                   batch_size, output_dir)
    run_extraction("political", political_ds,
                   model_wrapper, batch_size, output_dir)

    print("\nFeature Extraction Complete.")


if __name__ == "__main__":
    main()
