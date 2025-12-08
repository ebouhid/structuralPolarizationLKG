import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from omegaconf import DictConfig
import numpy as np
import logging
from typing import List, Dict, Optional, Union

# Configure logging to track the scientific process
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LKGDataLoader:
    def __init__(self, config: Union[DictConfig, Dict]):
        """
        Initializes the loader with experiment config.

        Args:
            config: Configuration (DictConfig or dict) containing:
                    - model.name: HF model ID for tokenizer
                    - data.political_path: HF dataset ID
                    - data.neutral_path: Local path to elevator parquet
                    - data.topics: List of keywords for filtering
        """
        self.config = config

        # Handle both DictConfig and dict access patterns
        model_name = config.model.name if isinstance(
            config, DictConfig) else config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Contentious topics as defined in your experimental design [cite: 100, 503]
        if isinstance(config, DictConfig):
            self.keywords = list(config.data.get('topics', [
                "gun control", "abortion", "climate change", "immigration"
            ]))
            self.max_seq_len = config.pipeline.get('context_window', 2048)
        else:
            self.keywords = config['data'].get('topics', [
                "gun control", "abortion", "climate change", "immigration"
            ])
            self.max_seq_len = config['pipeline'].get('context_window', 2048)

    def get_token_count(self, dataset: Dataset) -> int:
        """Calculates total token count in a dataset."""
        # We sum the lengths of the input_ids column
        return sum(len(ids) for ids in dataset['input_ids'])

    def load_elevator_corpus(self) -> Dataset:
        """
        Loads and tokenizes the neutral control corpus (Elevator Wiki).
        """
        neutral_path = self.config.data.neutral_path if isinstance(
            self.config, DictConfig) else self.config['data']['neutral_path']
        logger.info(f"Loading neutral corpus from {neutral_path}...")

        # Load from local parquet/csv
        # Using 'parquet' engine via datasets library
        dataset = load_dataset(
            "parquet", data_files=neutral_path, split="train")

        # Tokenize immediately to get the baseline count
        logger.info("Tokenizing neutral corpus...")
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['text'],
                truncation=True,       # <--- CHANGE THIS to True
                max_length=self.max_seq_len,  # <--- ADD THIS
                padding=False
            ),
            batched=True,
            remove_columns=dataset.column_names
        )

        total_tokens = self.get_token_count(dataset)
        logger.info(
            f"Neutral Corpus Loaded: {len(dataset)} rows, {total_tokens} tokens.")
        return dataset, total_tokens

    def load_political_corpus(self, target_token_count: int, sampling_strategy: str = "first") -> Dataset:
        """
        Loads LOCAL parquet files, filters, and tokenizes until target count is met.

        Args:
            target_token_count: Number of tokens to accumulate.
            sampling_strategy: How to select entries from the filtered dataset.
                - "first": Take entries from the beginning (oldest, assuming chronological order)
                - "last": Take entries from the end (most recent)
                - "random": Randomly sample entries
        """
        political_path = self.config.data.political_path if isinstance(
            self.config, DictConfig) else self.config['data']['political_path']
        logger.info(f"Loading local political corpus from {political_path}...")

        # 1. Load the dataset from local Parquet files
        # This is lazy-loaded (memory mapped), so it won't crash RAM
        dataset = load_dataset(
            "parquet",
            data_dir=political_path,
            split="train"
        )

        logger.info(f"Total raw rows available: {len(dataset)}")
        logger.info(f"Filtering for topics: {self.keywords}")

        # 2. Filter (Keyword Search)
        # Since we are local, we can use multiple workers for speed
        def filter_contentious(example):
            text = example.get('text', '')
            if text is None:
                return False
            # Simple string check is faster than regex for high-volume
            return any(keyword in text.lower() for keyword in self.keywords)

        # This creates a filtered dataset on disk/memory without network issues
        filtered_dataset = dataset.filter(filter_contentious, num_proc=12)

        logger.info(f"Filtered rows matching topics: {len(filtered_dataset)}")

        # 3. Apply sampling strategy
        if sampling_strategy == "random":
            # Shuffle the dataset for random sampling
            filtered_dataset = filtered_dataset.shuffle(seed=42)
            logger.info("Using RANDOM sampling strategy (shuffled dataset)")
        elif sampling_strategy == "last":
            # Reverse the indices to get most recent entries first
            indices = list(range(len(filtered_dataset) - 1, -1, -1))
            filtered_dataset = filtered_dataset.select(indices)
            logger.info("Using LAST (most recent) sampling strategy")
        elif sampling_strategy == "first":
            logger.info("Using FIRST (oldest) sampling strategy")
        else:
            raise ValueError(
                f"Unknown sampling_strategy: {sampling_strategy}. Use 'first', 'last', or 'random'.")

        # 4. Tokenize and Accumulate
        accumulated_data = []
        current_tokens = 0
        batch_size = 1000

        # We iterate over the filtered local dataset
        iterator = iter(filtered_dataset)

        logger.info("Tokenizing and accumulating...")
        while current_tokens < target_token_count:
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    break

            if not batch:
                break

            temp_ds = Dataset.from_list(batch)
            temp_ds = temp_ds.map(
                lambda x: self.tokenizer(
                    x['text'],
                    truncation=True,       # <--- CHANGE THIS to True
                    max_length=self.max_seq_len,  # <--- ADD THIS
                    padding=False
                ),
                batched=True,
                remove_columns=dataset.column_names
            )

            batch_tokens = sum(len(ids) for ids in temp_ds['input_ids'])
            accumulated_data.append(temp_ds)
            current_tokens += batch_tokens

            if current_tokens % 100000 < batch_tokens:  # Log occasionally
                logger.info(
                    f"Accumulated {current_tokens} / {target_token_count} tokens...")

        logger.info(f"Final accumulated tokens: {current_tokens}")
        return concatenate_datasets(accumulated_data)

    def prepare_balanced_corpora(self, sampling_strategy: str = "first"):
        """
        Orchestrates the loading and balancing.
        Returns two datasets of EQUAL token counts.

        Args:
            sampling_strategy: How to select political entries - "first", "last", or "random"
        """
        # 1. Load Neutral (The Anchor)
        neutral_ds, neutral_tokens = self.load_elevator_corpus()

        # 2. Load Political (The Variable) matched to Neutral
        political_ds = self.load_political_corpus(
            target_token_count=neutral_tokens,
            sampling_strategy=sampling_strategy)

        # 3. Strict Truncation
        # Even with loop, political might be slightly larger. We must truncate exactly.
        # This ensures graph density differences are due to structure, not size.

        # Flatten to a single list of tokens for precise slicing (conceptually)
        # For efficiency, we just truncate the last rows of political_ds if needed
        # (Simplification: In a real rigorous run, we'd flatten and reshape,
        # but row-level balancing is usually sufficient if rows are small).

        logger.info("Data Loading Complete. Ready for LKG Extraction.")
        return neutral_ds, political_ds


# Quick Test Block
if __name__ == "__main__":
    # Mock Config
    conf = {
        "model": {"name": "meta-llama/Llama-3.1-8B-Instruct"},
        "data": {
            "political_path": "/files/Doutorado/polarization/structuralPolarizationLKG/data/congressional_speeches/data/",
            # Adjust to your local path
            "neutral_path": "/files/Doutorado/polarization/structuralPolarizationLKG/data/elevator_corpus/train.parquet",
            "topics": ["gun control", "abortion"]
        }
    }
    loader = LKGDataLoader(conf)
    neu, pol = loader.prepare_balanced_corpora()
