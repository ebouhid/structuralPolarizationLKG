from src.graph_builder import LKGBuilder
import hydra
from omegaconf import DictConfig
import sys
import os
import pickle
import networkx as nx
from pathlib import Path

# Path fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def load_pickle(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()

    input_dir = Path(orig_cwd) / "results/activation_caches"
    output_dir = Path(orig_cwd) / "results/graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Features
    neutral_path = input_dir / "neutral_features.pkl"
    political_path = input_dir / "political_features.pkl"

    if not neutral_path.exists() or not political_path.exists():
        raise FileNotFoundError("Run 01_extract_features.py first!")

    neutral_feats = load_pickle(neutral_path)
    political_feats = load_pickle(political_path)

    # 3. SCIENTIFIC CONTROL: Truncate to Match Lengths
    # We must compare graphs built from the same amount of "time" (tokens)
    min_len = min(len(neutral_feats), len(political_feats))

    print(f"\n--- Data Balancing ---")
    print(f"Neutral Length: {len(neutral_feats)}")
    print(f"Political Length: {len(political_feats)}")
    print(f"Truncating both to: {min_len} contexts")

    neutral_feats = neutral_feats[:min_len]
    political_feats = political_feats[:min_len]

    # 4. Build Graphs
    builder = LKGBuilder(cfg)

    # Build Neutral
    print("\n--- Building Neutral LKG ---")
    G_neutral = builder.build_graph(neutral_feats, "Neutral")
    nx.write_gexf(G_neutral, output_dir / "neutral_lkg.gexf")
    print(f"Saved to {output_dir / 'neutral_lkg.gexf'}")

    # Build Political
    print("\n--- Building Political LKG ---")
    G_political = builder.build_graph(political_feats, "Political")
    nx.write_gexf(G_political, output_dir / "political_lkg.gexf")
    print(f"Saved to {output_dir / 'political_lkg.gexf'}")

    print("\nGraph Construction Complete.")


if __name__ == "__main__":
    main()
