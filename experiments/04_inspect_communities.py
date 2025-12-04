import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import hydra
from omegaconf import DictConfig
from src.topology_metrics import TopologyAnalyzer
import networkx as nx
import torch
import json
import numpy as np
from sae_lens import SAE, SAEConfig
from transformer_lens import HookedTransformer

def load_sae_and_model(cfg: DictConfig, orig_cwd: str):
    # We need the model to get the Unembed matrix (vocab projection)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Model ({device})...")
    model = HookedTransformer.from_pretrained(
        cfg.model.name,
        device=device,
        fold_ln=False,
        dtype="float16"
    )

    print("Loading SAE...")
    sae_path = os.path.join(orig_cwd, cfg.sae.release, cfg.sae.id)
    with open(os.path.join(sae_path, "cfg.json"), 'r') as f:
        cfg_dict = json.load(f)
    cfg_dict['device'] = device
    cfg_dict['dtype'] = "float16"

    sae = SAE.from_dict(cfg_dict)
    state_dict = torch.load(os.path.join(
        sae_path, "sae_weights.pt"), map_location=device)
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    sae.load_state_dict(state_dict)

    return sae, model


def interpret_feature(feature_idx, sae, model, k=5):
    """
    Projects the SAE decoder direction into the vocabulary 
    to see which tokens this feature promotes.
    """
    # 1. Get Decoder Vector for this feature [d_model]
    # Check shape: [d_sae, d_model] or [d_model, d_sae]
    # SAELens W_dec is usually [d_sae, d_model]
    decoder_vec = sae.W_dec[feature_idx]

    # 2. Unembed (Project to Vocab)
    # logits: [d_vocab]
    logits = decoder_vec @ model.W_U

    # 3. Top K
    vals, indices = torch.topk(logits, k)

    tokens = [model.tokenizer.decode(i.item()) for i in indices]
    return tokens


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()

    # 1. Load Resources
    sae, model = load_sae_and_model(cfg, orig_cwd)

    # 2. Load Political Graph
    graph_path = os.path.join(orig_cwd, "results/graphs/political_lkg.gexf")
    print(f"Loading Graph: {graph_path}")
    G = nx.read_gexf(graph_path)

    # 3. Get Top Communities
    print("Detecting Communities...")
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight='weight', seed=42)

    # Sort by size
    communities = sorted(communities, key=len, reverse=True)

    print("\n--- TOP 5 POLITICAL COMMUNITIES ---")
    for i, comm in enumerate(communities[:5]):
        print(f"\nCommunity {i+1} (Size: {len(comm)} nodes)")

        # Get 'Central' nodes in this community (highest degree within subgraph)
        subgraph = G.subgraph(comm)
        # Sort nodes by weighted degree
        top_nodes = sorted(subgraph.degree(weight='weight'),
                           key=lambda x: x[1], reverse=True)[:5]

        for feat_idx, degree in top_nodes:
            # GEXF stores IDs as strings, convert to int
            idx = int(feat_idx)
            tokens = interpret_feature(idx, sae, model)
            print(f"  Feature {idx} (Deg: {degree:.2f}): {tokens}")


if __name__ == "__main__":
    main()
