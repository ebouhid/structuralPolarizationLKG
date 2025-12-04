import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import json
import torch
from sae_lens import SAE

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def load_sae_metadata(config):
    """Load SAE to GPU in Float16."""
    sae_path = config['sae']['release']
    if not os.path.exists(os.path.join(sae_path, "cfg.json")):
        sae_path = os.path.join(sae_path, config['sae']['id'])

    cfg_file = os.path.join(sae_path, "cfg.json")
    with open(cfg_file, 'r') as f:
        cfg_dict = json.load(f)

    # --- OPTIMIZATION: Use GPU + Float16 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dict['device'] = device
    cfg_dict['dtype'] = "float16"

    sae = SAE.from_dict(cfg_dict)

    weight_path = os.path.join(os.path.dirname(cfg_file), "sae_weights.pt")
    state_dict = torch.load(weight_path, map_location=device)

    # Cast to Float16
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}

    sae.load_state_dict(state_dict)
    sae.to(device, dtype=torch.float16)

    return sae


def main():
    # 1. Configuration
    config_path = os.path.join(parent_dir, "config/experiment.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Target Feature
    # Use Feature 58074 (The Semantic Gun Feature)
    TARGET_FEAT_ID = "58074"
    TARGET_LABEL = "GUN (Concept)"

    # Prepare output directory
    sanitized_label = TARGET_LABEL.split()[0].lower()
    output_dir = os.path.join(parent_dir, "results", sanitized_label)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load Model & SAE (GPU Mode) - Moved up to load once
    print("Loading SAE & Model to GPU (Float16)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae = load_sae_metadata(config)

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        config['model']['name'],
        device=device,
        fold_ln=False,
        # FIX: Use float16 on GPU to save VRAM and match SAE
        dtype=torch.float16
    )

    graphs = [
        ("neutral", os.path.join(parent_dir, "results/graphs/neutral_lkg.gexf")),
        ("polarized", os.path.join(parent_dir, "results/graphs/political_lkg.gexf"))
    ]

    for graph_name, graph_path in graphs:
        # 4. Load Graph
        print(f"Loading Graph ({graph_name}): {graph_path}")
        if not os.path.exists(graph_path):
            print(f"Graph {graph_path} not found, skipping.")
            continue

        G = nx.read_gexf(graph_path)

        if TARGET_FEAT_ID not in G:
            print(
                f"Error: Feature {TARGET_FEAT_ID} not found in {graph_name} graph!")
            continue

        # 5. Extract Ego Graph
        print(
            f"Extracting neighborhood for {TARGET_LABEL} in {graph_name} graph...")
        ego_G = nx.ego_graph(G, TARGET_FEAT_ID, radius=1)
        print(f"Nodes: {len(ego_G.nodes())}, Edges: {len(ego_G.edges())}")

        # 6. Decode Labels
        labels = {}
        node_colors = []
        node_sizes = []

        print("Decoding labels...")
        for node in ego_G.nodes():
            idx = int(node)

            # Calculation happens on GPU (Fast)
            decoder_vec = sae.W_dec[idx]
            logits = decoder_vec @ model.W_U
            top_idx = torch.argmax(logits).item()

            # Move result to CPU only for string decoding
            top_tok = model.tokenizer.decode(top_idx)
            clean_label = top_tok.replace("Ġ", "").strip()

            labels[node] = clean_label

            if node == TARGET_FEAT_ID:
                node_colors.append('#ff7f0e')  # Orange
                node_sizes.append(3000)
                labels[node] = f"{TARGET_LABEL}\\n({clean_label})"
            else:
                node_colors.append('#1f77b4')  # Blue
                if ego_G.has_edge(TARGET_FEAT_ID, node):
                    # Handle weight if it's string or float
                    w_raw = ego_G[TARGET_FEAT_ID][node]['weight']
                    weight = float(w_raw)
                    node_sizes.append(1000 + int(weight * 5000))
                else:
                    node_sizes.append(1000)

        # 7. Plotting (CPU)
        print(f"Generating Plot for {graph_name}...")
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(ego_G, k=0.5, seed=42)

        nx.draw_networkx_nodes(
            ego_G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(ego_G, pos, width=2,
                               alpha=0.4, edge_color="gray")
        nx.draw_networkx_labels(ego_G, pos, labels=labels, font_size=11,
                                font_family="sans-serif", font_weight="bold")

        plt.title(
            f"Latent Knowledge Graph ({graph_name}): {TARGET_LABEL}", fontsize=20)
        plt.axis('off')

        output_img = os.path.join(output_dir, f"concept_{graph_name}.png")
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_img}")
        plt.close()


if __name__ == "__main__":
    main()
