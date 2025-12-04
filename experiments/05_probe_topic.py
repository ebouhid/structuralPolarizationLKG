import sys
import os
import networkx as nx
import torch
import json
import yaml
import pandas as pd
from sae_lens import SAE
from transformer_lens import HookedTransformer

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)  # Expanded for better reading
pd.set_option('display.width', 1000)


def interpret_feature(feature_idx, sae, model, k=4):
    """Decodes a feature into its top K tokens."""
    decoder_vec = sae.W_dec[feature_idx]
    logits = decoder_vec @ model.W_U
    vals, indices = torch.topk(logits, k)
    return [model.tokenizer.decode(i.item()).replace('\n', '\\n') for i in indices]


def get_top_features_for_prompt(prompt, target_word, model, sae, hook_name, k=3):
    """
    Runs a prompt and finds the features that activate MOST on the 'target_word'.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    # Find the position of the target word in the prompt
    # This is a simple heuristic: we assume the target word is the last one or close to it
    # Ideally, we look for the token index.
    token_strs = model.to_str_tokens(prompt)
    try:
        # Find index containing the target substring
        target_idx = next(i for i, t in enumerate(
            token_strs) if target_word.strip() in t)
    except StopIteration:
        target_idx = -1  # Default to last token if not found

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name]
        feature_acts = sae.encode(resid)

        # Get activations at the specific target token position
        # shape: [batch, pos, features]
        position_acts = feature_acts[0, target_idx, :]

        # Get Top K features
        top_vals, top_indices = torch.topk(position_acts, k)

    return top_indices.tolist(), top_vals.tolist()


def main():
    # 1. Setup
    config_path = os.path.join(parent_dir, "config/experiment.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading Model & SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(
        config['model']['name'],
        device=device,
        fold_ln=False,
        dtype=torch.float16
    )

    sae_path = os.path.join(config['sae']['release'], config['sae']['id'])
    cfg_file = os.path.join(sae_path, "cfg.json")
    if not os.path.exists(cfg_file):
        sae_path = config['sae']['release']
        cfg_file = os.path.join(sae_path, config['sae']['id'], "cfg.json")

    with open(cfg_file, 'r') as f:
        cfg_dict = json.load(f)
    cfg_dict['device'] = device
    cfg_dict['dtype'] = "float16"

    sae = SAE.from_dict(cfg_dict)
    weight_path = os.path.join(os.path.dirname(cfg_file), "sae_weights.pt")
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    sae.load_state_dict(state_dict)

    # 2. Contextual Probes
    # Format: (Prompt, Target_Token_To_Inspect)
    probes = [
        ("The legislation forces strict gun control", "gun"),
        # ("The Supreme Court overturned Roe v. Wade on abortion", "abortion"),
        # ("The global crisis of climate change", "climate"),
        # ("The border crisis and illegal immigration", "immigration"),
        # ("The government raised the income tax", "tax"),
        # ("The right to same-sex marriage", "marriage"),
        # ("The legalized use of recreational drugs like marijuana", "drugs"),
        # ("The freedom of speech and expression", "speech"),
        # ("The impact of social media on society", "social"),
        # ("The debate over healthcare reform", "healthcare"),
        # ("Everybody listened when the bossy manager spoke", "bossy"),
        ("Everbody listened when the assertive manager spoke", "assertive"),
        ("The child was very happy with the colorful toy", "colorful"),
    ]

    hook_name = config['sae']['id']

    print("\nRunning Contextual Probes...")

    # Pre-calculate probe results
    probe_results_data = []
    for prompt, target in probes:
        top_feats, acts = get_top_features_for_prompt(
            prompt, target, model, sae, hook_name, k=3)
        probe_results_data.append({
            "prompt": prompt,
            "target": target,
            "top_feats": top_feats,
            "acts": acts
        })

    # 3. Process Graphs
    graphs = [
        ("neutral", os.path.join(parent_dir, "results/graphs/neutral_lkg.gexf")),
        ("political", os.path.join(parent_dir, "results/graphs/political_lkg.gexf"))
    ]

    for graph_name, graph_path in graphs:
        print(f"Processing {graph_name} graph: {graph_path}")
        if not os.path.exists(graph_path):
            print("Graph not found.")
            continue
        G = nx.read_gexf(graph_path)

        results = []
        for item in probe_results_data:
            prompt = item['prompt']
            target = item['target']
            top_feats = item['top_feats']
            acts = item['acts']

            for i, (feat_id, act) in enumerate(zip(top_feats, acts)):
                # Decode
                tokens = interpret_feature(feat_id, sae, model)
                decoding_str = ", ".join(tokens)

                # Check Neighbors
                node_id = str(feat_id)
                neighbors_str = "ISOLATED"

                if node_id in G:
                    neighbors = list(G.neighbors(node_id))
                    try:
                        neighbors = sorted(neighbors, key=lambda x: float(
                            G[node_id][x]['weight']), reverse=True)[:4]
                    except:
                        neighbors = neighbors[:4]

                    neighbor_desc = []
                    for n_id in neighbors:
                        n_tokens = interpret_feature(
                            int(n_id), sae, model, k=1)
                        try:
                            n_weight = float(G[node_id][n_id]['weight'])
                            neighbor_desc.append(
                                f"{n_tokens[0]}({n_weight:.2f})")
                        except:
                            neighbor_desc.append(f"{n_tokens[0]}")

                    if neighbor_desc:
                        neighbors_str = " | ".join(neighbor_desc)

                results.append({
                    "Context": f"{target} (Rank {i+1})",
                    "Feature ID": feat_id,
                    "Act": f"{act:.1f}",
                    "Decoding": decoding_str,
                    "Neighbors": neighbors_str
                })

        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"   CONTEXTUAL FEATURE ANALYSIS ({graph_name.upper()})")
        print("="*80)
        print(df.to_string(index=False))
        output_csv = os.path.join(
            parent_dir, f"results/05_probe_topic_results_{graph_name}.csv")
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
