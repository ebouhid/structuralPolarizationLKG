import sys
import os
import torch
import pandas as pd
import numpy as np
import json
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformer_lens import HookedTransformer
from sae_lens import SAE
import logging
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def interpret_feature(feature_idx, sae, model, k=1):
    decoder_vec = sae.W_dec[feature_idx]
    logits = decoder_vec @ model.W_U
    vals, indices = torch.topk(logits, k)
    return model.tokenizer.decode(indices[0].item()).strip()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Resources
    logger.info("Loading Model & SAE...")
    model = HookedTransformer.from_pretrained(
        cfg.model.name,
        device=device,
        fold_ln=False,
        dtype="float16"
    )

    # Load SAE
    orig_cwd = hydra.utils.get_original_cwd()
    sae_path = os.path.join(orig_cwd, cfg.sae.release)
    if not os.path.exists(os.path.join(sae_path, "cfg.json")):
        sae_path = os.path.join(sae_path, cfg.sae.id)

    cfg_file = os.path.join(sae_path, "cfg.json")
    with open(cfg_file, 'r') as f:
        sae_cfg = json.load(f)
    sae_cfg['device'] = device
    sae_cfg['dtype'] = "float16"

    sae = SAE.from_dict(sae_cfg)
    weight_path = os.path.join(os.path.dirname(cfg_file), "sae_weights.pt")
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    sae.load_state_dict(state_dict)

    hook_point = cfg.sae.id

    # 2. Load Contrastive Topics from Config
    topics = {}

    # Inspect first item to determine keys
    first_topic = next(iter(cfg.contrastive.topics.values()))
    if 'mechanical' in first_topic and 'political' in first_topic:
        key_m, key_p = 'mechanical', 'political'
        label_m, label_p = 'Mech', 'Pol'
        short_m, short_p = 'M', 'P'
    elif 'view_a' in first_topic and 'view_b' in first_topic:
        key_m, key_p = 'view_a', 'view_b'
        label_m, label_p = 'View A', 'View B'
        short_m, short_p = 'A', 'B'
    else:
        # Fallback: take first two keys sorted
        keys = sorted(list(first_topic.keys()))
        if len(keys) >= 2:
            key_m, key_p = keys[0], keys[1]
            label_m, label_p = key_m, key_p
            short_m, short_p = '1', '2'
        else:
            raise ValueError(
                "Each topic must have at least two prompts defined.")

    logger.info(f"Using prompt keys: {key_m} (Base) vs {key_p} (Contrast)")

    for topic_name, prompts in cfg.contrastive.topics.items():
        topics[topic_name] = (
            prompts[key_m],
            prompts[key_p]
        )

    # 3. Extract Delta Vectors & Features
    deltas = {}
    feature_sets = {}
    residual_vectors = {}  # Store for PCA visualization

    logger.info("\n--- Extracting Contrastive Shifts ---")

    for topic, (mech_prompt, pol_prompt) in topics.items():
        # Run Mechanical
        _, cache_m = model.run_with_cache(
            mech_prompt, names_filter=[hook_point])
        resid_m = cache_m[hook_point][0, -1, :]  # Last token

        # Run Political
        _, cache_p = model.run_with_cache(
            pol_prompt, names_filter=[hook_point])
        resid_p = cache_p[hook_point][0, -1, :]  # Last token

        # Store vectors for later PCA
        residual_vectors[topic] = (resid_m, resid_p)

        # Calculate Vector Delta (The "Direction of Polarization" for this topic)
        delta = resid_p - resid_m
        deltas[topic] = delta

        # Extract Features specifically active in Political but NOT in Mechanical
        # We project the DELTA into SAE space to see what features constitute the difference
        # This is cleaner than subtracting sets.
        with torch.no_grad():
            # Encode the delta directly?
            # Better: Encode both, subtract activations
            acts_m = sae.encode(resid_m)
            acts_p = sae.encode(resid_p)

            # Find features that increased significantly
            act_diff = acts_p - acts_m
            # Get top 5 features that grew the most
            top_vals, top_inds = torch.topk(act_diff, 5)

            feats = []
            feats_detailed = []
            for idx, val in zip(top_inds, top_vals):
                if val.item() > 1.0:  # Threshold
                    label = interpret_feature(idx, sae, model)
                    feats.append(f"{label}({val.item():.1f})")
                    feats_detailed.append({
                        'feature_idx': idx.item(),
                        'token': label,
                        'activation_increase': val.item()
                    })

            feature_sets[topic] = {
                'formatted': feats, 'detailed': feats_detailed}
            logger.info(f"[{topic}] Top Shift Features: {', '.join(feats)}")

    # Save SAE feature interpretations to CSV
    orig_cwd = hydra.utils.get_original_cwd()
    # Hydra's managed experiment directory
    output_dir = HydraConfig.get().runtime.output_dir
    feature_interpretation_data = []
    for topic, features_dict in feature_sets.items():
        for feat_info in features_dict['detailed']:
            feature_interpretation_data.append({
                'topic': topic,
                'feature_idx': feat_info['feature_idx'],
                'token': feat_info['token'],
                'activation_increase': feat_info['activation_increase']
            })

    df_features = pd.DataFrame(feature_interpretation_data)
    features_path = os.path.join(
        output_dir, "08_sae_feature_interpretation.csv")
    df_features.to_csv(features_path, index=False)
    # 4. Compute Cosine Similarity between Topics
    logger.info(f"SAE feature interpretations saved to {features_path}")
    # Do "Gun Politics" and "Abortion Politics" point in the same direction?
    print("\n" + "="*60)
    print("   GEOMETRIC DISENTANGLEMENT ANALYSIS")
    print("="*60)
    print("Cosine Similarity between Topic Shifts (1.0 = Same Direction, 0.0 = Orthogonal)")

    topic_list = list(topics.keys())
    similarity_data = []

    for i in range(len(topic_list)):
        for j in range(i+1, len(topic_list)):
            t1 = topic_list[i]
            t2 = topic_list[j]

            v1 = deltas[t1].float().cpu().numpy()
            v2 = deltas[t2].float().cpu().numpy()

            # 1 - cosine distance = cosine similarity
            sim = 1 - cosine(v1, v2)
            print(f"{t1} vs {t2}: {sim:.4f}")
            similarity_data.append(
                {'topic1': t1, 'topic2': t2, 'cosine_similarity': sim})

    # Save similarity matrix to CSV
    df_similarity = pd.DataFrame(similarity_data)
    similarity_path = os.path.join(
        output_dir, "08_cosine_similarity_matrix.csv")
    df_similarity.to_csv(similarity_path, index=False)
    logger.info(f"Cosine similarity matrix saved to {similarity_path}")

    # Build full square similarity matrix for heatmap
    n_topics = len(topic_list)
    similarity_matrix = np.eye(n_topics)  # Initialize with 1s on diagonal

    # Fill in the similarity values
    topic_idx_map = {topic: i for i, topic in enumerate(topic_list)}
    for _, row in df_similarity.iterrows():
        i = topic_idx_map[row['topic1']]
        j = topic_idx_map[row['topic2']]
        sim = row['cosine_similarity']
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim  # Make symmetric

    # Save full square matrix as CSV
    df_matrix = pd.DataFrame(
        similarity_matrix,
        index=topic_list,
        columns=topic_list
    )
    matrix_path = os.path.join(output_dir, "08_similarity_matrix_full.csv")
    df_matrix.to_csv(matrix_path)
    logger.info(f"Full similarity matrix saved to {matrix_path}")

    # Create heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        vmin=-0.2,
        vmax=1.0,
        cbar_kws={'label': 'Cosine Similarity'},
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=True,
        yticklabels=True,
        annot_kws={'size': 8}
    )
    plt.title('Cosine Similarity Matrix: Topic Polarization Directions\n(1.0 = Same Direction, 0.0 = Orthogonal, -1.0 = Opposite)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Topic', fontsize=12, fontweight='bold')
    plt.ylabel('Topic', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "08_similarity_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logger.info(f"Similarity heatmap saved to {heatmap_path}")
    plt.close()

    # 5. PCA Visualization of Trajectories
    logger.info("\n--- Running PCA on Residual Trajectories ---")

    labels = []
    vectors = []
    colors_list = []

    # Define color map for topics
    num_topics = len(topics)
    colors_map = cm.get_cmap('tab20')(np.linspace(0, 1, num_topics))
    topic_colors = {topic: colors_map[i] for i, topic in enumerate(topic_list)}

    for topic in topic_list:
        vec_m, vec_p = residual_vectors[topic]

        vectors.append(vec_m.float().cpu().numpy())
        vectors.append(vec_p.float().cpu().numpy())

        labels.append(f"{topic} ({label_m})")
        labels.append(f"{topic} ({label_p})")

        colors_list.append(topic_colors[topic])
        colors_list.append(topic_colors[topic])

    X = np.array(vectors)

    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_
    logger.info(f"Explained Variance: PC1={var[0]:.4f}, PC2={var[1]:.4f}")

    # Plot Trajectories
    plt.figure(figsize=(14, 12))

    for i in range(0, len(X_pca), 2):
        idx_mech = i
        idx_pol = i + 1

        start = X_pca[idx_mech]
        end = X_pca[idx_pol]

        topic_name = labels[idx_mech].split(" (")[0]
        c = topic_colors[topic_name]

        # Draw arrow
        plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                  head_width=0.3, head_length=0.3, fc=c, ec=c, alpha=0.7,
                  length_includes_head=True, width=0.03)

        # Plot points
        plt.scatter(start[0], start[1], color=c, marker='o', s=100, alpha=0.8)
        plt.scatter(end[0], end[1], color=c, marker='^', s=150, alpha=0.8)

        # Add text labels
        plt.text(start[0]-0.5, start[1]+0.3, short_m,
                 fontsize=9, color=c, fontweight='bold')
        plt.text(end[0]-0.5, end[1]+0.3, short_p,
                 fontsize=9, color=c, fontweight='bold')

    plt.title(f"Latent Trajectories of Polarization\\nPC1: {var[0]*100:.2f}% var, PC2: {var[1]*100:.2f}% var",
              fontsize=14, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({var[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var[1]*100:.1f}%)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Custom legend
    legend_elements = [Line2D([0], [0], color=topic_colors[t], lw=2.5, label=t)
                       for t in topic_list]
    plt.legend(handles=legend_elements, loc='best',
               fontsize=10, framealpha=0.9)

    # Save figure
    plot_path = os.path.join(output_dir, "08_pca_trajectory_map.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"PCA trajectory map saved to {plot_path}")
    plt.close()

    # Save trajectory data to CSV
    df_trajectory = pd.DataFrame({
        'topic': [label.split(' (')[0] for label in labels],
        'type': [label_m if f'({label_m})' in label else label_p for label in labels],
        'label': labels,
        'pc1': X_pca[:, 0],
        'pc2': X_pca[:, 1]
    })

    trajectory_path = os.path.join(output_dir, "08_trajectory_data.csv")
    df_trajectory.to_csv(trajectory_path, index=False)
    logger.info(f"Trajectory data saved to {trajectory_path}")

    # Save PCA statistics
    df_pca_stats = pd.DataFrame({
        'component': ['PC1', 'PC2'],
        'explained_variance_ratio': var,
        'explained_variance_percent': var * 100
    })

    stats_path = os.path.join(output_dir, "08_pca_statistics.csv")
    df_pca_stats.to_csv(stats_path, index=False)
    logger.info(f"PCA statistics saved to {stats_path}")

    # 6. PCA Visualization of Difference Vectors (Deltas)
    logger.info("\n--- Running PCA on Difference Vectors ---")

    delta_vectors = []
    delta_labels = []
    delta_colors = []

    for topic in topic_list:
        delta_vectors.append(deltas[topic].float().cpu().numpy())
        delta_labels.append(topic)
        delta_colors.append(topic_colors[topic])

    X_delta = np.array(delta_vectors)

    # Run PCA on Deltas
    pca_delta = PCA(n_components=2)
    X_delta_pca = pca_delta.fit_transform(X_delta)

    var_delta = pca_delta.explained_variance_ratio_
    logger.info(
        f"Delta PCA Explained Variance: PC1={var_delta[0]:.4f}, PC2={var_delta[1]:.4f}")

    # Plot Deltas
    plt.figure(figsize=(12, 10))

    for i, topic in enumerate(delta_labels):
        x, y = X_delta_pca[i]
        c = delta_colors[i]
        plt.scatter(x, y, color=c, s=150, alpha=0.9, label=topic)
        # Add offset to text to avoid overlapping with point
        plt.text(x, y + (max(X_delta_pca[:, 1]) - min(X_delta_pca[:, 1]))*0.02,
                 topic, fontsize=9, alpha=0.8, ha='center')

    plt.title(f"PCA of Polarization Directions (Delta Vectors)\\nPC1: {var_delta[0]*100:.2f}% var, PC2: {var_delta[1]*100:.2f}% var",
              fontsize=14, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({var_delta[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_delta[1]*100:.1f}%)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    # Save figure
    delta_plot_path = os.path.join(output_dir, "08_pca_deltas_map.png")
    plt.tight_layout()
    plt.savefig(delta_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"PCA delta map saved to {delta_plot_path}")
    plt.close()

    # Save delta PCA data
    df_delta_pca = pd.DataFrame({
        'topic': delta_labels,
        'pc1': X_delta_pca[:, 0],
        'pc2': X_delta_pca[:, 1]
    })
    delta_data_path = os.path.join(output_dir, "08_delta_pca_data.csv")
    df_delta_pca.to_csv(delta_data_path, index=False)
    logger.info(f"Delta PCA data saved to {delta_data_path}")


if __name__ == "__main__":
    main()
