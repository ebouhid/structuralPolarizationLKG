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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from icecream import ic

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Load Model & SAE
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
    logger.info("Loading contrastive topics...")
    topics = cfg.contrastive.topics
    topic_names = list(topics.keys())
    
    logger.info(f"Found {len(topic_names)} topics: {', '.join(topic_names)}")
    
    # 3. Extract Residual Vectors for all topics
    logger.info("\n--- Extracting Residual Vectors ---")
    residual_vectors = {}
    
    for topic_name in topic_names:
        ic(topic_name)
        ic(topics[topic_name])
        view_a_prompt = topics[topic_name]["view_a"]
        view_b_prompt = topics[topic_name]["view_b"]
        
        # Run view_a (left)
        _, cache_a = model.run_with_cache(
            view_a_prompt, names_filter=[hook_point])
        resid_a = cache_a[hook_point][0, -1, :].cpu().numpy()  # Last token
        
        # Run view_b (right)
        _, cache_b = model.run_with_cache(
            view_b_prompt, names_filter=[hook_point])
        resid_b = cache_b[hook_point][0, -1, :].cpu().numpy()  # Last token
        
        residual_vectors[topic_name] = {
            'view_a': resid_a,
            'view_b': resid_b
        }
        
        logger.info(f"[{topic_name}] Extracted residual vectors")
    
    # 4. Create Diff Vectors
    # Right diff: view_b - view_a
    # Left diff: view_a - view_b
    logger.info("\n--- Creating Difference Vectors ---")
    
    diff_vectors = []
    labels = []
    topic_labels = []
    
    for topic_name in topic_names:
        resid_a = residual_vectors[topic_name]['view_a']
        resid_b = residual_vectors[topic_name]['view_b']
        
        # Right diff vector (view_b - view_a)
        right_diff = resid_b - resid_a
        diff_vectors.append(right_diff)
        labels.append(1)  # 1 for right
        topic_labels.append(topic_name)
        
        # Left diff vector (view_a - view_b)
        left_diff = resid_a - resid_b
        diff_vectors.append(left_diff)
        labels.append(0)  # 0 for left
        topic_labels.append(topic_name)
    
    # Convert to numpy arrays
    X = np.array(diff_vectors)
    y = np.array(labels)
    topic_array = np.array(topic_labels)
    
    logger.info(f"Created {len(X)} difference vectors ({np.sum(y == 0)} left, {np.sum(y == 1)} right)")
    
    # 5. Split topics evenly and randomly for train/test
    # We want to split by topics, not by individual samples
    # This ensures the model generalizes to unseen topics
    n_topics = len(topic_names)
    n_train_topics = n_topics // 2
    
    # Random permutation of topics
    shuffled_topics = np.random.permutation(topic_names)
    train_topics = set(shuffled_topics[:n_train_topics])
    test_topics = set(shuffled_topics[n_train_topics:])
    
    logger.info(f"\nTrain topics ({len(train_topics)}): {', '.join(sorted(train_topics))}")
    logger.info(f"Test topics ({len(test_topics)}): {', '.join(sorted(test_topics))}")
    
    # Create train/test split based on topics
    train_mask = np.array([topic in train_topics for topic in topic_array])
    test_mask = ~train_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"\nTrain set: {len(X_train)} samples ({np.sum(y_train == 0)} left, {np.sum(y_train == 1)} right)")
    logger.info(f"Test set: {len(X_test)} samples ({np.sum(y_test == 0)} left, {np.sum(y_test == 1)} right)")
    
    # 6. Standardize features
    logger.info("\n--- Standardizing Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train SVM Classifier
    logger.info("\n--- Training SVM Classifier ---")
    # Try different kernels
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    for kernel in kernels:
        logger.info(f"\nTraining SVM with {kernel} kernel...")
        
        if kernel == 'linear':
            svm = SVC(kernel=kernel, random_state=42)
        elif kernel == 'rbf':
            svm = SVC(kernel=kernel, gamma='scale', random_state=42)
        elif kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, gamma='scale', random_state=42)
        
        # Train
        svm.fit(X_train_scaled, y_train)
        
        # Predict on train and test
        y_train_pred = svm.predict(X_train_scaled)
        y_test_pred = svm.predict(X_test_scaled)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results[kernel] = {
            'model': svm,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'y_test_pred': y_test_pred
        }
        
        logger.info(f"{kernel.upper()} - Train Accuracy: {train_acc:.4f}")
        logger.info(f"{kernel.upper()} - Test Accuracy: {test_acc:.4f}")
    
    # Select best model (by test accuracy)
    best_kernel = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_kernel]['model']
    best_test_acc = results[best_kernel]['test_accuracy']
    y_test_pred = results[best_kernel]['y_test_pred']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_kernel.upper()} kernel")
    logger.info(f"Test Accuracy: {best_test_acc:.4f}")
    logger.info(f"{'='*60}")
    
    # 8. Detailed Classification Report
    logger.info("\n--- Classification Report (Test Set) ---")
    class_names = ['Left', 'Right']
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # 9. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # 10. Save Results
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Save metrics for all kernels
    metrics_data = []
    for kernel, result in results.items():
        metrics_data.append({
            'kernel': kernel,
            'train_accuracy': result['train_accuracy'],
            'test_accuracy': result['test_accuracy']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, "09_svm_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"\nSVM metrics saved to {metrics_path}")
    
    # Save train/test split information
    split_data = []
    for topic in topic_names:
        split_data.append({
            'topic': topic,
            'split': 'train' if topic in train_topics else 'test'
        })
    
    df_split = pd.DataFrame(split_data)
    split_path = os.path.join(output_dir, "09_topic_split.csv")
    df_split.to_csv(split_path, index=False)
    logger.info(f"Topic split information saved to {split_path}")
    
    # Save predictions
    pred_data = []
    for i, topic in enumerate(topic_array[test_mask]):
        pred_data.append({
            'topic': topic,
            'true_label': 'Left' if y_test[i] == 0 else 'Right',
            'predicted_label': 'Left' if y_test_pred[i] == 0 else 'Right',
            'correct': y_test[i] == y_test_pred[i]
        })
    
    df_pred = pd.DataFrame(pred_data)
    pred_path = os.path.join(output_dir, "09_predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")
    
    # 11. Visualize Results
    # Plot 1: Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of accuracies
    kernels_list = list(results.keys())
    train_accs = [results[k]['train_accuracy'] for k in kernels_list]
    test_accs = [results[k]['test_accuracy'] for k in kernels_list]
    
    x = np.arange(len(kernels_list))
    width = 0.35
    
    axes[0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    axes[0].bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('SVM Performance by Kernel')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([k.upper() for k in kernels_list])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title(f'Confusion Matrix ({best_kernel.upper()} kernel)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "09_svm_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Results visualization saved to {plot_path}")
    plt.close()
    
    # Plot 2: Per-topic accuracy breakdown
    topic_accuracy = {}
    for topic in test_topics:
        topic_mask = topic_array[test_mask] == topic
        topic_y_true = y_test[topic_mask]
        topic_y_pred = y_test_pred[topic_mask]
        topic_accuracy[topic] = accuracy_score(topic_y_true, topic_y_pred)
    
    # Sort by accuracy
    sorted_topics = sorted(topic_accuracy.items(), key=lambda x: x[1], reverse=True)
    topics_sorted, accs_sorted = zip(*sorted_topics)
    
    plt.figure(figsize=(12, 6))
    colors = ['green' if acc == 1.0 else 'orange' if acc >= 0.5 else 'red' 
              for acc in accs_sorted]
    plt.barh(range(len(topics_sorted)), accs_sorted, color=colors, alpha=0.7)
    plt.yticks(range(len(topics_sorted)), topics_sorted)
    plt.xlabel('Accuracy')
    plt.title('Per-Topic Classification Accuracy (Test Set)')
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    plt.legend()
    plt.tight_layout()
    
    topic_plot_path = os.path.join(output_dir, "09_topic_accuracy.png")
    plt.savefig(topic_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Topic accuracy plot saved to {topic_plot_path}")
    plt.close()
    
    logger.info("\n=== SVM Classification Complete ===")


if __name__ == "__main__":
    main()
