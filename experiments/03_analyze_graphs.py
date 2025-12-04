from src.topology_metrics import TopologyAnalyzer
import hydra
from omegaconf import DictConfig
import sys
import os
import networkx as nx
import logging

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()

    graph_dir = os.path.join(orig_cwd, "results/graphs")
    neutral_path = os.path.join(graph_dir, "neutral_lkg.gexf")
    political_path = os.path.join(graph_dir, "political_lkg.gexf")

    if not os.path.exists(neutral_path) or not os.path.exists(political_path):
        print("Error: Graph files not found. Run 02_build_graphs.py first.")
        return

    analyzer = TopologyAnalyzer()

    # 1. Load Neutral
    print(f"Loading Neutral Graph from {neutral_path}...")
    G_neutral = nx.read_gexf(neutral_path)
    # Convert string weights back to float (GEXF sometimes stores as string)
    for u, v, d in G_neutral.edges(data=True):
        if 'weight' in d:
            d['weight'] = float(d['weight'])

    neutral_stats = analyzer.analyze_graph(G_neutral, "Neutral")

    # 2. Load Political
    print(f"Loading Political Graph from {political_path}...")
    G_political = nx.read_gexf(political_path)
    for u, v, d in G_political.edges(data=True):
        if 'weight' in d:
            d['weight'] = float(d['weight'])

    political_stats = analyzer.analyze_graph(G_political, "Political")

    # 3. Verdict
    analyzer.print_comparison(neutral_stats, political_stats)


if __name__ == "__main__":
    main()
