import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TopologyAnalyzer:
    def __init__(self):
        pass

    def analyze_graph(self, G: nx.Graph, label: str):
        """
        Calculates topological metrics for a given graph.
        """
        logger.info(f"--- Analyzing {label} Topology ---")
        
        # 1. Basic Stats
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_edges == 0:
            logger.warning(f"[{label}] Graph is empty! Try lowering phi_threshold in graph_builder.py")
            return {
                "nodes": num_nodes, "edges": 0, "density": 0, 
                "modularity": 0, "communities": 0
            }

        # 2. Density
        # The ratio of actual edges to possible edges
        density = nx.density(G)
        
        # 3. Community Detection (Louvain)
        # We use the edge weights (Phi coefficients) to drive clustering
        logger.info(f"[{label}] Running Louvain Community Detection...")
        communities = louvain_communities(G, weight='weight', seed=42)
        num_communities = len(communities)
        
        # 4. Modularity (Q Score)
        # Measures the strength of division between communities
        # Range: [-0.5, 1.0]. Higher > 0.4 implies strong echo chambers.
        Q = modularity(G, communities, weight='weight')
        
        logger.info(f"[{label}] Results: Q={Q:.4f}, Communities={num_communities}")
        
        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "density": density,
            "modularity": Q,
            "num_communities": num_communities
        }

    def print_comparison(self, neutral_stats, political_stats):
        print("\n" + "="*40)
        print("   FINAL STRUCTURAL AUDIT REPORT")
        print("="*40)
        print(f"{'Metric':<20} | {'Neutral':<12} | {'Political':<12}")
        print("-" * 50)
        
        metrics = ["nodes", "edges", "density", "modularity", "num_communities"]
        
        for m in metrics:
            n_val = neutral_stats[m]
            p_val = political_stats[m]
            
            # Format floats
            if isinstance(n_val, float):
                n_str = f"{n_val:.5f}"
                p_str = f"{p_val:.5f}"
            else:
                n_str = str(n_val)
                p_str = str(p_val)
                
            print(f"{m.capitalize():<20} | {n_str:<12} | {p_str:<12}")
            
        print("="*40)
        
        # The Hypothesis Test
        delta_q = political_stats['modularity'] - neutral_stats['modularity']
        print(f"\nPolarization Delta (ΔQ): {delta_q:.4f}")
        
        if delta_q > 0.05:
             print(">> CONCLUSION: The model exhibits STRUCTURAL POLARIZATION.")
             print("   The political concepts form tighter, more segregated echo chambers")
             print("   than the neutral control concepts.")
        elif delta_q < -0.05:
             print(">> CONCLUSION: The model is UNEXPECTEDLY INTEGRATED.")
             print("   Political concepts are more interconnected than mechanical ones.")
        else:
             print(">> CONCLUSION: No significant structural difference observed.")
