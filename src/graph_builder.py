import numpy as np
import scipy.sparse as sp
import networkx as nx
import logging
from tqdm import tqdm
from typing import List, Set

logger = logging.getLogger(__name__)


class LKGBuilder:
    def __init__(self, config):
        self.config = config
        self.min_cooccurrence = config['graph'].get('min_cooccurrence', 5)
        self.phi_threshold = config['graph'].get('phi_threshold', 0.05)

    def build_graph(self, feature_lists: List[Set[int]], label: str) -> nx.Graph:
        """
        Converts a list of feature sets into a NetworkX graph using Phi coefficient.
        """
        logger.info(
            f"[{label}] Building Sparse Matrix from {len(feature_lists)} contexts...")

        # 1. Determine Dictionary Size (Max Feature ID)
        # We scan once to find the largest feature ID to shape the matrix
        max_feat_id = 0
        for fset in feature_lists:
            if fset:
                max_feat_id = max(max_feat_id, max(fset))

        dict_size = max_feat_id + 1
        num_samples = len(feature_lists)
        logger.info(
            f"[{label}] Vocabulary Size: {dict_size}, Samples: {num_samples}")

        # 2. Construct Sparse Indicator Matrix X (Rows=Samples, Cols=Features)
        # We use a "Lil" matrix first for efficient construction, then CSR for math
        # Or better: construct coords directly for CSR
        indptr = [0]
        indices = []
        data = []

        for fset in tqdm(feature_lists, desc="Vectorizing"):
            indices.extend(list(fset))
            data.extend([1] * len(fset))
            indptr.append(len(indices))

        X = sp.csr_matrix((data, indices, indptr), shape=(
            num_samples, dict_size), dtype=float)

        # 3. Compute Stats
        # Column sums = frequency of each feature (n_i)
        n_i = np.array(X.sum(axis=0)).flatten()

        # 4. Compute Co-occurrence Matrix (C = X.T @ X)
        logger.info(
            f"[{label}] Computing Co-occurrence Matrix (Sparse MatMul)...")
        # This is the heavy lift:
        C = X.T @ X

        # 5. Calculate Phi (Vectorized on Sparse Data)
        logger.info(f"[{label}] Calculating Phi Coefficients...")

        # We iterate only over non-zero co-occurrences
        # C is Upper Triangular symmetric mostly, but X.T @ X gives full symmetric.
        # We convert to COO to iterate edges.
        C = sp.triu(C, k=1).tocoo()  # Upper triangle only, ignore self-loops

        edges = []

        # Pre-calculate terms for the denominator to speed up loop
        # D = sqrt(n_i * (N - n_i))
        # We add epsilon to avoid div by zero
        with np.errstate(invalid='ignore'):
            D_terms = np.sqrt(n_i * (num_samples - n_i))

        # Iterate over candidate edges (where co-occurrence > 0)
        # Zip is faster than iterating indices in Python
        current_edges = 0

        for i, j, n_ij in zip(C.row, C.col, C.data):
            if n_ij < self.min_cooccurrence:
                continue

            # Phi Formula
            numerator = (num_samples * n_ij) - (n_i[i] * n_i[j])
            denominator = D_terms[i] * D_terms[j]

            if denominator == 0:
                continue

            phi = numerator / denominator

            if phi > self.phi_threshold:
                edges.append((int(i), int(j), float(phi)))
                current_edges += 1

        logger.info(
            f"[{label}] Graph Constructed: {len(edges)} strong edges found.")

        # 6. Build NetworkX Graph
        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        # Add node attributes (Frequency)
        # Only add nodes that actually have edges
        active_nodes = set(G.nodes())
        node_attrs = {n: {"frequency": int(n_i[n])} for n in active_nodes}
        nx.set_node_attributes(G, node_attrs)

        return G
