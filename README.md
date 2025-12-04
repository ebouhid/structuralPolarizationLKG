# Structural Polarization in LLMs via Latent Knowledge Graphs (LKG)

A framework for detecting and measuring **structural polarization** in Large Language Models by analyzing the topology of their internal feature co-activation networks.

## Overview

This project investigates whether LLMs exhibit measurable structural differences in how they represent contentious (political) vs. neutral (mechanical) concepts at the level of internal feature representations. We use **Sparse Autoencoders (SAEs)** to extract interpretable features from an LLM's residual stream, build **Latent Knowledge Graphs (LKGs)** from feature co-activations, and apply network science metrics to compare topological properties.

### Key Hypothesis

> *If a language model has absorbed polarized conceptual structures, its internal feature co-activation graph for political topics will exhibit higher modularity (more "echo chamber" structure) than an equivalent graph built from neutral topics.*

### Methodology

1. **Feature Extraction**: Run text through Llama-3.1-8B-Instruct and capture feature activations via a pre-trained SAE at layer 19
2. **Graph Construction**: Build co-occurrence graphs where nodes are SAE features and edges are weighted by the Phi coefficient (correlation)
3. **Topology Analysis**: Compare modularity (Q-score), community structure, and other network metrics between political and neutral domains

## 🔬 Datasets

| Corpus | Description | Purpose |
|--------|-------------|---------|
| **Congressional Speeches** | U.S. political discourse dataset | Contentious domain (gun control, abortion, climate, immigration) |
| **Elevator Wiki** | Technical Wikipedia articles about elevators/lifts | Neutral control domain (mechanical, non-partisan) |

## 🏗️ Project Structure

```
structuralPolarizationLKG/
├── config/
│   └── experiment.yaml           # Experiment configuration
├── data/
│   ├── congressional_speeches/   # Political corpus (parquet files)
│   ├── elevator_wiki/            # Neutral corpus
│   └── sae/                      # Pre-trained SAE weights (Goodfire layer 19)
├── experiments/
│   ├── 01_extract_features.py    # Run LLM+SAE to extract feature activations
│   ├── 02_build_graphs.py        # Construct LKGs from feature co-occurrences
│   ├── 03_analyze_graphs.py      # Compute topological metrics
│   ├── 04_inspect_communities.py # Interpret community semantics
│   ├── 05_probe_topic.py         # Contextual probing of specific concepts
│   └── 06_visualize_ego.py       # Visualize ego-graphs around concepts
├── src/
│   ├── data_loader.py            # Corpus loading and balancing
│   ├── graph_builder.py          # Phi-coefficient graph construction
│   ├── model_utils.py            # LLM + SAE wrapper
│   └── topology_metrics.py       # Network analysis (modularity, communities)
├── results/
│   ├── activation_caches/        # Cached feature activations
│   └── graphs/                   # Generated .gexf graph files
├── notebooks/                    # Jupyter exploration notebooks
└── utils/                        # Helper scripts
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Llama-3.1-8B)
- ~50GB disk space for model weights and cached activations

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/structuralPolarizationLKG.git
cd structuralPolarizationLKG

# Create virtual environment
conda create -n structuralPolarizationLKG python=3.13
conda activate structuralPolarizationLKG

# Install dependencies
pip install -r requirements.txt

### Configuration

Edit `config/experiment.yaml` to adjust:

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  device: "cuda"
  target_layer: 19          # Layer to hook for SAE

sae:
  release: "data/sae/goodfire_l19"
  id: "blocks.19.hook_resid_post"

data:
  topics: ["gun control", "abortion", "climate change", "immigration"]
  max_tokens: 2000000       # Token budget for each corpus

graph:
  phi_threshold: 0.05       # Minimum Phi coefficient for edge creation
  min_cooccurrence: 5       # Minimum co-occurrence count
```

### Running the Pipeline

Execute the experiments in order:

```bash
# Step 1: Extract SAE features from both corpora
python experiments/01_extract_features.py

# Step 2: Build Latent Knowledge Graphs
python experiments/02_build_graphs.py

# Step 3: Analyze graph topology and compare
python experiments/03_analyze_graphs.py

# Step 4: Inspect community semantics
python experiments/04_inspect_communities.py

# Step 5: Probe specific topics
python experiments/05_probe_topic.py

# Step 6: Visualize ego-graphs
python experiments/06_visualize_ego.py
```

## Key Metrics

### Modularity (Q-Score)
- Measures the strength of division into communities
- Range: [-0.5, 1.0]
- **Q > 0.4**: Strong community structure ("echo chambers")
- **ΔQ = Q_political - Q_neutral**: Polarization delta

### Phi Coefficient
- Measures feature co-activation correlation
- Used as edge weights in the LKG
- Formula: $\phi = \frac{N \cdot n_{ij} - n_i \cdot n_j}{\sqrt{n_i(N-n_i) \cdot n_j(N-n_j)}}$

### Community Detection
- Uses Louvain algorithm with edge weights
- Identifies clusters of semantically related features

## Example Output

```
==========================================
   FINAL STRUCTURAL AUDIT REPORT
==========================================
Metric               | Neutral      | Political
--------------------------------------------------
Nodes                | 8234         | 7891
Edges                | 45623        | 38291
Density              | 0.00134      | 0.00123
Modularity           | 0.52341      | 0.67892
Num_communities      | 45           | 72
==========================================

Polarization Delta (ΔQ): 0.1555
>> CONCLUSION: The model exhibits STRUCTURAL POLARIZATION.
   The political concepts form tighter, more segregated echo chambers
   than the neutral control concepts.
```

## Core Components

### LKGModelWrapper (`src/model_utils.py`)
Wraps TransformerLens + SAE-Lens to extract sparse feature activations:
- Loads Llama-3.1-8B in FP16
- Hooks into residual stream at specified layer
- Applies SAE encoding with configurable activation threshold
- Handles padding tokens correctly

### LKGBuilder (`src/graph_builder.py`)
Constructs graphs from feature co-occurrences:
- Uses sparse matrix operations for efficiency
- Computes Phi coefficients vectorized
- Applies thresholds for edge filtering
- Outputs NetworkX graphs in GEXF format

### TopologyAnalyzer (`src/topology_metrics.py`)
Computes network science metrics:
- Louvain community detection
- Modularity scoring
- Comparative analysis between graphs

## Dependencies

| Package | Purpose |
|---------|---------|
| `transformer_lens` | Hook into LLM residual stream |
| `sae_lens` | Load pre-trained Sparse Autoencoders |
| `networkx` | Graph data structures and algorithms |
| `cdlib` / `python-louvain` | Community detection |
| `scipy` | Sparse matrix operations |
| `torch` | Deep learning backend |
| `datasets` | HuggingFace data loading |
