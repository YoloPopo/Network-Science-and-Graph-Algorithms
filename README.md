# Network-Science-and-Graph-Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository contains a comprehensive collection of Jupyter notebooks designed as a hands-on course in Network Science and Graph Machine Learning. The material guides users from the foundational principles of graph theory to advanced, state-of-the-art techniques in graph embeddings and Graph Neural Networks (GNNs).

Each notebook is structured as a practical assignment, complete with theoretical explanations, code skeletons, and assertion tests to validate the implementations.

## Key Features

*   **Comprehensive Coverage:** Spans the full spectrum of graph analytics, from classic network properties and generative models to modern deep learning on graphs.
*   **Hands-On Learning:** Filled with coding exercises (`# YOUR CODE HERE`) that challenge the user to implement algorithms from scratch, reinforcing theoretical understanding.
*   **Practical Applications:** Explores real-world problems and datasets, including social networks, epidemic modeling, knowledge graphs, and link prediction.
*   **Modern Techniques:** Includes dedicated notebooks on Graph Embeddings (DeepWalk, Node2Vec), Knowledge Graph Embeddings (TransE), and a full suite of Graph Neural Networks (GCN, GAE, GAT, GraphSAGE).
*   **Standard Tooling:** Utilizes industry-standard libraries such as `NetworkX`, `scikit-learn`, `PyTorch`, and the Deep Graph Library (`DGL`).

## Structure of the Notebooks

The 15 notebooks are thematically organized into four distinct parts, creating a logical learning progression.

### Part 1: Foundations of Network Science

These notebooks introduce the fundamental concepts of graph theory and analysis.

1.  **Introduction to Network Science (`01-Introduction-to-network-science.ipynb`)**
    *   Reading graphs from files (adjlists, edgelists).
    *   Basic graph manipulation with `NetworkX`.
    *   Graph visualization, layouts, and customizing node/edge appearance.
2.  **Power Law & Degree Distributions (`02-Power-Law.ipynb`)**
    *   Understanding degree distributions and CDFs.
    *   Generating and fitting Power Law distributions using PDF, CDF, and PPF.
    *   Parameter estimation with Maximum Likelihood Estimation (MLE) and Kolmogorov-Smirnov tests.
3.  **Random Graphs (`03-Random-graphs.ipynb`)**
    *   Implementing the Erdős-Rényi (ER) model.
    *   Analyzing the emergence of the giant component and phase transitions.
    *   Comparing degree distributions of ER graphs with real-world networks.
4.  **Generative Network Models (`04-Generative-network-models.ipynb`)**
    *   Implementing the Watts-Strogatz (small-world) and Barabási-Albert (preferential attachment) models.
    *   Analyzing their structural properties, such as average path length and degree distribution.

### Part 2: Structural Analysis & Community Detection

This section dives into methods for understanding node importance and discovering community structures.

5.  **Node Centrality Measures (`05-Node-centrality-measures.ipynb`)**
    *   Implementing and analyzing Degree, Closeness, and Betweenness centrality.
    *   Calculating Katz and Eigenvector centrality.
    *   Analyzing centrality correlation on a Moscow Metro dataset.
6.  **Structural Properties (`06-Structural-properties.ipynb`)**
    *   Implementing the PageRank and HITS algorithms.
    *   Analyzing personalized PageRank for recommendations.
    *   Exploring node similarity metrics (Pearson, Jaccard, Cosine) and matrix reordering with Cuthill-McKee.
7.  **Graph Partitioning (`07-Graph-partitioning.ipynb`)**
    *   Understanding the Graph Laplacian and its eigenvalues.
    *   Implementing Spectral Partitioning and Laplacian Eigenmaps with k-means.
    *   Using k-core decomposition and clique-finding for community analysis.
8.  **Network Communities (`08-Network-communities.ipynb`)**
    *   Implementing the Girvan-Newman algorithm using edge betweenness.
    *   Calculating and optimizing Modularity.
    *   Implementing Label Propagation (synchronous and asynchronous) and the Ego-Splitting framework.

### Part 3: Dynamic Processes & Applications

These notebooks focus on applying graph-based models to solve real-world problems.

9.  **Epidemic Models (`09-Epidemic-models.ipynb`)**
    *   Modeling SI, SIR, and SIS dynamics using differential equations and network-based simulations.
    *   Fitting model parameters to data.
    *   Simulating immunization and self-isolation strategies on networks.
10. **Node Classification (`10-Node-classification.ipynb`)**
    *   Analyzing assortativity in networks.
    *   Implementing classic label propagation algorithms: Relational Neighbor, Label Propagation, and MultiRankWalk.
    *   Applying Ridge Regression on graphs for node-level regression tasks.
11. **Cascades and Influence Maximization (`11-Cascades-and-influence-maximization.ipynb`)**
    *   Implementing the Linear Threshold and Independent Cascade models.
    *   Solving the influence maximization problem with a greedy algorithm.
12. **Link Prediction (`12-Link-prediction.ipynb`)**
    *   Creating time-split datasets and performing negative sampling.
    *   Using similarity-based predictors (Jaccard, Adamic-Adar).
    *   Implementing link prediction using node embeddings and evaluating with ROC AUC and HR@k.

### Part 4: Graph Representation Learning & GNNs

The final part covers modern techniques for learning low-dimensional representations of graphs.

13. **Graph Embeddings (`13-Graph-Embeddings.ipynb`)**
    *   Implementing DeepWalk with Skip-Gram negative sampling and Hierarchical Softmax.
    *   Implementing Node2Vec with biased random walks.
    *   Implementing GraRep for multi-scale relationship learning.
14. **Graph Neural Networks (`14-Graph-Neural-Networks.ipynb`)**
    *   Building a Graph Convolutional Network (GCN) from scratch using both matrix multiplication and message passing paradigms.
    *   Implementing a Graph Autoencoder (GAE) for unsupervised embedding.
    *   Implementing a Graph Attention Network (GAT) and GraphSAGE with neighbor sampling using DGL.
15. **Knowledge Graph Embedding (`15-Knowledge-graph-embedding.ipynb`)**
    *   Working with the Freebase KG dataset.
    *   Implementing the TransE model for knowledge graph embedding.
    *   Evaluating embeddings on a link prediction task using metrics like Hit@k and MRR.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Jupyter Notebook or Jupyter Lab

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    A `requirements.txt` file is provided for easy setup.
    ```bash
    pip install -r requirements.txt
    ```
    **Note on DGL:** The Deep Graph Library (`dgl`) often requires a specific version matching your PyTorch and CUDA installation. The command included in the `requirements.txt` is a general one. If you encounter issues, please install it manually by following the official instructions at the [DGL website](https://www.dgl.ai/pages/start.html). For example, for PyTorch 2.x and no CUDA:
    ```bash
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
    ```

4.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```

## Contributing

Contributions are welcome! If you find any issues, have suggestions for improvements, or want to add new material, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
