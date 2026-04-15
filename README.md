# CoTDA: Freshness-Aware V2X Coordination

**CoTDA** (Cross-Layer Coordination for Fresh and Reliable Data Acquisition) is a framework designed to handle the end-to-end data pipeline in V2X systems where multiple vehicles compete for bandwidth. It coordinates four key functions: admission control, adaptive compression, multi-hop routing, and reliability feedback.

The framework optimizes the **Age of Information (AoI)** across the network by making intelligent trade-offs between data freshness, acquisition value, and delivery reliability.

## Table of Contents
- [Description](#description)
- [Methodology](#methodology)
- [Dataset Information](#dataset-information)
- [Code Information](#code-information)
- [Usage Instructions](#usage-instructions)
- [Requirements](#requirements)
- [Citations](#citations)
- [License](#license)

## Description
This repository contains the implementation of the CoTDA framework for smart transportation systems. It addresses the challenges of data staleness (high AoI) and link unreliability in V2V/V2I communication environments. By integrating a four-stage pipeline, it ensures that only the most urgent and informative data is admitted, compressed based on its value, and routed through reliable links.

## Methodology
The CoTDA pipeline executes in four sequential stages per time slot:
1.  **Stage (A) Urgency-Aware Acquisition**: Filters sensing agents based on AoI-driven gating scores to prioritize stale and novel information.
2.  **Stage (B) Adaptive Compression**: Selects discrete compression levels through a learned projection to balance reconstruction fidelity against bandwidth.
3.  **Stage (C) Reliability-Weighted Routing**: Assigns multi-hop paths using Lagrangian decomposition with dual-variable warm-starts.
4.  **Stage (D) Delivery-Reliability Feedback**: Closes the loop by updating per-agent reliability scores and global context based on delivery outcomes.

## Dataset Information
The experiments utilize two main data sources:
-   **SUMO (Simulation of Urban MObility)**: Used to generate synthetic traffic traces for Urban-Small, Urban-Large, and Highway scenarios.
-   **pNEUMA Dataset**: Drone-captured vehicle trajectories from Athens, Greece. The raw data must be obtained from [EPFL Open Traffic](https://open-traffic.epfl.ch/) under a Creative Commons license. Replay scripts for the Location 1 (Athens) scenario are provided in `src/data/`.

## Code Information
-   `src/models/`: Core implementation of the gating critic, bottleneck projector, and Lagrangian router.
-   `src/training/`: PPO algorithm implementation and policy update logic.
-   `src/data/`: V2X simulator and trajectory replayer.
-   `configs/`: YAML files containing scenario-specific hyperparameters.
-   `figures/`: Scripts and output files for all experimental charts.

## Usage Instructions

### Setup
1.  Clone the repository and create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Experiments
-   **Training**: Train the PPO policy for a specific scenario:
    ```bash
    python scripts/train.py --config configs/experiment_main.yaml
    ```
-   **Evaluation**: Evaluate a trained checkpoint:
    ```bash
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --scenario urban_large
    ```
-   **Visualization**: Regenerate all manuscript figures:
    ```bash
    python figures/scripts/generate_all_figures.py
    ```

## Requirements
-   Python 3.9+
-   PyTorch 1.12+
-   NumPy, Matplotlib, NetworkX, PyYAML
-   SUMO 1.18+ (for synthetic trace generation)

## Citations
If you use this code or dataset in your research, please cite:
```bibtex
@article{CoTDA2026,
  title={Cross-Layer Coordination for Fresh and Reliable Data Acquisition in Smart Transportation Systems},
  author={Xia, S. and Zhang, B. and Tai, J.},
  journal={Academic Manuscript},
  year={2026}
}
```

## License
This project is released under the **MIT License**. SUMO is used under the EPL v2 license.
