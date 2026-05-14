# Project NexUS: Federated Multi-Agent Reinforcement Learning for Traffic Signal Control

**NexUS** is a distributed AI framework designed to optimize urban mobility. By combining **Federated Learning** with **Multi-Agent Reinforcement Learning (MARL)**, the system enables traffic signals at different intersections to learn optimal timing strategies collaboratively without sharing raw traffic data. This project was developed as part of **COMP4320: Machine Learning III**.

---

## 🚀 Overview
The core objective of NexUS is to reduce congestion and wait times in complex urban environments, utilizing models representative of the **Sydney CBD**.

### Key Features
* **Federated Learning (FedAvg)**: Implements decentralized training where local agents compute weight updates and a central server aggregates them using Federated Averaging to form a robust global model.
* **Transformer-Based Agents**: Utilizes a `TransformerAgent` architecture to process environmental states and determine optimal traffic signal phases.
* **SUMO Integration**: Direct interaction with the **Simulation of Urban MObility (SUMO)** suite for high-fidelity traffic simulation.
* **Custom Reward Function**: Agents are incentivized to minimize both the number of stopped vehicles and the accumulated waiting time at intersections.

---

## 🛠️ System Architecture

1.  **Local Environment**: Individual agents control specific traffic lights within a SUMO-simulated environment, such as the Sydney CBD map provided in the project.
2.  **Local Training**: Agents undergo local training episodes to optimize their policy based on local traffic patterns before communicating with the server.
3.  **Aggregation**: The `FederatedServer` collects local model weights and applies **FedAvg** to update the global model.
4.  **Synchronization**: The improved global weights are distributed back to all local agents for the next round of training.

---

## 💻 Getting Started

### Prerequisites
* Python 3.10+
* SUMO (Simulation of Urban MObility) installed and the `SUMO_HOME` environment variable set.

### Installation
1.  **Open PowerShell** in the project root directory.
2.  **Activate the virtual environment**:
    ```powershell
    .\test.venv\Scripts\Activate.ps1
    ```
3.  **Install dependencies**:
    ```powershell
    python -m pip install -r requirements.txt
    ```

---

## 🏃 Running the Simulation

### Full Training with SUMO
To run a real-world simulation using the configured SUMO network and route files:
```powershell
python train.py --use-env