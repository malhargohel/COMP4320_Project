import numpy as np
import matplotlib.pyplot as plt
import json

class MetricsLogger:
    """Helper to track rewards and wait times for the USYD project report."""
    def __init__(self):
        self.history = {"global_reward": [], "avg_wait_time": []}

    def log(self, reward, wait_time):
        self.history["global_reward"].append(reward)
        self.history["avg_wait_time"].append(wait_time)

    def save_plot(self, filename="training_results.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["global_reward"], label="Cumulative Reward")
        plt.title("F-MARL Training Progress - Project NexUS")
        plt.xlabel("Federated Rounds")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)

    def save_history(self, filename="training_history.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


def preprocess_state(raw_state):
    """
    Converts raw SUMO observations into normalized NumPy arrays ready for Transformer input.
    """
    if isinstance(raw_state, dict):
        values = []
        for item in raw_state.values():
            if isinstance(item, dict):
                values.extend(np.asarray(list(item.values()), dtype=np.float32).ravel())
            else:
                values.extend(np.asarray(item, dtype=np.float32).ravel())
        normalized_state = np.asarray(values, dtype=np.float32)
    else:
        normalized_state = np.asarray(raw_state, dtype=np.float32).ravel()

    if normalized_state.size == 0:
        normalized_state = np.zeros(1, dtype=np.float32)

    normalized_state = normalized_state / 50.0
    return normalized_state.astype(np.float32)
