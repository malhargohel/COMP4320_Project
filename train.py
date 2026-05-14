import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim
import yaml

from src.environment import UrbanMobilityWrapper
from src.federated import FederatedServer
from src.model import TransformerAgent
from src.utils import MetricsLogger, preprocess_state


def load_config(config_path: str = "configs/params.yaml") -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_tensor(raw_state: Any, state_dim: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
    state = preprocess_state(raw_state)
    if state.ndim > 1:
        state = state.ravel()

    if state.size < state_dim:
        padded = torch.zeros(state_dim, dtype=torch.float32)
        padded[: state.size] = torch.from_numpy(state)
        state = padded
    else:
        state = torch.from_numpy(state[:state_dim]).float()

    if seq_len == 1:
        state = state.unsqueeze(0)
    else:
        state = state[: state_dim * seq_len].view(seq_len, state_dim)

    return state.unsqueeze(0).to(device)


def select_action(agent: TransformerAgent, state_tensor: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
    action_probs, state_value = agent(state_tensor)
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), state_value.squeeze(0)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns: list[float] = []
    R = 0.0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def build_action_dict(agent_ids: list[str], action_idx: int) -> Any:
    if agent_ids:
        return {agent_id: action_idx for agent_id in agent_ids}
    return action_idx


def prepare_local_envs(config: Dict[str, Any], use_env: bool) -> list[Optional[UrbanMobilityWrapper]]:
    num_agents = config.get("federated_params", {}).get("num_agents", 3)
    if not use_env:
        return [None] * num_agents

    net_file = config.get("simulation", {}).get("net_file")
    route_file = config.get("simulation", {}).get("route_file")
    use_gui = config.get("simulation", {}).get("gui", False)

    if not net_file or not route_file:
        raise ValueError("SUMO net_file and route_file must be configured in configs/params.yaml")

    net_path = Path(net_file)
    route_path = Path(route_file)
    if not net_path.exists() or not route_path.exists():
        print("WARNING: SUMO net/route files not found. Falling back to placeholder training.")
        return [None] * num_agents

    envs: list[Optional[UrbanMobilityWrapper]] = []
    for _ in range(num_agents):
        try:
            envs.append(UrbanMobilityWrapper(net_file, route_file, use_gui=use_gui))
        except Exception as error:
            print(f"WARNING: Unable to initialize SUMO environment ({error}). Falling back to placeholder training.")
            return [None] * num_agents

    return envs


def simulate_local_training(agent: TransformerAgent, local_epochs: int, batch_size: int, seq_len: int, state_dim: int, learning_rate: float, device: str = "cpu") -> float:
    agent.to(device)
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    total_reward = 0.0

    for _ in range(local_epochs):
        states = torch.randn(batch_size, seq_len, state_dim, device=device)
        _, state_value = agent(states)
        loss = -state_value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward += -loss.item()

    return total_reward / local_epochs


def train_local_agent(
    agent: TransformerAgent,
    env: Optional[UrbanMobilityWrapper],
    config: Dict[str, Any],
    local_epochs: int,
    device: str = "cpu",
) -> float:
    rl_config = config.get("rl_params", {})
    gamma = float(rl_config.get("gamma", 0.99))
    learning_rate = float(rl_config.get("learning_rate", 3e-4))
    state_dim = int(rl_config.get("state_dim", 10))
    seq_len = int(rl_config.get("seq_len", 1))
    max_steps = int(config.get("simulation", {}).get("max_steps_per_episode", 200))

    if env is None:
        return simulate_local_training(agent, local_epochs, batch_size=1, seq_len=seq_len, state_dim=state_dim, learning_rate=learning_rate, device=device)

    agent.to(device)
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    episode_reward = 0.0

    for episode in range(local_epochs):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        step = 0

        while not done and step < max_steps:
            state_tensor = to_tensor(state, state_dim=state_dim, seq_len=seq_len, device=device)
            action, log_prob, value = select_action(agent, state_tensor)
            actions = build_action_dict(env.agent_ids, action)
            next_state, reward, done, _ = env.step(actions)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
            step += 1

        episode_reward += sum(rewards)
        returns = compute_returns(rewards, gamma).to(device)
        values_tensor = torch.stack(values).squeeze(-1)
        advantages = returns - values_tensor
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()

        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return episode_reward / local_epochs


def train_federated_rl(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rl_config = config.get("rl_params", {})
    federated_config = config.get("federated_params", {})
    state_dim = int(rl_config.get("state_dim", 10))
    action_dim = int(rl_config.get("action_dim", 4))
    num_agents = int(federated_config.get("num_agents", 3))

    local_epochs = args.local_epochs if args.local_epochs is not None else int(federated_config.get("local_epochs", 5))
    envs = prepare_local_envs(config, args.use_env)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    global_model = TransformerAgent(state_dim=state_dim, action_dim=action_dim)
    local_agents = [TransformerAgent(state_dim, action_dim) for _ in range(num_agents)]
    metrics_logger = MetricsLogger()

    print("Starting federated training...")
    for round_idx in range(args.rounds):
        print(f"--- Federated Round {round_idx + 1}/{args.rounds} ---")
        for agent in local_agents:
            agent.load_state_dict(global_model.state_dict())

        local_rewards = []
        for i, agent in enumerate(local_agents):
            reward = train_local_agent(agent, envs[i], config, local_epochs, device=args.device)
            local_rewards.append(reward)
            print(f"  Agent {i} average episode reward={reward:.4f}")

        avg_reward = sum(local_rewards) / len(local_rewards)
        metrics_logger.log(avg_reward, wait_time=0.0)
        global_model = FederatedServer.aggregate_weights(local_agents)
        print(f"  Global model synchronized. avg_reward={avg_reward:.4f}")

    model_path = results_dir / "global_model.pth"
    torch.save(global_model.state_dict(), model_path)

    plot_path = results_dir / "training_results.png"
    history_path = results_dir / "training_history.json"

    metrics_logger.save_plot(str(plot_path))
    metrics_logger.save_history(str(history_path))

    print("\nTraining complete.")
    print(f"Results saved to: {results_dir.resolve()}")
    print(f"  - model: {model_path.name}")
    print(f"  - plot: {plot_path.name}")
    print(f"  - history: {history_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated MARL training for Project NexUS.")
    parser.add_argument("--config", default="configs/params.yaml", help="Path to config file.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=None, help="Local training episodes per local agent.")
    parser.add_argument("--output-dir", default="results", help="Directory for saved outputs.")
    parser.add_argument("--use-env", action="store_true", help="Use the SUMO environment when map files are available.")
    parser.add_argument("--device", default="cpu", help="Device for training: cpu or cuda.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_federated_rl(args)
