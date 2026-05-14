import gymnasium as gym
from typing import Any

class UrbanMobilityWrapper:
    """
    A wrapper for the SUMO environment to handle multi-agent interactions.
    Calculates the reward based on pressure and wait time.
    """
    def __init__(self, net_file: str, route_file: str, use_gui: bool = False):
        try:
            from sumo_rl import SumoEnvironment
        except ImportError as error:
            raise ImportError(
                "SUMO environment is not available. Make sure SUMO_HOME is set and sumo_rl is installed."
            ) from error

        self.env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=3600,
            delta_time=5,
            single_agent=False,
        )
        self.agent_ids: list[str] = []

    def compute_custom_rewards(self, info: dict, ts_ids: list[str]) -> dict:
        rewards = {}
        for ts in ts_ids:
            stopped = info.get(f"{ts}_stopped", 0)
            accumulated_waiting_time = info.get(f"{ts}_accumulated_waiting_time", 0)
            rewards[ts] = -(stopped + accumulated_waiting_time)
        return rewards

    def _extract_agent_ids(self, obs: Any) -> list[str]:
        if isinstance(obs, dict):
            return [str(key) for key in obs.keys()]
        return []

    def reset(self) -> tuple[Any, dict]:
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output
        else:
            obs, info = reset_output, {}
        self.agent_ids = self._extract_agent_ids(obs)
        return obs, info

    def step(self, actions: Any) -> tuple[Any, dict, dict, dict]:
        obs, rewards, dones, info = self.env.step(actions)
        self.agent_ids = self._extract_agent_ids(obs)
        custom_rewards = {}
        if info is not None and isinstance(info, dict) and self.agent_ids:
            custom_rewards = self.compute_custom_rewards(info, self.agent_ids)
        elif isinstance(rewards, dict):
            custom_rewards = rewards
        else:
            custom_rewards = {ts: float(rewards) for ts in self.agent_ids}

        return obs, custom_rewards, dones, info 