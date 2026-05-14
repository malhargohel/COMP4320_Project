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

    def compute_custom_reward(self, info: dict) -> float:
        reward = 0.0
        if info is not None and isinstance(info, dict):
            waiting_time = info.get("system_total_waiting_time", 0)
            stopped_vehicles = info.get("system_total_stopped", 0)
            reward = -(waiting_time + stopped_vehicles)
        return float(reward)

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

    def step(self, actions: Any) -> tuple[Any, float, bool, dict]:
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.agent_ids = self._extract_agent_ids(obs)
        custom_reward = self.compute_custom_reward(info)
        done = bool(terminated or truncated)
        return obs, custom_reward, done, info