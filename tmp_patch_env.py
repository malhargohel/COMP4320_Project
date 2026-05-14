from pathlib import Path
path = Path('src/environment.py').resolve()
if not path.exists():
    raise SystemExit(f'path not found: {path}')
text = path.read_text(encoding='utf-8')
start = text.index('    def step(self, actions: Any) -> tuple[Any, dict, dict, dict]:')
end = text.index('        return obs, custom_reward, done, info', start) + len('        return obs, custom_reward, done, info')
old_block = text[start:end]
new_block = '''    def step(self, actions: Any) -> tuple[Any, dict, dict, dict]:
        obs, rewards, dones, info = self.env.step(actions)
        self.agent_ids = self._extract_agent_ids(obs)
        custom_rewards = {}
        if info is not None and isinstance(info, dict) and self.agent_ids:
            custom_rewards = self.compute_custom_rewards(info, self.agent_ids)
        elif isinstance(rewards, dict):
            custom_rewards = rewards
        else:
            custom_rewards = {ts: float(rewards) for ts in self.agent_ids}

        return obs, custom_rewards, dones, info'''
with open(path, 'w', encoding='utf-8') as f:
    f.write(text[:start] + new_block + text[end:])
print('patched')
