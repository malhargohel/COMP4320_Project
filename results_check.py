import json
with open('results/training_history.json') as f:
    history = json.load(f)

rounds = len(history['global_reward'])
initial = history['global_reward'][0]
final = history['global_reward'][-1]
improvement = final - initial

print('Rounds completed:', rounds)
print('Initial reward:  ', round(initial, 4))
print('Final reward:    ', round(final, 4))
print('Improvement:     ', round(improvement, 4))
