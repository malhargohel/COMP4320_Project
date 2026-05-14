import inspect
import os
from sumo_rl import SumoEnvironment

print('SUMO_HOME', os.environ.get('SUMO_HOME'))
print('SumoEnvironment.__init__', inspect.signature(SumoEnvironment.__init__))
print('SumoEnvironment.step', inspect.signature(SumoEnvironment.step))
print('SumoEnvironment.reset', inspect.signature(SumoEnvironment.reset))
