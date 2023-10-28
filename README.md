# mt-genetic-reinforcement
My Minor Thesis project investigating the viability of merging attributes of reinforcement learning and genetic algorithms to combat poor generalization in reward sparse enviroments.

This branch has code that was used to help generate results for a Mixed Population Experiement with a single DQN agent in a genetic algorithms population while the GA Agents provide experience to a population memory buffer.

This buffer has replaced the internal buffer of the DQN, and can be found referenced in the  ```scripts/ga/util/Holder.py``` file.

## Setup
Python Version 3.7 <br>
Import dependencies in ```scripts/requirements.txt```

## Usage
Test a model using ```scripts/environment/MarioEnvironment.py```<br>
Train a new model using ```scripts/ga/TrainMarioGA.py```

When training, do not use odd population numbers and the minimum population must be larger than '5'<br>
Due to the method used to select parents and ensure genetic performance isn't lost this restriction is in place<br>
See ```scripts/ga/util/MarioGAUtil.py``` for details


### Configuration
Settings for Population are in ```scripts/ga/TrainMarioGA.py```<br>
Settings for Agents are in ```scripts/nn/setup/AgentParameters.py```

#### Notes
- When using Multithreading, you must disable the human render mode and gpu usage as it will freeze otherwise.
- The multithreading functionality is inconsistent, I've had successful runs with it, however access violations are
somewhat common and have caused crashes, I'm not sure of the exact cause at the moment.

## Credits
The following projects were used to help me develop this codebase:

* https://github.com/robertjankowski/ga-openai-gym 
* https://pypi.org/project/gym-super-mario-bros/

