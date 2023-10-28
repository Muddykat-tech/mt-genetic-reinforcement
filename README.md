# mt-genetic-reinforcement
My Minor Thesis project investigating the viability of merging attributes of reinforcement learning and genetic algorithms to combat poor generalization in reward sparse enviroments.

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

## Credits
The following projects were used to help me develop this codebase:

* https://github.com/robertjankowski/ga-openai-gym 
* https://pypi.org/project/gym-super-mario-bros/

