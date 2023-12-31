class MarioCudaAgent:
    def __init__(self):
        self.agent_parameters = {}

        # How many buttons can the agent 'press'
        self.agent_parameters["n_actions"] = 7

        # As the Neural Network is a Convolutional Neural Network a selected downsample size is needed
        # Changing this may require Tweaking the value of the layers in CNN.py
        self.agent_parameters["downsample_w"] = 84
        self.agent_parameters["downsample_h"] = 84

        # When Training the agent, how many episodes should it get?
        # 750 should be enough time to pass a level and perhaps a bit of the next
        self.agent_parameters["n_episodes"] = 800

        # How should the agent train? using the GPU?
        self.agent_parameters["gpu"] = 0

        # Does the environment that the agent is running in have raw pixel rgb values as the observation?
        self.agent_parameters["use_rgb_for_raw_state"] = True

        # How many times should the agent repeat a chosen action? This can help train the agent faster in some
        # scenarios, marios jump height is tied to how long the button is held, this can be achieved by NN on its
        # own, however repeating actions can help drop the complexity of understanding this.
        self.agent_parameters["n_repeat"] = 8

        # How many frames should be included in the CNN? Multiple frames are included to help the network understand
        # inertia and momentum, including more may eventually result in a more accurate network, but may slow the
        # training process.
        # Tweaking this Value may require you to change 'input_size' as input_size is the number of channels taken in.
        # and we use the in channels to accept multiple frames
        self.agent_parameters["n_frames"] = 4

        # Hyper Parameters
        self.agent_parameters['input_size'] = 4
        self.agent_parameters['hidden_size'] = 32
        self.agent_parameters['output_size'] = 7
        self.agent_parameters['action_conf'] = 100

        # Reinforcement Parameters
        self.agent_parameters['learn_start'] = 50
        self.agent_parameters['experience_episodes'] = 10
        self.agent_parameters['memory_size'] = 1000000
        self.agent_parameters['batch_size'] = 32
        self.agent_parameters['ep_end'] = 0.05
        self.agent_parameters['ep_start'] = 0.9
        self.agent_parameters['ep_decay'] = 1000
        self.agent_parameters['gamma'] = 0.99
        self.agent_parameters['tau'] = 0.005
        self.agent_parameters['target_update_freq'] = 100

    def get_parameter(self, param_id):
        return self.agent_parameters[param_id]

    def set_parameter(self, param_id, value):
        self.agent_parameters[param_id] = value
