class MarioCudaAgent:
    def __init__(self):
        self.agent_parameters = {}
        self.agent_parameters["gpu"] = True

        # How many buttons can the agent 'press'
        self.agent_parameters["n_actions"] = 7

        # As the Neural Network is a Convolutional Neural Network a selected downsample size is needed
        self.agent_parameters["downsample_w"] = 128
        self.agent_parameters["downsample_h"] = 128

        # When Training the agent, how many episodes should it get?
        self.agent_parameters["n_episodes"] = 400

        # How should the agent train? using the GPU?
        self.agent_parameters["gpu"] = 0

        # Does the environment that the agent is running in have raw pixel rgb values as the observation?
        self.agent_parameters["use_rgb_for_raw_state"] = True

        # How many times should the agent repeat a chosen action? This can help train the agent faster in some
        # scenarios, marios jump height is tied to how long the button is held, this can be achieved by NN on its
        # own, however repeating actions can help drop the complexity of understanding this.
        self.agent_parameters["n_repeat"] = 4

        # How many frames should be included in the CNN? Multiple frames are included to help the network understand
        # inertia and momentum, including more may eventually result in a more accurate network, but may slow the
        # training process.
        self.agent_parameters["n_frames"] = 4

        # Hyper Parameters
        self.agent_parameters['input_size'] = 4
        self.agent_parameters['hidden_size'] = 256
        self.agent_parameters['output_size'] = 7
        self.agent_parameters['action_conf'] = 50

    def get_parameter(self, param_id):
        return self.agent_parameters[param_id]

    def set_parameter(self, param_id, value):
        self.agent_parameters[param_id] = value
