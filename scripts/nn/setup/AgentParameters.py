
class MarioCudaAgent:

    def __init__(self, args):
        self.agent_parameters = {}
        self.agent_parameters["gpu"] = True

        # How many buttons can the agent 'press'
        self.agent_parameters["n_actions"] = 7

        # As the Neural Network is a Convolutional Neural Network a selected downsample size is needed
        self.agent_parameters["downsample_w"] = 84
        self.agent_parameters["downsample_h"] = 84

        # When Training the agent, how many episodes should it get?
        self.agent_parameters["n_episodes"] = 800

        # How many times should the agent repeat a chosen action? This can help train the agent faster in some
        # scenarios, marios jump height is tied to how long the button is held, this can be achieved by NN on its
        # own, however repeating actions can help drop the complexity of understanding this.
        self.agent_parameters["n_repeat"] = 4

        # How many frames should be included in the CNN? Multiple frames are included to help the network understand
        # inertia and momentum, including more may eventually result in a more accurate network, but may slow the
        # training process.
        self.agent_parameters["n_frames"] = 4

    def get_parameters(self):
        return self.agent_parameters
