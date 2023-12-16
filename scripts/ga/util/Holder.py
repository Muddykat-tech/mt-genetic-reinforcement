from ga.util.ReplayMemory import ReplayMemory
from nn.setup import AgentParameters

memory_buffer_history = []
replay_memory = ReplayMemory(AgentParameters.MarioCudaAgent().agent_parameters['memory_size'])

fitness_memory = []
fitness_memory_ticks = []