import numpy as np
def get_spike_times(data):
    neurons_data = data.get("neurons", [])
    spike_times_list = [np.array(neuron[2]) for neuron in neurons_data if len(neuron) > 2]
    return spike_times_list