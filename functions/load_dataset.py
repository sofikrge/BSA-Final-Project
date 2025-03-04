import pickle

def load_dataset(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    neurons = data["neurons"]
    sacc_start = data.get("sacc drinking session start time", 0)
    non_stimuli_time = (0, sacc_start)
    return data, neurons, non_stimuli_time
