import pickle

def load_dataset(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    neurons = data["neurons"]
    sacc_start = data.get("sacc drinking session start time", 0)
    non_stimuli_time = (0, sacc_start)
    
    # Explicitly add the computed non_stimuli_time back into data bc it did not work without this
    data["non_stimuli_time"] = non_stimuli_time

    return data, neurons, non_stimuli_time
