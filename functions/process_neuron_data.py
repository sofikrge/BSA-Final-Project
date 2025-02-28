import os
import pickle
import numpy as np
import pandas as pd

def process_neuron_data():
    """
    Processes all .pkl files in the 'data/raw' directory.
    
    Returns:
        structured_data (dict): Dictionary of DataFrames containing neuron metadata.
        spiking_data (dict): Dictionary of neuron spike times per file.
    """
    # Define path to 'data/raw' folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

    # List all .pkl files in the folder
    pkl_files = [f for f in os.listdir(data_folder) if f.endswith(".pkl")]

    # Dictionaries to store results
    structured_data = {}  # Stores DataFrames per file
    spiking_data = {}  # Stores full spike time data per file

    # Process each file separately
    for pkl_file in pkl_files:
        file_path = os.path.join(data_folder, pkl_file)

        # Load the data
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Extract neuron spike data
        neurons_data = data.get("neurons", [])

        # Process neurons and extract key information
        neurons_list = []
        all_spike_times = {}  # Dictionary to store all spike times per neuron

        for neuron in neurons_data:
            electrode, cluster, spike_times = neuron[:3]  # Extract first 3 elements

            # Ensure spike_times is a list or array
            spike_times = np.array(spike_times) if isinstance(spike_times, list) else spike_times

            # Store spike times for this neuron
            neuron_id = f"Electrode_{electrode}_Cluster_{cluster}"
            all_spike_times[neuron_id] = spike_times  # Stores all spike times for later analysis

            # Append structured neuron data
            neurons_list.append({
                "Electrode": electrode,
                "Cluster": cluster,
                "Spike_Count": len(spike_times),  # Number of spikes
                "First_Spike": spike_times[0] if len(spike_times) > 0 else None,  # First spike time
                "Last_Spike": spike_times[-1] if len(spike_times) > 0 else None  # Last spike time
        
            })

        # Convert to Pandas DataFrame
        df_neurons = pd.DataFrame(neurons_list)
        print(df_neurons.head())
        
        # Generate a unique identifier for this dataset
        file_key = os.path.splitext(pkl_file)[0]  # Remove .pkl extension
        structured_data[file_key] = df_neurons  # Store DataFrame per file
        spiking_data[file_key] = all_spike_times  # Store full spike time dictionary per file

        # Print summary
        print(f"Processed {pkl_file}: {df_neurons.shape[0]} neurons found.")
        print(f"Stored structured data as 'df_neurons_{file_key}' and full spike times as 'neurons_data_{file_key}'.")
        
    return structured_data, spiking_data
