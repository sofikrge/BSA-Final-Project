import numpy as np
import os
import pickle
from functions.load_dataset import load_dataset

def apply_manual_fusion(datasets, manual_fusion, fusion_file_mapping, raw_dir, processed_dir):
    """
    Applies manual neuron fusion based on predefined groupings.
    """
    for dataset_name, dataset in datasets.items():
        neurons_data = dataset["neurons"]
        time_window = dataset["non_stimuli_time"]
        
        print(f"\nProcessing dataset: {dataset_name}")
        total_neurons = len(neurons_data)
        
        # Get the fusion groups for this dataset.
        fusion_groups = manual_fusion.get(dataset_name, [])
        print("Manual fusion groups for", dataset_name, ":", [sorted(group) for group in fusion_groups])
        
        # Create a set of all indices that will be fused (union of all groups).
        indices_to_fuse_all = set()
        for group in fusion_groups:
            indices_to_fuse_all.update(group)
        
        # Keep neurons that are not part of any fusion group.
        neurons_not_fused = [neuron for idx, neuron in enumerate(neurons_data)
                               if idx not in indices_to_fuse_all]
        
        # Now fuse neurons in each fusion group.
        fused_neurons = []
        for group in fusion_groups:
            # Concatenate spike times from all neurons in this group.
            group_spike_times = np.concatenate([neurons_data[idx][2] for idx in sorted(group)])
            group_spike_times.sort()
            
            # Create a new neuron from one of the neurons in the group (using the smallest index).
            fused_neuron = neurons_data[min(group)].copy()
            fused_neuron[2] = group_spike_times  # Replace spike times with the fused spike times.
            fused_neurons.append(fused_neuron)
            print(f"Fused neurons {sorted(group)} into one.")
        
        # The final neuron list is the neurons not fused plus the newly fused neurons.
        filtered_neurons_data = neurons_not_fused + fused_neurons
        new_count = len(filtered_neurons_data)
        filtered_out_count = total_neurons - new_count
        print(f"Original neuron count: {total_neurons}, New neuron count: {new_count}.")
        print(f"{filtered_out_count} neurons were fused (removed and replaced with {len(fused_neurons)} fused neurons).")
        
        # Reload the original full data to preserve metadata.
        original_file = os.path.join(raw_dir, fusion_file_mapping[dataset_name][0])
        data, _, _ = load_dataset(original_file)
        data["neurons"] = filtered_neurons_data
        
        # Build the output filename with a "_filtered" suffix.
        output_filename = dataset_name + "_CCfiltered.pkl"
        output_path = os.path.join(processed_dir, output_filename)
        
        # Save the updated dictionary as a pickle file.
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved fused data for {dataset_name} to {output_path}")