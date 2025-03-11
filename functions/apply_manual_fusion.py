import numpy as np
import os
import pickle
from functions.load_dataset import load_dataset

def apply_manual_fusion(datasets, manual_fusion, fusion_file_mapping, raw_dir, processed_dir):
    """
    Applies manual neuron fusion based on predefined groupings
    """
    for dataset_name, dataset in datasets.items(): # loop over datasets
        neurons_data = dataset["neurons"]
        total_neurons = len(neurons_data)
        
        # Get the fusion groups
        fusion_groups = manual_fusion.get(dataset_name, [])
        print("Manual fusion groups for", dataset_name, ":", [sorted(group) for group in fusion_groups])
        
        # Identify neurons to be fused
        indices_to_fuse_all = set()
        for group in fusion_groups:
            indices_to_fuse_all.update(group)
        
        # Keep neurons that arent fused
        neurons_not_fused = [neuron for idx, neuron in enumerate(neurons_data)
                               if idx not in indices_to_fuse_all]
        
        # Perform fusion
        fused_neurons = []
        for group in fusion_groups:
            group_spike_times = np.concatenate([neurons_data[idx][2] for idx in sorted(group)]) # concat spike times across neurons
            group_spike_times.sort()
            
            # Copy of the neuron with smallest index, spike times replaced with newly fused ones
            fused_neuron = neurons_data[min(group)].copy()
            fused_neuron[2] = group_spike_times
            fused_neurons.append(fused_neuron)
        
        # Combine fused and unfused
        filtered_neurons_data = neurons_not_fused + fused_neurons
        # new_count = len(filtered_neurons_data)
        # print(f"Original neuron count: {total_neurons}, New neuron count: {new_count}.")
        # print(f"{filtered_out_count} neurons were fused (removed and replaced with {len(fused_neurons)} fused neurons).")
        
        # Reload the original full data to general structure
        original_file = os.path.join(raw_dir, fusion_file_mapping[dataset_name][0])
        data, _, _ = load_dataset(original_file)
        data["neurons"] = filtered_neurons_data
        
        # Save new files
        output_filename = dataset_name + "_CCfiltered.pkl"
        output_path = os.path.join(processed_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved fused data for {dataset_name} to {output_path}")