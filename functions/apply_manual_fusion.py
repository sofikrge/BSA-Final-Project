import numpy as np
import os
import pickle
from functions.load_dataset import load_dataset

def apply_manual_modification(datasets, manual_fusion, manual_deletion, file_mapping, raw_dir, processed_dir):
    """
    Applies manual neuron modifications (fusion and deletion) based on explicitly assigned neuron numbers.
    """
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        neurons_data = dataset["neurons"]
        total_neurons = len(neurons_data)
        # print(f"Total neurons loaded: {total_neurons}")

        # Assign neuron numbers explicitly
        neuron_indices_mapping = {idx: neuron for idx, neuron in enumerate(neurons_data)}

        # Compute deletion indices
        deletion_groups = manual_deletion.get(dataset_name, [])
        deletion_indices = set()
        for group in deletion_groups:
            deletion_indices.update(group)
        print(f"Deletion indices: {sorted(deletion_indices)}")

        # Fusion groups after removing deleted neurons
        fusion_groups_raw = manual_fusion.get(dataset_name, [])
        valid_fusion_groups = []
        for group in fusion_groups_raw:
            filtered_group = sorted(set(group) - deletion_indices)
            if len(filtered_group) >= 2:
                valid_fusion_groups.append(filtered_group)
        # print(f"Valid fusion groups after deletion exclusion: {valid_fusion_groups}")

        # Fusion indices
        fusion_indices = set()
        for group in valid_fusion_groups:
            fusion_indices.update(group)

        # Remaining neurons
        remaining_neurons = [neuron_indices_mapping[idx] for idx in neuron_indices_mapping
                             if idx not in deletion_indices and idx not in fusion_indices]
        # print(f"Remaining neurons count (excluding deletion/fusion): {len(remaining_neurons)}")

        # Fuse neurons explicitly
        fused_neurons = []
        for group in valid_fusion_groups:
            print(f"Fusing neuron group: {group}")
            group_spike_times = np.concatenate([neuron_indices_mapping[idx][2] for idx in sorted(group)])
            group_spike_times.sort()
            fused_neuron = neuron_indices_mapping[min(group)].copy()
            fused_neuron[2] = group_spike_times
            fused_neurons.append(fused_neuron)
        # print(f"Number of neurons after fusion: {len(fused_neurons)}")

        # Final neuron set
        final_neurons = remaining_neurons + fused_neurons
        print(f"Dataset {dataset_name}: Original count: {total_neurons}, final count: {len(final_neurons)}")

        # Load original dataset
        original_file = os.path.join(raw_dir, file_mapping[dataset_name])
        data, _, non_stimuli_time = load_dataset(original_file)
        data["neurons"] = final_neurons
        data["non_stimuli_time"] = non_stimuli_time

        # Save the modified dataset
        output_filename = dataset_name + "_CCFiltered.pkl"
        output_path = os.path.join(processed_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved modified dataset for {dataset_name} to {output_path}")

        # Print explicit final neuron order summary
        final_neuron_order = [idx for idx in neuron_indices_mapping if idx not in deletion_indices and idx not in fusion_indices]
        fused_neuron_indices = [min(group) for group in valid_fusion_groups]
        final_neuron_order += fused_neuron_indices
        #final_neuron_order.sort()
        # print(f"Final neuron order (indices): {final_neuron_order}")

    print("\nSummary: Neuron modifications completed explicitly using assigned neuron numbers.")
