import numpy as np
import matplotlib.pyplot as plt

def plot_correlogram_matrix(neurons_data, binsize=0.001, limit=0.02):
    num_neurons = len(neurons_data)
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))
    
    for i, neuron_i in enumerate(neurons_data):
        t1 = neuron_i[:3][2]  # Extract spike times

        for j, neuron_j in enumerate(neurons_data):
            t2 = neuron_j[:3][2]  # Extract spike times

            # Compute correlogram
            counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=(i == j), density=False)

            # Ensure counts and bins align correctly
            if len(counts) > len(bins) - 1:
                counts = counts[:-1]

            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot in the matrix
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color='b', alpha=0.7, edgecolor='k')
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels for the first row and first column
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.tight_layout()
    plt.show()
