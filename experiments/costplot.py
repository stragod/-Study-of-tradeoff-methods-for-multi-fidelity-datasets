# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:35:50 2023

@author: ragha
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import psutil
import os

# Define the three fidelity levels
fidelity_levels = {
    "low": ("HF", "sto-3g"),
    "intermediate": ("MP2", "6-31g"),
    "high": ("CCSD(T)", "cc-pvdz"),
}

# Define the sample sizes to consider
sample_sizes = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]

# Initialize dictionaries to store the CPU times and memory usage
cpu_times = {level: [] for level in fidelity_levels}
memory_usages = {level: [] for level in fidelity_levels}

# Get a list of all molecule IDs
molecule_ids = list(molecules.keys())

# Loop over each sample size
for sample_size in sample_sizes:
    # Randomly select a subset of molecules
    subset_ids = random.sample(molecule_ids, sample_size)

    # Loop over each fidelity level
    for level, (method, basis) in fidelity_levels.items():
        # Initialize lists to store the CPU time and memory usage for this sample size and fidelity level
        cpu_times_sample = []
        memory_usages_sample = []

        # Loop over each molecule in the subset
        for molecule_id in subset_ids:
            # Run the quantum chemistry calculation and measure the CPU time and memory usage
            start_time = time.time()
            start_mem = memory_usage()
            # Replace 'run_calculation' with your actual function
            energy = run_calculation(molecules[molecule_id], method, basis)
            end_time = time.time()
            end_mem = memory_usage()

            # Record the CPU time and memory usage
            cpu_times_sample.append(end_time - start_time)
            memory_usages_sample.append(end_mem - start_mem)

        # Record the average CPU time and memory usage for this sample size and fidelity level
        cpu_times[level].append(np.mean(cpu_times_sample))
        memory_usages[level].append(np.mean(memory_usages_sample))

# Create the plots
for level in fidelity_levels:
    plt.plot(sample_sizes, cpu_times[level], label=f"{level} fidelity")
plt.xlabel("Sample size")
plt.ylabel("Average CPU time (s)")
plt.legend()
plt.show()

for level in fidelity_levels:
    plt.plot(sample_sizes, memory_usages[level], label=f"{level} fidelity")
plt.xlabel("Sample size")
plt.ylabel("Average memory usage (KB)")
plt.legend()
plt.show()
