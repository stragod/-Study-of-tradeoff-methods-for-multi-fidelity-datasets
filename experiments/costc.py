# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:07:51 2023

@author: ragha
"""

import psi4
import json
import numpy as np
import time
with open("/mnt/c/Users/ragha/Downloads/qm7b.json", "r") as f:
    data = json.load(f)

molecules = data["molecules"]
sample_size = 1500  # Adjust this number based on your needs and available resources
sample_molecules = np.random.choice(list(molecules.values()), size=sample_size, replace=False)
def run_calculation(molecule, method, basis):
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)


methods = ["HF", "MP2", "CCSD(T)"]
basis_sets = ["sto-3g", "6-31g", "cc-pvdz"]
results = {}

for method in methods:
    for basis in basis_sets:
        key = f"{method}/{basis}"
        results[key] = {"cpu_time": [], "energies": []}

        for mol in sample_molecules:
            start_time = time.time()
            energy = run_calculation(mol, method, basis)
            end_time = time.time()

            cpu_time = end_time - start_time
            results[key]["cpu_time"].append(cpu_time)
            results[key]["energies"].append(energy)

# Calculate average CPU time and memory usage (if applicable) for each method and basis set combination
for key, value in results.items():
    avg_cpu_time = np.mean(value["cpu_time"])
    # avg_memory = ...  # Calculate average memory usage if you want to include it in your analysis
    print(f"{key}: Average CPU time = {avg_cpu_time} seconds")

import matplotlib.pyplot as plt

sample_sizes = [50, 100, 200, 500, 1000,1250,1500,1750,2000,2250,2500,2750,3000,4000,5000,6000,7000]  # Adjust this based on your available resources and time constraints

methods = ["HF", "MP2", "CCSD(T)"]
basis_sets = ["sto-3g", "6-31g", "cc-pvdz"]
fidelity_levels = [("HF", "sto-3g"), ("MP2", "6-31g"), ("CCSD(T)", "cc-pvdz")]

results = {}

for method in methods:
    for basis in basis_sets:
        key = f"{method}/{basis}"
        results[key] = {"sample_size": [], "cpu_time": []}

        for sample_size in sample_sizes:
            sample_molecules = np.random.choice(list(molecules.values()), size=sample_size, replace=False)
            cpu_times = []

            for mol in sample_molecules:
                start_time = time.time()
                energy = run_calculation(mol, method, basis)
                end_time = time.time()

                cpu_time = end_time - start_time
                cpu_times.append(cpu_time)

            avg_cpu_time = np.mean(cpu_times)
            results[key]["sample_size"].append(sample_size)
            results[key]["cpu_time"].append(avg_cpu_time)

fig, ax = plt.subplots()

for method, basis in fidelity_levels:
    key = f"{method}/{basis}"
    cpu_times = results[key]["cpu_time"]
    sample_sizes = results[key]["sample_size"]
    ax.plot(sample_sizes, cpu_times, label=key)

ax.set_xlabel("Sample size")
ax.set_ylabel("Average CPU time (s)")
ax.set_title("Performance comparison of different fidelity levels")
ax.legend()

plt.show()
